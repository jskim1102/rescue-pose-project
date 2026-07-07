"""모든 영상 소스 + 단일 InferenceWorker 의 통합 매니저.

- 캡처 스레드들이 모두 같은 worker 에 frame 제출
- worker 결과는 dispatch 스레드가 source_id 별로 캐싱
- 캡처 스레드는 캐시에서 자기 source_id 의 최신 결과만 조회

rtsp-detection: JPEG/get_frame 경로 제거(영상은 mediamtx WHEP). detections_to_json
은 `frame:{w,h}`(SEAM) 동봉 — frontend BboxOverlay 좌표 스케일용.
"""

from __future__ import annotations

import base64
import json
import logging
import threading
import time
from typing import Optional, Union

import numpy as np
import cv2

from app.config import INFERENCE_INTERVAL, MAX_INFER_PER_SEC, SYNCED_POSE_STREAM_KEYS
from app.inference import FrameRequest, InferenceResult, InferenceWorker
from app.rescue import PersonRescue, RescueTracker
from app.streaming.capture import SourceType, VideoCaptureThread

logger = logging.getLogger("rtsp-streaming.streaming.manager")

INFERENCE_IMGSZ = 640
_IPCAM_SOURCE_PREFIX = "ipcam-"


def _resize_for_inference(frame: np.ndarray, imgsz: int = INFERENCE_IMGSZ) -> np.ndarray:
    """추론 제출 전 프레임을 모델 입력크기로 다운스케일 — mp.Queue IPC pickle 비용 절감
    (1080p 6.2MB/27ms → 640 691KB/1.6ms). 종횡비 보존, 업스케일 금지. YOLO 가 어차피
    내부 letterbox 로 imgsz 축소하므로 정확도 동등. 좌표 SEAM(frame_w/h=req.frame.shape)이
    자동으로 축소 치수를 따라가 프론트 BboxOverlay 가 video.videoWidth/frame.w 로 스케일업."""
    h, w = frame.shape[:2]
    scale = imgsz / max(h, w)
    if scale >= 1.0:
        return frame
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _wants_annotated_frame(source_id: str) -> bool:
    if not source_id.startswith(_IPCAM_SOURCE_PREFIX):
        return False
    stream_key = source_id[len(_IPCAM_SOURCE_PREFIX):]
    return stream_key in SYNCED_POSE_STREAM_KEYS


class StreamManager:
    """비디오 소스(웹캠/IP CAM) + 추론 워커 통합 관리.

    싱글톤으로 사용 (`from app.streaming.manager import manager`).
    서버 lifespan 시작/종료에서 `startup()` / `shutdown()` 호출.
    """

    # dispatch 루프의 idle sleep
    _DISPATCH_IDLE_SEC = 0.01

    # H1 tombstone TTL(초): 최근 삭제된 source_id 를 이 시간 동안 기억해 resurrect 를 막는다.
    # delete_ipcam(commit+mediamtx round-trip)~WS start_capture 경합 창(~1 RTT)보다 넉넉히 크면 된다.
    # stream_key 는 재사용되지 않으므로 만료돼 잊혀도 안전(같은 sid 로 새 카메라가 생기지 않음).
    _TOMBSTONE_TTL = 30.0

    def __init__(self, inference_interval: float = 0.1) -> None:
        self._captures: dict[str, VideoCaptureThread] = {}
        self._lock = threading.Lock()
        # 추론 FPS 제한 — 매 프레임 추론은 GPU 낭비. 기본 10fps (interval 0.1s).
        self._inference_interval = inference_interval

        self._worker = InferenceWorker()
        self._latest_results: dict[str, InferenceResult] = {}
        self._results_lock = threading.Lock()

        # [DEMO-ONLY 임시] 합성 detection override — sid 있으면 실 detection 대신 이걸 WS 로 송출.
        self._demo_results: dict[str, InferenceResult] = {}

        # rescue 판정 — 카메라별 RescueTracker(연속 lying 누적) + 최신 상태 캐시.
        # dispatch 루프가 새 result 마다 update(프레임당 1회); detections_to_json 이 ts 정합 시 소비.
        self._rescue_trackers: dict[str, RescueTracker] = {}
        self._latest_rescue: dict[str, tuple[float, list[PersonRescue]]] = {}
        # rescue 종료 이벤트 버퍼(source_id → ["recovered"|"lost", ...]). WS 직렬화가 drain 소비.
        self._rescue_end_events: dict[str, list[str]] = {}
        self._rescue_lock = threading.Lock()

        # source_id 별 추론 enabled — key 없으면 False 가 기본 (명시적으로 켜야 추론)
        self._per_source_enabled: dict[str, bool] = {}
        # source_id 별 confidence threshold — key 없으면 worker 의 global 값 사용
        self._per_source_conf: dict[str, float] = {}
        # source_id 별 사용 모델 목록.
        #   key 없음(None) → 미설정 = 추론 안 함 (모델을 명시 선택해야 추론)
        #   [] (빈 리스트) → 이 카메라 추론 안 함 (bbox 없음)
        #   [m1, m2, ...] → 해당 모델들 사용 (Phase 1 에선 첫 항목만, Phase 2 에 다중 적용 예정)
        self._per_source_models: dict[str, list[str]] = {}
        # source_id 별 annotated JPEG 생성 여부. URL 기반 자동 sync_pose 판정은 ipcam 라우터가 알고,
        # StreamManager 는 source_id 만 보므로 여기로 동적 설정을 전달한다.
        self._per_source_annotated: dict[str, bool] = {}
        self._per_source_lock = threading.Lock()

        # H1: 최근 삭제 source_id → 삭제 monotonic 시각. start_capture 의 create 분기가 참조해
        # 삭제된 카메라 캡처 resurrection(delete↔WS TOCTOU)을 막는다. self._lock 로 보호.
        self._tombstones: dict[str, float] = {}

        # fix19: worker 프로세스 사망 감지 watchdog — 마지막 respawn monotonic 시각(백오프 기준).
        self._last_worker_respawn: float = 0.0

        self._dispatch_thread: Optional[threading.Thread] = None
        self._dispatch_running = False

    # ── lifecycle ────────────────────────────────────────────────

    def startup(self) -> None:
        """워커 + dispatch 스레드 기동."""
        self._worker.start()
        self._dispatch_running = True
        self._dispatch_thread = threading.Thread(
            target=self._dispatch_loop, daemon=True, name="inference-dispatch"
        )
        self._dispatch_thread.start()
        logger.info("StreamManager started (inference worker + dispatch)")

    def shutdown(self) -> None:
        """모든 캡처 + 워커 + dispatch 정리."""
        # 1) 캡처 강제 종료
        with self._lock:
            captures = list(self._captures.values())
            self._captures.clear()
        for cap in captures:
            cap.force_stop()
            logger.info("Capture %s 강제 종료", cap.source_id)

        # 2) dispatch 종료
        self._dispatch_running = False
        if self._dispatch_thread:
            self._dispatch_thread.join(timeout=2)

        # 3) worker 종료
        self._worker.stop()
        logger.info("StreamManager shutdown 완료")

    # ── 캡처 시작/종료 (라우터에서 호출) ─────────────────────────

    def _is_tombstoned_locked(self, source_id: str) -> bool:
        """self._lock 보유 상태에서 호출 — source_id 가 TTL 내 삭제됐으면 True. 조회된 sid 가 만료됐으면
        청소한다(단, 삭제 후 재조회 안 되는 sid 는 이 lazy 경로로 안 지워짐 → 전역 청소는
        _prune_tombstones_locked 가 remove_capture 에서 담당)."""
        ts = self._tombstones.get(source_id)
        if ts is None:
            return False
        if time.monotonic() - ts >= self._TOMBSTONE_TTL:
            del self._tombstones[source_id]
            return False
        return True

    def _prune_tombstones_locked(self) -> None:
        """self._lock 하 — 만료된 tombstone 전부 청소. stream_key 는 재사용 안 되고 삭제된 sid 는
        start_capture 로 재조회되지 않아 lazy 청소가 안 먹으므로, 삭제 때마다 여기서 전역 청소해
        무한증식을 막는다(dict 는 최근 TTL 창의 삭제분만 유지)."""
        now = time.monotonic()
        for sid in [s for s, ts in self._tombstones.items() if now - ts >= self._TOMBSTONE_TTL]:
            del self._tombstones[sid]

    def start_capture(self, source_id: str, source: SourceType) -> bool:
        """source_id 캡처 시작(**없을 때만** 생성) + 뷰어 1명 증가. open 성공 시 True.

        create-only-if-absent: 엔트리가 이미 있으면 넘어온 `source` 를 **무시하고 기존 엔트리를 재사용**한다.
        URL 권위는 오직 update_ipcam→replace_source 로 단일화되어 있고, WS 핸들러가 넘기는 `source` 는
        `await accept()` 前에 읽혀 stale 일 수 있다(URL 편집 in-flight). 옛 "source 상이 시 교체" 방어망은
        stale WS 가 정확한 new_url 캡처를 옛 url 로 되돌려 모든 뷰어에게 딴 카메라를 지속 노출하는
        F1-무력화 SHOULD 였다(독립 게이트 확인) → 제거. 신규 url 반영은 replace_source 가 담당한다.

        증가(acquire_viewer)는 get/create 와 함께 **manager 락 안에서** 원자 수행 → replace_source 의
        adopt-read/swap 과 직렬화(증가side BLOCKER 제거, 감소side release_viewer 대칭). open(블로킹)은 락 밖.
        open 후 엔트리가 바뀌었으면(orphan: 기동 중 remove/replace) 또는 open 실패면, 증가분을 '현재
        엔트리'에서 release_viewer 로 되돌리고 방금 만든 캡처를 정리한 뒤 False.
        """
        with self._lock:
            cap = self._captures.get(source_id)
            if cap is None:
                if self._is_tombstoned_locked(source_id):
                    # H1: 최근 삭제된 카메라 — resurrect 거부(삭제된 RTSP 를 디코드하지 않는다). WS 는 닫힌다.
                    return False
                cap = VideoCaptureThread(
                    source_id=source_id,
                    source=source,
                    frame_callback=self._submit_frame,
                    inference_interval=self._inference_interval,
                )
                self._captures[source_id] = cap
            cap.acquire_viewer()   # 증가를 get/create 와 원자적으로 (락 하)
            target = cap
        ok = target._ensure_running()
        # orphan(기동 중 remove/replace) 또는 open 실패 → 증가분을 '현재 엔트리'에서 되돌린다.
        rollback_thread = None
        with self._lock:
            current = self._captures.get(source_id)
            orphaned = current is not target
            if orphaned:
                # 우리 증가는 (adopt 로) 현재 엔트리에 접혀 있으니 거기서 감소 (감소side 대칭).
                # NOTE: current 가 adopt-후손이 아닌 경우(remove_capture 로 dict 가 비워진 뒤 다른 WS 가
                # fresh 엔트리 생성)엔 남의 뷰어를 감소시킬 수 있으나, remove_capture 는 delete 전용이고
                # delete sweep(ipcam.py)이 그 fresh 엔트리도 force_stop 하므로 오늘은 무해(생존 피해자 없음).
                # 향후 non-delete remove_capture 호출자(idle-eviction 등)를 추가하면 이 가정 재검토 필요.
                if current is not None:
                    rollback_thread = current.release_viewer()
            elif not ok:
                # 현재 엔트리인데 open 실패 → 우리 증가 되돌림 (WS 는 False 에 stop_capture 를 안 부름).
                rollback_thread = target.release_viewer()
        if orphaned or not ok:
            if orphaned:
                target.force_stop()   # 밀려난 캡처 정리 (untracked 스레드 방지)
            if rollback_thread is not None:
                rollback_thread.join(timeout=5)
            self._recompute_cadence()
            return False
        self._recompute_cadence()
        return True

    def stop_capture(self, source_id: str) -> None:
        # 감소는 manager 락 안에서 '현재 엔트리'에 수행 → replace_source 의 swap 과 직렬화되어
        # 감소가 옛 캡처로 오배정되지 않는다(misdirected-decrement/ref 팽창 방지). join 은 락 밖.
        thread = None
        with self._lock:
            cap = self._captures.get(source_id)
            if cap is not None:
                thread = cap.release_viewer()
        if cap is None:
            return
        if thread is not None:
            thread.join(timeout=5)
        self._recompute_cadence()

    def replace_source(self, source_id: str, new_source: SourceType) -> bool:
        """URL 편집 시 캡처를 새 source 로 원자 교체하거나, 캡처가 없으면(idle 카메라) 새 source 를
        **URL 권위로 기록**한다(idle ref-0 엔트리 신설). 변경했으면 True.

        옛 캡처가 있으면 새 캡처로 락 안에서 원자 교체 + 뷰어 수(ref_count) 이관(adopt_viewers) 후 옛
        캡처 force-stop — 교체 도중 WS 가 연결/해제돼도 decrement 유실·ref 팽창이 없다(부재 창 없음).
        옛 캡처가 없으면(old=None: 뷰어 없는 idle 카메라) no-op 하지 않고 ref-0 idle 엔트리를 신설한다 —
        이렇게 URL 권위를 기록해야, 편집 중 stale 값을 읽은 첫 연결이 create-only-if-absent 로 이
        엔트리(=새 url)를 재사용해 옛 url 로 persistent stale 엔트리를 만들지 않는다(SHOULD-4,
        safety-critical: rescue-pose 로 전파되는 정본. round-2 의 old=None no-op 은 이 self-heal 을
        없앴었다). idle 엔트리는 ref-0·미기동이라 cadence·리소스 영향 없고, 첫 연결 시 open 되거나
        재편집/삭제로 교체·제거된다.
        NOTE(accept): 옛 스레드가 force_stop 직전에 워커에 제출한 프레임이 dispatch 루프에서 pop 이후
        다시 캐시될 수 있어 새 카메라 WS 가 ~1 추론간격 동안 옛 bbox 를 볼 수 있다 — transient 수용.
        """
        with self._lock:
            old = self._captures.get(source_id)
            if old is not None and old.source == new_source:
                return False  # 변화 없음
            if old is None and self._is_tombstoned_locked(source_id):
                # H1 대칭: 삭제된(tombstoned) 카메라엔 idle 엔트리도 신설하지 않는다 (update↔delete
                # 경합 → 삭제 카메라의 dormant resurrection 방지). 살아있는 카메라의 실 교체(old not
                # None)는 삭제가 아니므로 tombstone 무관.
                return False
            new = VideoCaptureThread(
                source_id=source_id,
                source=new_source,
                frame_callback=self._submit_frame,
                inference_interval=self._inference_interval,
            )
            if old is not None:
                new.adopt_viewers(old.ref_count)   # 뷰어 수 원자 이관 (idle 신설이면 ref-0 유지)
            self._captures[source_id] = new    # 엔트리 원자 교체/신설 (부재 창 없음)
        if old is not None:
            old.force_stop()
            with self._results_lock:
                self._latest_results.pop(source_id, None)  # 옛 카메라 검출 잔상 제거
            # ★ 안전 핸드머지: rescue 상태도 폐기 — 옛 카메라의 needs-rescue 가 새 source 로 눌러앉아
            # 허위 구조경보를 내지 않게(OFF-path set_source_inference_enabled 미러). 누락=안전사고.
            with self._rescue_lock:
                self._rescue_trackers.pop(source_id, None)
                self._latest_rescue.pop(source_id, None)
                self._rescue_end_events.pop(source_id, None)
        new._ensure_running()                  # ref>0 이면 1회 open, ref-0(idle 신설)이면 미기동
        # 기동 중 remove/재replace 로 엔트리가 바뀌었으면 방금 만든 new 는 orphan → 정리.
        orphan = None
        with self._lock:
            if self._captures.get(source_id) is not new:
                orphan = new
        if orphan is not None:
            orphan.force_stop()
        self._recompute_cadence()
        return True

    def remove_capture(self, source_id: str) -> None:
        """카메라 삭제 시 캡처를 ref_count 무시하고 완전 제거한다.

        stop_capture 는 ref-count 기반이라 viewer 2+ 면 삭제해도 스레드가 살아 삭제된 카메라를
        계속 디코드한다(F2). 삭제는 force-stop + dict 제거 + 검출/per-source 캐시 정리로 즉시 끝낸다.
        H1: pop 과 함께 tombstone 을 등록해, 아직 start_capture 를 안 부른 in-flight WS 가 삭제된
        카메라를 resurrect 하지 못하게 한다(start_capture create 분기가 self._lock 하에서 확인).
        """
        with self._lock:
            cap = self._captures.pop(source_id, None)
            self._tombstones[source_id] = time.monotonic()  # H1: resurrect 방지 (pop 과 원자)
            self._prune_tombstones_locked()  # 만료분 전역 청소 (무한증식 방지)
        if cap is not None:
            cap.force_stop()
        with self._results_lock:
            self._latest_results.pop(source_id, None)
        # ★ 안전 핸드머지: rescue 상태도 폐기 — 삭제된 카메라의 needs-rescue 가 눌러앉아 허위
        # 구조경보를 내지 않게(OFF-path set_source_inference_enabled 미러). 누락=안전사고.
        with self._rescue_lock:
            self._rescue_trackers.pop(source_id, None)
            self._latest_rescue.pop(source_id, None)
            self._rescue_end_events.pop(source_id, None)
        with self._per_source_lock:
            self._per_source_enabled.pop(source_id, None)
            self._per_source_conf.pop(source_id, None)
            self._per_source_models.pop(source_id, None)
            self._per_source_annotated.pop(source_id, None)
        self._recompute_cadence()

    def _recompute_cadence(self) -> None:
        """활성(is_running) 캡처 수 N 으로 per-camera 제출 간격을 재계산.
        interval = max(INFERENCE_INTERVAL_floor, N / MAX_INFER_PER_SEC).
        총 제출 ≤ 워커 예산 → 다수캠서도 GPU 미포화. 소수캠은 floor(30fps) 유지."""
        with self._lock:
            active = [c for c in self._captures.values() if c.is_running]
        n = len(active)
        if n == 0:
            return
        interval = max(self._inference_interval, n / MAX_INFER_PER_SEC)
        for c in active:
            c.set_inference_interval(interval)
        logger.info("cadence 재계산: 활성 %d캠 → per-camera interval=%.3fs (%.1ffps)", n, interval, 1.0/interval)

    # (get_frame 제거 — JPEG 버퍼 없음. 영상은 mediamtx WHEP, detection 은 좌표 WS)
    # (get_capture_stats 제거 — stats endpoint 가 mediamtx readers 기반으로 바뀌어 死코드였고,
    #  그게 읽던 _inference_ts deque 가 dispatch 루프에서 영구 append 돼 source 당 메모리 누수였다.)

    # ── 캡처 ↔ 워커 bridge ───────────────────────────────────────

    def _submit_frame(self, source_id: str, frame: np.ndarray, captured_at: Optional[float] = None) -> None:
        """캡처 스레드 callback — global AND per-source 둘 다 ON 인 경우만 워커에 제출.

        per-source conf 가 설정돼 있으면 FrameRequest 에 포함 → 워커가 그 값으로 추론.
        없으면 None 이라 워커가 global 값 사용.
        """
        status = self._worker.get_status()
        if not status.get("enabled", True):
            return
        if not self.is_source_inference_enabled(source_id):
            return
        with self._per_source_lock:
            conf = self._per_source_conf.get(source_id)
            models_list = self._per_source_models.get(source_id)  # None or list
            want_annotated = self._per_source_annotated.get(source_id, False)

        # 미설정(None)/빈 리스트([]) = 추론 안 함 — 모델을 명시 선택해야만 추론
        if not models_list:
            return

        # fix13: 가드 통과(추론 ON)한 프레임만 모델 입력크기로 다운스케일 — IPC pickle 비용 절감.
        frame = _resize_for_inference(frame)

        # Phase 2: 모델 list 전체를 worker 에 전달 → 다중 모델 detection 합침
        self._worker.submit(
            FrameRequest(
                source_id=source_id,
                frame=frame,
                timestamp=captured_at if captured_at is not None else time.time(),
                conf_threshold=conf,
                model_names=models_list,  # None or non-empty list
                want_annotated=want_annotated or _wants_annotated_frame(source_id),
            )
        )

    def _get_latest_result(self, source_id: str) -> Optional[InferenceResult]:
        """source_id 의 최신 추론 결과 (없으면 None)."""
        with self._results_lock:
            return self._latest_results.get(source_id)

    # WS 핸들러용 public alias — frontend overlay 가 detections JSON 으로 받음 (§4.19)
    def get_source_latest_detections(self, source_id: str) -> Optional[InferenceResult]:
        # [DEMO-ONLY 임시] 데모 주입이 있으면 실 detection 대신 반환(ts 갱신해 매 폴링 재송출).
        demo = self._demo_results.get(source_id)
        if demo is not None:
            import dataclasses
            return dataclasses.replace(demo, timestamp=time.time())
        return self._get_latest_result(source_id)

    def set_demo_detection(self, source_id: str, result: Optional[InferenceResult]) -> None:
        """[DEMO-ONLY 임시] 합성 detection 주입/해제(None=해제)."""
        if result is None:
            self._demo_results.pop(source_id, None)
        else:
            self._demo_results[source_id] = result

    def _maybe_respawn_worker(self) -> None:
        """worker 프로세스가 죽었으면 respawn 한다 (fix19). 큐/shared state 는 재사용되고
        worker_main 이 새 프로세스에서 모델을 lazy 재로드한다. respawn 폭주를 막으려 직전
        respawn 후 최소 10s 백오프를 둔다. _dispatch_loop(데몬 스레드, 이벤트루프 아님 →
        H2 무관)이 ~5s 주기로 호출한다."""
        if self._worker.is_alive():
            return
        nowm = time.monotonic()
        if nowm - self._last_worker_respawn < 10.0:
            return  # 백오프 — 직전 respawn 후 10s 미경과
        logger.error("Inference worker 프로세스 사망 감지 — respawn 시도")
        self._worker.start()
        self._last_worker_respawn = nowm

    def _dispatch_loop(self) -> None:
        """worker.out_q → _latest_results 캐시 + rescue 판정 + worker 사망 watchdog(fix19). 별도 스레드.

        fix21(G1): while-body 전체를 try/except 로 감싼다 — worker 가 out_q.put 도중 killed(OOM,
        watchdog 의 표적 시나리오)되면 drain_results 가 queue.Empty 가 아니라 EOFError/OSError/
        UnpicklingError 를 던질 수 있고, 그러면 이 스레드가 죽어 결과캐싱+rescue경보+watchdog 이 함께
        죽는다(watchdog 이 감시하던 바로 그 죽음에 자신이 피살). rescue update 예외도 armor 안에 포함해
        dispatch 사망→lying 경보 무음사망을 막는다. 어떤 예외든 살아남아야 한다(watchdog 계약).
        """
        next_worker_check = 0.0
        while self._dispatch_running:
            try:
                # worker 사망 감지 — drain 정상경로와 독립, ~5s 주기 게이트로 저부하.
                nowm = time.monotonic()
                if nowm >= next_worker_check:
                    next_worker_check = nowm + 5.0
                    self._maybe_respawn_worker()

                results = self._worker.drain_results()
                if results:
                    with self._results_lock:
                        for r in results:
                            self._latest_results[r.source_id] = r
                    # rescue 누적 — 새 result(프레임)마다 카메라별 tracker.update. WS 클라이언트
                    # 수/유무와 무관하게 여기서 딱 1회 처리해야 lying 지속이 정확히 쌓인다.
                    # ★ armor try 안 — rescue update 예외로 dispatch 가 죽으면 lying 경보 무음사망.
                    for r in results:
                        self._update_rescue(r)
                else:
                    time.sleep(self._DISPATCH_IDLE_SEC)
            except Exception:
                logger.exception("Dispatch loop 예외 — 복구 후 계속")
                time.sleep(1.0)
                continue
        logger.info("Dispatch loop 종료")

    def _update_rescue(self, result: InferenceResult) -> None:
        """새 추론 result 로 카메라 rescue tracker 갱신 + 상태 캐시(result.timestamp 로 정합)."""
        with self._rescue_lock:
            tracker = self._rescue_trackers.get(result.source_id)
            if tracker is None:
                tracker = RescueTracker()
                self._rescue_trackers[result.source_id] = tracker
            states = tracker.update(result.detections, result.frame_w, result.frame_h)
            self._latest_rescue[result.source_id] = (result.timestamp, states)
            evs = tracker.drain_events()
            if evs:
                buf = self._rescue_end_events.setdefault(result.source_id, [])
                buf.extend(evs)
                del buf[:-20]  # WS 미연결 시 무한증식 방지 — 최근 20건만 보관.

    def get_rescue_states(self, source_id: str, timestamp: float) -> Optional[list[PersonRescue]]:
        """source_id 의 rescue 상태(입력 detection 순서에 정렬) — 주어진 result timestamp 와
        일치할 때만 반환(정합). 불일치(더 새 result 로 갱신)면 None → 직렬화가 rescue 필드 생략."""
        with self._rescue_lock:
            entry = self._latest_rescue.get(source_id)
        if entry is not None and entry[0] == timestamp:
            return entry[1]
        return None

    def drain_rescue_events(self, source_id: str) -> list[str]:
        """source_id 의 rescue 종료 이벤트(recovered/lost) 반환 후 비운다. WS 송출 시 1회 소비."""
        with self._rescue_lock:
            return self._rescue_end_events.pop(source_id, [])

    # ── 추론 제어 (FastAPI 라우터에서 호출) ──────────────────────

    def get_inference_config(self) -> dict:
        return self._worker.get_status()

    def set_inference_enabled(self, enabled: bool) -> None:
        self._worker.set_enabled(enabled)
        if not enabled:
            # OFF 시 캐시 비워서 raw 프레임으로 회귀
            with self._results_lock:
                self._latest_results.clear()
            # rescue 상태도 폐기 — OFF(멈춘 feed)에 needs-rescue 가 눌러앉지 않게. 재개 시 fresh.
            with self._rescue_lock:
                self._rescue_trackers.clear()
                self._latest_rescue.clear()
                self._rescue_end_events.clear()

    def set_inference_model(self, model_name: str) -> None:
        self._worker.set_model(model_name)

    def set_inference_conf_threshold(self, threshold: float) -> None:
        self._worker.set_conf_threshold(threshold)

    # ── per-source 추론 ON/OFF (각 카메라마다 독립적으로 제어) ──

    def is_source_inference_enabled(self, source_id: str) -> bool:
        """key 없으면 False (기본 OFF — 명시적으로 켜야 추론)."""
        with self._per_source_lock:
            return self._per_source_enabled.get(source_id, False)

    def set_source_inference_enabled(self, source_id: str, enabled: bool) -> None:
        """source_id 의 추론 ON/OFF. OFF 시 기존 결과 캐시 비워서 bbox 즉시 사라지게."""
        with self._per_source_lock:
            self._per_source_enabled[source_id] = enabled
        if not enabled:
            with self._results_lock:
                self._latest_results.pop(source_id, None)
            # 이 카메라 rescue 상태도 폐기(OFF → 눌러앉는 needs-rescue 방지).
            with self._rescue_lock:
                self._rescue_trackers.pop(source_id, None)
                self._latest_rescue.pop(source_id, None)
                self._rescue_end_events.pop(source_id, None)
        logger.info("Per-source inference: %s = %s", source_id, enabled)

    def get_source_conf_threshold(self, source_id: str) -> Optional[float]:
        """source_id 의 per-source conf. 없으면 None (= global 사용)."""
        with self._per_source_lock:
            return self._per_source_conf.get(source_id)

    def set_source_conf_threshold(self, source_id: str, conf: float) -> None:
        """source_id 의 per-source conf 설정 (0~1)."""
        conf = max(0.0, min(1.0, float(conf)))
        with self._per_source_lock:
            self._per_source_conf[source_id] = conf
        logger.info("Per-source conf: %s = %.2f", source_id, conf)

    def get_source_models(self, source_id: str) -> Optional[list[str]]:
        """source_id 의 per-source 모델 목록.

        - None: 미설정 = 추론 안 함 (모델 명시 선택 필요)
        - []  : 명시적 추론 안 함
        - [m1, m2, ...]: 해당 모델들 (Phase 1 에선 [0] 만 적용)
        """
        with self._per_source_lock:
            models = self._per_source_models.get(source_id)
            return list(models) if models is not None else None  # 복사 반환

    def set_source_models(self, source_id: str, models: list[str]) -> None:
        """source_id 의 per-source 모델 목록 설정. 빈 리스트면 추론 안 함."""
        with self._per_source_lock:
            self._per_source_models[source_id] = list(models)
        logger.info("Per-source models: %s = %s", source_id, models)

    def set_source_annotated_frame(self, source_id: str, enabled: bool) -> None:
        """source_id 의 annotated JPEG 생성 여부 설정."""
        with self._per_source_lock:
            self._per_source_annotated[source_id] = bool(enabled)
        logger.info("Per-source annotated frame: %s = %s", source_id, enabled)


def detections_to_json(result: InferenceResult) -> str:
    """InferenceResult → WS 로 보낼 JSON 문자열. frontend overlay 가 파싱.

    좌표 keypoints 는 추론 캡처 frame 픽셀 기준. `frame:{w,h}` 동봉(SEAM) — frontend
    KeypointOverlay 가 video.videoWidth/frame.w 스케일로 변환(두 해상도 같으면 identity).
    사람마다 17 COCO keypoint `[[x,y,conf]×17]`, bbox(xyxy)는 미전송.

    계약: item{keypoints:[[x,y,c]×17], model, posture?, rescueNeeded?:bool, lyingSec?:number}.
    posture/rescueNeeded 는 해당될 때만 동봉. rescue 상태는 dispatch 루프가 채운 tracker 결과를
    result.timestamp 로 정합해 가져온다(module 싱글톤 manager). ts 불일치(더 새 result)면 생략.
    """
    rescue_states = manager.get_rescue_states(result.source_id, result.timestamp)
    items = []
    for i, d in enumerate(result.detections):
        item = {
            "class_id": d.class_id,
            "name": d.class_name,
            "conf": d.confidence,
            "keypoints": [[x, y, c] for (x, y, c) in d.keypoints],
            "model": d.model,
        }
        if d.posture:
            item["posture"] = d.posture
        rs = rescue_states[i] if rescue_states is not None and i < len(rescue_states) else None
        if rs is not None:
            if rs.rescue_needed:
                item["rescueNeeded"] = True
            if rs.lying_sec > 0:
                item["lyingSec"] = round(rs.lying_sec, 1)
        items.append(item)
    payload = {
        "type": "detections",
        "timestamp": result.timestamp,
        "frame": {"w": result.frame_w, "h": result.frame_h},
        "items": items,
    }
    if result.annotated_jpeg is not None:
        payload["annotatedFrame"] = (
            "data:image/jpeg;base64,"
            + base64.b64encode(result.annotated_jpeg).decode("ascii")
        )
    # rescue 종료 이벤트(recovered/lost) — 있으면 이번 메시지에 동봉(1회 소비). 프론트 이벤트 로그용.
    rescue_events = manager.drain_rescue_events(result.source_id)
    if rescue_events:
        payload["rescueEvents"] = rescue_events
    return json.dumps(payload, ensure_ascii=False)


# 싱글톤 — main.py 에서 startup/shutdown 호출.
# INFERENCE_INTERVAL(env) 을 캡처 throttle 로 배선 — spec §5: env 로 추론 fps 조절 가능.
manager = StreamManager(inference_interval=INFERENCE_INTERVAL)
