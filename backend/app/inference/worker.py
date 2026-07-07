"""YOLO 추론 워커 — backend 메인 프로세스와 분리된 별도 프로세스에서 동작.

CLAUDE.md §6: "AI 추론은 반드시 별도 프로세스" 원칙 구현.

흐름:
    main process                        worker process (별도)
    ─────────────                       ─────────────────────
    capture thread                       (이 모듈)
       │                                    │
       │  frame ──submit()───► in_q ──────►│  YOLO 추론
       │                                    │
       │  ◄───── out_q ◄──── result ◄──────│
       │
       └─ raw JPEG → WebSocket → 브라우저 (브라우저 canvas 가 bbox 오버레이, §4.19)
"""

from __future__ import annotations

import logging
import math
import multiprocessing as mp
import os
import queue
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import cv2

from app import config

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
# 데이터 구조 (in/out queue 에 실리는 메시지)
# ─────────────────────────────────────────────────────────────────


@dataclass
class Detection:
    """단일 사람 pose 결과 (17 COCO keypoint)."""

    class_id: int
    class_name: str
    confidence: float
    # 사람마다 17개 COCO keypoint. 각 점 = (x, y, per-kpt-conf), 추론 캡처 frame 픽셀 좌표.
    keypoints: list[tuple[float, float, float]]
    # 어떤 모델이 이 detection 을 만들었는지. 다중 모델 추론 시 라벨 prefix 결정에 사용.
    model: str = ""
    # COCO17 keypoints 기반 자세 (standing/sitting/lying). None = keypoints 부재/미분류.
    # tunnel worker._classify_posture 차용(rescue seam) — _parse_results 가 계산해 채운다.
    posture: Optional[str] = None


@dataclass
class FrameRequest:
    """워커에 보낼 프레임 요청. capture thread → worker."""

    source_id: str  # 카메라 식별자 ("webcam-0", "ipcam-<stream_key>")
    frame: np.ndarray  # OpenCV BGR 이미지
    timestamp: float  # time.time() 캡처 시각
    # per-source confidence threshold. None 이면 worker 의 global state 값 사용.
    conf_threshold: Optional[float] = None
    # per-source 모델 목록. None 이면 worker 의 global state model 1개 사용.
    # 리스트면 그 모델들을 모두 돌려 detection 결과 합침 (다중 모델 추론).
    model_names: Optional[list[str]] = None
    # True 면 이 추론 프레임 자체에 keypoint 를 그린 JPEG 를 함께 반환한다.
    # cam2 동기화 표시 같은 allowlist source 에만 켠다.
    want_annotated: bool = False


@dataclass
class InferenceResult:
    """워커 결과. worker → capture thread."""

    source_id: str
    timestamp: float
    detections: list[Detection] = field(default_factory=list)
    # SEAM(하이브리드): YOLO 가 본 프레임 치수(추론 캡처 해상도). detections_to_json 이
    # `frame:{w,h}` 로 동봉 → 프론트 BboxOverlay 가 video.videoWidth/frame_w 스케일(같으면 identity).
    frame_w: int = 0
    frame_h: int = 0
    # want_annotated=True 요청에서만 채워지는 JPEG bytes. 같은 req.frame 에 detections 를 그린 결과.
    annotated_jpeg: Optional[bytes] = None


_KPT_CONF_THRESHOLD = 0.5
_SKELETON_EDGES = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
    (5, 11), (6, 12), (5, 6), (5, 7), (6, 8),
    (7, 9), (8, 10), (1, 2), (0, 1), (0, 2),
    (1, 3), (2, 4), (3, 5), (4, 6),
]
_POSE_PALETTE_RGB = [
    (255, 128, 0), (255, 153, 51), (255, 178, 102), (230, 230, 0), (255, 153, 255),
    (153, 204, 255), (255, 102, 255), (255, 51, 255), (102, 178, 255), (51, 153, 255),
    (255, 153, 153), (255, 102, 102), (255, 51, 51), (153, 255, 153), (102, 255, 102),
    (51, 255, 51), (0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 255),
]
_KPT_COLOR_IDX = [16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]
_LIMB_COLOR_IDX = [9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]
_POSTURE_LABEL_COLOR_BGR = {
    "lying": (77, 72, 229),
    "sitting": (59, 162, 224),
    "standing": (80, 185, 63),
}
_POSTURE_TEXT_COLOR_BGR = (18, 13, 10)


def _clamped_point(x: float, y: float, w: int, h: int) -> tuple[int, int]:
    return (
        int(round(max(0.0, min(float(w - 1), x)))),
        int(round(max(0.0, min(float(h - 1), y)))),
    )


def _rgb_to_bgr(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    r, g, b = rgb
    return (b, g, r)


def _kpt_color_bgr(i: int) -> tuple[int, int, int]:
    return _rgb_to_bgr(_POSE_PALETTE_RGB[_KPT_COLOR_IDX[i]])


def _limb_color_bgr(i: int) -> tuple[int, int, int]:
    return _rgb_to_bgr(_POSE_PALETTE_RGB[_LIMB_COLOR_IDX[i]])


def _draw_posture_badge(
    annotated: np.ndarray,
    det: Detection,
    w: int,
    h: int,
    scale: float,
) -> None:
    posture = (det.posture or "").lower()
    badge_color = _POSTURE_LABEL_COLOR_BGR.get(posture)
    if badge_color is None:
        return

    valid = [(x, y) for x, y, conf in det.keypoints if conf >= _KPT_CONF_THRESHOLD]
    if not valid:
        return

    min_x = min(x for x, _ in valid)
    min_y = min(y for _, y in valid)
    text = posture.upper()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.4, 0.42 * scale)
    thickness = max(1, int(round(scale)))
    (text_w, text_h), _baseline = cv2.getTextSize(text, font, font_scale, thickness)
    pad_x = max(5, int(round(5 * scale)))
    pad_y = max(3, int(round(3 * scale)))
    label_w = text_w + pad_x * 2
    label_h = text_h + pad_y * 2

    x1 = int(round(min_x))
    y1 = int(round(min_y)) - label_h - 2
    x1 = max(0, min(max(0, w - label_w - 1), x1))
    y1 = max(0, min(max(0, h - label_h - 1), y1))
    x2 = min(w - 1, x1 + label_w)
    y2 = min(h - 1, y1 + label_h)

    cv2.rectangle(annotated, (x1, y1), (x2, y2), badge_color, -1)
    cv2.putText(
        annotated,
        text,
        (x1 + pad_x, y1 + pad_y + text_h),
        font,
        font_scale,
        _POSTURE_TEXT_COLOR_BGR,
        thickness,
        lineType=cv2.LINE_AA,
    )


def _encode_annotated_pose_jpeg(frame: np.ndarray, detections: list[Detection]) -> Optional[bytes]:
    """YOLO 가 본 frame 위에 pose 를 직접 그려 JPEG 로 반환한다."""
    if frame.size == 0:
        return None
    annotated = frame.copy()
    h, w = annotated.shape[:2]
    scale = max(1.0, min(w, h) / 600.0)
    line_width = max(2, int(round(2 * scale)))
    point_radius = max(3, int(round(3 * scale)))

    for det in detections:
        if len(det.keypoints) < 17:
            continue
        for edge_idx, (a, b) in enumerate(_SKELETON_EDGES):
            ax, ay, ac = det.keypoints[a]
            bx, by, bc = det.keypoints[b]
            if ac < _KPT_CONF_THRESHOLD or bc < _KPT_CONF_THRESHOLD:
                continue
            cv2.line(
                annotated,
                _clamped_point(ax, ay, w, h),
                _clamped_point(bx, by, w, h),
                _limb_color_bgr(edge_idx),
                line_width,
                lineType=cv2.LINE_AA,
            )
        for kpt_idx, (x, y, conf) in enumerate(det.keypoints):
            if conf < _KPT_CONF_THRESHOLD:
                continue
            cv2.circle(
                annotated,
                _clamped_point(x, y, w, h),
                point_radius,
                _kpt_color_bgr(kpt_idx),
                thickness=-1,
                lineType=cv2.LINE_AA,
            )
        _draw_posture_badge(annotated, det, w, h, scale)
    ok, buf = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    return buf.tobytes() if ok else None


# ─────────────────────────────────────────────────────────────────
# 워커 클래스
# ─────────────────────────────────────────────────────────────────


class InferenceWorker:
    """별도 프로세스로 YOLO 추론을 수행한다.

    사용 예:
        worker = InferenceWorker()                    # YOLO_DEFAULT_MODEL · YOLO_DEVICE 환경변수 자동 사용
        worker.start()
        worker.submit(FrameRequest("webcam-0", frame, time.time()))
        result = worker.get_result()                  # non-blocking
        worker.set_model("yolo26s.pt")                # 런타임 모델 전환
        worker.set_enabled(False)                      # 추론 OFF (raw 스트리밍 회귀)
        worker.stop()
    """

    # 큐 크기 — 최신 프레임만 처리하기 위해 in_q=1, 결과 약간 버퍼링 out_q=8
    _IN_QUEUE_SIZE = 1
    _OUT_QUEUE_SIZE = 8

    def __init__(
        self,
        model_name: Optional[str] = None,
        conf_threshold: Optional[float] = None,
        device: Optional[str] = None,
    ) -> None:
        self.model_name: str = model_name or config.YOLO_DEFAULT_MODEL
        self.conf_threshold: float = float(
            conf_threshold
            if conf_threshold is not None
            else config.YOLO_CONF_THRESHOLD
        )
        # device=None 이면 워커가 torch.cuda.is_available() 로 자동 감지
        self.device: Optional[str] = device or os.getenv("YOLO_DEVICE") or None

        ctx = mp.get_context("spawn")  # CUDA 호환을 위해 spawn 강제 (fork 시 CUDA 초기화 충돌)
        self.in_q: mp.Queue = ctx.Queue(maxsize=self._IN_QUEUE_SIZE)
        self.out_q: mp.Queue = ctx.Queue(maxsize=self._OUT_QUEUE_SIZE)

        # 런타임 제어용 shared state (Manager dict)
        self._manager = ctx.Manager()
        self._state = self._manager.dict()
        self._state["model_name"] = self.model_name
        self._state["conf_threshold"] = self.conf_threshold
        self._state["enabled"] = True
        self._state["stop"] = False

        self._proc: Optional[mp.Process] = None
        self._ctx = ctx

    # ── lifecycle ────────────────────────────────────────────────

    def start(self) -> None:
        """워커 프로세스 spawn. 모델 로딩은 워커 안에서 (메인 프로세스 메모리 절약)."""
        if self._proc is not None and self._proc.is_alive():
            return
        # fix19: stop() 이 켠 stop 플래그를 리셋 — 안 하면 respawn 된 프로세스가 즉시
        # state["stop"]==True 를 읽고 탈출해 watchdog respawn 이 무의미해진다.
        self._state["stop"] = False
        # fix21(G1): respawn 마다 큐 재생성 — worker 가 out_q.put 도중 killed(OOM)되면 mp 파이프가
        # 손상돼 drain_results 가 EOFError/OSError 를 계속 던진다. 새 워커가 손상 큐를 재사용하면
        # 회복 불가 → 재생성이 실제 복구 수단. 옛 큐는 close 안 함(동시 capture submit 이 self.in_q
        # 를 읽는데 close 하면 거기서 예외 → 참조소멸 후 GC 회수). _state(Manager dict)는 유지 →
        # model/conf/enabled 가 respawn 넘어 지속. in-flight submit 이 옛 큐로 가도 무해(버려짐).
        self.in_q = self._ctx.Queue(maxsize=self._IN_QUEUE_SIZE)
        self.out_q = self._ctx.Queue(maxsize=self._OUT_QUEUE_SIZE)
        self._proc = self._ctx.Process(
            target=_worker_main,
            args=(self.in_q, self.out_q, self._state, self.device),
            daemon=True,
            name="yolo-inference-worker",
        )
        self._proc.start()
        logger.info(
            "Inference worker started: pid=%s, model=%s, device=%s",
            self._proc.pid,
            self.model_name,
            self.device or "auto",
        )

    def stop(self) -> None:
        """워커 종료 및 정리."""
        self._state["stop"] = True
        if self._proc is not None:
            self._proc.join(timeout=5)
            if self._proc.is_alive():
                logger.warning("Worker did not exit gracefully, terminating")
                self._proc.terminate()
            self._proc = None
        logger.info("Inference worker stopped")

    def is_alive(self) -> bool:
        """워커 프로세스가 살아있는지 (manager 의 사망 감지 watchdog 용). `_proc` 자체는
        외부로 노출하지 않는다 — 이 불리언만 공개 계약."""
        return self._proc is not None and self._proc.is_alive()

    # ── 메인 → 워커 (제출) ────────────────────────────────────────

    def submit(self, req: FrameRequest) -> None:
        """프레임 제출. 큐 가득차있으면 가장 오래된 것 drop 후 새 것 삽입."""
        try:
            self.in_q.put_nowait(req)
        except queue.Full:
            try:
                self.in_q.get_nowait()  # 오래된 것 drop
            except queue.Empty:
                pass
            try:
                self.in_q.put_nowait(req)
            except queue.Full:
                pass  # 그 사이에 다른 producer 가 채웠을 수 있음. 다음 기회에.

    # ── 워커 → 메인 (결과 수신) ──────────────────────────────────

    def get_result(self, timeout: float = 0.0) -> Optional[InferenceResult]:
        """결과 1건 가져오기. 기본 non-blocking (timeout=0)."""
        try:
            if timeout > 0:
                return self.out_q.get(timeout=timeout)
            return self.out_q.get_nowait()
        except queue.Empty:
            return None

    def drain_results(self) -> list[InferenceResult]:
        """현재 큐의 모든 결과를 비워서 반환 (capture thread 가 한 tick 에 처리)."""
        results: list[InferenceResult] = []
        while True:
            try:
                results.append(self.out_q.get_nowait())
            except queue.Empty:
                break
        return results

    # ── 런타임 제어 (API 에서 호출) ───────────────────────────────

    def set_model(self, model_name: str) -> None:
        """런타임 모델 전환. 워커가 다음 iteration 에서 reload."""
        self._state["model_name"] = model_name
        logger.info("Model switch requested: %s", model_name)

    def set_enabled(self, enabled: bool) -> None:
        """추론 ON/OFF 토글. OFF 면 워커는 in_q 만 비우고 결과 송출 안 함."""
        self._state["enabled"] = bool(enabled)
        logger.info("Inference enabled=%s", enabled)

    def set_conf_threshold(self, threshold: float) -> None:
        """confidence 임계값 변경."""
        self._state["conf_threshold"] = float(threshold)

    def get_status(self) -> dict:
        """현재 상태 (FastAPI `/api/inference/config` GET 용)."""
        return {
            "enabled": bool(self._state.get("enabled", True)),
            "model": str(self._state.get("model_name", self.model_name)),
            "conf_threshold": float(self._state.get("conf_threshold", self.conf_threshold)),
            "device": self.device or "auto",
        }


# ─────────────────────────────────────────────────────────────────
# 워커 프로세스 메인 — module-level 함수 (spawn 으로 picklable)
# ─────────────────────────────────────────────────────────────────


def _worker_main(in_q: mp.Queue, out_q: mp.Queue, state, device_override: Optional[str]) -> None:
    """워커 프로세스 entry point. import 도 여기서 — 메인 프로세스 메모리 절약."""
    # 워커 프로세스 안에서만 import (torch/ultralytics 무거움)
    import torch
    from ultralytics import YOLO

    from app.inference.models_dir import is_preset, resolve_model_path

    # 로깅 (워커는 별도 프로세스라 핸들러 별도 설정 필요할 수 있음)
    worker_logger = logging.getLogger("inference.worker")

    # device 결정
    if device_override:
        device = device_override
    elif torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"
    worker_logger.info("Worker device: %s", device)

    # 모델 cache — 여러 모델을 동시에 GPU 메모리에 보유하여 per-source 모델을 즉시 사용
    # key: 모델 이름 (예: "yolo26n.pt"), value: 로드된 YOLO 객체
    models_cache: dict[str, "YOLO"] = {}

    def get_or_load_model(name: str):
        """name 의 YOLO 모델을 cache 에서 가져오거나 새로 로드. 로드 실패 시 None 반환."""
        if name in models_cache:
            return models_cache[name]
        if not is_preset(name):  # allowlist (codex #1) — preset 외 임의 .pt 로드 거부
            worker_logger.error("거부된 모델 (preset 아님): %s", name)
            return None
        try:
            path = resolve_model_path(name)
            worker_logger.info("Loading YOLO model into cache: %s (path: %s)", name, path)
            m = YOLO(path)
            try:
                m.to(device)
            except Exception as e:
                worker_logger.warning("model.to(%s) failed for %s: %s — keeping CPU", device, name, e)
            models_cache[name] = m
            return m
        except Exception as e:
            worker_logger.error("Model load failed: %s — %s", name, e)
            return None

    # 시작 시 global 기본 모델 미리 로드
    global_model_name = state["model_name"]
    get_or_load_model(global_model_name)

    # 메인 루프
    while not state.get("stop", False):
        # 1) global 기본 모델 전환 요청 — per-source 가 없을 때 fallback
        requested_global = state.get("model_name", global_model_name)
        if requested_global != global_model_name:
            worker_logger.info("Global model switch: %s → %s", global_model_name, requested_global)
            if get_or_load_model(requested_global) is not None:
                global_model_name = requested_global
            else:
                state["model_name"] = global_model_name  # rollback

        # 2) OFF 모드 — 큐만 비우고 휴식
        if not state.get("enabled", True):
            try:
                in_q.get(timeout=0.1)
            except queue.Empty:
                pass
            continue

        # 3) 프레임 가져오기
        try:
            req: FrameRequest = in_q.get(timeout=0.1)
        except queue.Empty:
            continue

        # 4) 이 frame 에 적용할 모델 list 결정 — per-source > global (단일 모델로 폴백)
        target_names = req.model_names if req.model_names else [global_model_name]

        # 5) 각 모델로 추론 → detections 합침 (Phase 2)
        conf = (
            req.conf_threshold
            if req.conf_threshold is not None
            else float(state.get("conf_threshold", 0.5))
        )
        detections: list[Detection] = []
        for target_name in target_names:
            model = get_or_load_model(target_name)
            if model is None:
                # 로드 실패한 모델은 skip — 다른 모델은 계속 처리
                worker_logger.warning("Skipping unavailable model: %s", target_name)
                continue
            try:
                results = model(req.frame, conf=conf, verbose=False)
                detections.extend(_parse_results(results[0], model.names, target_name))
            except Exception as e:
                worker_logger.error("Inference error (%s): %s", target_name, e)
                continue

        # 6) 결과 송출 (detections 가 비어있어도 send — 클라이언트가 raw 표시)
        result = InferenceResult(
            source_id=req.source_id,
            timestamp=req.timestamp,
            detections=detections,
            # SEAM: 추론한 프레임 치수 동봉 (req.frame.shape = (H, W, C)).
            frame_w=int(req.frame.shape[1]),
            frame_h=int(req.frame.shape[0]),
            annotated_jpeg=(
                _encode_annotated_pose_jpeg(req.frame, detections)
                if req.want_annotated
                else None
            ),
        )
        try:
            out_q.put_nowait(result)
        except queue.Full:
            # 결과 큐 가득 차면 오래된 거 drop
            try:
                out_q.get_nowait()
                out_q.put_nowait(result)
            except (queue.Empty, queue.Full):
                pass

    worker_logger.info("Worker exiting")


def _parse_results(result, names: dict, model_name: str = "") -> list[Detection]:
    """ultralytics Results 객체 → Detection 리스트.

    `model_name` 은 다중 모델 추론에서 어떤 모델이 만든 detection 인지 표시하기 위해 사용.
    """
    detections: list[Detection] = []
    kpts = result.keypoints
    if kpts is None or len(kpts) == 0:
        return detections
    xy_arr = kpts.xy.cpu().numpy()  # [N, 17, 2] (x, y) 픽셀 좌표
    kconf = kpts.conf.cpu().numpy() if kpts.conf is not None else None  # [N, 17]
    # person confidence: pose 결과의 boxes.conf 있으면 사용, 없으면 per-kpt conf 평균.
    boxes = result.boxes
    pconf = (
        boxes.conf.cpu().numpy()
        if boxes is not None and boxes.conf is not None
        else None
    )
    n = xy_arr.shape[0]
    for i in range(n):
        keypoints: list[tuple[float, float, float]] = []
        for j in range(xy_arr.shape[1]):
            c = float(kconf[i, j]) if kconf is not None else 0.0
            keypoints.append((float(xy_arr[i, j, 0]), float(xy_arr[i, j, 1]), c))
        if pconf is not None:
            person_conf = float(pconf[i])
        elif kconf is not None:
            person_conf = float(kconf[i].mean())
        else:
            person_conf = 0.0
        posture = _classify_posture(keypoints) if keypoints else None
        detections.append(
            Detection(
                class_id=0,  # pose 단일 class = person
                class_name="person",
                confidence=person_conf,
                keypoints=keypoints,
                model=model_name,
                posture=posture,
            )
        )
    return detections


def _angle_three_points(
    a: list[float], b: list[float], c: list[float]
) -> float:
    """점 b 에서 a-b-c 가 이루는 각도 (degrees)."""
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    dot = ba[0] * bc[0] + ba[1] * bc[1]
    mag_ba = math.hypot(*ba)
    mag_bc = math.hypot(*bc)
    if mag_ba == 0 or mag_bc == 0:
        return 180.0
    cos_angle = max(-1.0, min(1.0, dot / (mag_ba * mag_bc)))
    return math.degrees(math.acos(cos_angle))


def _classify_posture(kpts: list[list[float]]) -> str:
    """COCO 17 keypoints → 자세 분류 (standing / sitting / lying).

    torso 각도 (어깨↔엉덩이 중점 벡터) + 다리 각도 (hip-knee-ankle) 로 판별.
    """
    if len(kpts) < 17:
        return "standing"

    _MIN_C = 0.3
    l_sh, r_sh = kpts[5], kpts[6]
    l_hip, r_hip = kpts[11], kpts[12]
    l_knee, r_knee = kpts[13], kpts[14]
    l_ankle, r_ankle = kpts[15], kpts[16]

    # 어깨·엉덩이 모두 신뢰도 충분해야 torso 각도 계산 가능
    if not all(k[2] > _MIN_C for k in [l_sh, r_sh, l_hip, r_hip]):
        return "standing"

    sh_mid = ((l_sh[0] + r_sh[0]) / 2, (l_sh[1] + r_sh[1]) / 2)
    hip_mid = ((l_hip[0] + r_hip[0]) / 2, (l_hip[1] + r_hip[1]) / 2)

    dx = hip_mid[0] - sh_mid[0]
    dy = hip_mid[1] - sh_mid[1]
    # 수직 = 90°, 수평 = 0° (이미지 좌표계: y 아래로 증가)
    torso_angle = math.degrees(math.atan2(abs(dy), abs(dx)))

    if torso_angle < 35:
        return "lying"

    # standing vs sitting: 다리 꺾임 (hip-knee-ankle 각도) 으로 판별
    leg_angles: list[float] = []
    for hip, knee, ankle in [(l_hip, l_knee, l_ankle), (r_hip, r_knee, r_ankle)]:
        if knee[2] > _MIN_C and ankle[2] > _MIN_C:
            leg_angles.append(_angle_three_points(hip, knee, ankle))

    if leg_angles:
        avg = sum(leg_angles) / len(leg_angles)
        if avg < 130:
            return "sitting"

    return "standing"
