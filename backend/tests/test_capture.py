"""capture 스레드 DECODE cadence throttle 테스트 (codex #4, backend half).

문제: _capture_loop 이 매 iteration 마다 cap.read()(full-fps 디코드)를 호출하고
worker 제출만 _next_submit_ts 로 스로틀했다 → 16 카메라에서 쓰지도 않는 프레임을
디코드하느라 CPU/네트워크 낭비. 스펙은 low-fps DECODE 를 원한다.

결정적(non-flaky, no real sleep/threads): monkeypatch 로 capture 모듈의 clock 을
고정 리스트로 치환, FakeCap 이 grab()/read() 호출마다 clock 을 DT 만큼 전진시키고
공유 iteration 카운터가 MAX 에 도달하면 loop 를 종료시킨다. _capture_loop 를 스레드
없이 직접(동기) 호출한다.
"""

from __future__ import annotations

import os
import app.streaming.capture as capture
from app.streaming.capture import VideoCaptureThread
import numpy as _np
import pytest


def test_capture_loop_throttles_decode(monkeypatch):
    clock = [0.0]
    DT = 0.02  # grab/read 한 번당 clock 전진량(초)
    MAX = 50  # 총 loop iteration 수 (grab+read 합산)
    INTERVAL = 0.1  # inference_interval → 목표 디코드 간격

    # capture 모듈의 clock 을 고정 리스트로 치환 (실제 sleep/실시간 의존 제거).
    monkeypatch.setattr(capture.time, "time", lambda: clock[0])

    collector: list = []

    t = VideoCaptureThread(
        "src-test",
        "rtsp://fake",
        frame_callback=lambda sid, frame, ts: collector.append((sid, frame)),
        inference_interval=INTERVAL,
    )

    counter = {"grabs": 0, "reads": 0, "iters": 0}

    class FakeCap:
        def isOpened(self):
            return True

        def _tick(self):
            # grab/read 각각이 loop iteration 한 번 = clock DT 전진.
            clock[0] += DT
            counter["iters"] += 1
            if counter["iters"] >= MAX:
                t._running = False  # 결정적 종료

        def grab(self):
            counter["grabs"] += 1
            self._tick()
            return True

        def read(self):
            counter["reads"] += 1
            self._tick()
            return True, object()

        def release(self):
            return None

    monkeypatch.setattr(t, "_open_capture", lambda: FakeCap())
    t._running = True
    t._capture_loop(t._generation)  # 동기 직접 호출 (스레드 X)

    # 총 시간 = MAX*DT = 1.0s, INTERVAL=0.1s → 디코드는 ~10 회로 스로틀.
    assert counter["reads"] <= 12, f"decode 스로틀 실패: reads={counter['reads']}"
    # 대부분 프레임은 디코드 없이 grab-drain 되어야 한다.
    assert counter["grabs"] > counter["reads"], (
        f"grab-drain 안 됨: grabs={counter['grabs']} reads={counter['reads']}"
    )
    # 디코드 1 회당 정확히 1 회 제출.
    assert len(collector) == counter["reads"], (
        f"제출/디코드 1:1 아님: submits={len(collector)} reads={counter['reads']}"
    )


def test_open_rtsp_capture_configures_low_latency(monkeypatch):
    """RTSP OpenCV 캡처는 기본적으로 low-latency FFmpeg 옵션과 1-frame buffer 를 요청한다."""
    monkeypatch.delenv("OPENCV_FFMPEG_CAPTURE_OPTIONS", raising=False)
    created = []

    class FakeCap:
        def __init__(self):
            self.set_calls = []

        def set(self, prop, value):
            self.set_calls.append((prop, value))
            return True

    def fake_video_capture(source, backend):
        created.append((source, backend, FakeCap()))
        return created[-1][2]

    monkeypatch.setattr(capture.cv2, "VideoCapture", fake_video_capture)

    cap = VideoCaptureThread("src-test", "rtsp://fake")._open_capture()

    assert created == [("rtsp://fake", capture.cv2.CAP_FFMPEG, cap)]
    opts = os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS", "")
    assert "fflags;nobuffer" in opts
    assert "flags;low_delay" in opts
    assert (capture.cv2.CAP_PROP_BUFFERSIZE, 1) in cap.set_calls


def test_capture_loop_uses_frame_pts_for_callback_timestamp(monkeypatch):
    """OpenCV 가 PTS 를 주면 read 완료 시간이 아니라 프레임 시간 간격으로 timestamp 를 보낸다."""
    clock = [1000.0]
    pts_msec = [1000.0, 1033.0]
    captured_ts = []

    monkeypatch.setattr(capture.time, "time", lambda: clock[0])

    t = VideoCaptureThread(
        "src-test",
        "rtsp://fake",
        frame_callback=lambda sid, frame, ts: captured_ts.append(ts),
        inference_interval=0.0,
    )

    class FakeCap:
        def __init__(self):
            self.reads = 0

        def isOpened(self):
            return True

        def read(self):
            clock[0] += 0.1
            self.reads += 1
            if self.reads >= 2:
                t._running = False
            return True, object()

        def get(self, prop):
            if prop == capture.cv2.CAP_PROP_POS_MSEC:
                return pts_msec[self.reads - 1]
            return 0

        def release(self):
            return None

    monkeypatch.setattr(t, "_open_capture", lambda: FakeCap())
    t._running = True
    t._capture_loop(t._generation)

    assert len(captured_ts) == 2
    assert captured_ts[1] - captured_ts[0] == pytest.approx(0.033, abs=0.005)


def test_capture_loop_reopens_after_consecutive_read_failures(monkeypatch):
    """OpenCV/FFmpeg 가 열린 cap 에서 ret=False 만 반복하면 cap 을 재오픈해 검출 제출을 복구한다."""
    monkeypatch.setattr(capture, "_MAX_CONSECUTIVE_CAPTURE_FAILURES", 3, raising=False)
    monkeypatch.setattr(capture, "_REOPEN_BACKOFF_SEC", 0.0, raising=False)
    monkeypatch.setattr(capture.time, "sleep", lambda _: None)

    clock = [1000.0]
    monkeypatch.setattr(capture.time, "time", lambda: clock[0])

    submitted = []
    t = VideoCaptureThread(
        "src",
        "rtsp://fake",
        frame_callback=lambda sid, frame, ts: submitted.append((sid, frame)),
        inference_interval=0.0,
    )

    class BadCap:
        def __init__(self):
            self.reads = 0
            self.released = False

        def isOpened(self):
            return True

        def read(self):
            self.reads += 1
            clock[0] += 0.01
            # 기존 구현이 재오픈하지 않으면 테스트가 무한 루프에 빠지지 않게 종료한다.
            if self.reads >= 5:
                t._running = False
            return False, None

        def grab(self):
            return False

        def get(self, prop):
            return 0

        def release(self):
            self.released = True

    class GoodCap:
        def __init__(self):
            self.released = False

        def isOpened(self):
            return True

        def read(self):
            clock[0] += 0.01
            t._running = False
            return True, object()

        def grab(self):
            return True

        def get(self, prop):
            return 0

        def release(self):
            self.released = True

    bad = BadCap()
    good = GoodCap()
    opened = [bad, good]
    monkeypatch.setattr(t, "_open_capture", lambda: opened.pop(0))

    t._generation = 1
    t._running = True
    t._capture_loop(1)

    assert len(submitted) == 1
    assert submitted[0][0] == "src"
    assert bad.released is True
    assert good.released is True
    assert opened == []


# ── fix20: 캡처 스레드 좀비 generation fencing ────────────────────
# _ensure_running 이 기동마다 _generation++ 하고 그 gen 을 스레드에 넘긴다. 느리게 열린
# 구세대 스레드(좀비)가 현세대 공유 상태(_cap/_running)를 덮어쓰지 못하게 _capture_loop
# 이 gen 을 fence 한다. 실 스레드/타이밍 없이 gen 값을 직접 넣어 결정적으로 검증한다.


class _GenFakeCap:
    """gen fencing 테스트용 FakeCap — grab/read 호출수 + release 추적."""

    def __init__(self):
        self.grabs = 0
        self.reads = 0
        self.released = False

    def isOpened(self):
        return True

    def grab(self):
        self.grabs += 1
        return True

    def read(self):
        self.reads += 1
        return True, object()

    def get(self, prop):
        return 0

    def release(self):
        self.released = True


def test_capture_loop_stale_generation_does_not_clobber_current(monkeypatch):
    """느리게 열린 구세대(gen1) 스레드가 현세대(gen2)의 _cap/_running 을 덮어쓰지 않는다.
    stale 은 자기 cap 만 release 하고 즉시 종료 → 활성 디코더 1개(gen2), 디코드/제출 0."""
    t = VideoCaptureThread("src", "rtsp://fake", frame_callback=None, inference_interval=0.0)

    # 현세대 = 2 (thread#2 가 소유, 이미 running + _cap 세팅됨).
    cap2 = _GenFakeCap()
    t._generation = 2
    t._running = True
    t._cap = cap2
    t._open_event.set()

    # 뒤늦게 open 을 마친 구세대 thread#1(gen=1) 을 동기 실행.
    cap1 = _GenFakeCap()
    monkeypatch.setattr(t, "_open_capture", lambda: cap1)
    t._capture_loop(1)

    # 클로버 0 — 공유 상태는 여전히 thread#2 것.
    assert t._cap is cap2
    assert t._running is True
    assert t._open_event.is_set() is True
    # stale 은 자기 cap 만 정리하고 루프 미진입(디코드/제출 0).
    assert cap1.released is True
    assert cap2.released is False
    assert cap1.grabs == 0 and cap1.reads == 0


def test_capture_loop_current_generation_opens_and_submits(monkeypatch):
    """fix20 회귀: 단일 세대(gen1) 정상 open — _cap 세팅 후 프레임 제출, 종료 시 정리."""
    clock = [0.0]
    monkeypatch.setattr(capture.time, "time", lambda: clock[0])
    submitted = []
    t = VideoCaptureThread(
        "src",
        "rtsp://fake",
        frame_callback=lambda sid, frame, ts: submitted.append(sid),
        inference_interval=0.0,
    )

    class FakeCap(_GenFakeCap):
        def read(self):
            self.reads += 1
            clock[0] += 0.05
            if self.reads >= 3:
                t._running = False
            return True, object()

    cap = FakeCap()
    monkeypatch.setattr(t, "_open_capture", lambda: cap)
    # _ensure_running 이 하는 gen 세팅 모사: gen=1, running.
    t._generation = 1
    t._running = True
    t._capture_loop(1)

    assert submitted == ["src", "src", "src"]     # 3 프레임 제출
    assert cap.released is True                     # 종료 시 release
    assert t._cap is None                           # finally(현세대) 정리
    assert t._running is False


# ─── StreamManager 소스 lifecycle: replace_source(F1) / remove_capture(F2) ───
# CEO #268 (계보 SHARED 버그 정본 수정):
#  F1(fix15): URL 편집 시 옛 source 캡처가 안정 source_id 로 재사용돼 잔존 → replace_source 가
#             옛 캡처 force-stop + 새 source 재생성(뷰어 ref 보존)으로 교체.
#  F2(fix16): 삭제 시 stop_capture 1회(ref-count 감소)만 → viewer 2+면 스레드 생존(계속 YOLO)
#             → remove_capture 가 ref 무시 force-stop + 완전 제거.
# 실 RTSP/스레드 없이 lifecycle 만 결정적으로 모사 (test_cadence_guard 의 object.__new__ idiom).

import threading

import app.streaming.manager as manager_mod
from app.streaming.manager import StreamManager


class _FakeThread:
    """VideoCaptureThread 대역 — 실 RTSP/스레드 없이 ref_count lifecycle 만 모사."""

    def __init__(self, source_id, source, *, frame_callback=None, inference_interval=0.0):
        self.source_id = source_id
        self.source = source
        self._ref = 0
        self._running = False
        self.force_stopped = False
        self._dead = False

    def acquire_viewer(self):
        self._ref += 1

    def start(self):
        self.acquire_viewer()
        self._ensure_running()
        return True

    def _ensure_running(self):
        if self._dead:
            return False
        if self._running or self._ref <= 0:
            return self._running
        self._running = True
        return True

    def adopt_viewers(self, n):
        self._ref = max(0, n)

    def release_viewer(self):
        self._ref = max(0, self._ref - 1)
        if self._ref > 0 or not self._running:
            return None
        self._running = False
        return None  # fake: 실제 스레드 없음 (join 불필요)

    def stop(self):
        self.release_viewer()

    def force_stop(self):
        self.force_stopped = True
        self._dead = True
        self._running = False

    @property
    def is_running(self):
        return self._running

    @property
    def ref_count(self):
        return self._ref

    def set_inference_interval(self, interval):
        pass


def _bare_manager():
    """InferenceWorker/dispatch 안 띄우고 lifecycle 상태만 갖춘 매니저."""
    m = object.__new__(StreamManager)
    m._captures = {}
    m._lock = threading.Lock()
    m._inference_interval = 0.033
    m._latest_results = {}
    m._results_lock = threading.Lock()
    m._per_source_enabled = {}
    m._per_source_conf = {}
    m._per_source_models = {}
    m._per_source_annotated = {}
    m._per_source_lock = threading.Lock()
    # rescue-pose adaptation: replace_source/remove_capture 의 rescue pop 핸드머지가 이 속성들을
    # 참조하므로 bare 매니저에도 갖춰야 AttributeError 없이 lifecycle 이 돈다(정본과의 divergence).
    m._rescue_trackers = {}
    m._latest_rescue = {}
    m._rescue_end_events = {}
    m._rescue_lock = threading.Lock()
    m._tombstones = {}  # H1 tombstone (CEO #289/#295) — start_capture/replace_source create 분기 참조
    return m


class _FakeWorker:
    def __init__(self):
        self.submitted = []

    def get_status(self):
        return {"enabled": True}

    def submit(self, req):
        self.submitted.append(req)


def test_submit_frame_requests_annotated_frame_only_for_synced_ipcam(monkeypatch):
    """cam2 같은 동기화 allowlist source 만 worker 에 annotated JPEG 를 요청한다."""
    monkeypatch.setattr(manager_mod, "SYNCED_POSE_STREAM_KEYS", {"rescue_pose__ipcam-cam2"})
    m = _bare_manager()
    worker = _FakeWorker()
    m._worker = worker
    sid = "ipcam-rescue_pose__ipcam-cam2"
    m._per_source_enabled[sid] = True
    m._per_source_models[sid] = ["yolo26n-pose.pt"]

    m._submit_frame(sid, _np.zeros((360, 640, 3), dtype=_np.uint8), captured_at=1.0)

    assert worker.submitted[-1].want_annotated is True


def test_submit_frame_keeps_nvr_default_without_annotated_frame(monkeypatch):
    """allowlist 밖(NVR/cam1) source 는 기존 좌표-only 경로를 유지한다."""
    monkeypatch.setattr(manager_mod, "SYNCED_POSE_STREAM_KEYS", {"rescue_pose__ipcam-cam2"})
    m = _bare_manager()
    worker = _FakeWorker()
    m._worker = worker
    sid = "ipcam-rescue_pose__ipcam-nvr"
    m._per_source_enabled[sid] = True
    m._per_source_models[sid] = ["yolo26n-pose.pt"]

    m._submit_frame(sid, _np.zeros((360, 640, 3), dtype=_np.uint8), captured_at=1.0)

    assert worker.submitted[-1].want_annotated is False


def test_submit_frame_requests_annotated_frame_for_dynamic_source_setting(monkeypatch):
    """URL 자동분류 결과는 stream_manager per-source 설정으로 worker 요청에 반영된다."""
    monkeypatch.setattr(manager_mod, "SYNCED_POSE_STREAM_KEYS", set())
    m = _bare_manager()
    worker = _FakeWorker()
    m._worker = worker
    sid = "ipcam-rescue_pose__ipcam-auto"
    m._per_source_enabled[sid] = True
    m._per_source_models[sid] = ["yolo26n-pose.pt"]

    m.set_source_annotated_frame(sid, True)
    m._submit_frame(sid, _np.zeros((360, 640, 3), dtype=_np.uint8), captured_at=1.0)

    assert worker.submitted[-1].want_annotated is True


# ── F2 fix16: remove_capture ─────────────────────────────────────

def test_remove_capture_force_stops_regardless_of_ref_count(monkeypatch):
    """viewer 2명(ref=2) 이어도 remove_capture 는 즉시 force-stop + 완전 제거."""
    monkeypatch.setattr(manager_mod, "VideoCaptureThread", _FakeThread)
    m = _bare_manager()
    m.start_capture("s", "rtsp://a")  # ref 1
    m.start_capture("s", "rtsp://a")  # ref 2 (뷰어 2명)
    cap = m._captures["s"]
    assert cap.ref_count == 2 and cap.is_running

    m.remove_capture("s")

    assert cap.force_stopped is True
    assert not cap.is_running
    assert "s" not in m._captures


def test_stop_capture_ref_counted_leaves_thread_running_with_two_viewers(monkeypatch):
    """대비(버그 근원): 기존 stop_capture 1회 = ref 감소만 → viewer 2 → 삭제해도 스레드 생존.
    remove_capture(F2 fix)가 이 누수를 없앤다. (stop_capture 자체는 WS 해제용이라 불변)"""
    monkeypatch.setattr(manager_mod, "VideoCaptureThread", _FakeThread)
    m = _bare_manager()
    m.start_capture("s", "rtsp://a")
    m.start_capture("s", "rtsp://a")  # ref 2
    cap = m._captures["s"]

    m.stop_capture("s")  # delete 가 예전에 하던 1회 감소

    assert cap.ref_count == 1
    assert cap.is_running  # ← 누수: 삭제된 카메라 스레드가 계속 YOLO


def test_remove_capture_clears_per_source_settings(monkeypatch):
    """카메라 소멸 시 per-source enabled/conf/models 캐시도 정리."""
    monkeypatch.setattr(manager_mod, "VideoCaptureThread", _FakeThread)
    m = _bare_manager()
    m.start_capture("s", "rtsp://a")
    m.set_source_conf_threshold("s", 0.7)
    m.set_source_models("s", ["yolo"])
    m.set_source_inference_enabled("s", False)

    m.remove_capture("s")

    assert m.get_source_conf_threshold("s") is None
    assert m.get_source_models("s") is None
    assert m.is_source_inference_enabled("s") is False  # 키 제거 → 기본값(OFF) 복귀


def test_remove_capture_absent_is_noop(monkeypatch):
    monkeypatch.setattr(manager_mod, "VideoCaptureThread", _FakeThread)
    m = _bare_manager()
    m.remove_capture("missing")  # 예외 없이 통과


# ── F1 fix15: replace_source + start_capture 하드닝 ───────────────

def test_replace_source_swaps_url_and_preserves_viewers(monkeypatch):
    """URL 편집 시 옛 캡처 force-stop + 새 source 재생성, 뷰어 ref 보존(연결 WS 재개)."""
    monkeypatch.setattr(manager_mod, "VideoCaptureThread", _FakeThread)
    m = _bare_manager()
    m.start_capture("s", "rtsp://old")
    m.start_capture("s", "rtsp://old")  # 뷰어 2명
    old = m._captures["s"]
    assert old.source == "rtsp://old" and old.ref_count == 2

    changed = m.replace_source("s", "rtsp://new")

    assert changed is True
    assert old.force_stopped is True  # 옛 스레드 종료
    new = m._captures["s"]
    assert new is not old
    assert new.source == "rtsp://new"  # 새 카메라
    assert new.ref_count == 2  # 뷰어 보존


def test_replace_source_noop_when_same_url(monkeypatch):
    monkeypatch.setattr(manager_mod, "VideoCaptureThread", _FakeThread)
    m = _bare_manager()
    m.start_capture("s", "rtsp://x")
    cap = m._captures["s"]
    assert m.replace_source("s", "rtsp://x") is False
    assert m._captures["s"] is cap  # 불필요 재생성 없음


def test_replace_source_creates_idle_entry_when_absent(monkeypatch):
    """SHOULD-4(CEO #276/277): idle 카메라(캡처 없음) URL 편집 시 no-op 대신 ref-0 idle 엔트리를
    신설해 새 URL 을 권위로 기록한다. (옛 old=None no-op 이면 False + 엔트리 없음)."""
    monkeypatch.setattr(manager_mod, "VideoCaptureThread", _FakeThread)
    m = _bare_manager()
    changed = m.replace_source("s", "rtsp://new")
    assert changed is True
    assert m._captures["s"].source == "rtsp://new"      # URL 권위 기록
    assert m._captures["s"].ref_count == 0              # idle
    assert not m._captures["s"].is_running              # 미기동


def test_idle_camera_url_edit_first_connect_uses_new_url(monkeypatch):
    """SHOULD-4 회귀락(CEO #276/277): idle 카메라 URL 편집 후 stale 첫 연결이 와도 NEW url 을 쓴다
    (replace_source 가 URL 권위를 기록 → create-only-if-absent 가 그 엔트리 재사용). 버그(old=None
    no-op)로 되돌리면 첫 연결이 stale url 로 persistent stale 엔트리를 만들어 FAIL."""
    monkeypatch.setattr(manager_mod, "VideoCaptureThread", _FakeThread)
    m = _bare_manager()
    assert "s" not in m._captures                       # idle: 캡처 없음
    m.replace_source("s", "rtsp://new")                 # URL 편집 (idle)
    # stale 값(옛 url)을 읽은 첫 연결
    started = m.start_capture("s", "rtsp://stale-old")
    assert started is True
    assert m._captures["s"].source == "rtsp://new"      # NEW url 사용 (stale 무시)
    assert m._captures["s"].ref_count == 1
    assert m._captures["s"].is_running


def test_replace_source_evicts_stale_detection_cache(monkeypatch):
    """교체 시 옛 source 의 최신 검출 캐시를 비운다(다른 카메라 bbox 잔상 방지)."""
    monkeypatch.setattr(manager_mod, "VideoCaptureThread", _FakeThread)
    m = _bare_manager()
    m.start_capture("s", "rtsp://old")
    m._latest_results["s"] = object()  # 옛 검출
    m.replace_source("s", "rtsp://new")
    assert "s" not in m._latest_results


def test_replace_source_no_viewers_swaps_without_starting(monkeypatch):
    """활성 뷰어 0(ref 0)이면 옛 캡처를 새 source 로 원자 교체하되 기동은 안 함
    (다음 연결이 reuse-open). 엔트리는 present-but-idle."""
    monkeypatch.setattr(manager_mod, "VideoCaptureThread", _FakeThread)
    m = _bare_manager()
    m.start_capture("s", "rtsp://old")
    m.stop_capture("s")  # ref 0
    old = m._captures["s"]
    changed = m.replace_source("s", "rtsp://new")
    assert changed is True
    assert old.force_stopped is True
    new = m._captures["s"]
    assert new is not old and new.source == "rtsp://new"
    assert new.ref_count == 0 and not new.is_running  # 뷰어 없음 → 기동 안 함


def test_replace_source_no_lost_decrement_on_concurrent_disconnect(monkeypatch):
    """BLOCKER 회귀락: 교체 도중(옛 캡처 force_stop 시점)에 뷰어가 끊겨도 엔트리가 항상 존재해
    decrement 가 유실되지 않는다 → 새 캡처 ref 팽창 없음. (옛 del-후-loop 설계면 ref=2 로 영구 누수)"""
    monkeypatch.setattr(manager_mod, "VideoCaptureThread", _FakeThread)
    m = _bare_manager()
    m.start_capture("s", "rtsp://old")
    m.start_capture("s", "rtsp://old")  # 뷰어 2명 (ref 2)
    old = m._captures["s"]

    orig_force = old.force_stop
    def force_then_one_viewer_leaves():
        orig_force()
        m.stop_capture("s")  # 교체 창에서 WS 1명 해제 — 엔트리(new)가 있어야 반영됨
    monkeypatch.setattr(old, "force_stop", force_then_one_viewer_leaves)

    m.replace_source("s", "rtsp://new")

    new = m._captures["s"]
    assert new is not old
    assert new.ref_count == 1, f"decrement 유실/팽창 (BLOCKER 재발): ref={new.ref_count}"
    assert new.is_running  # 뷰어 1 남음 → 기동


def test_stop_capture_decrements_current_entry_after_replace(monkeypatch):
    """SHOULD 회귀락: stop_capture 는 manager 락 하 '현재 엔트리'에 감소 → 교체 후에도 새 캡처를
    올바로 감소(옛 캡처로 오배정 안 함). misdirected-decrement/ref 팽창 방지."""
    monkeypatch.setattr(manager_mod, "VideoCaptureThread", _FakeThread)
    m = _bare_manager()
    m.start_capture("s", "rtsp://old")
    m.start_capture("s", "rtsp://old")  # ref 2
    m.replace_source("s", "rtsp://new")  # new adopts 2
    new = m._captures["s"]
    assert new.ref_count == 2

    m.stop_capture("s")  # 현재 엔트리(new) 감소
    assert new.ref_count == 1
    m.stop_capture("s")
    assert new.ref_count == 0
    assert not new.is_running  # 0 → 종료


def test_start_capture_cleans_orphan_when_entry_swapped_during_start(monkeypatch):
    """SHOULD 회귀락: open(_ensure_running) 도중 remove/replace 로 엔트리가 바뀌면 방금 만든 캡처는
    orphan → force_stop + False 반환(어떤 조회로도 못 멈추는 untracked 스레드 방지)."""
    fired = []
    holder = {}

    class _InjectFake(_FakeThread):
        def _ensure_running(self):
            r = super()._ensure_running()
            if not fired:  # 최초 1회만: open 직후 엔트리 제거(remove 경합 모사)
                fired.append(1)
                holder["target"] = self
                _mgr[0]._captures.pop(self.source_id, None)
            return r

    monkeypatch.setattr(manager_mod, "VideoCaptureThread", _InjectFake)
    m = _bare_manager()
    _mgr = [m]
    started = m.start_capture("s", "rtsp://x")
    assert started is False  # orphan 감지 → False
    assert "s" not in m._captures
    # leak guard 락인: 밀려난 target 이 force_stop 돼야 untracked 디코드 스레드가 안 남는다
    # (start_capture 의 target.force_stop() 을 pass 로 바꾸면 이 단언이 FAIL).
    assert holder["target"].force_stopped is True


def test_replace_source_cleans_orphan_when_removed_during_open(monkeypatch):
    """replace_source 의 orphan.force_stop leak guard 락인: 새 캡처 open 중 remove 로 엔트리가 비면
    방금 만든 new 는 orphan → force_stop (guard 를 pass 로 바꾸면 FAIL — untracked 스레드 방지)."""
    fired = []
    holder = {}

    class _InjectFake(_FakeThread):
        def _ensure_running(self):
            r = super()._ensure_running()
            if fired and "new" not in holder:  # replace 가 만든 new 의 open 시점에만
                holder["new"] = self
                _mgr[0]._captures.pop(self.source_id, None)  # 기동 중 remove 경합
            return r

    monkeypatch.setattr(manager_mod, "VideoCaptureThread", _InjectFake)
    m = _bare_manager()
    _mgr = [m]
    m.start_capture("s", "rtsp://old")  # 엔트리 E (fired empty → inject 안 함)
    fired.append(1)                     # 이제부터 inject 활성
    m.replace_source("s", "rtsp://new")  # new._ensure_running 에서 remove → new orphan
    assert holder["new"].force_stopped is True
    assert "s" not in m._captures


def test_start_capture_no_phantom_ref_on_orphan_during_open(monkeypatch):
    """BLOCKER 회귀락(증가side, CEO #274): start_capture 의 open 중 URL 교체(replace)가 끼어
    target 이 밀려나면(orphan-return-False), 증가한 뷰어를 '현재 엔트리'에서 되돌려 ref 팽창(유령
    디코드 스레드)이 없다. 옛 설계(cap.start ref++ 락 밖 + orphan 시 current 미감소)면 new.ref=1 유령."""
    m = None
    arm = {"on": False, "done": False}

    class _ReplaceOnOpen(_FakeThread):
        def _ensure_running(self):
            # X 가 u1 을 open 하려는 순간(현재 엔트리일 때) URL 교체가 끼어든다.
            if arm["on"] and not arm["done"] and self.source == "rtsp://u1" \
                    and m._captures.get(self.source_id) is self:
                arm["done"] = True
                m.replace_source("s", "rtsp://u2")
            return super()._ensure_running()

    monkeypatch.setattr(manager_mod, "VideoCaptureThread", _ReplaceOnOpen)
    m = _bare_manager()
    m.start_capture("s", "rtsp://u1")  # E: 뷰어1 running (arm off)
    m.stop_capture("s")               # E: ref0 idle, 엔트리 잔존
    assert m._captures["s"].ref_count == 0

    arm["on"] = True
    started = m.start_capture("s", "rtsp://u1")  # X: acquire E→1, open중 replace→orphan
    assert started is False
    new = m._captures["s"]
    assert new.source == "rtsp://u2"
    assert new.ref_count == 0, f"증가side 유령 ref (BLOCKER 재발): {new.ref_count}"


def test_start_capture_reuses_entry_ignoring_stale_source(monkeypatch):
    """SHOULD 회귀락(CEO #274 + 게이트): create-only-if-absent — 엔트리가 있으면 넘어온 source 를
    무시하고 기존 엔트리 재사용(url 권위=replace_source 단일화). stale WS 가 옛 url 로 정확한 캡처를
    덮지 않는다(F1 무력화 SHOULD 제거). 옛 mismatch-replace 설계면 여기서 stale 캡처로 교체돼 FAIL."""
    monkeypatch.setattr(manager_mod, "VideoCaptureThread", _FakeThread)
    m = _bare_manager()
    m.start_capture("s", "rtsp://current")
    cap = m._captures["s"]
    started = m.start_capture("s", "rtsp://stale-old")  # 다른(stale) source 재연결
    assert started is True
    assert m._captures["s"] is cap          # 교체 없음 (정확한 캡처 보존)
    assert cap.source == "rtsp://current"   # 옛 url 로 안 덮임 (F1 유지)
    assert cap.ref_count == 2
    assert not cap.force_stopped


def test_start_capture_open_fail_rolls_back_viewer(monkeypatch):
    """SHOULD-tdd 회귀락(게이트): open 실패(_ensure_running False, non-orphan)면 증가한 뷰어를
    되돌린다 — WS 는 False 에 stop_capture 를 안 부르므로 롤백 안 하면 유령 +1 누수. rollback 줄 지우면 FAIL."""
    class _OpenFailFake(_FakeThread):
        def _ensure_running(self):
            return False  # RTSP 도달 불가

    monkeypatch.setattr(manager_mod, "VideoCaptureThread", _OpenFailFake)
    m = _bare_manager()
    started = m.start_capture("s", "rtsp://down")
    assert started is False
    assert m._captures["s"].ref_count == 0, f"open 실패 롤백 안 됨(유령 +1): {m._captures['s'].ref_count}"


def test_start_capture_open_fail_joins_rollback_thread(monkeypatch):
    """release_viewer 가 0-crossing 스레드를 반환하면 start_capture 가 join 한다 (join 핸드오프 커버)."""
    joined = {"n": 0}

    class _JoinThread:
        def join(self, timeout=None):
            joined["n"] += 1

    class _OpenFailJoinFake(_FakeThread):
        def _ensure_running(self):
            return False

        def release_viewer(self):
            self._ref = max(0, self._ref - 1)
            self._running = False
            return _JoinThread()

    monkeypatch.setattr(manager_mod, "VideoCaptureThread", _OpenFailJoinFake)
    m = _bare_manager()
    m.start_capture("s", "rtsp://down")
    assert joined["n"] == 1


def test_stop_capture_joins_thread_on_last_viewer(monkeypatch):
    """마지막 뷰어 stop 시 release_viewer 반환 스레드를 stop_capture 가 join (join 핸드오프 커버)."""
    joined = {"n": 0}

    class _JoinThread:
        def join(self, timeout=None):
            joined["n"] += 1

    class _JoinFake(_FakeThread):
        def release_viewer(self):
            self._ref = max(0, self._ref - 1)
            if self._ref > 0 or not self._running:
                return None
            self._running = False
            return _JoinThread()

    monkeypatch.setattr(manager_mod, "VideoCaptureThread", _JoinFake)
    m = _bare_manager()
    m.start_capture("s", "rtsp://x")  # ref1 running
    m.stop_capture("s")               # 마지막 뷰어 → release 반환 스레드 join
    assert joined["n"] == 1


def test_start_capture_reuses_when_same_source(monkeypatch):
    """동일 source 재연결은 기존 스레드 재사용(ref 증가) — 하위호환."""
    monkeypatch.setattr(manager_mod, "VideoCaptureThread", _FakeThread)
    m = _bare_manager()
    m.start_capture("s", "rtsp://x")
    cap = m._captures["s"]
    m.start_capture("s", "rtsp://x")
    assert m._captures["s"] is cap
    assert cap.ref_count == 2


def test_ref_count_property_reflects_internal_counter():
    """실 VideoCaptureThread.ref_count property 가 내부 _ref_count 를 락 하에 그대로 반영
    (replace_source 의 뷰어 보존 근거 — manager 테스트는 _FakeThread 라 실 property 를 직접 확인)."""
    t = VideoCaptureThread("s", "rtsp://x")
    assert t.ref_count == 0
    t._ref_count = 3
    assert t.ref_count == 3


# ── rescue-pose 안전 핸드머지 락인 (정본 divergence: canonical 엔 rescue 없음) ──
# 삭제/URL편집 시 rescue tracker/상태를 폐기하지 않으면 소멸/교체된 카메라의 needs-rescue 가
# 눌러앉아 허위 구조경보를 낸다(안전사고). remove_capture/replace_source 의 pop 을 지우면 FAIL.

def test_remove_capture_pops_rescue_state(monkeypatch):
    """★ 안전: 삭제 시 rescue tracker/최신상태 폐기 — 삭제된 카메라 needs-rescue 잔존(허위경보) 방지."""
    monkeypatch.setattr(manager_mod, "VideoCaptureThread", _FakeThread)
    m = _bare_manager()
    m.start_capture("s", "rtsp://a")
    m._rescue_trackers["s"] = object()
    m._latest_rescue["s"] = (1.0, [])

    m.remove_capture("s")

    assert "s" not in m._rescue_trackers
    assert "s" not in m._latest_rescue


def test_replace_source_pops_rescue_state(monkeypatch):
    """★ 안전: URL 교체 시 옛 카메라 rescue 상태 폐기 — 새 source 로 needs-rescue 잔상 이월(허위경보) 방지."""
    monkeypatch.setattr(manager_mod, "VideoCaptureThread", _FakeThread)
    m = _bare_manager()
    m.start_capture("s", "rtsp://old")
    m._rescue_trackers["s"] = object()
    m._latest_rescue["s"] = (1.0, [])

    m.replace_source("s", "rtsp://new")

    assert "s" not in m._rescue_trackers
    assert "s" not in m._latest_rescue


# ── H1: delete↔WS start_capture TOCTOU tombstone (CEO #289/#295) ───────

def test_remove_capture_tombstones_source_blocking_resurrection(monkeypatch):
    """H1 회귀락: remove_capture 가 tombstone 등록 → 삭제 직후 in-flight(stale) WS 의 start_capture 가
    삭제된 카메라 캡처를 resurrect 하지 못한다(create 거부, False). tombstone 체크를 지우면 재생성돼 FAIL."""
    monkeypatch.setattr(manager_mod, "VideoCaptureThread", _FakeThread)
    m = _bare_manager()
    m.start_capture("s", "rtsp://x")     # 캡처 존재
    m.remove_capture("s")                # 삭제 → tombstone 등록
    assert "s" not in m._captures

    started = m.start_capture("s", "rtsp://x")  # 삭제 직후 stale WS 연결
    assert started is False              # resurrect 거부
    assert "s" not in m._captures        # 재생성 안 됨


def test_tombstone_expires_allowing_recreate(monkeypatch):
    """tombstone 은 TTL 후 만료 → create 허용 + 만료 항목 청소(무한증식 방지)."""
    monkeypatch.setattr(manager_mod, "VideoCaptureThread", _FakeThread)
    m = _bare_manager()
    m.start_capture("s", "rtsp://x")
    m.remove_capture("s")
    # tombstone 을 TTL+1 초 이전으로 조작(만료 모사)
    m._tombstones["s"] = m._tombstones["s"] - (StreamManager._TOMBSTONE_TTL + 1)

    started = m.start_capture("s", "rtsp://x")
    assert started is True
    assert m._captures["s"].source == "rtsp://x"
    assert "s" not in m._tombstones      # 만료 항목 청소됨


def test_tombstone_does_not_block_reuse_of_live_entry(monkeypatch):
    """tombstone 은 create 분기만 막는다 — 이미 살아있는 엔트리 재사용(2번째 뷰어)은 통과."""
    monkeypatch.setattr(manager_mod, "VideoCaptureThread", _FakeThread)
    m = _bare_manager()
    m.start_capture("s", "rtsp://x")     # 엔트리 존재
    m._tombstones["s"] = float("inf")    # (부자연스럽지만) tombstone 있어도 재사용은 통과해야
    started = m.start_capture("s", "rtsp://x")
    assert started is True
    assert m._captures["s"].ref_count == 2


def test_replace_source_refuses_idle_create_for_tombstoned(monkeypatch):
    """H1 대칭 회귀락(gate): replace_source 의 idle-create(old=None)도 tombstone 된 sid 엔 엔트리를
    신설하지 않는다 (update↔delete 경합으로 삭제된 카메라 dormant resurrection 방지). 체크 없으면
    idle 엔트리 신설돼 FAIL."""
    monkeypatch.setattr(manager_mod, "VideoCaptureThread", _FakeThread)
    m = _bare_manager()
    m.start_capture("s", "rtsp://x")
    m.remove_capture("s")               # 삭제 → tombstone
    assert "s" not in m._captures

    changed = m.replace_source("s", "rtsp://new")  # 삭제된 카메라 URL 편집 경합
    assert changed is False             # idle 엔트리 신설 거부
    assert "s" not in m._captures


def test_remove_capture_prunes_expired_tombstones(monkeypatch):
    """H1 회귀락(gate): remove_capture 가 만료된 tombstone 을 전역 청소 → 무한증식 방지
    (삭제된 sid 는 재조회 안 돼 lazy 청소가 안 먹으므로)."""
    monkeypatch.setattr(manager_mod, "VideoCaptureThread", _FakeThread)
    m = _bare_manager()
    m._tombstones["old1"] = -1e9        # 확실히 만료(now - (-1e9) ≫ TTL)
    m._tombstones["old2"] = -1e9
    m.start_capture("s", "rtsp://x")
    m.remove_capture("s")               # 새 tombstone 등록 + 만료 전역 청소
    assert "old1" not in m._tombstones and "old2" not in m._tombstones  # 만료 청소됨
    assert "s" in m._tombstones         # 방금 것은 유지(만료 전)


def test_ensure_running_refuses_after_force_stop(monkeypatch):
    """H1 근본 회귀락(gate): force_stop 은 영구 dead 표시 → _ensure_running 이 캡처를 다시 못 켠다
    (reuse-branch 에서 delete 가 lock-밖 open 과 경합해도 삭제된 RTSP 재디코드 안 함). _dead 없으면
    _running False + ref>0 라 스레드를 새로 만들어(재기동) FAIL."""
    from app.streaming.capture import VideoCaptureThread

    opened = {"n": 0}

    class _FakeCap:
        def isOpened(self):
            return False

        def grab(self):
            return False

        def read(self):
            return False, None

        def release(self):
            pass

    def _fake_open():
        opened["n"] += 1
        return _FakeCap()

    t = VideoCaptureThread("s", "rtsp://x")
    monkeypatch.setattr(t, "_open_capture", _fake_open)
    t._ref_count = 1
    t.force_stop()                        # 영구 dead
    assert t._ensure_running() is False   # 재기동 거부
    assert t._thread is None              # 스레드 아예 안 만듦 (short-circuit)
    assert opened["n"] == 0               # _open_capture 미호출 → 삭제 카메라 재디코드 없음


def test_start_capture_refuses_resurrect_when_force_stopped_during_open(monkeypatch):
    """H1 신규 race 회귀락: reuse-branch(엔트리 존재)에서 2번째 뷰어의 open(_ensure_running) 도중 그
    캡처가 force_stop(delete 경합)되면 _dead 로 _ensure_running False → 삭제 카메라를 resurrect 하지
    않는다. _dead 가드 없으면 죽은 캡처가 다시 running 으로 되살아나 FAIL."""
    holder = {}

    class _DieDuringOpen(_FakeThread):
        def _ensure_running(self):
            # reuse 2번째 뷰어 open 시점에 delete 가 이 캡처를 force_stop(_dead) — 경합 모사.
            if holder.get("arm") and not holder.get("fired"):
                holder["fired"] = True
                self.force_stop()
            return super()._ensure_running()

    monkeypatch.setattr(manager_mod, "VideoCaptureThread", _DieDuringOpen)
    m = _bare_manager()
    m.start_capture("s", "rtsp://x")     # 엔트리 running (arm off → inject 안 함)
    cap = m._captures["s"]
    holder["arm"] = True

    started = m.start_capture("s", "rtsp://x")  # 2번째 뷰어 reuse; open 중 force_stop→_dead
    assert started is False              # _ensure_running False → resurrect 거부
    assert not cap.is_running            # 죽은 캡처 재기동 안 됨


# ── fix19: worker 사망 감지 watchdog (respawn + 10s 백오프) ────────
# _dispatch_loop(데몬 스레드, 이벤트루프 아님 → H2 무관)이 ~5s 주기로 _maybe_respawn_worker 를
# 호출한다. 여기선 그 respawn 결정(is_alive→ERROR→start→백오프)만 결정적으로 잠근다.


def test_dispatch_watchdog_respawns_dead_worker_with_backoff():
    """worker 가 죽었으면 respawn(start) 하되, 직전 respawn 후 10s 백오프 창에선 재기동하지
    않는다(respawn 폭주 방지). 창이 지나면 다시 respawn 허용."""
    m = _bare_manager()
    m._last_worker_respawn = 0.0

    starts = {"n": 0}

    class _DeadWorker:
        def is_alive(self):
            return False

        def start(self):
            starts["n"] += 1

    m._worker = _DeadWorker()

    # 첫 tick — 죽음 감지 → respawn 1회.
    m._maybe_respawn_worker()
    assert starts["n"] == 1

    # 10s 백오프 창 내 두 번째 tick — respawn 안 함.
    m._maybe_respawn_worker()
    assert starts["n"] == 1

    # 백오프(10s) 경과 후엔 다시 respawn 허용.
    m._last_worker_respawn -= 11.0
    m._maybe_respawn_worker()
    assert starts["n"] == 2


def test_dispatch_watchdog_leaves_live_worker_untouched():
    """살아있는 worker 는 respawn 하지 않는다 (정상 상태 무개입)."""
    m = _bare_manager()
    m._last_worker_respawn = 0.0

    starts = {"n": 0}

    class _LiveWorker:
        def is_alive(self):
            return True

        def start(self):
            starts["n"] += 1

    m._worker = _LiveWorker()
    m._maybe_respawn_worker()
    assert starts["n"] == 0


def test_dispatch_loop_survives_drain_exception_and_resumes(monkeypatch):
    """fix21(G1): drain_results 가 EOFError(큐 손상)를 던져도 _dispatch_loop 이 죽지 않고
    복구 후 계속 tick + 캐싱 재개한다. worker 가 out_q.put 중 killed → mp 파이프 손상 시나리오
    (watchdog 이 감시하는 바로 그 죽음에 dispatch 스레드가 동반사망하면 안 된다)."""
    m = _bare_manager()
    m._last_worker_respawn = 0.0
    m._dispatch_running = True

    class _StubResult:
        source_id = "camX"

    res = _StubResult()
    calls = {"n": 0}

    class _Worker:
        def is_alive(self):
            return True  # watchdog no-op

        def drain_results(self):
            calls["n"] += 1
            if calls["n"] == 1:
                raise EOFError("corrupted mp pipe")   # 첫 tick — 큐 손상 예외
            if calls["n"] >= 3:
                m._dispatch_running = False            # 결정적 종료
            return [res] if calls["n"] == 2 else []

    m._worker = _Worker()
    # sleep no-op — armor 의 1.0s 및 idle 대기 제거(빠르고 결정적).
    monkeypatch.setattr(manager_mod.time, "sleep", lambda s: None)

    m._dispatch_loop()  # 예외에도 안 죽고 돌다가 _dispatch_running=False 로 종료

    assert calls["n"] >= 3                          # 1차 예외 뒤에도 계속 tick(생존)
    assert m._latest_results.get("camX") is res     # 복구 후 캐싱 재개
