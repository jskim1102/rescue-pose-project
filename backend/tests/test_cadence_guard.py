"""fix14: _recompute_cadence — 활성 캠 수 N 으로 per-camera 제출 간격 자동 하향.

interval = max(INFERENCE_INTERVAL_floor, N / MAX_INFER_PER_SEC).
가드는 fps 를 *낮추기만* 한다(interval 을 올림) — floor(30fps) 위로는 절대 안 올림.
실 GPU/RTSP 없이 순수 수학 + 배선만 검증(fake capture 주입).
"""

import threading

import pytest

from app.config import INFERENCE_INTERVAL, MAX_INFER_PER_SEC
from app.streaming.manager import StreamManager


class FakeCapture:
    """VideoCaptureThread 대역 — is_running 속성 + set_inference_interval 기록만."""

    def __init__(self, running: bool = True) -> None:
        self.is_running = running
        self.interval: float | None = None

    def set_inference_interval(self, interval: float) -> None:
        self.interval = interval


def _mgr(caps: dict) -> StreamManager:
    """InferenceWorker(Manager 프로세스) spawn 없이 _recompute_cadence 만 실 검증.

    __init__ 우회(object.__new__) — 가드가 실제로 읽는 3개 속성만 세팅.
    """
    m = object.__new__(StreamManager)
    m._lock = threading.Lock()
    m._captures = caps
    m._inference_interval = INFERENCE_INTERVAL
    return m


def _expected(n: int) -> float:
    return max(INFERENCE_INTERVAL, n / MAX_INFER_PER_SEC)


@pytest.mark.parametrize("n", [1, 2, 4, 8, 16])
def test_interval_scales_with_active_count(n):
    caps = {f"c{i}": FakeCapture() for i in range(n)}
    _mgr(caps)._recompute_cadence()
    exp = _expected(n)
    for c in caps.values():
        assert c.interval == pytest.approx(exp)


def test_n1_stays_at_floor():
    """N=1 → 1/52 < 0.033 이라 floor(INFERENCE_INTERVAL) 유지 — fps 를 cap 위로 안 올림."""
    c = FakeCapture()
    _mgr({"c0": c})._recompute_cadence()
    assert c.interval == pytest.approx(INFERENCE_INTERVAL)


def test_guard_only_lowers_fps_never_raises_above_cap():
    """모든 N 에서 interval >= floor (fps <= 30fps cap)."""
    for n in range(1, 17):
        caps = {f"c{i}": FakeCapture() for i in range(n)}
        _mgr(caps)._recompute_cadence()
        for c in caps.values():
            assert c.interval >= INFERENCE_INTERVAL


def test_all_active_receive_same_interval():
    caps = {f"c{i}": FakeCapture() for i in range(5)}
    _mgr(caps)._recompute_cadence()
    vals = {c.interval for c in caps.values()}
    assert len(vals) == 1


def test_inactive_captures_excluded():
    """is_running=False 캡처는 N 에서 제외 — 활성 2대만 세어 2/52 적용."""
    active = {f"a{i}": FakeCapture(running=True) for i in range(2)}
    idle = {f"i{i}": FakeCapture(running=False) for i in range(3)}
    caps = {**active, **idle}
    _mgr(caps)._recompute_cadence()
    exp = _expected(2)
    for c in active.values():
        assert c.interval == pytest.approx(exp)
    for c in idle.values():
        assert c.interval is None  # 비활성엔 setter 미호출


def test_zero_active_is_noop():
    """활성 0 → no-op(크래시 없음, setter 미호출)."""
    idle = {"x": FakeCapture(running=False)}
    _mgr(idle)._recompute_cadence()
    assert idle["x"].interval is None
    _mgr({})._recompute_cadence()  # 빈 dict 도 크래시 없음
