"""TDD — RescueTracker (rescue core, U1 rule).

규칙: 한 사람의 posture 가 `lying` 로 연속 ≥ N초 지속되면 needs-rescue.
사람 식별(D5) = centroid 최근접 association (track ID 없음). 감지 gap>expiry → track 폐기.
시계 주입(Clock) 으로 실제 sleep 없이 ≥N초 경과를 시뮬레이션한다.

실제 감지는 ~10fps 연속이므로 시간 경과는 expiry 보다 작은 step 으로 update 를 계속
호출해(=계속 감지) 시뮬한다(`feed`). 단일 점프 = "감지 gap"(별도 테스트).
"""
from dataclasses import dataclass
from typing import Optional

from app.rescue import RescueTracker


class Clock:
    """주입용 가짜 단조시계 — advance 로 시간 경과를 시뮬(sleep 없음)."""

    def __init__(self, t: float = 0.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


@dataclass
class P:
    """update 입력용 최소 person — keypoints((x,y,c)) + posture (worker.Detection 과 duck-호환)."""

    keypoints: list
    posture: Optional[str]


def person(cx: float, cy: float, posture: Optional[str] = "lying") -> P:
    # 17 keypoint 를 모두 (cx,cy,0.9) 로 → centroid = (cx,cy) (association 결정적).
    return P(keypoints=[(float(cx), float(cy), 0.9)] * 17, posture=posture)


FRAME = {"frame_w": 1920, "frame_h": 1080}
N = 10.0
STEP = 1.0  # < expiry(3.0): 연속 감지 시뮬 (track 이 gap 으로 안 끊기게).


def _tracker(clk: Clock) -> RescueTracker:
    return RescueTracker(lying_threshold_s=N, expiry_s=3.0, now=clk)


def feed(tr: RescueTracker, clk: Clock, persons, seconds: float, step: float = STEP):
    """persons 를 step 간격으로 계속 감지시키며 총 `seconds` 경과. 마지막 states 반환.
    (실제 ~10fps 연속 감지 시뮬 — step<expiry 라 track 이 gap 으로 안 끊긴다.)"""
    t_end = clk.t + seconds
    states = tr.update(persons, **FRAME)
    while clk.t < t_end - 1e-9:
        clk.advance(min(step, t_end - clk.t))
        states = tr.update(persons, **FRAME)
    return states


def test_lying_below_threshold_no_rescue():
    clk = Clock()
    tr = _tracker(clk)
    s = feed(tr, clk, [person(100, 100, "lying")], 5.0)
    assert s[0].rescue_needed is False
    assert 4.9 <= s[0].lying_sec <= 5.1


def test_lying_reaches_threshold_fires():
    clk = Clock()
    tr = _tracker(clk)
    s = feed(tr, clk, [person(100, 100, "lying")], N)
    assert s[0].rescue_needed is True
    assert s[0].lying_sec >= N


def test_posture_change_resets():
    clk = Clock()
    tr = _tracker(clk)
    feed(tr, clk, [person(100, 100, "lying")], 8.0)  # 8초 lying, 아직 미발화
    s = tr.update([person(100, 100, "standing")], **FRAME)  # posture 변경 → 리셋
    assert s[0].rescue_needed is False
    assert s[0].lying_sec == 0.0
    s = feed(tr, clk, [person(100, 100, "standing")], 20.0)  # standing 지속
    assert s[0].rescue_needed is False


def test_detection_gap_expires_state():
    clk = Clock()
    tr = _tracker(clk)
    feed(tr, clk, [person(100, 100, "lying")], 8.0)  # 8초 lying
    # 감지 gap — 사람이 expiry(3s) 넘게 안 보임 → track 폐기.
    clk.advance(4.0)
    s = tr.update([], **FRAME)
    assert s == []
    # 재등장 — 누적이 8초를 이어받지 않고 fresh 시작.
    s = tr.update([person(100, 100, "lying")], **FRAME)
    assert s[0].rescue_needed is False
    assert s[0].lying_sec < 1.0
    # fresh 10초 연속 누적 후 발화.
    s = feed(tr, clk, [person(100, 100, "lying")], N)
    assert s[0].rescue_needed is True


def test_per_person_independence():
    clk = Clock()
    tr = _tracker(clk)
    # 멀리 떨어진 두 사람: A=오래 lying, B=standing.
    people = [person(100, 100, "lying"), person(1600, 950, "standing")]
    s = feed(tr, clk, people, N + 1.0)
    assert s[0].rescue_needed is True   # A 장기 lying → 발화
    assert s[1].rescue_needed is False  # B standing → 미발화
    assert s[1].lying_sec == 0.0


def test_lying_sec_reported_before_threshold():
    # lyingSec 은 임계 전에도 누적 보고(UI 우선순위 패널용).
    clk = Clock()
    tr = _tracker(clk)
    s = feed(tr, clk, [person(100, 100, "lying")], 3.0)
    assert 2.9 <= s[0].lying_sec <= 3.1
    assert s[0].rescue_needed is False


def test_reset_clears_state():
    clk = Clock()
    tr = _tracker(clk)
    feed(tr, clk, [person(100, 100, "lying")], 8.0)
    tr.reset()  # 추론 OFF/카메라 제거 시뮬
    s = tr.update([person(100, 100, "lying")], **FRAME)  # 폐기 후 fresh
    assert s[0].lying_sec == 0.0
    assert s[0].rescue_needed is False


def test_moving_lying_person_stays_associated():
    # 누운 사람이 약간 움직여도(프레임마다 centroid 소폭 이동) 같은 track 으로 누적 유지.
    clk = Clock()
    tr = _tracker(clk)
    states = tr.update([person(100, 100, "lying")], **FRAME)
    x = 100.0
    while clk.t < N - 1e-9:
        clk.advance(STEP)
        x += 5.0  # 프레임당 5px 이동 (max_dist=0.2*대각선≈440 이내 → 연관 유지)
        states = tr.update([person(x, 100, "lying")], **FRAME)
    assert states[0].rescue_needed is True
    assert states[0].lying_sec >= N


# ── rescue 종료 이벤트 (recovered=기립), (b) 배선 ──

def test_recovered_event_on_standup():
    clk = Clock()
    tr = _tracker(clk)
    feed(tr, clk, [person(100, 100, "lying")], N)       # rescue active
    assert tr.drain_events() == []                       # 발화 중엔 종료 이벤트 없음
    tr.update([person(100, 100, "standing")], **FRAME)   # 기립 → 회복
    assert tr.drain_events() == ["recovered"]
    assert tr.drain_events() == []                        # 소비 후 비움


def test_no_event_on_expiry():
    # 대상 소실(lost) 제거 — rescue active track 이 만료돼도 종료 이벤트 없음(옮김/센서끊김 구분 불가).
    clk = Clock()
    tr = _tracker(clk)
    feed(tr, clk, [person(100, 100, "lying")], N)        # rescue active
    clk.advance(3.1)                                      # > expiry(3.0)
    tr.update([], **FRAME)                                # 감지 소실 → track 만료
    assert tr.drain_events() == []


def test_no_event_if_never_rescue_active():
    clk = Clock()
    tr = _tracker(clk)
    feed(tr, clk, [person(100, 100, "lying")], 5.0)      # <N → 미발화(active 아님)
    clk.advance(3.1)
    tr.update([], **FRAME)                                # 만료되지만 active 였던 적 없음
    assert tr.drain_events() == []
