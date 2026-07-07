"""InferenceWorker lifecycle 회귀 — fix19 (worker 사망 감지 respawn 의 기반 계약).

실 torch/ultralytics/멀티프로세스 없이(object.__new__ + fake Process) 두 계약을 잠근다:
  (a) is_alive(): _proc 존재 && 살아있음. _proc 은 외부로 직접 노출하지 않는다.
  (b) stop()→start() respawn: start() 가 state['stop'] 를 리셋해야 respawn 된 프로세스가
      즉시 탈출하지 않는다(선행버그: stop() 이 켠 플래그를 아무도 안 껐다).
"""

from __future__ import annotations

import types

from app.inference.worker import InferenceWorker


class _FakeProc:
    """mp.Process 대역 — start()/is_alive()/join()/terminate() 만 흉내."""

    def __init__(self, alive: bool = False) -> None:
        self._alive = alive
        self.pid = 4321

    def start(self) -> None:
        self._alive = True

    def is_alive(self) -> bool:
        return self._alive

    def join(self, timeout=None) -> None:
        self._alive = False

    def terminate(self) -> None:
        self._alive = False


def test_is_alive_reflects_proc_state():
    """is_alive() = (_proc is not None and _proc.is_alive()) — watchdog 공개 계약."""
    w = object.__new__(InferenceWorker)
    w._proc = None
    assert w.is_alive() is False           # 프로세스 없음

    w._proc = _FakeProc(alive=True)
    assert w.is_alive() is True            # 살아있음

    w._proc._alive = False
    assert w.is_alive() is False           # 죽음


def test_start_after_stop_resets_stop_flag_enabling_respawn():
    """회귀락: stop() 이 state['stop']=True 로 워커를 종료시킨 뒤 start()(respawn)가 이 플래그를
    리셋하지 않으면 새 프로세스가 즉시 stop==True 를 읽고 탈출 → respawn 이 무의미해진다."""
    w = object.__new__(InferenceWorker)
    w._proc = _FakeProc(alive=True)
    w._state = {"stop": False, "model_name": "yolo26n.pt"}
    w.model_name = "yolo26n.pt"
    w.device = None
    w.in_q = None
    w.out_q = None

    created: list[_FakeProc] = []

    def _make(**kwargs) -> _FakeProc:
        p = _FakeProc()
        created.append(p)
        return p

    w._ctx = types.SimpleNamespace(Process=_make, Queue=lambda maxsize: object())

    # 1) stop() — 워커 루프 종료 신호 ON.
    w.stop()
    assert w._state["stop"] is True
    assert w._proc is None

    # 2) start() (respawn) — stop 플래그 리셋 + 새 프로세스 spawn.
    w.start()
    assert w._state["stop"] is False, "start() must reset stop flag so a respawned worker runs"
    assert len(created) == 1
    assert w.is_alive() is True


def test_start_respawn_recreates_queues_preserving_state():
    """fix21(G1): respawn(alive 체크 통과)마다 in_q/out_q 를 새 객체로 재생성한다(손상 큐 회복).
    _state(Manager dict)는 유지 → model/conf/enabled 가 respawn 을 넘어 지속."""
    w = object.__new__(InferenceWorker)
    w._proc = None
    w._state = {"stop": False, "model_name": "yolo26n.pt", "conf_threshold": 0.4, "enabled": True}
    w.model_name = "yolo26n.pt"
    w.device = None
    old_in = object()
    old_out = object()
    w.in_q = old_in
    w.out_q = old_out

    made_queues: list = []

    def _make_queue(maxsize):
        q = object()
        made_queues.append((q, maxsize))
        return q

    created: list = []

    def _make_proc(**kwargs):
        created.append(kwargs)
        return _FakeProc()

    w._ctx = types.SimpleNamespace(Queue=_make_queue, Process=_make_proc)

    w.start()

    # 큐 재생성 — 새 identity (옛 큐와 다름).
    assert w.in_q is not old_in
    assert w.out_q is not old_out
    assert len(made_queues) == 2
    # _state 유지 — model/conf/enabled 가 respawn 넘어 보존.
    assert w._state["model_name"] == "yolo26n.pt"
    assert w._state["conf_threshold"] == 0.4
    assert w._state["enabled"] is True
    assert w._state["stop"] is False
    # Process 는 새(재생성된) 큐를 받는다.
    assert created[0]["args"][0] is w.in_q
    assert created[0]["args"][1] is w.out_q
