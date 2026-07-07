"""per-camera rescue-need 판정 (rescue core).

규칙(U1): 한 사람의 posture 가 `lying` 로 **연속 ≥ N초**(기본 10, env knob) 지속되면 needs-rescue.
단위 = 사람. lying-즉시·넘어짐 전이 감지 없음. 알림은 UI 전용(U2, 여기선 상태만 산출).

사람 식별(D5): worker 는 track ID 없이 plain `model(frame)` — lineage parity 로 worker 추론 경로를
건드리지 않는다. 대신 여기서 프레임별 사람 detection 을 **centroid 최근접 association** 으로 track 에
연관해 사람별 연속 lying 지속을 누적한다(누워있는 사람은 정지 → 연관이 견고). 감지 gap 이 expiry 를
넘으면 track 을 폐기(stale/멈춘 feed 에 rescue 상태가 눌러앉지 않도록 — rescue-safety).

시계는 주입(`now`)한다 — 테스트가 sleep 없이 ≥N초 경과를 시뮬레이션.
"""
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import Callable, Optional, Sequence


def _env_float(name: str, default: float) -> float:
    """양수 env float 파싱 — 미설정/비숫자/비양수면 default."""
    try:
        v = float(os.getenv(name) or "")
        return v if v > 0 else default
    except (TypeError, ValueError):
        return default


# ── knobs (env or default constant) ──
# needs-rescue 임계 N — 연속 lying 지속(초). env RESCUE_LYING_THRESHOLD_S 로 조절.
DEFAULT_LYING_THRESHOLD_S: float = _env_float("RESCUE_LYING_THRESHOLD_S", 10.0)
# track 만료(초) — 이 시간 넘게 안 보이면 track 폐기(감지 gap → 상태 리셋). env RESCUE_EXPIRY_S.
DEFAULT_EXPIRY_S: float = _env_float("RESCUE_EXPIRY_S", 3.0)
# centroid 연관 최대 거리 = 프레임 대각선 × 이 비율. 정지한 누운 사람엔 충분히 넉넉.
DEFAULT_MATCH_FRACTION: float = 0.2

# centroid 에 포함할 keypoint 최소 conf(c > 이 값). worker 가 이미 사람 conf 를 적용하므로 0.
_MIN_KP_CONF = 0.0


@dataclass
class PersonRescue:
    """한 detection(프레임 내 사람)의 rescue 상태 — update 입력 순서에 정렬해 반환."""

    rescue_needed: bool
    lying_sec: float  # 현재 연속 lying 지속(초). lying 아니면 0.0.


@dataclass
class _Track:
    cx: float
    cy: float
    lying_since: Optional[float]  # 연속 lying 시작 시각(now 단위). lying 아니면 None.
    last_seen: float
    rescue_active: bool = False  # 현재 needs-rescue 상태(종료 사유 판별용: 회복 vs 소실).


def _centroid(keypoints: Sequence[Sequence[float]]) -> Optional[tuple[float, float]]:
    """conf>임계 keypoint 들의 평균 좌표(사람 위치). 유효 점 없으면 None."""
    sx = sy = 0.0
    n = 0
    for kp in keypoints:
        if kp[2] > _MIN_KP_CONF:
            sx += kp[0]
            sy += kp[1]
            n += 1
    return (sx / n, sy / n) if n else None


class RescueTracker:
    """카메라 1대의 rescue 상태 추적기. 프레임(detections batch)마다 update() 호출."""

    def __init__(
        self,
        lying_threshold_s: float = DEFAULT_LYING_THRESHOLD_S,
        expiry_s: float = DEFAULT_EXPIRY_S,
        match_fraction: float = DEFAULT_MATCH_FRACTION,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        self._threshold = lying_threshold_s
        self._expiry = expiry_s
        self._match_fraction = match_fraction
        self._now = now
        self._tracks: list[_Track] = []
        # needs-rescue track 종료 이벤트("recovered"=기립). update 마다 누적, drain 으로 소비.
        self._pending_events: list[str] = []

    def update(self, persons, frame_w: float = 0.0, frame_h: float = 0.0) -> list[PersonRescue]:
        """이번 프레임의 사람 detection 을 처리 → 입력 순서에 정렬된 PersonRescue 리스트.

        persons: `.keypoints`((x,y,c) 시퀀스) + `.posture`(Optional[str]) 를 가진 객체들.
        """
        t = self._now()

        # 1) 만료 — 오래 안 보인 track 폐기(감지 gap → 상태 리셋, stale feed 안전).
        #    대상 소실(lost) 제거 — 만료돼도 종료 이벤트 없음(옮김/센서끊김 구분 불가).
        alive: list[_Track] = []
        for tr in self._tracks:
            if (t - tr.last_seen) <= self._expiry:
                alive.append(tr)
        self._tracks = alive

        cents = [_centroid(p.keypoints) for p in persons]
        postures = [getattr(p, "posture", None) for p in persons]

        diag = math.hypot(frame_w, frame_h)
        max_dist = self._match_fraction * diag if diag > 0 else math.inf

        # 2) 전역 최근접 greedy 연관 (사람 i → 기존 track j). 가장 가까운 쌍부터 배정.
        pairs: list[tuple[float, int, int]] = []
        for i, ci in enumerate(cents):
            if ci is None:
                continue
            for j, tr in enumerate(self._tracks):
                d = math.hypot(ci[0] - tr.cx, ci[1] - tr.cy)
                if d <= max_dist:
                    pairs.append((d, i, j))
        pairs.sort(key=lambda pr: pr[0])
        person_track: list[Optional[int]] = [None] * len(persons)
        used_tracks: set[int] = set()
        for _, i, j in pairs:
            if person_track[i] is not None or j in used_tracks:
                continue
            person_track[i] = j
            used_tracks.add(j)

        # 3) track 갱신/생성 + 연속 lying 누적 → 상태.
        states: list[PersonRescue] = []
        for i in range(len(persons)):
            lying = postures[i] == "lying"
            ci = cents[i]
            j = person_track[i]
            if j is not None:
                tr = self._tracks[j]
                if ci is not None:
                    tr.cx, tr.cy = ci
                tr.last_seen = t
                if lying:
                    if tr.lying_since is None:  # lying 시작 — 누적 개시
                        tr.lying_since = t
                    # 이미 lying 이면 lying_since 유지(연속 누적)
                else:  # non-lying(standing/sitting/None) → 연속 끊김, 리셋
                    tr.lying_since = None
            else:
                # 신규 사람 — track 생성. centroid 없으면 원점 취급(무 keypoint 엣지).
                cx, cy = ci if ci is not None else (0.0, 0.0)
                tr = _Track(cx=cx, cy=cy, lying_since=(t if lying else None), last_seen=t)
                self._tracks.append(tr)

            lying_sec = (t - tr.lying_since) if tr.lying_since is not None else 0.0
            needed = tr.lying_since is not None and lying_sec >= self._threshold
            # rescue 상태 전이 추적 — active 였다가 non-lying 되면 "recovered"(기립=회복) 발행.
            if needed and not tr.rescue_active:
                tr.rescue_active = True
            elif not needed and tr.rescue_active:
                self._pending_events.append("recovered")
                tr.rescue_active = False
            states.append(PersonRescue(rescue_needed=needed, lying_sec=lying_sec))
        return states

    def drain_events(self) -> list[str]:
        """지난 update 들에서 쌓인 rescue 종료 이벤트(recovered) 반환 후 비운다."""
        evs = self._pending_events
        self._pending_events = []
        return evs

    def reset(self) -> None:
        """추적 상태 초기화 — manager 가 추론 OFF/카메라 제거 시 호출(stale rescue 방지)."""
        self._tracks = []
        self._pending_events = []
