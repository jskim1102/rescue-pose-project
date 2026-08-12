"""TDD — 카메라별 바닥/원근 보정 기반 세로 누움 판정."""

import threading
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from app.posture_calibration import PostureCalibration
from app.rescue import RescueTracker
from app.streaming.manager import StreamManager


def _payload() -> dict:
    return {
        "frame_width": 1000,
        "frame_height": 1000,
        "floor_image_points": [[0, 0], [1000, 0], [1000, 1000], [0, 1000]],
        "floor_world_points": [[0, 0], [4, 0], [4, 4], [0, 4]],
        "standing_references": [
            {"foot_px": [500, 800], "keypoint_height_px": 600, "height_m": 1.7},
            {"foot_px": [500, 600], "keypoint_height_px": 500, "height_m": 1.7},
            {"foot_px": [500, 400], "keypoint_height_px": 400, "height_m": 1.7},
        ],
    }


def _keypoints(*, top_y: float, foot_y: float, scale: float = 1.0) -> list[tuple[float, float, float]]:
    """화면상 세로인 COCO17 뼈대. scale은 추론 프레임 축소 회귀용."""
    center_x = 500.0
    span = foot_y - top_y
    points = [(0.0, 0.0, 0.0)] * 17
    points[0] = (center_x * scale, top_y * scale, 0.95)
    for index, x_offset, fraction in [
        (5, -20, 0.22),
        (6, 20, 0.22),
        (11, -15, 0.48),
        (12, 15, 0.48),
        (13, -12, 0.73),
        (14, 12, 0.73),
        (15, -10, 1.0),
        (16, 10, 1.0),
    ]:
        points[index] = (
            (center_x + x_offset) * scale,
            (top_y + span * fraction) * scale,
            0.95,
        )
    return points


def test_vertical_lying_overrides_legacy_standing_only_with_floor_evidence():
    calibration = PostureCalibration.from_dict(_payload())

    # 화면 세로 길이는 기립 기준의 절반이지만, 바닥으로 투영하면 약 1.2m인 사람 형상.
    posture = calibration.classify(
        _keypoints(top_y=500, foot_y=800),
        fallback="standing",
        frame_width=1000,
        frame_height=1000,
    )

    assert posture == "lying"


def test_normal_standing_keeps_fallback():
    calibration = PostureCalibration.from_dict(_payload())

    posture = calibration.classify(
        _keypoints(top_y=200, foot_y=800),
        fallback="standing",
        frame_width=1000,
        frame_height=1000,
    )

    assert posture == "standing"


def test_sitting_is_not_overridden_by_ambiguous_calibration_evidence():
    calibration = PostureCalibration.from_dict(_payload())

    posture = calibration.classify(
        _keypoints(top_y=500, foot_y=800),
        fallback="sitting",
        frame_width=1000,
        frame_height=1000,
    )

    assert posture == "sitting"


def test_keypoints_are_scaled_from_inference_frame_to_calibration_frame():
    calibration = PostureCalibration.from_dict(_payload())

    posture = calibration.classify(
        _keypoints(top_y=500, foot_y=800, scale=0.5),
        fallback="standing",
        frame_width=500,
        frame_height=500,
    )

    assert posture == "lying"


def test_person_outside_calibrated_floor_area_keeps_fallback():
    payload = _payload()
    payload["floor_image_points"] = [
        [100, 100],
        [900, 100],
        [900, 900],
        [100, 900],
    ]
    calibration = PostureCalibration.from_dict(payload)

    posture = calibration.classify(
        _keypoints(top_y=650, foot_y=950),
        fallback="standing",
        frame_width=1000,
        frame_height=1000,
    )

    assert posture == "standing"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p.update(floor_world_points=[[0, 0], [1, 0], [2, 0], [3, 0]]),
        lambda p: p.update(floor_world_points=p["floor_world_points"][:-1]),
        lambda p: p.update(standing_references=p["standing_references"][:2]),
        lambda p: p["standing_references"][0].update(keypoint_height_px=float("nan")),
    ],
)
def test_invalid_calibration_is_rejected(mutate):
    payload = _payload()
    mutate(payload)

    with pytest.raises(ValueError):
        PostureCalibration.from_dict(payload)


@dataclass
class _Person:
    keypoints: list[tuple[float, float, float]]
    posture: str


def _bare_manager() -> StreamManager:
    manager = StreamManager.__new__(StreamManager)
    manager._rescue_lock = threading.Lock()
    manager._rescue_trackers = {
        "calibrated": RescueTracker(posture_stability_s=0),
        "plain": RescueTracker(posture_stability_s=0),
    }
    manager._latest_rescue = {}
    manager._posture_calibration_lock = threading.Lock()
    manager._posture_calibrations = {}
    return manager


def test_manager_applies_calibration_per_source_before_rescue_tracking():
    manager = _bare_manager()
    manager.set_posture_calibration(
        "calibrated", PostureCalibration.from_dict(_payload())
    )
    calibrated_person = _Person(_keypoints(top_y=500, foot_y=800), "standing")
    plain_person = _Person(_keypoints(top_y=500, foot_y=800), "standing")

    manager._update_rescue(
        SimpleNamespace(
            source_id="calibrated",
            timestamp=1.0,
            frame_w=1000,
            frame_h=1000,
            detections=[calibrated_person],
        )
    )
    manager._update_rescue(
        SimpleNamespace(
            source_id="plain",
            timestamp=1.0,
            frame_w=1000,
            frame_h=1000,
            detections=[plain_person],
        )
    )

    assert calibrated_person.posture == "lying"
    assert plain_person.posture == "standing"
