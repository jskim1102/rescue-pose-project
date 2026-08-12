"""카메라별 바닥 homography와 위치별 기립 스케일을 이용한 자세 보정.

단안 2D keypoint만으로 모든 세로 누움과 기립을 수학적으로 구분할 수는 없다. 이 모듈은
바닥상 신체 길이가 사람 범위이고, 같은 위치의 기립 기준보다 화면상 뼈대가 뚜렷하게 압축된
경우에만 기존 ``standing``을 ``lying``으로 교정한다. 근거가 약하면 기존 분류를 보존한다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np


_MIN_POINT_CONFIDENCE = 0.3
_REFERENCE_HEIGHT_M = 1.7
_MIN_FLOOR_BODY_LENGTH_M = 0.9
_MAX_FLOOR_BODY_LENGTH_M = 2.3
_MAX_STANDING_COMPRESSION = 0.60


@dataclass(frozen=True)
class StandingReference:
    """한 위치에서 관측한 기립자 keypoint 높이와 실제 키."""

    foot_px: tuple[float, float]
    keypoint_height_px: float
    height_m: float


@dataclass(frozen=True)
class PostureCalibration:
    """검증·전처리된 카메라 보정값. 생성 후 여러 추론 스레드에서 읽기 전용으로 쓴다."""

    frame_width: int
    frame_height: int
    floor_image_points: tuple[tuple[float, float], ...]
    floor_world_points: tuple[tuple[float, float], ...]
    standing_references: tuple[StandingReference, ...]
    reprojection_rmse_m: float
    _image_to_floor: np.ndarray = field(repr=False, compare=False)
    _floor_image_hull: np.ndarray = field(repr=False, compare=False)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "PostureCalibration":
        """API/DB payload를 검증하고 image→floor homography를 계산한다."""
        try:
            frame_width = int(raw["frame_width"])
            frame_height = int(raw["frame_height"])
            image_points = _points(raw["floor_image_points"], "floor_image_points")
            world_points = _points(raw["floor_world_points"], "floor_world_points")
            refs_raw = raw["standing_references"]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("보정값 형식이 올바르지 않습니다") from exc

        if frame_width <= 0 or frame_height <= 0:
            raise ValueError("frame_width와 frame_height는 양수여야 합니다")
        if not 4 <= len(image_points) <= 16:
            raise ValueError("바닥 기준점은 4~16개가 필요합니다")
        if len(image_points) != len(world_points):
            raise ValueError("화면 바닥점과 실제 바닥점 개수가 같아야 합니다")
        if not all(
            0.0 <= x <= frame_width and 0.0 <= y <= frame_height
            for x, y in image_points
        ):
            raise ValueError("화면 바닥점은 보정 프레임 안에 있어야 합니다")
        if _hull_area(image_points) < 1.0 or _hull_area(world_points) < 1e-4:
            raise ValueError("바닥 기준점이 한 직선에 몰려 있습니다")

        image_array = np.asarray(image_points, dtype=np.float64)
        world_array = np.asarray(world_points, dtype=np.float64)
        matrix, _ = cv2.findHomography(image_array, world_array, method=0)
        if matrix is None or not np.isfinite(matrix).all():
            raise ValueError("바닥 원근 행렬을 계산할 수 없습니다")
        condition = float(np.linalg.cond(matrix))
        if not math.isfinite(condition) or condition > 1e12:
            raise ValueError("바닥 기준점 배치가 불안정합니다")

        projected = cv2.perspectiveTransform(
            image_array.reshape(-1, 1, 2), matrix
        ).reshape(-1, 2)
        rmse = float(np.sqrt(np.mean(np.sum((projected - world_array) ** 2, axis=1))))
        world_diagonal = float(np.linalg.norm(world_array.max(axis=0) - world_array.min(axis=0)))
        if rmse > max(0.1, world_diagonal * 0.05):
            raise ValueError("바닥 기준점 오차가 너무 큽니다")

        if not isinstance(refs_raw, list) or not 3 <= len(refs_raw) <= 20:
            raise ValueError("서로 다른 위치의 기립 기준 샘플이 3~20개 필요합니다")
        image_hull = cv2.convexHull(np.asarray(image_points, dtype=np.float32))
        references: list[StandingReference] = []
        for item in refs_raw:
            if not isinstance(item, dict):
                raise ValueError("기립 기준 샘플 형식이 올바르지 않습니다")
            foot = _points([item.get("foot_px")], "foot_px")[0]
            try:
                keypoint_height = float(item["keypoint_height_px"])
                height_m = float(item["height_m"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("기립 기준 샘플 형식이 올바르지 않습니다") from exc
            if not all(math.isfinite(value) for value in (*foot, keypoint_height, height_m)):
                raise ValueError("보정값은 유한한 숫자여야 합니다")
            if not (0.0 <= foot[0] <= frame_width and 0.0 <= foot[1] <= frame_height):
                raise ValueError("기립 기준 발 위치는 보정 프레임 안에 있어야 합니다")
            if cv2.pointPolygonTest(image_hull, foot, False) < 0:
                raise ValueError("기립 기준 발 위치는 보정한 바닥 영역 안에 있어야 합니다")
            if keypoint_height <= 1.0:
                raise ValueError("기립 keypoint 높이는 1px보다 커야 합니다")
            if not 0.5 <= height_m <= 2.5:
                raise ValueError("기립 기준 실제 키는 0.5~2.5m여야 합니다")
            references.append(
                StandingReference(
                    foot_px=foot,
                    keypoint_height_px=keypoint_height,
                    height_m=height_m,
                )
            )

        ref_floor = [_project(matrix, reference.foot_px) for reference in references]
        if _max_pair_distance(ref_floor) < 0.25:
            raise ValueError("기립 기준 샘플은 바닥의 서로 다른 위치에서 측정해야 합니다")

        return cls(
            frame_width=frame_width,
            frame_height=frame_height,
            floor_image_points=tuple(image_points),
            floor_world_points=tuple(world_points),
            standing_references=tuple(references),
            reprojection_rmse_m=rmse,
            _image_to_floor=matrix,
            _floor_image_hull=image_hull,
        )

    def to_dict(self) -> dict[str, Any]:
        """DB/API에 저장 가능한 canonical payload."""
        return {
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "floor_image_points": [list(point) for point in self.floor_image_points],
            "floor_world_points": [list(point) for point in self.floor_world_points],
            "standing_references": [
                {
                    "foot_px": list(reference.foot_px),
                    "keypoint_height_px": reference.keypoint_height_px,
                    "height_m": reference.height_m,
                }
                for reference in self.standing_references
            ],
        }

    def classify(
        self,
        keypoints: list[tuple[float, float, float]],
        *,
        fallback: str,
        frame_width: float,
        frame_height: float,
    ) -> str:
        """강한 바닥 증거가 있는 세로 누움만 교정하고, 나머지는 fallback을 보존한다."""
        if fallback != "standing" or len(keypoints) < 17:
            return fallback
        if frame_width <= 0 or frame_height <= 0:
            return fallback

        scale_x = self.frame_width / float(frame_width)
        scale_y = self.frame_height / float(frame_height)
        visible = [
            (float(point[0]) * scale_x, float(point[1]) * scale_y)
            for point in keypoints
            if len(point) >= 3
            and float(point[2]) > _MIN_POINT_CONFIDENCE
            and math.isfinite(float(point[0]))
            and math.isfinite(float(point[1]))
        ]
        # 얼굴/어깨·골반·하체가 충분하지 않으면 크기 비교가 쉽게 왜곡된다.
        visible_core = [
            (float(keypoints[index][0]) * scale_x, float(keypoints[index][1]) * scale_y)
            for index in [0, 5, 6, 11, 12, 13, 14, 15, 16]
            if len(keypoints[index]) >= 3
            and float(keypoints[index][2]) > _MIN_POINT_CONFIDENCE
            and math.isfinite(float(keypoints[index][0]))
            and math.isfinite(float(keypoints[index][1]))
        ]
        if len(visible) < 6 or len(visible_core) < 5:
            return fallback

        lower = [
            (float(keypoints[index][0]) * scale_x, float(keypoints[index][1]) * scale_y)
            for index in [15, 16]
            if float(keypoints[index][2]) > _MIN_POINT_CONFIDENCE
        ]
        if not lower:
            return fallback
        foot = (
            sum(point[0] for point in lower) / len(lower),
            sum(point[1] for point in lower) / len(lower),
        )
        if cv2.pointPolygonTest(self._floor_image_hull, foot, False) < 0:
            return fallback

        observed_extent_px = _max_pair_distance(visible_core)
        expected_height_px = self._expected_keypoint_height_px(foot)
        if expected_height_px <= 0.0:
            return fallback
        compression = observed_extent_px / expected_height_px

        try:
            floor_points = [_project(self._image_to_floor, point) for point in visible_core]
        except ValueError:
            return fallback
        floor_body_length = _max_pair_distance(floor_points)

        if (
            _MIN_FLOOR_BODY_LENGTH_M <= floor_body_length <= _MAX_FLOOR_BODY_LENGTH_M
            and compression <= _MAX_STANDING_COMPRESSION
        ):
            return "lying"
        return fallback

    def _expected_keypoint_height_px(self, foot_px: tuple[float, float]) -> float:
        """발 위치 주변 기립 샘플을 바닥거리 IDW로 보간한 1.7m 기준 keypoint 높이."""
        foot_world = _project(self._image_to_floor, foot_px)
        weighted_sum = 0.0
        weight_total = 0.0
        for reference in self.standing_references:
            ref_world = _project(self._image_to_floor, reference.foot_px)
            distance = math.dist(foot_world, ref_world)
            weight = 1.0 / (distance * distance + 0.05)
            normalized_height = (
                reference.keypoint_height_px
                * _REFERENCE_HEIGHT_M
                / reference.height_m
            )
            weighted_sum += weight * normalized_height
            weight_total += weight
        return weighted_sum / weight_total if weight_total else 0.0


def _points(raw: Any, name: str) -> list[tuple[float, float]]:
    if not isinstance(raw, (list, tuple)):
        raise ValueError(f"{name} 형식이 올바르지 않습니다")
    points: list[tuple[float, float]] = []
    for point in raw:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError(f"{name}의 각 점은 [x, y]여야 합니다")
        try:
            x, y = float(point[0]), float(point[1])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name}의 좌표는 숫자여야 합니다") from exc
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("보정값은 유한한 숫자여야 합니다")
        points.append((x, y))
    return points


def _hull_area(points: list[tuple[float, float]]) -> float:
    array = np.asarray(points, dtype=np.float32)
    return float(cv2.contourArea(cv2.convexHull(array)))


def _project(matrix: np.ndarray, point: tuple[float, float]) -> tuple[float, float]:
    vector = matrix @ np.asarray([point[0], point[1], 1.0], dtype=np.float64)
    if abs(float(vector[2])) < 1e-9:
        raise ValueError("원근 투영이 유효하지 않습니다")
    projected = (float(vector[0] / vector[2]), float(vector[1] / vector[2]))
    if not all(math.isfinite(value) for value in projected):
        raise ValueError("원근 투영이 유효하지 않습니다")
    return projected


def _max_pair_distance(points: list[tuple[float, float]]) -> float:
    return max(
        (math.dist(first, second) for i, first in enumerate(points) for second in points[i + 1 :]),
        default=0.0,
    )
