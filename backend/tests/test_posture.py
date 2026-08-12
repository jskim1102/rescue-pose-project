"""Regression tests — worker._classify_posture (tunnel worker 차용, rescue seam).

합성 COCO17 keypoint 로 lying/sitting/standing 기본 동작과 실내 테스트에서 확인한
대각 누움·허리 숙임·일부 가림 회귀, guard 경로(부족 keypoint·저신뢰 shoulder)를 고정한다.
keypoint shape = (x, y, conf) 튜플 (rtsp-keypoint 정본 shape — 인덱스 호환).

이미지 좌표계: y 아래로 증가. lying은 몸통 방향에 전체 몸축/골반 펴짐을 함께 사용한다.
sitting은 보이는 다리 하나의 무릎각을 사용하고, 발목이 가리면 직립 몸통의 골반각으로 보완한다.
"""
from app.inference.worker import _classify_posture


def _kp(x: float, y: float, c: float = 0.9) -> tuple[float, float, float]:
    return (float(x), float(y), float(c))


def _body(shoulders, hips, knees, ankles):
    """17-COCO keypoint 리스트 조립. 얼굴(0-4)·팔(7-10)은 분류에 미사용 → 채우기용."""
    face = [_kp(100, 60)] * 5            # 0 nose .. 4 ear
    l_sh, r_sh = shoulders
    l_el, r_el, l_wr, r_wr = [_kp(80, 150, 0.5)] * 4  # 7-10 arms (unused)
    l_hip, r_hip = hips
    l_kn, r_kn = knees
    l_an, r_an = ankles
    return [
        *face,
        l_sh, r_sh,                      # 5, 6 shoulders
        l_el, r_el, l_wr, r_wr,          # 7-10 arms
        l_hip, r_hip,                    # 11, 12 hips
        l_kn, r_kn,                      # 13, 14 knees
        l_an, r_an,                      # 15, 16 ankles
    ]


def test_standing():
    # torso 수직(어깨 위 / 엉덩이 아래, 같은 x → torso_angle≈90°) + 다리 직립(hip-knee-ankle≈180°)
    kpts = _body(
        shoulders=(_kp(90, 100), _kp(110, 100)),
        hips=(_kp(90, 200), _kp(110, 200)),
        knees=(_kp(90, 300), _kp(110, 300)),
        ankles=(_kp(90, 400), _kp(110, 400)),
    )
    assert _classify_posture(kpts) == "standing"


def test_sitting():
    # torso 수직(≈90°, lying 아님) 이지만 다리 꺾임(hip↓knee, ankle 앞으로 → ≈90° <130) → sitting
    kpts = _body(
        shoulders=(_kp(90, 100), _kp(110, 100)),
        hips=(_kp(90, 200), _kp(110, 200)),
        knees=(_kp(90, 280), _kp(110, 280)),
        ankles=(_kp(160, 280), _kp(180, 280)),
    )
    assert _classify_posture(kpts) == "sitting"


def test_lying():
    # torso 수평(≈0°)이고 골반각이 펴짐(≈180°) → lying
    kpts = _body(
        shoulders=(_kp(100, 90), _kp(100, 110)),
        hips=(_kp(300, 90), _kp(300, 110)),
        knees=(_kp(350, 90), _kp(350, 110)),
        ankles=(_kp(400, 90), _kp(400, 110)),
    )
    assert _classify_posture(kpts) == "lying"


def test_diagonal_lying_above_45_degrees():
    # 실내 테스트 회귀: 화면 축 기준 약 50°로 누워도 몸통과 다리가 펴져 있으면 lying.
    kpts = _body(
        shoulders=(_kp(95, 105), _kp(105, 95)),
        hips=(_kp(175, 200), _kp(185, 190)),
        knees=(_kp(255, 295), _kp(265, 285)),
        ankles=(_kp(335, 390), _kp(345, 380)),
    )
    assert _classify_posture(kpts) == "lying"


def test_bending_forward_at_the_waist_is_not_lying():
    # 실내 테스트 회귀: 상체만 수평이어도 골반에서 상체-다리가 꺾이면 lying 이 아니다.
    kpts = _body(
        shoulders=(_kp(90, 90), _kp(90, 110)),
        hips=(_kp(200, 90), _kp(200, 110)),
        knees=(_kp(200, 190), _kp(200, 210)),
        ankles=(_kp(200, 290), _kp(200, 310)),
    )
    assert _classify_posture(kpts) == "standing"


def test_forward_leaning_sitting_stays_sitting():
    # 상체가 대각선이어도 골반·무릎이 함께 굽혀진 자세는 lying 이 아니라 sitting.
    kpts = _body(
        shoulders=(_kp(70, 100), _kp(90, 100)),
        hips=(_kp(170, 180), _kp(190, 180)),
        knees=(_kp(170, 260), _kp(190, 260)),
        ankles=(_kp(240, 260), _kp(260, 260)),
    )
    assert _classify_posture(kpts) == "sitting"


def test_lying_uses_the_visible_leg_when_the_other_is_occluded():
    # 한쪽 무릎·발목이 가려져도 보이는 쪽의 펴진 골반각으로 lying 을 유지한다.
    kpts = _body(
        shoulders=(_kp(95, 105), _kp(105, 95)),
        hips=(_kp(175, 185), _kp(185, 175)),
        knees=(_kp(245, 255), _kp(255, 245, 0.1)),
        ankles=(_kp(315, 325), _kp(325, 315, 0.1)),
    )
    assert _classify_posture(kpts) == "lying"


def test_curled_lying_is_not_mistaken_for_sitting():
    # 누운 채 다리를 굽히면 골반각이 135°보다 작아질 수 있다. 몸 전체가 화면 수평으로
    # 놓인 경우에는 그 굽힘만으로 sitting/standing 으로 되돌리면 안 된다.
    kpts = _body(
        shoulders=(_kp(90, 90), _kp(90, 110)),
        hips=(_kp(190, 90), _kp(190, 110)),
        knees=(_kp(235, 150), _kp(235, 170)),
        ankles=(_kp(300, 150), _kp(300, 170)),
    )
    assert _classify_posture(kpts) == "lying"


def test_lying_uses_the_visible_torso_side_when_the_other_is_occluded():
    # PDF의 일부 포인트 가림 회귀: 한쪽 어깨·엉덩이가 가려져도 반대편 몸통과 하체가
    # 충분히 보이면 누움 방향을 계산할 수 있다.
    kpts = _body(
        shoulders=(_kp(100, 100), _kp(100, 120, 0.1)),
        hips=(_kp(200, 100), _kp(200, 120, 0.1)),
        knees=(_kp(270, 100), _kp(270, 120, 0.1)),
        ankles=(_kp(340, 100), _kp(340, 120, 0.1)),
    )
    assert _classify_posture(kpts) == "lying"


def test_sitting_uses_one_clearly_bent_leg_instead_of_two_leg_mean():
    # 한쪽 무릎은 약 90°, 반대쪽은 곧게 펴진 자세. 평균은 경계값을 넘지만 PDF 기준은
    # 명확히 굽은 다리 하나가 있으면 sitting 이다.
    kpts = _body(
        shoulders=(_kp(90, 100), _kp(110, 100)),
        hips=(_kp(90, 200), _kp(110, 200)),
        knees=(_kp(90, 280), _kp(110, 280)),
        ankles=(_kp(170, 280), _kp(110, 360)),
    )
    assert _classify_posture(kpts) == "sitting"


def test_sitting_falls_back_to_hip_angle_when_ankles_are_occluded():
    # 발목 없이 어깨-엉덩이-무릎이 약 90°로 꺾인 경우도 sitting 으로 유지한다.
    kpts = _body(
        shoulders=(_kp(90, 100), _kp(110, 100)),
        hips=(_kp(90, 200), _kp(110, 200)),
        knees=(_kp(170, 200), _kp(190, 200)),
        ankles=(_kp(240, 200, 0.1), _kp(260, 200, 0.1)),
    )
    assert _classify_posture(kpts) == "sitting"


def test_forward_bend_with_occluded_ankles_stays_standing():
    # 발목 누락 fallback 이 허리를 숙인 사람까지 sitting 으로 바꾸지 않아야 한다.
    kpts = _body(
        shoulders=(_kp(90, 90), _kp(90, 110)),
        hips=(_kp(200, 90), _kp(200, 110)),
        knees=(_kp(200, 190), _kp(200, 210)),
        ankles=(_kp(200, 290, 0.1), _kp(200, 310, 0.1)),
    )
    assert _classify_posture(kpts) == "standing"


def test_missing_knees_and_ankles_does_not_invent_sitting():
    # PDF 기준: 하체 관절 근거가 전혀 없으면 sitting 으로 추정하지 않는다.
    kpts = _body(
        shoulders=(_kp(90, 100), _kp(110, 100)),
        hips=(_kp(90, 200), _kp(110, 200)),
        knees=(_kp(90, 280, 0.1), _kp(110, 280, 0.1)),
        ankles=(_kp(170, 280, 0.1), _kp(190, 280, 0.1)),
    )
    assert _classify_posture(kpts) == "standing"


def test_fewer_than_17_keypoints_defaults_standing():
    assert _classify_posture([_kp(0, 0)] * 10) == "standing"


def test_low_confidence_torso_defaults_standing():
    # 어깨/엉덩이 신뢰도 <0.3 → torso 각도 계산 불가 → standing 폴백 (원본 guard)
    kpts = _body(
        shoulders=(_kp(100, 90, 0.1), _kp(100, 110, 0.1)),
        hips=(_kp(300, 90, 0.1), _kp(300, 110, 0.1)),
        knees=(_kp(350, 90), _kp(350, 110)),
        ankles=(_kp(400, 90), _kp(400, 110)),
    )
    assert _classify_posture(kpts) == "standing"
