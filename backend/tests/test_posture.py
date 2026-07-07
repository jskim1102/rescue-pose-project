"""Characterization test — worker._classify_posture (tunnel worker 차용, rescue seam).

verbatim 이식 코드라 test-after: 합성 COCO17 keypoint 로 lying/sitting/standing 3케이스와
guard 경로(부족 keypoint·저신뢰 shoulder)를 고정해 원본 tunnel 동작을 보존한다.
keypoint shape = (x, y, conf) 튜플 (rtsp-keypoint 정본 shape — 인덱스 호환).

이미지 좌표계: y 아래로 증가. torso_angle: 수직=90°, 수평=0° (<35°=lying).
standing vs sitting: hip-knee-ankle 다리각 평균 <130°=sitting.
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
    # torso 수평(어깨·엉덩이 y 동일, x 크게 벌어짐 → torso_angle≈0° <35°) → lying
    kpts = _body(
        shoulders=(_kp(100, 90), _kp(100, 110)),
        hips=(_kp(300, 90), _kp(300, 110)),
        knees=(_kp(350, 90), _kp(350, 110)),
        ankles=(_kp(400, 90), _kp(400, 110)),
    )
    assert _classify_posture(kpts) == "lying"


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
