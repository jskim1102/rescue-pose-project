import numpy as np

from app.inference import worker as worker_mod
from app.inference.worker import Detection, _encode_annotated_pose_jpeg


def _person(posture: str = "sitting") -> Detection:
    keypoints = [(40.0 + i * 5.0, 50.0 + i * 3.0, 0.9) for i in range(17)]
    return Detection(
        class_id=0,
        class_name="person",
        confidence=0.9,
        keypoints=keypoints,
        model="yolo26n-pose.pt",
        posture=posture,
    )


def test_annotated_pose_uses_frontend_palette_and_posture_badge(monkeypatch):
    lines = []
    circles = []
    rectangles = []
    texts = []

    def fake_line(_img, _p1, _p2, color, _thickness, **_kwargs):
        lines.append(tuple(color))

    def fake_circle(_img, _center, _radius, color, **_kwargs):
        circles.append(tuple(color))

    def fake_rectangle(_img, _p1, _p2, color, _thickness, **_kwargs):
        rectangles.append(tuple(color))

    def fake_put_text(_img, text, _org, _font, _font_scale, color, _thickness, **_kwargs):
        texts.append((text, tuple(color)))

    monkeypatch.setattr(worker_mod.cv2, "line", fake_line)
    monkeypatch.setattr(worker_mod.cv2, "circle", fake_circle)
    monkeypatch.setattr(worker_mod.cv2, "rectangle", fake_rectangle)
    monkeypatch.setattr(worker_mod.cv2, "putText", fake_put_text)
    monkeypatch.setattr(worker_mod.cv2, "getTextSize", lambda *_args: ((56, 12), 3))
    monkeypatch.setattr(
        worker_mod.cv2,
        "imencode",
        lambda *_args: (True, np.array([1, 2, 3], dtype=np.uint8)),
    )

    jpeg = _encode_annotated_pose_jpeg(np.zeros((120, 180, 3), dtype=np.uint8), [_person()])

    assert jpeg == b"\x01\x02\x03"
    assert lines[0] == (255, 153, 51)  # frontend POSE_PALETTE[9] as BGR
    assert lines[4] == (255, 51, 255)  # frontend POSE_PALETTE[7] as BGR
    assert circles[0] == (0, 255, 0)  # nose uses KPT_COLOR_IDX[0] = 16
    assert circles[5] == (0, 128, 255)  # shoulder uses KPT_COLOR_IDX[5] = 0
    assert circles[11] == (255, 153, 51)  # hip uses KPT_COLOR_IDX[11] = 9
    assert rectangles[0] == (59, 162, 224)  # sitting #e0a23b as BGR
    assert texts == [("SITTING", (18, 13, 10))]  # cam1 dark label text as BGR
