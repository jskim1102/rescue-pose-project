"""Docker 기본 실행이 rtsp-keypoint와 같은 GPU 계약을 유지하는지 검증한다."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _backend_block(path: Path, *, has_frontend: bool) -> str:
    text = path.read_text(encoding="utf-8")
    block = text.split("\n  backend:\n", 1)[1]
    if has_frontend:
        block = block.split("\n  frontend:\n", 1)[0]
    return block


def test_default_compose_reserves_nvidia_gpu_and_weight_cache() -> None:
    compose = PROJECT_ROOT / "docker-compose.yml"
    backend = _backend_block(compose, has_frontend=True)

    assert backend.count("driver: nvidia") == 1
    assert "capabilities: [gpu]" in backend
    assert "- yolo_weights:/root/.cache/ultralytics" in backend


def test_legacy_gpu_override_does_not_duplicate_gpu_reservation() -> None:
    override = PROJECT_ROOT / "docker-compose.gpu.yml"
    backend = _backend_block(override, has_frontend=False)

    assert "YOLO_DEVICE: cuda:0" in backend
    assert "deploy:" not in backend
    assert "volumes:" not in backend
