import logging
import os
import sys
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import load_dotenv

# 런타임 env 로드 — backend/.env(있으면) + 프로젝트 루트 .env 둘 다.
# self-host MediaMTX의 offset 포트·자격증명은 프로젝트 루트 .env 에 두므로, bare import(pytest·검증
# `python -c "import app.main"`)에서도 MEDIAMTX_API 등이 resolve 돼 아래 import-time fail-fast 가
# 정상 설정에선 발화하지 않는다. override=False(기본): dev.sh/compose 가 이미 export 한 프로세스
# env 가 우선이며 .env 가 덮어쓰지 않는다.
_backend_env = Path(__file__).resolve().parent.parent / ".env"
_root_env = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(_backend_env)
load_dotenv(_root_env)

# 빈 문자열(set-but-empty)도 미설정으로 취급해 안전기본 폴백 (codex #3, rtsp-streaming 정본 parity).
#   os.getenv(k, default) 는 k="" 면 "" 를 그대로 돌려준다 → CORS_ORIGINS="" → CORS 차단,
#   MAX_IPCAMS="" → int("") → ValueError(import 크래시). `or` 로 빈값=거짓 → 기본값 폴백.
CORS_ORIGINS: str = os.getenv("CORS_ORIGINS") or "*"

_raw_max_ipcams = int(os.getenv("MAX_IPCAMS") or "16")
MAX_IPCAMS: int = max(1, min(64, _raw_max_ipcams))

# mediamtx API 주소 — 미설정/빈값이면 import 시점에 즉시 fail-fast (원본 deepeye 동작 복원).
# 빈값을 허용하면 stats/delete/lifespan 의 mediamtx 호출이 호출 시점마다 산발적
# uncaught RuntimeError(요청 500·부팅 실패)로 번진다. 설정 누락을 가장 이른 지점(import)에서
# 한 번에 드러낸다. Docker 는 compose environment 블록, 로컬은 루트 .env(또는 backend/.env)로 주입.
MEDIAMTX_API: str = os.getenv("MEDIAMTX_API", "").strip()
if not MEDIAMTX_API:
    raise RuntimeError(
        "MEDIAMTX_API required — 환경변수 MEDIAMTX_API 가 설정되지 않았습니다. "
        "Docker 는 docker-compose.yml 의 environment, 로컬은 프로젝트 루트 .env 를 확인하세요."
    )


def _derive_mediamtx_rtsp_base(api_url: str) -> str:
    """MEDIAMTX_API host 에서 backend-internal RTSP base URL 을 만든다."""
    parsed = urlsplit(api_url)
    host = parsed.hostname or "127.0.0.1"
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    return f"rtsp://{host}:8554"


# Detection 캡처용 mediamtx RTSP fan-out base. 미설정이면 API host 의 8554 포트를 사용한다.
MEDIAMTX_RTSP_BASE_URL: str = (
    os.getenv("MEDIAMTX_RTSP_BASE_URL", "").strip()
    or _derive_mediamtx_rtsp_base(MEDIAMTX_API)
)

# mediamtx 인증(#100) — backend user 로 API 호출 시 Basic auth. 비번 비우면 무인증(로컬/테스트 하위호환).
MEDIAMTX_BACKEND_USER: str = os.getenv("MEDIAMTX_BACKEND_USER", "backend")
MEDIAMTX_BACKEND_PASS: str = os.getenv("MEDIAMTX_BACKEND_PASS", "")

# WebRTC 외부접속 광고 호스트(공인 IP). mediamtx.yml 의 webrtcAdditionalHosts 로
# 주입된다. 미설정이면 mediamtx 가 컨테이너 내부 주소만 광고 → 외부에서 영상 안 나옴.
MEDIAMTX_WEBRTC_HOST: str = os.getenv("MEDIAMTX_WEBRTC_HOST", "")

# ── detection (YOLO 추론 — deepeye-lite 차용) ──
# 기본 모델 + conf. 워커(InferenceWorker)도 같은 env 를 직접 읽음(worker.py).
YOLO_DEFAULT_MODEL: str = os.getenv("YOLO_DEFAULT_MODEL", "yolo26n-pose.pt")


def _env_float(name: str, default: float) -> tuple[float, str | None]:
    """env float 파싱 — 비숫자/빈값이면 default 로 폴백(bare float() import crash 방지).

    logger 가 이 시점엔 아직 미설정이라 경고를 직접 emit 하지 않고 메시지로 반환만 한다 —
    호출부가 logger 설정 후 한꺼번에 emit.
    """
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default, None
    try:
        return float(raw), None
    except ValueError:
        return default, f"{name}={raw!r} 가 숫자가 아니라 기본값 {default} 로 폴백"


YOLO_CONF_THRESHOLD, _conf_warn = _env_float("YOLO_CONF_THRESHOLD", 0.5)
# 빈값 = 워커가 torch.cuda.is_available() 자동감지 (GPU→cuda:0 / 없으면 CPU 폴백).
YOLO_DEVICE: str = os.getenv("YOLO_DEVICE", "")

# rtsp-keypoint cadence autotune 정본. INFERENCE_INTERVAL은 telemetry 수렴 전 초기값이고,
# MIN/MAX는 폭주 방지 경계다. CAPTURE_INTERVAL은 detection WS polling 주기다.
MIN_INFERENCE_INTERVAL: float = 0.01
MAX_INFERENCE_INTERVAL: float = 1.0
INFERENCE_INTERVAL: float = 0.033
CAPTURE_INTERVAL: float = 0.01

# 실제 budget은 worker infer_ms EWMA에서 매초 계산한다. 이 값은 telemetry 표본이
# 모이기 전 bootstrap 상한이며, source별 latest queue + micro-batch가 burst를 흡수한다.
MAX_INFER_PER_SEC: float = 52.0
AUTOTUNE_HEADROOM: float = 0.95
AUTOTUNE_EWMA_ALPHA: float = 0.2
AUTOTUNE_MIN_SAMPLES: int = 5
AUTOTUNE_TARGET_FPS_MAX: float = MAX_INFER_PER_SEC

# 모델별 worker pool + adaptive inference resolution 내부 정책.
INFERENCE_BATCH_MAX: int = 8
INFERENCE_BATCH_TIMEOUT_SEC: float = 0.008
INFERENCE_AGGREGATE_TIMEOUT_SEC: float = 2.0
INFERENCE_IMGSZ_STAGES: tuple[int, ...] = (320, 416, 512, 640)
ADAPTIVE_DOWNSHIFT_TICKS: int = 2
ADAPTIVE_UPSHIFT_TICKS: int = 5
ADAPTIVE_OVERLOAD_RATIO: float = 0.85
ADAPTIVE_UNDERLOAD_RATIO: float = 0.65

# custom .pt 모델 디렉토리 — 미설정 시 backend/models(네이티브 dev) = /app/models(컨테이너)
# 로 자동 결정(절대경로, cwd 무관). models_dir 가 import 시 CUSTOM_MODELS_DIR 를 읽으므로
# 반드시 그 전에(여기, config import 시점) setdefault. compose 가 명시하면 그 값이 우선.
os.environ.setdefault(
    "CUSTOM_MODELS_DIR", str(Path(__file__).resolve().parent.parent / "models")
)

LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()


def setup_logging() -> logging.Logger:
    """애플리케이션 로거 설정"""
    logger = logging.getLogger("rtsp-streaming")
    logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(levelname)s  %(name)s — %(message)s",
                              datefmt="%Y-%m-%d %H:%M:%S")
        )
        logger.addHandler(handler)

    return logger


logger = setup_logging()

if _raw_max_ipcams != MAX_IPCAMS:
    logger.warning("MAX_IPCAMS=%d → %d 로 보정됨 (허용 범위: 1~64)", _raw_max_ipcams, MAX_IPCAMS)
for _w in (_conf_warn,):
    if _w:
        logger.warning("%s", _w)
