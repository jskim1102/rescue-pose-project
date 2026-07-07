import logging
import os
import sys
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import load_dotenv

# 런타임 env 로드 — backend/.env(있으면) + 프로젝트 루트 .env 둘 다.
# 공유 인프라(CEO #115)는 런타임 env 를 프로젝트 루트 .env 에 두므로, bare import(pytest·검증
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


def _env_csv_set(name: str) -> set[str]:
    return {
        key.strip()
        for key in (os.getenv(name, "") or "").split(",")
        if key.strip()
    }


MEDIAMTX_DETECTION_FANOUT_STREAM_KEYS: set[str] = _env_csv_set("MEDIAMTX_DETECTION_FANOUT_STREAM_KEYS")
# 같은 추론 프레임에 keypoint 를 그려 보내는 동기화 표시 강제 allowlist/denylist.
# 자동 URL 분류가 애매한 카메라를 수동으로 보정할 때 사용한다.
SYNCED_POSE_STREAM_KEYS: set[str] = _env_csv_set("SYNCED_POSE_STREAM_KEYS")
SYNCED_POSE_DISABLED_STREAM_KEYS: set[str] = _env_csv_set("SYNCED_POSE_DISABLED_STREAM_KEYS")

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
# 추론 샘플 간격(초). 0.1=10fps — StreamManager/캡처 throttle (GPU 낭비 방지).
# clamp [0.01,1.0]: 0/음수면 throttle 이 무력화돼 매 프레임 추론 → GPU 과부하. 상한 1.0(=1fps).
_raw_inference_interval, _inference_warn = _env_float("INFERENCE_INTERVAL", 0.1)
INFERENCE_INTERVAL: float = max(0.01, min(1.0, _raw_inference_interval))
# detection WS 폴링 송출 간격(초). 영상은 WHEP, 이 WS 는 좌표 JSON 만.
# clamp [0.01,1.0]: 0/음수면 asyncio.sleep(0) busy-loop, 너무 크면 좌표 갱신 지연.
_raw_capture_interval, _capture_warn = _env_float("CAPTURE_INTERVAL", 0.03)
CAPTURE_INTERVAL: float = max(0.01, min(1.0, _raw_capture_interval))
# 단일 inference 워커의 지속 infer/s 예산. 활성 캠 수 N 으로 나눠 per-camera 제출 케이던스를
# 자동 하향(interval=max(INFERENCE_INTERVAL, N/MAX_INFER_PER_SEC)) → 다수캠서도 워커 미포화.
# clamp 하한 1.0: 0/음수면 interval=N/budget 가 음수/무한대라 케이던스 붕괴.
_raw_max_infer, _max_infer_warn = _env_float("MAX_INFER_PER_SEC", 65.0)
MAX_INFER_PER_SEC: float = max(1.0, _raw_max_infer)

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
for _w in (_conf_warn, _inference_warn, _capture_warn, _max_infer_warn):
    if _w:
        logger.warning("%s", _w)
if _raw_inference_interval != INFERENCE_INTERVAL:
    logger.warning("INFERENCE_INTERVAL=%.3f → %.3f 로 보정됨 (허용 범위: 0.01~1.0)",
                   _raw_inference_interval, INFERENCE_INTERVAL)
if _raw_capture_interval != CAPTURE_INTERVAL:
    logger.warning("CAPTURE_INTERVAL=%.3f → %.3f 로 보정됨 (허용 범위: 0.01~1.0)",
                   _raw_capture_interval, CAPTURE_INTERVAL)
if _raw_max_infer != MAX_INFER_PER_SEC:
    logger.warning("MAX_INFER_PER_SEC=%.1f → %.1f 로 보정됨 (허용 범위: >=1.0)",
                   _raw_max_infer, MAX_INFER_PER_SEC)
