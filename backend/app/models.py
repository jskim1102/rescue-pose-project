import os
import uuid
from datetime import datetime

from sqlalchemy import DateTime, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


def _generate_stream_key() -> str:
    """등록용 고유 스트림 키 생성.

    MEDIAMTX_PATH_PREFIX 로 path 네임스페이싱 → `<prefix>__ipcam-<hex>`.
    self-host 전환 뒤에도 기존 DB 키 호환과 viewer path 권한 경계를 위해 prefix 를 유지한다.
    mediamtx.py/transcode($MTX_PATH)/WHEP 는 stream_key 를 그대로 쓰므로 자동 네임스페이싱된다.
    """
    suffix = f"ipcam-{uuid.uuid4().hex[:8]}"
    prefix = os.getenv("MEDIAMTX_PATH_PREFIX")
    return f"{prefix}__{suffix}" if prefix else suffix


class IpCam(Base):
    __tablename__ = "ip_cams"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    rtsp_url: Mapped[str] = mapped_column(String(500), nullable=False)
    stream_key: Mapped[str] = mapped_column(String(100), nullable=False, unique=True, default=_generate_stream_key)
    # 카메라별 바닥 homography + 기립 스케일. API가 검증한 canonical JSON만 저장한다.
    posture_calibration: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
