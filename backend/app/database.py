import os
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

# DB 경로 — 기본은 backend/deepeye.db (로컬/테스트), compose 는 named volume 을
# 가리키도록 DATABASE_URL 환경변수로 override (12-factor).
_db_path = Path(__file__).resolve().parent.parent / "deepeye.db"
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{_db_path}")

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)


class Base(DeclarativeBase):
    pass


def init_db() -> None:
    """모든 테이블 생성 (없는 경우에만)"""
    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI Depends용 DB 세션 제너레이터"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
