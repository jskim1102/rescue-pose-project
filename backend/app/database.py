import os
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.orm import DeclarativeBase, sessionmaker

# DB 경로 — 기본은 backend/deepeye.db (로컬/테스트), compose 는 named volume 을
# 가리키도록 DATABASE_URL 환경변수로 override (12-factor).
_db_path = Path(__file__).resolve().parent.parent / "deepeye.db"
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{_db_path}")

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})


@event.listens_for(engine, "connect")
def _set_sqlite_busy_timeout(dbapi_connection, connection_record):
    """동시 카메라 등록의 BEGIN IMMEDIATE 쓰기 락을 최대 5초 기다린다."""
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA busy_timeout=5000")
    cursor.close()


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
