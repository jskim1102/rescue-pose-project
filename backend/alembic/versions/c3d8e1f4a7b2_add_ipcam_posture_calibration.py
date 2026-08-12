"""add per-camera posture calibration

Revision ID: c3d8e1f4a7b2
Revises: f7c2a9d3e1b4
Create Date: 2026-08-11 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "c3d8e1f4a7b2"
down_revision: Union[str, Sequence[str], None] = "f7c2a9d3e1b4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """카메라별 검증된 보정 payload를 저장하는 nullable 컬럼을 추가한다."""
    op.add_column(
        "ip_cams",
        sa.Column("posture_calibration", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    """보정 기능 롤백 시 추가 컬럼을 제거한다."""
    op.drop_column("ip_cams", "posture_calibration")
