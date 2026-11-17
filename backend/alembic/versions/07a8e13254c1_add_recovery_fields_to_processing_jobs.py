"""add_recovery_fields_to_processing_jobs

Revision ID: 07a8e13254c1
Revises: bf033e49b411
Create Date: 2026-11-15 23:53:54.772757

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '07a8e13254c1'
down_revision: Union[str, Sequence[str], None] = 'bf033e49b411'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema to add job recovery columns to processing_jobs."""
    with op.batch_alter_table('processing_jobs', schema=None) as batch_op:
        batch_op.add_column(
            sa.Column('attempt_count', sa.Integer(), nullable=False, server_default='0')
        )
        batch_op.add_column(
            sa.Column('max_attempts', sa.Integer(), nullable=False, server_default='2')
        )
        batch_op.add_column(
            sa.Column('payload', sa.Text(), nullable=True)
        )
        batch_op.add_column(
            sa.Column('started_at', sa.DateTime(), nullable=True)
        )
        batch_op.add_column(
            sa.Column('completed_at', sa.DateTime(), nullable=True)
        )


def downgrade() -> None:
    """Downgrade schema to remove job recovery columns from processing_jobs."""
    with op.batch_alter_table('processing_jobs', schema=None) as batch_op:
        batch_op.drop_column('completed_at')
        batch_op.drop_column('started_at')
        batch_op.drop_column('payload')
        batch_op.drop_column('max_attempts')
        batch_op.drop_column('attempt_count')
