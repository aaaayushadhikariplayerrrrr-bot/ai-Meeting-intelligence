"""add_token_version_to_users

Revision ID: bf033e49b411
Revises: 67f21d7a4121
Create Date: 2026-11-15 23:12:33.461157

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'bf033e49b411'
down_revision: Union[str, Sequence[str], None] = '67f21d7a4121'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema to add token_version column to users."""
    with op.batch_alter_table('users', schema=None) as batch_op:
        batch_op.add_column(
            sa.Column('token_version', sa.Integer(), nullable=False, server_default='1')
        )


def downgrade() -> None:
    """Downgrade schema to remove token_version column from users."""
    with op.batch_alter_table('users', schema=None) as batch_op:
        batch_op.drop_column('token_version')
