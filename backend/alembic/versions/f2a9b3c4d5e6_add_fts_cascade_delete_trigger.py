"""add_fts_cascade_delete_trigger

Revision ID: f2a9b3c4d5e6
Revises: e1f8b2c4a9d0
Create Date: 2026-11-16 17:35:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f2a9b3c4d5e6'
down_revision: Union[str, Sequence[str], None] = 'e1f8b2c4a9d0'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create SQLite trigger to synchronize FTS5 deletions on cascading or direct deletes."""
    bind = op.get_bind()
    if bind.dialect.name == "sqlite":
        bind.execute(sa.text(
            "CREATE TRIGGER IF NOT EXISTS trg_transcript_chunks_fts_delete "
            "AFTER DELETE ON transcript_chunks "
            "BEGIN "
            "    DELETE FROM transcript_chunks_fts WHERE chunk_id = old.id; "
            "END;"
        ))


def downgrade() -> None:
    """Drop SQLite trigger for FTS5 deletion synchronization."""
    bind = op.get_bind()
    if bind.dialect.name == "sqlite":
        bind.execute(sa.text("DROP TRIGGER IF EXISTS trg_transcript_chunks_fts_delete"))
