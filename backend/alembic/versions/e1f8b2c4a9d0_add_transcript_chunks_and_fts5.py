"""add_transcript_chunks_and_fts5

Revision ID: e1f8b2c4a9d0
Revises: d407b60ca77f
Create Date: 2026-11-16 17:15:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e1f8b2c4a9d0'
down_revision: Union[str, Sequence[str], None] = 'd407b60ca77f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema: create transcript_chunks table, indexes, and FTS5 virtual table."""
    op.create_table(
        'transcript_chunks',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('organization_id', sa.String(length=36), nullable=False),
        sa.Column('meeting_id', sa.String(length=36), nullable=False),
        sa.Column('transcript_id', sa.String(length=36), nullable=False),
        sa.Column('chunk_index', sa.Integer(), nullable=False),
        sa.Column('text', sa.Text(), nullable=False),
        sa.Column('start_time', sa.Float(), nullable=False),
        sa.Column('end_time', sa.Float(), nullable=False),
        sa.Column('segment_ids', sa.Text(), nullable=False),
        sa.Column('language_codes', sa.Text(), nullable=False),
        sa.Column('dominant_language', sa.String(length=20), server_default='en', nullable=False),
        sa.Column('token_count', sa.Integer(), nullable=False),
        sa.Column('is_split_utterance', sa.Boolean(), server_default=sa.text('0'), nullable=False),
        sa.Column('extra_metadata', sa.Text(), nullable=True),
        sa.Column('model_name', sa.String(length=100), server_default='intfloat/multilingual-e5-small', nullable=False),
        sa.Column('dimension', sa.Integer(), server_default='384', nullable=False),
        sa.Column('embedding', sa.LargeBinary(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(['meeting_id'], ['meetings.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['organization_id'], ['organizations.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['transcript_id'], ['transcripts.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_transcript_chunks_created_at', 'transcript_chunks', ['created_at'], unique=False)
    op.create_index('ix_transcript_chunks_id', 'transcript_chunks', ['id'], unique=False)
    op.create_index('ix_transcript_chunks_meeting_id', 'transcript_chunks', ['meeting_id'], unique=False)
    op.create_index('ix_transcript_chunks_organization_id', 'transcript_chunks', ['organization_id'], unique=False)
    op.create_index('ix_transcript_chunks_transcript_id', 'transcript_chunks', ['transcript_id'], unique=False)
    op.create_index('ix_transcript_chunks_org_meeting', 'transcript_chunks', ['organization_id', 'meeting_id'], unique=False)
    op.create_index('ix_transcript_chunks_org_transcript', 'transcript_chunks', ['organization_id', 'transcript_id'], unique=False)

    # Check FTS5 availability and create SQLite FTS5 virtual table
    bind = op.get_bind()
    if bind.dialect.name == "sqlite":
        try:
            bind.execute(sa.text("CREATE VIRTUAL TABLE IF NOT EXISTS _fts5_test USING fts5(x)"))
            bind.execute(sa.text("DROP TABLE _fts5_test"))
        except Exception as exc:
            raise RuntimeError(
                f"SQLite FTS5 extension is not available in the database environment: {exc}"
            ) from exc

        bind.execute(sa.text(
            "CREATE VIRTUAL TABLE IF NOT EXISTS transcript_chunks_fts USING fts5("
            "chunk_id UNINDEXED, text, tokenize='unicode61')"
        ))


def downgrade() -> None:
    """Downgrade schema: drop FTS5 virtual table, indexes, and transcript_chunks table."""
    bind = op.get_bind()
    if bind.dialect.name == "sqlite":
        bind.execute(sa.text("DROP TABLE IF EXISTS transcript_chunks_fts"))

    op.drop_index('ix_transcript_chunks_org_transcript', table_name='transcript_chunks')
    op.drop_index('ix_transcript_chunks_org_meeting', table_name='transcript_chunks')
    op.drop_index('ix_transcript_chunks_transcript_id', table_name='transcript_chunks')
    op.drop_index('ix_transcript_chunks_organization_id', table_name='transcript_chunks')
    op.drop_index('ix_transcript_chunks_meeting_id', table_name='transcript_chunks')
    op.drop_index('ix_transcript_chunks_id', table_name='transcript_chunks')
    op.drop_index('ix_transcript_chunks_created_at', table_name='transcript_chunks')
    op.drop_table('transcript_chunks')
