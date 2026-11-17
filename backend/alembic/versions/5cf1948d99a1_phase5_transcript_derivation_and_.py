"""phase5_transcript_derivation_and_redaction

Revision ID: 5cf1948d99a1
Revises: 77e8e5bedff2
Create Date: 2026-11-15 00:29:14.979535

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '5cf1948d99a1'
down_revision: Union[str, Sequence[str], None] = '77e8e5bedff2'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    with op.batch_alter_table('transcript_segments', schema=None) as batch_op:
        batch_op.add_column(sa.Column('source_segment_id', sa.String(length=36), nullable=True))
        batch_op.add_column(sa.Column('redaction_metadata', sa.Text(), nullable=True))
        batch_op.create_index(batch_op.f('ix_transcript_segments_source_segment_id'), ['source_segment_id'], unique=False)
        batch_op.create_foreign_key('fk_transcript_segments_source_segment_id', 'transcript_segments', ['source_segment_id'], ['id'], ondelete='SET NULL')

    with op.batch_alter_table('transcripts', schema=None) as batch_op:
        batch_op.add_column(sa.Column('parent_transcript_id', sa.String(length=36), nullable=True))
        batch_op.create_index(batch_op.f('ix_transcripts_parent_transcript_id'), ['parent_transcript_id'], unique=False)
        batch_op.create_foreign_key('fk_transcripts_parent_transcript_id', 'transcripts', ['parent_transcript_id'], ['id'], ondelete='SET NULL')


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table('transcripts', schema=None) as batch_op:
        batch_op.drop_constraint('fk_transcripts_parent_transcript_id', type_='foreignkey')
        batch_op.drop_index(batch_op.f('ix_transcripts_parent_transcript_id'))
        batch_op.drop_column('parent_transcript_id')

    with op.batch_alter_table('transcript_segments', schema=None) as batch_op:
        batch_op.drop_constraint('fk_transcript_segments_source_segment_id', type_='foreignkey')
        batch_op.drop_index(batch_op.f('ix_transcript_segments_source_segment_id'))
        batch_op.drop_column('redaction_metadata')
        batch_op.drop_column('source_segment_id')
