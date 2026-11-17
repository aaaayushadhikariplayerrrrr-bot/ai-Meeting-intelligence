"""add_action_item_reviews_table

Revision ID: d5e6f7a8b9c0
Revises: c3b2a1d0e9f8
Create Date: 2026-11-16 21:10:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd5e6f7a8b9c0'
down_revision: Union[str, Sequence[str], None] = 'c3b2a1d0e9f8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create action_item_reviews table for immutable cross-meeting commitment tracking."""
    op.create_table(
        'action_item_reviews',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('action_item_id', sa.String(length=36), nullable=False),
        sa.Column('meeting_id', sa.String(length=36), nullable=False),
        sa.Column('evidence_segment_id', sa.String(length=36), nullable=True),
        sa.Column('reviewer_user_id', sa.String(length=36), nullable=True),
        sa.Column('status_at_review', sa.String(length=50), nullable=True),
        sa.Column('review_notes', sa.Text(), nullable=True),
        sa.Column('reviewed_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['action_item_id'], ['action_items.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['meeting_id'], ['meetings.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['evidence_segment_id'], ['transcript_segments.id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['reviewer_user_id'], ['users.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id'),
    )
    with op.batch_alter_table('action_item_reviews', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_action_item_reviews_id'), ['id'], unique=False)
        batch_op.create_index(batch_op.f('ix_action_item_reviews_action_item_id'), ['action_item_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_action_item_reviews_meeting_id'), ['meeting_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_action_item_reviews_evidence_segment_id'), ['evidence_segment_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_action_item_reviews_reviewer_user_id'), ['reviewer_user_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_action_item_reviews_created_at'), ['created_at'], unique=False)


def downgrade() -> None:
    """Drop action_item_reviews table."""
    with op.batch_alter_table('action_item_reviews', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_action_item_reviews_created_at'))
        batch_op.drop_index(batch_op.f('ix_action_item_reviews_reviewer_user_id'))
        batch_op.drop_index(batch_op.f('ix_action_item_reviews_evidence_segment_id'))
        batch_op.drop_index(batch_op.f('ix_action_item_reviews_meeting_id'))
        batch_op.drop_index(batch_op.f('ix_action_item_reviews_action_item_id'))
        batch_op.drop_index(batch_op.f('ix_action_item_reviews_id'))

    op.drop_table('action_item_reviews')
