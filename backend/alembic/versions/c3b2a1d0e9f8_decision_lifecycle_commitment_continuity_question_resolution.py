"""decision_lifecycle_commitment_continuity_question_resolution

Revision ID: c3b2a1d0e9f8
Revises: f2a9b3c4d5e6
Create Date: 2026-11-16 21:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c3b2a1d0e9f8'
down_revision: Union[str, Sequence[str], None] = 'f2a9b3c4d5e6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade database schema for decision lifecycle, commitment continuity, and question resolution."""
    # 1. Update decisions table
    with op.batch_alter_table('decisions', schema=None) as batch_op:
        batch_op.add_column(
            sa.Column('status', sa.String(length=50), nullable=False, server_default='ACTIVE')
        )
        batch_op.add_column(
            sa.Column('supersedes_decision_id', sa.String(length=36), nullable=True)
        )
        batch_op.create_index(batch_op.f('ix_decisions_status'), ['status'], unique=False)
        batch_op.create_index(
            batch_op.f('ix_decisions_supersedes_decision_id'),
            ['supersedes_decision_id'],
            unique=False
        )
        batch_op.create_foreign_key(
            'fk_decisions_supersedes_decision_id',
            'decisions',
            ['supersedes_decision_id'],
            ['id'],
            ondelete='SET NULL'
        )
        batch_op.create_check_constraint(
            'ck_decisions_no_self_supersede',
            'supersedes_decision_id IS NULL OR supersedes_decision_id != id'
        )

    # 2. Update action_items table
    with op.batch_alter_table('action_items', schema=None) as batch_op:
        batch_op.add_column(
            sa.Column('last_reviewed_meeting_id', sa.String(length=36), nullable=True)
        )
        batch_op.add_column(
            sa.Column('last_evidence_segment_id', sa.String(length=36), nullable=True)
        )
        batch_op.add_column(
            sa.Column('last_reviewed_at', sa.DateTime(timezone=True), nullable=True)
        )
        batch_op.create_index(
            batch_op.f('ix_action_items_last_reviewed_meeting_id'),
            ['last_reviewed_meeting_id'],
            unique=False
        )
        batch_op.create_index(
            batch_op.f('ix_action_items_last_evidence_segment_id'),
            ['last_evidence_segment_id'],
            unique=False
        )
        batch_op.create_foreign_key(
            'fk_action_items_last_reviewed_meeting_id',
            'meetings',
            ['last_reviewed_meeting_id'],
            ['id'],
            ondelete='SET NULL'
        )
        batch_op.create_foreign_key(
            'fk_action_items_last_evidence_segment_id',
            'transcript_segments',
            ['last_evidence_segment_id'],
            ['id'],
            ondelete='SET NULL'
        )

    # 3. Update unresolved_questions table
    with op.batch_alter_table('unresolved_questions', schema=None) as batch_op:
        batch_op.add_column(
            sa.Column('resolved_at', sa.DateTime(timezone=True), nullable=True)
        )
        batch_op.add_column(
            sa.Column('resolving_meeting_id', sa.String(length=36), nullable=True)
        )
        batch_op.add_column(
            sa.Column('resolving_decision_id', sa.String(length=36), nullable=True)
        )
        batch_op.add_column(
            sa.Column('resolution_notes', sa.Text(), nullable=True)
        )
        batch_op.create_index(
            batch_op.f('ix_unresolved_questions_resolving_meeting_id'),
            ['resolving_meeting_id'],
            unique=False
        )
        batch_op.create_index(
            batch_op.f('ix_unresolved_questions_resolving_decision_id'),
            ['resolving_decision_id'],
            unique=False
        )
        batch_op.create_foreign_key(
            'fk_unresolved_questions_resolving_meeting_id',
            'meetings',
            ['resolving_meeting_id'],
            ['id'],
            ondelete='SET NULL'
        )
        batch_op.create_foreign_key(
            'fk_unresolved_questions_resolving_decision_id',
            'decisions',
            ['resolving_decision_id'],
            ['id'],
            ondelete='SET NULL'
        )


def downgrade() -> None:
    """Downgrade schema."""
    # 1. Downgrade unresolved_questions table
    with op.batch_alter_table('unresolved_questions', schema=None) as batch_op:
        batch_op.drop_constraint('fk_unresolved_questions_resolving_decision_id', type_='foreignkey')
        batch_op.drop_constraint('fk_unresolved_questions_resolving_meeting_id', type_='foreignkey')
        batch_op.drop_index(batch_op.f('ix_unresolved_questions_resolving_decision_id'))
        batch_op.drop_index(batch_op.f('ix_unresolved_questions_resolving_meeting_id'))
        batch_op.drop_column('resolution_notes')
        batch_op.drop_column('resolving_decision_id')
        batch_op.drop_column('resolving_meeting_id')
        batch_op.drop_column('resolved_at')

    # 2. Downgrade action_items table
    with op.batch_alter_table('action_items', schema=None) as batch_op:
        batch_op.drop_constraint('fk_action_items_last_evidence_segment_id', type_='foreignkey')
        batch_op.drop_constraint('fk_action_items_last_reviewed_meeting_id', type_='foreignkey')
        batch_op.drop_index(batch_op.f('ix_action_items_last_evidence_segment_id'))
        batch_op.drop_index(batch_op.f('ix_action_items_last_reviewed_meeting_id'))
        batch_op.drop_column('last_reviewed_at')
        batch_op.drop_column('last_evidence_segment_id')
        batch_op.drop_column('last_reviewed_meeting_id')

    # 3. Downgrade decisions table
    with op.batch_alter_table('decisions', schema=None) as batch_op:
        batch_op.drop_constraint('ck_decisions_no_self_supersede', type_='check')
        batch_op.drop_constraint('fk_decisions_supersedes_decision_id', type_='foreignkey')
        batch_op.drop_index(batch_op.f('ix_decisions_supersedes_decision_id'))
        batch_op.drop_index(batch_op.f('ix_decisions_status'))
        batch_op.drop_column('supersedes_decision_id')
        batch_op.drop_column('status')
