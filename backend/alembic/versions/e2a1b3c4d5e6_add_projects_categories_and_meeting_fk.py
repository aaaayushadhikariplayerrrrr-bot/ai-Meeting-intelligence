"""add_projects_categories_and_meeting_fk

Revision ID: e2a1b3c4d5e6
Revises: d5e6f7a8b9c0
Create Date: 2026-11-22 16:10:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e2a1b3c4d5e6'
down_revision: Union[str, Sequence[str], None] = 'd5e6f7a8b9c0'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create projects table and add category/project_id to meetings table."""
    # 1. Create projects table
    op.create_table(
        'projects',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('organization_id', sa.String(length=36), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('slug', sa.String(length=255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('created_by_user_id', sa.String(length=36), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['created_by_user_id'], ['users.id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['organization_id'], ['organizations.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('organization_id', 'slug', name='uq_projects_organization_slug'),
    )
    with op.batch_alter_table('projects', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_projects_id'), ['id'], unique=False)
        batch_op.create_index(batch_op.f('ix_projects_organization_id'), ['organization_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_projects_slug'), ['slug'], unique=False)
        batch_op.create_index(batch_op.f('ix_projects_created_by_user_id'), ['created_by_user_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_projects_created_at'), ['created_at'], unique=False)

    # 2. Add category and project_id to meetings table
    with op.batch_alter_table('meetings', schema=None) as batch_op:
        batch_op.add_column(sa.Column('category', sa.String(length=50), nullable=True))
        batch_op.add_column(sa.Column('project_id', sa.String(length=36), nullable=True))
        batch_op.create_index(batch_op.f('ix_meetings_category'), ['category'], unique=False)
        batch_op.create_index(batch_op.f('ix_meetings_project_id'), ['project_id'], unique=False)
        batch_op.create_foreign_key('fk_meetings_project_id', 'projects', ['project_id'], ['id'], ondelete='SET NULL')


def downgrade() -> None:
    """Drop category/project_id from meetings and drop projects table."""
    with op.batch_alter_table('meetings', schema=None) as batch_op:
        batch_op.drop_constraint('fk_meetings_project_id', type_='foreignkey')
        batch_op.drop_index(batch_op.f('ix_meetings_project_id'))
        batch_op.drop_index(batch_op.f('ix_meetings_category'))
        batch_op.drop_column('project_id')
        batch_op.drop_column('category')

    with op.batch_alter_table('projects', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_projects_created_at'))
        batch_op.drop_index(batch_op.f('ix_projects_created_by_user_id'))
        batch_op.drop_index(batch_op.f('ix_projects_slug'))
        batch_op.drop_index(batch_op.f('ix_projects_organization_id'))
        batch_op.drop_index(batch_op.f('ix_projects_id'))

    op.drop_table('projects')
