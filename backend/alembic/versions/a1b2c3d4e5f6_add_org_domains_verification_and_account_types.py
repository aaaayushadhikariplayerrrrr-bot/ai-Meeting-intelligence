"""add_org_domains_verification_and_account_types

Revision ID: a1b2c3d4e5f6
Revises: e2a1b3c4d5e6
Create Date: 2026-11-23 23:25:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, Sequence[str], None] = 'e2a1b3c4d5e6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add company domain, verification status, and account type to organizations and users."""
    with op.batch_alter_table('organizations', schema=None) as batch_op:
        batch_op.add_column(sa.Column('company_domain', sa.String(length=255), nullable=True))
        batch_op.add_column(sa.Column('is_domain_verified', sa.Boolean(), server_default='1', nullable=False))
        batch_op.add_column(sa.Column('domain_verification_token', sa.String(length=255), nullable=True))
        batch_op.add_column(sa.Column('domain_verification_method', sa.String(length=50), nullable=True))
        batch_op.add_column(sa.Column('account_type', sa.String(length=50), server_default='ORGANIZATION', nullable=False))
        batch_op.create_index(batch_op.f('ix_organizations_company_domain'), ['company_domain'], unique=False)

    with op.batch_alter_table('users', schema=None) as batch_op:
        batch_op.add_column(sa.Column('is_verified', sa.Boolean(), server_default='1', nullable=False))
        batch_op.add_column(sa.Column('verification_token', sa.String(length=255), nullable=True))


def downgrade() -> None:
    """Revert domain verification and account type fields."""
    with op.batch_alter_table('users', schema=None) as batch_op:
        batch_op.drop_column('verification_token')
        batch_op.drop_column('is_verified')

    with op.batch_alter_table('organizations', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_organizations_company_domain'))
        batch_op.drop_column('account_type')
        batch_op.drop_column('domain_verification_method')
        batch_op.drop_column('domain_verification_token')
        batch_op.drop_column('is_domain_verified')
        batch_op.drop_column('company_domain')
