"""phase6_clean_org_roles_and_seed_dev_membership

Revision ID: 67f21d7a4121
Revises: 5cf1948d99a1
Create Date: 2026-11-15 01:17:32.646268

Step 3B:
- Refines role model semantics by migrating legacy 'SUPER_ADMIN' membership roles
  to 'ORG_ADMIN'.
- Seeds the missing OrganizationMembership linking the existing 'dev@local.internal'
  user to 'default-dev-org' with role 'ORG_ADMIN'.
- Preserves all existing users, organizations, meetings, media assets, processing jobs,
  and audit logs intact without altering password hashes or creating orphan records.
"""
from typing import Sequence, Union
import uuid
from datetime import datetime, timezone

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '67f21d7a4121'
down_revision: Union[str, Sequence[str], None] = '5cf1948d99a1'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade database data to clean org roles and seed development membership."""
    conn = op.get_bind()

    # 1. Handle legacy 'SUPER_ADMIN' role conversions
    # Track converted membership IDs to ensure exact, non-destructive restoration on downgrade
    conn.execute(sa.text("""
        CREATE TABLE IF NOT EXISTS _migration_phase6_legacy_roles (
            membership_id VARCHAR(36) PRIMARY KEY
        )
    """))

    legacy_rows = conn.execute(
        sa.text("SELECT id FROM organization_memberships WHERE role = 'SUPER_ADMIN'")
    ).fetchall()
    for row in legacy_rows:
        conn.execute(
            sa.text("INSERT OR IGNORE INTO _migration_phase6_legacy_roles (membership_id) VALUES (:mid)"),
            {"mid": row[0]}
        )
    if legacy_rows:
        conn.execute(
            sa.text("UPDATE organization_memberships SET role = 'ORG_ADMIN' WHERE role = 'SUPER_ADMIN'")
        )

    # 2. Seed development membership linking dev@local.internal to default-dev-org
    # Track inserted ID to ensure only the inserted record is deleted on downgrade
    conn.execute(sa.text("""
        CREATE TABLE IF NOT EXISTS _migration_phase6_seeded_membership (
            membership_id VARCHAR(36) PRIMARY KEY
        )
    """))

    dev_user = conn.execute(
        sa.text("SELECT id FROM users WHERE email = :email"),
        {"email": "dev@local.internal"}
    ).fetchone()

    dev_org = conn.execute(
        sa.text("SELECT id FROM organizations WHERE slug = :slug"),
        {"slug": "default-dev-org"}
    ).fetchone()

    # Only insert if both existing stable entities are present
    if dev_user and dev_org:
        user_id = dev_user[0]
        org_id = dev_org[0]

        # Verify whether membership already exists to prevent duplicate insertion
        existing = conn.execute(
            sa.text("SELECT id FROM organization_memberships WHERE organization_id = :org_id AND user_id = :user_id"),
            {"org_id": org_id, "user_id": user_id}
        ).fetchone()

        if not existing:
            new_id = str(uuid.uuid4())
            now_iso = datetime.now(timezone.utc).isoformat()

            conn.execute(
                sa.text("""
                    INSERT INTO organization_memberships (
                        id, organization_id, user_id, role, is_active, created_at, updated_at
                    ) VALUES (
                        :id, :org_id, :user_id, :role, :is_active, :created_at, :updated_at
                    )
                """),
                {
                    "id": new_id,
                    "org_id": org_id,
                    "user_id": user_id,
                    "role": "ORG_ADMIN",
                    "is_active": True,
                    "created_at": now_iso,
                    "updated_at": now_iso,
                }
            )
            conn.execute(
                sa.text("INSERT OR IGNORE INTO _migration_phase6_seeded_membership (membership_id) VALUES (:mid)"),
                {"mid": new_id}
            )


def downgrade() -> None:
    """Downgrade schema.
    
    Safety guarantees:
    - Deletes ONLY the specific development membership inserted by this migration.
    - Never deletes pre-existing memberships.
    - Restores ONLY legacy rows that were explicitly converted from SUPER_ADMIN to ORG_ADMIN.
    - Preserves all users, organizations, meetings, media assets, processing jobs, and audit logs.
    - Cleans up temporary tracking tables.
    """
    conn = op.get_bind()

    # 1. Remove only the seeded development membership inserted by upgrade
    has_seed_tracker = conn.execute(
        sa.text("SELECT name FROM sqlite_master WHERE type='table' AND name='_migration_phase6_seeded_membership'")
    ).fetchone()
    if has_seed_tracker:
        seeded_ids = [
            r[0] for r in conn.execute(sa.text("SELECT membership_id FROM _migration_phase6_seeded_membership")).fetchall()
        ]
        for mid in seeded_ids:
            conn.execute(
                sa.text("DELETE FROM organization_memberships WHERE id = :mid"),
                {"mid": mid}
            )
        conn.execute(sa.text("DROP TABLE IF EXISTS _migration_phase6_seeded_membership"))

    # 2. Restore only legacy rows that were explicitly converted from SUPER_ADMIN
    has_legacy_tracker = conn.execute(
        sa.text("SELECT name FROM sqlite_master WHERE type='table' AND name='_migration_phase6_legacy_roles'")
    ).fetchone()
    if has_legacy_tracker:
        legacy_ids = [
            r[0] for r in conn.execute(sa.text("SELECT membership_id FROM _migration_phase6_legacy_roles")).fetchall()
        ]
        for mid in legacy_ids:
            conn.execute(
                sa.text("UPDATE organization_memberships SET role = 'SUPER_ADMIN' WHERE id = :mid"),
                {"mid": mid}
            )
        conn.execute(sa.text("DROP TABLE IF EXISTS _migration_phase6_legacy_roles"))
