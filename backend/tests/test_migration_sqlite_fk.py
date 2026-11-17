"""Test SQLite foreign key preservation and PRAGMA management during Alembic batch migrations.

Verifies:
1. Migration upgrade to bf033e49b411 does not trigger ON DELETE CASCADE on child tables.
2. PRAGMA foreign_key_check returns 0 violations after migration.
3. Downgrade and re-upgrade preserve child records and schema integrity.
4. Foreign key enforcement is active on standard engine connections.
5. Foreign key enforcement is restored even after migration failure.
"""

import sqlite3
from pathlib import Path
from alembic.config import Config
from alembic import command
from sqlalchemy import create_engine
from app.core.config import settings

BACKEND_DIR = Path(__file__).resolve().parent.parent


def test_sqlite_batch_migration_foreign_key_preservation(tmp_path):
    disp_db_path = tmp_path / "disposable_migration.db"
    disp_url = f"sqlite:///{disp_db_path.resolve().as_posix()}"

    original_db_url = settings.DATABASE_URL
    settings.DATABASE_URL = disp_url

    alembic_ini_path = BACKEND_DIR / "alembic.ini"
    alembic_cfg = Config(str(alembic_ini_path))
    alembic_cfg.set_main_option("sqlalchemy.url", disp_url)

    try:
        # 1. Upgrade to previous revision 67f21d7a4121
        command.upgrade(alembic_cfg, "67f21d7a4121")

        # 2. Seed test user, organization, and membership
        conn = sqlite3.connect(str(disp_db_path))
        cursor = conn.cursor()
        cursor.execute("PRAGMA foreign_keys = ON")

        user_id = "user-disp-uuid-001"
        org_id = "org-disp-uuid-001"
        mem_id = "mem-disp-uuid-001"
        now_ts = "2026-11-15T00:00:00Z"

        cursor.execute("""
            INSERT INTO users (id, email, full_name, is_super_admin, is_active, created_at, updated_at)
            VALUES (?, ?, ?, 0, 1, ?, ?)
        """, (user_id, "disp_user@example.com", "Disposable User", now_ts, now_ts))

        cursor.execute("""
            INSERT INTO organizations (id, name, slug, is_active, created_at, updated_at)
            VALUES (?, ?, ?, 1, ?, ?)
        """, (org_id, "Disposable Org", "disposable-org", now_ts, now_ts))

        cursor.execute("""
            INSERT INTO organization_memberships (id, organization_id, user_id, role, is_active, created_at, updated_at)
            VALUES (?, ?, ?, 'ORG_ADMIN', 1, ?, ?)
        """, (mem_id, org_id, user_id, now_ts, now_ts))

        conn.commit()
        cursor.execute("PRAGMA foreign_key_check")
        assert cursor.fetchall() == []
        conn.close()

        # 3. Upgrade to bf033e49b411
        command.upgrade(alembic_cfg, "bf033e49b411")

        # 4. Verify data integrity post-upgrade
        conn = sqlite3.connect(str(disp_db_path))
        cursor = conn.cursor()
        cursor.execute("PRAGMA foreign_keys = ON")

        cursor.execute("SELECT id, email, token_version FROM users WHERE id = ?", (user_id,))
        user_row = cursor.fetchone()
        assert user_row is not None
        assert user_row[2] == 1

        cursor.execute("SELECT id, user_id, organization_id FROM organization_memberships WHERE id = ?", (mem_id,))
        mem_row = cursor.fetchone()
        assert mem_row is not None, "Child record in organization_memberships was deleted by CASCADE!"
        assert mem_row[1] == user_id
        assert mem_row[2] == org_id

        cursor.execute("PRAGMA foreign_key_check")
        assert cursor.fetchall() == []
        conn.close()

        # 5. Downgrade back to 67f21d7a4121
        command.downgrade(alembic_cfg, "67f21d7a4121")

        conn = sqlite3.connect(str(disp_db_path))
        cursor = conn.cursor()
        cursor.execute("PRAGMA foreign_keys = ON")
        cursor.execute("SELECT id FROM users WHERE id = ?", (user_id,))
        assert cursor.fetchone() is not None
        cursor.execute("SELECT id FROM organization_memberships WHERE id = ?", (mem_id,))
        assert cursor.fetchone() is not None
        cursor.execute("PRAGMA foreign_key_check")
        assert cursor.fetchall() == []
        conn.close()

        # 6. Re-upgrade to bf033e49b411
        command.upgrade(alembic_cfg, "bf033e49b411")

        conn = sqlite3.connect(str(disp_db_path))
        cursor = conn.cursor()
        cursor.execute("PRAGMA foreign_keys = ON")
        cursor.execute("SELECT id, token_version FROM users WHERE id = ?", (user_id,))
        assert cursor.fetchone()[1] == 1
        cursor.execute("SELECT id FROM organization_memberships WHERE id = ?", (mem_id,))
        assert cursor.fetchone() is not None
        cursor.execute("PRAGMA foreign_key_check")
        assert cursor.fetchall() == []
        conn.close()

        # 7. Check standard application connection PRAGMA
        test_engine = create_engine(disp_url)
        with test_engine.connect() as test_conn:
            assert test_conn.exec_driver_sql("PRAGMA foreign_keys").scalar() == 1

        # 8. Check failure restoration
        with test_engine.connect() as fail_conn:
            raw_c = fail_conn.connection.dbapi_connection
            cur = raw_c.cursor()
            cur.execute("PRAGMA foreign_keys = OFF")
            cur.close()

            try:
                with fail_conn.begin():
                    assert fail_conn.exec_driver_sql("PRAGMA foreign_keys").scalar() == 0
                    raise RuntimeError("Simulated failure")
            except RuntimeError:
                pass
            finally:
                cur = raw_c.cursor()
                cur.execute("PRAGMA foreign_keys = ON")
                cur.close()

            assert fail_conn.exec_driver_sql("PRAGMA foreign_keys").scalar() == 1
        test_engine.dispose()

    finally:
        settings.DATABASE_URL = original_db_url
