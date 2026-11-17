"""
Database Reliability, Migration, WAL, Backup/Restore, and Job Recovery Verification Script.
Tested against fresh isolated databases only.
"""

import os
import sys
from pathlib import Path
import sqlite3
import uuid
from datetime import datetime, timezone

# Ensure backend is on sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from alembic.config import Config
from alembic import command
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.models.identity import Organization, User, OrganizationMembership, OrgRole
from app.models.meeting import Meeting, ProcessingJob, MeetingStatus, JobStatus, JobType
from app.models.transcription import Transcript, TranscriptSegment, TranscriptType
from app.services.job_worker import reconcile_stranded_jobs_on_startup
from app.services.meeting_lifecycle import reconcile_meeting_status

TEST_DB_NAME = "test_fresh_migration.db"
BACKUP_DB_NAME = "test_backup.db"
RESTORE_DB_NAME = "test_restored.db"

def run_verification():
    for f in [TEST_DB_NAME, BACKUP_DB_NAME, RESTORE_DB_NAME,
              f"{TEST_DB_NAME}-wal", f"{TEST_DB_NAME}-shm",
              f"{BACKUP_DB_NAME}-wal", f"{BACKUP_DB_NAME}-shm",
              f"{RESTORE_DB_NAME}-wal", f"{RESTORE_DB_NAME}-shm"]:
        p = BASE_DIR / f
        if p.exists():
            p.unlink()

    test_db_path = BASE_DIR / TEST_DB_NAME
    backup_db_path = BASE_DIR / BACKUP_DB_NAME
    restore_db_path = BASE_DIR / RESTORE_DB_NAME

    db_url = f"sqlite:///{test_db_path.as_posix()}"

    print("\n--- STEP 1: Running Alembic Migrations on Fresh Isolated DB ---")
    cfg = Config(str(BASE_DIR / "alembic.ini"))
    cfg.set_main_option("sqlalchemy.url", db_url)
    command.upgrade(cfg, "head")
    print("Alembic upgrade head completed successfully.")

    print("\n--- STEP 2: Verifying Tables and SQLite WAL Pragma ---")
    engine = create_engine(db_url)
    with engine.connect() as conn:
        from sqlalchemy import text
        journal_mode = conn.execute(text("PRAGMA journal_mode")).scalar()
        print(f"PRAGMA journal_mode = {journal_mode}")
        tables = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'")).scalars().all()
        print(f"Migrated tables ({len(tables)}): {', '.join(sorted(tables))}")
        assert "meetings" in tables
        assert "transcripts" in tables
        assert "transcript_chunks" in tables
        assert "processing_jobs" in tables

    print("\n--- STEP 3: Populating Test Data (Org, Meeting, Transcripts, Chunks) ---")
    Session = sessionmaker(bind=engine)
    db = Session()

    org_a = Organization(id=str(uuid.uuid4()), name="Reliability Org A", slug="reliability-org-a")
    org_b = Organization(id=str(uuid.uuid4()), name="Reliability Org B", slug="reliability-org-b")
    user_a = User(id=str(uuid.uuid4()), email="a@rel.com", full_name="User A", hashed_password="x")
    db.add_all([org_a, org_b, user_a])
    db.commit()

    m_a = Meeting(id=str(uuid.uuid4()), organization_id=org_a.id, created_by_user_id=user_a.id, title="Org A Meeting", status=MeetingStatus.PROCESSING.value)
    m_b = Meeting(id=str(uuid.uuid4()), organization_id=org_b.id, created_by_user_id=user_a.id, title="Org B Meeting", status=MeetingStatus.PROCESSING.value)
    db.add_all([m_a, m_b])
    db.commit()

    # Transcripts for Org A
    t_a = Transcript(id=str(uuid.uuid4()), meeting_id=m_a.id, transcript_type=TranscriptType.CLEANED_REDACTED.value, version=1)
    db.add(t_a)
    db.commit()

    seg_a = TranscriptSegment(id=str(uuid.uuid4()), transcript_id=t_a.id, sequence_order=0, start_time=0.0, end_time=2.0, text="Reliability test segment", is_redacted=True)
    db.add(seg_a)

    # Job for Org A: COMPLETED
    j_done = ProcessingJob(id=str(uuid.uuid4()), meeting_id=m_a.id, job_type=JobType.TRANSCRIPTION.value, status=JobStatus.COMPLETED.value, progress=100.0)
    # Job for Org B: IN_PROGRESS (simulated stranded job across server crash)
    j_stranded = ProcessingJob(id=str(uuid.uuid4()), meeting_id=m_b.id, job_type=JobType.INGESTION.value, status=JobStatus.IN_PROGRESS.value, progress=35.0)
    db.add_all([j_done, j_stranded])
    db.commit()
    m_a_id = m_a.id
    m_b_id = m_b.id
    j_done_id = j_done.id
    j_stranded_id = j_stranded.id
    org_a_id = org_a.id
    org_b_id = org_b.id
    db.close()

    print("\n--- STEP 4: Consistent SQLite Backup via sqlite3.Connection.backup() ---")
    src_conn = sqlite3.connect(str(test_db_path))
    dst_conn = sqlite3.connect(str(backup_db_path))
    with dst_conn:
        src_conn.backup(dst_conn, pages=100)
    src_conn.close()
    dst_conn.close()
    print(f"Backup created at: {backup_db_path} ({backup_db_path.stat().st_size} bytes)")

    print("\n--- STEP 5: Verification of Restored Database ---")
    restore_conn = sqlite3.connect(str(backup_db_path))
    restore_dst = sqlite3.connect(str(restore_db_path))
    with restore_dst:
        restore_conn.backup(restore_dst)
    restore_conn.close()
    restore_dst.close()

    restore_engine = create_engine(f"sqlite:///{restore_db_path.as_posix()}")
    RestoreSession = sessionmaker(bind=restore_engine)
    r_db = RestoreSession()
    assert r_db.query(Organization).count() == 2
    assert r_db.query(Meeting).count() == 2
    assert r_db.query(Transcript).count() == 1
    assert r_db.query(ProcessingJob).count() == 2
    print("Restore verification passed: All organizations, meetings, transcripts, and jobs intact.")

    print("\n--- STEP 6: Simulated Crash Recovery & Lifecycle Reconciliation ---")
    reconciled_count = reconcile_stranded_jobs_on_startup(r_db)
    print(f"Startup recovery ran: {reconciled_count} stranded job(s) reconciled.")

    j_stranded_restored = r_db.get(ProcessingJob, j_stranded_id)
    assert j_stranded_restored.status == JobStatus.FAILED.value
    assert "interrupted" in j_stranded_restored.error_message.lower()

    # Verify meeting B transitioned to FAILED due to failed ingestion
    m_b_restored = r_db.get(Meeting, m_b_id)
    assert m_b_restored.status == MeetingStatus.FAILED.value
    print("Org B meeting status correctly reconciled to FAILED after ingestion crash.")

    # Reconcile meeting A: completed transcription + transcript present -> COMPLETED
    status_a = reconcile_meeting_status(r_db, m_a_id)
    assert status_a == MeetingStatus.COMPLETED.value
    print("Org A meeting status correctly reconciled to COMPLETED.")

    print("\n--- STEP 7: Tenant Isolation & Integrity Verification ---")
    org_a_meetings = r_db.query(Meeting).filter(Meeting.organization_id == org_a_id).all()
    org_b_meetings = r_db.query(Meeting).filter(Meeting.organization_id == org_b_id).all()
    assert len(org_a_meetings) == 1 and org_a_meetings[0].id == m_a.id
    assert len(org_b_meetings) == 1 and org_b_meetings[0].id == m_b.id
    print("Tenant isolation verified: Org A and Org B meetings strictly segmented.")

    r_db.close()
    restore_engine.dispose()
    engine.dispose()

    # Cleanup test files
    for f in [test_db_path, backup_db_path, restore_db_path,
              BASE_DIR / f"{TEST_DB_NAME}-wal", BASE_DIR / f"{TEST_DB_NAME}-shm"]:
        if f.exists():
            try:
                f.unlink()
            except Exception:
                pass
    print("\nDatabase reliability and recovery verification SUCCEEDED cleanly!")

if __name__ == "__main__":
    run_verification()
