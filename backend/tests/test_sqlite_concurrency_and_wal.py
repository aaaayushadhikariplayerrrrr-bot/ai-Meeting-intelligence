"""Phase 9.2: SQLite Concurrency & WAL Hardening Tests.

Verifies:
1. SQLite PRAGMAs (WAL journal mode, busy_timeout=30000, synchronous=NORMAL, foreign_keys=ON).
2. Foreign key enforcement rejection of orphan records.
3. Concurrent reads and writes execute without locking errors under WAL mode.
4. Background memory indexing write coexists with simultaneous API-side reads.
5. Concurrent job claims do not process the same job twice (atomic claiming).
6. Transaction rollback on induced error restores clean state.
7. FTS5 and vector BLOB records remain consistent.
8. Tenant isolation remains strictly intact under concurrent sessions.
9. Remote / network filesystem detection prevents inappropriate WAL configuration.
"""

import os
import sys
import tempfile
import struct
import concurrent.futures
from datetime import datetime, timezone
from pathlib import Path
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import IntegrityError

from app.database.base import Base
from app.database.session import is_network_filesystem
from app.models.identity import User, Organization, OrganizationMembership, OrgRole
from app.models.meeting import Meeting, ProcessingJob, JobStatus, JobType, MeetingStatus
from app.models.transcription import Transcript, TranscriptType
from app.models.memory import TranscriptChunkRecord
from app.services.job_worker import claim_job_atomically
import app.models  # noqa: F401


@pytest.fixture
def temp_db():
    """Create an isolated, temporary file-based SQLite database for concurrency testing."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db_url = f"sqlite:///{Path(path).as_posix()}"

    engine = create_engine(
        db_url,
        connect_args={"check_same_thread": False, "timeout": 30.0},
        echo=False
    )
    Base.metadata.create_all(bind=engine)
    SessionFactory = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    yield {"path": path, "url": db_url, "engine": engine, "SessionFactory": SessionFactory}

    engine.dispose()
    for ext in ["", "-wal", "-shm"]:
        p = path + ext
        if os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass


def test_sqlite_wal_pragmas_configured(temp_db):
    """Test 1: Verify journal mode is WAL, busy_timeout is 30000ms, synchronous is NORMAL, and FK is ON."""
    engine = temp_db["engine"]
    with engine.connect() as conn:
        journal_mode = conn.exec_driver_sql("PRAGMA journal_mode;").scalar()
        busy_timeout = conn.exec_driver_sql("PRAGMA busy_timeout;").scalar()
        synchronous = conn.exec_driver_sql("PRAGMA synchronous;").scalar()
        foreign_keys = conn.exec_driver_sql("PRAGMA foreign_keys;").scalar()

    assert journal_mode.lower() == "wal", f"Expected WAL journal mode, got {journal_mode}"
    assert busy_timeout == 30000, f"Expected 30000ms busy timeout, got {busy_timeout}"
    assert synchronous == 1, f"Expected synchronous=NORMAL (1), got {synchronous}"
    assert foreign_keys == 1, f"Expected foreign_keys=ON (1), got {foreign_keys}"


def test_foreign_key_enforcement_intact(temp_db):
    """Test 2: Verify foreign key enforcement rejects orphan records and permits valid relationships."""
    SessionFactory = temp_db["SessionFactory"]
    db = SessionFactory()

    try:
        # Attempt to insert an orphan ProcessingJob without a parent Meeting
        orphan_job = ProcessingJob(
            id="job-orphan-001",
            meeting_id="non-existent-meeting-id",
            job_type=JobType.TRANSCRIPTION.value,
            status=JobStatus.PENDING.value,
            payload="{}",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        db.add(orphan_job)
        with pytest.raises(IntegrityError):
            db.commit()
        db.rollback()

        # Create valid parent hierarchy: Org -> Meeting -> ProcessingJob
        org = Organization(id="org-fk-001", name="FK Org", slug="fk-org", is_active=True)
        db.add(org)
        db.commit()

        meeting = Meeting(
            id="meeting-fk-001",
            organization_id=org.id,
            title="FK Meeting",
            status=MeetingStatus.COMPLETED.value,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        db.add(meeting)
        db.commit()

        valid_job = ProcessingJob(
            id="job-valid-001",
            meeting_id=meeting.id,
            job_type=JobType.TRANSCRIPTION.value,
            status=JobStatus.PENDING.value,
            payload="{}",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        db.add(valid_job)
        db.commit()

        queried_job = db.get(ProcessingJob, "job-valid-001")
        assert queried_job is not None
        assert queried_job.meeting_id == meeting.id
    finally:
        db.close()


def test_concurrent_reads_and_writes_no_lock_errors(temp_db):
    """Test 3: Verify multiple concurrent readers and writers operate without 'database is locked' errors."""
    SessionFactory = temp_db["SessionFactory"]

    # Seed an initial organization
    init_db = SessionFactory()
    org = Organization(id="org-conc-001", name="Concurrency Org", slug="conc-org", is_active=True)
    init_db.add(org)
    init_db.commit()
    init_db.close()

    errors = []

    def writer_task(worker_id: int):
        for i in range(10):
            db = SessionFactory()
            try:
                meeting = Meeting(
                    id=f"meeting-w{worker_id}-{i}",
                    organization_id="org-conc-001",
                    title=f"Worker {worker_id} Meeting {i}",
                    status=MeetingStatus.COMPLETED.value,
                    created_at=datetime.now(timezone.utc),
                    updated_at=datetime.now(timezone.utc),
                )
                db.add(meeting)
                db.commit()
            except Exception as e:
                errors.append(f"Writer {worker_id} iter {i} error: {e}")
            finally:
                db.close()

    def reader_task(worker_id: int):
        for i in range(20):
            db = SessionFactory()
            try:
                meetings = db.query(Meeting).filter(Meeting.organization_id == "org-conc-001").all()
                count = len(meetings)
                assert count >= 0
            except Exception as e:
                errors.append(f"Reader {worker_id} iter {i} error: {e}")
            finally:
                db.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        for w in range(4):
            futures.append(executor.submit(writer_task, w))
        for r in range(4):
            futures.append(executor.submit(reader_task, r))

        for f in concurrent.futures.as_completed(futures):
            f.result()

    assert errors == [], f"Encountered concurrency errors: {errors}"

    verify_db = SessionFactory()
    total_meetings = verify_db.query(Meeting).filter(Meeting.organization_id == "org-conc-001").count()
    verify_db.close()
    assert total_meetings == 40, f"Expected 40 meetings written concurrently, got {total_meetings}"


def test_background_memory_write_coexists_with_api_reads(temp_db):
    """Test 4: Background memory indexing writes coexist with simultaneous API-side reads."""
    SessionFactory = temp_db["SessionFactory"]

    # Seed organization, meeting, and primary transcript
    init_db = SessionFactory()
    now = datetime.now(timezone.utc)
    org = Organization(id="org-mem-001", name="Mem Org", slug="mem-org", is_active=True)
    init_db.add(org)
    init_db.commit()

    meeting = Meeting(
        id="meeting-mem-001",
        organization_id=org.id,
        title="Memory Sync Meeting",
        status=MeetingStatus.COMPLETED.value,
        created_at=now,
        updated_at=now,
    )
    init_db.add(meeting)
    init_db.commit()

    transcript = Transcript(
        id="trans-mem-001",
        meeting_id=meeting.id,
        transcript_type=TranscriptType.CLEANED.value,
        version=1,
        is_primary=True,
        created_at=now,
        updated_at=now,
    )
    init_db.add(transcript)
    init_db.commit()
    init_db.close()

    errors = []
    embedding_dim = 384

    def background_indexer():
        for i in range(15):
            db = SessionFactory()
            try:
                synthetic_vector = [0.01 * (i + j) for j in range(embedding_dim)]
                blob = struct.pack(f"{embedding_dim}f", *synthetic_vector)
                chunk = TranscriptChunkRecord(
                    id=f"chunk-{i}",
                    organization_id="org-mem-001",
                    meeting_id="meeting-mem-001",
                    transcript_id="trans-mem-001",
                    chunk_index=i,
                    text=f"Transcript chunk content for segment {i} discussing architecture.",
                    start_time=float(i * 10),
                    end_time=float(i * 10 + 9),
                    segment_ids="[]",
                    language_codes="[\"en\"]",
                    dominant_language="en",
                    token_count=10,
                    is_split_utterance=False,
                    model_name="all-MiniLM-L6-v2",
                    dimension=embedding_dim,
                    embedding=blob,
                    created_at=datetime.now(timezone.utc),
                    updated_at=datetime.now(timezone.utc),
                )
                db.add(chunk)
                db.commit()
            except Exception as e:
                errors.append(f"Indexer error iter {i}: {e}")
            finally:
                db.close()

    def api_reader():
        for i in range(30):
            db = SessionFactory()
            try:
                m = db.get(Meeting, "meeting-mem-001")
                assert m is not None
                chunks = db.query(TranscriptChunkRecord).filter(TranscriptChunkRecord.meeting_id == "meeting-mem-001").all()
                assert len(chunks) >= 0
            except Exception as e:
                errors.append(f"API reader error iter {i}: {e}")
            finally:
                db.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        f_idx = executor.submit(background_indexer)
        f_read1 = executor.submit(api_reader)
        f_read2 = executor.submit(api_reader)

        f_idx.result()
        f_read1.result()
        f_read2.result()

    assert errors == [], f"Encountered errors during indexing & reading: {errors}"

    verify_db = SessionFactory()
    chunk_count = verify_db.query(TranscriptChunkRecord).filter(TranscriptChunkRecord.meeting_id == "meeting-mem-001").count()
    verify_db.close()
    assert chunk_count == 15, f"Expected 15 chunks indexed, got {chunk_count}"


def test_concurrent_job_claims_atomic(temp_db):
    """Test 5: Concurrent workers calling claim_job_atomically process a job exactly once."""
    SessionFactory = temp_db["SessionFactory"]

    init_db = SessionFactory()
    org = Organization(id="org-atomic-001", name="Atomic Org", slug="atomic-org", is_active=True)
    init_db.add(org)
    init_db.commit()

    meeting = Meeting(
        id="meeting-atomic-001",
        organization_id=org.id,
        title="Atomic Meeting",
        status=MeetingStatus.INGESTING.value,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    init_db.add(meeting)
    init_db.commit()

    target_job_id = "job-atomic-claim-001"
    job = ProcessingJob(
        id=target_job_id,
        meeting_id=meeting.id,
        job_type=JobType.TRANSCRIPTION.value,
        status=JobStatus.PENDING.value,
        attempt_count=0,
        payload="{}",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    init_db.add(job)
    init_db.commit()
    init_db.close()

    claimed_by = []
    failed_claims = []

    def attempt_claim(worker_id: int):
        db = SessionFactory()
        try:
            claimed = claim_job_atomically(db, target_job_id)
            if claimed:
                claimed_by.append(worker_id)
            else:
                failed_claims.append(worker_id)
        finally:
            db.close()

    # 10 workers simultaneously attempt to claim the exact same pending job
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(attempt_claim, w) for w in range(10)]
        for f in concurrent.futures.as_completed(futures):
            f.result()

    assert len(claimed_by) == 1, f"Expected exactly 1 claim winner, got {len(claimed_by)}: {claimed_by}"
    assert len(failed_claims) == 9, f"Expected 9 failed claims, got {len(failed_claims)}"

    verify_db = SessionFactory()
    final_job = verify_db.get(ProcessingJob, target_job_id)
    assert final_job.status == JobStatus.IN_PROGRESS.value
    assert final_job.attempt_count == 1
    verify_db.close()


def test_transaction_rollback_on_induced_error(temp_db):
    """Test 6: Verify transaction rollback cleans up uncommitted data and preserves engine sanity."""
    SessionFactory = temp_db["SessionFactory"]

    init_db = SessionFactory()
    org = Organization(id="org-rb-001", name="Rollback Org", slug="rb-org", is_active=True)
    init_db.add(org)
    init_db.commit()
    init_db.close()

    # Open a session and simulate a failure during multi-entity write
    db = SessionFactory()
    try:
        meeting = Meeting(
            id="meeting-rb-001",
            organization_id="org-rb-001",
            title="Meeting To Rollback",
            status=MeetingStatus.COMPLETED.value,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        db.add(meeting)
        db.flush()

        # Induce an explicit error before commit
        raise RuntimeError("Simulated mid-transaction failure")
    except RuntimeError:
        db.rollback()
    finally:
        db.close()

    # Verify meeting was NOT persisted
    check_db = SessionFactory()
    rolled_back = check_db.get(Meeting, "meeting-rb-001")
    assert rolled_back is None, "Uncommitted meeting was found after rollback!"

    # Verify subsequent transactions succeed normally
    new_meeting = Meeting(
        id="meeting-rb-clean",
        organization_id="org-rb-001",
        title="Clean Meeting After Rollback",
        status=MeetingStatus.COMPLETED.value,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    check_db.add(new_meeting)
    check_db.commit()
    persisted = check_db.get(Meeting, "meeting-rb-clean")
    assert persisted is not None
    check_db.close()


def test_fts5_and_vector_records_consistency(temp_db):
    """Test 7: Verify FTS5 virtual table and vector BLOB storage integrity under concurrent session use."""
    SessionFactory = temp_db["SessionFactory"]
    engine = temp_db["engine"]

    # Seed organization, meeting, and transcript
    db = SessionFactory()
    now = datetime.now(timezone.utc)
    org = Organization(id="org-vec-001", name="Vec Org", slug="vec-org", is_active=True)
    db.add(org)
    meeting = Meeting(
        id="meeting-vec-001",
        organization_id=org.id,
        title="Vector & FTS5 Meeting",
        status=MeetingStatus.COMPLETED.value,
        created_at=now,
        updated_at=now,
    )
    db.add(meeting)
    db.commit()

    transcript = Transcript(
        id="trans-vec-001",
        meeting_id=meeting.id,
        transcript_type=TranscriptType.CLEANED.value,
        version=1,
        is_primary=True,
        created_at=now,
        updated_at=now,
    )
    db.add(transcript)
    db.commit()

    # Insert transcript chunk record with 384-dimensional vector embedding stored as BLOB
    embedding_dim = 384
    synthetic_vector = [0.05 * i for i in range(embedding_dim)]
    vector_blob = struct.pack(f"{embedding_dim}f", *synthetic_vector)

    chunk = TranscriptChunkRecord(
        id="chunk-vec-001",
        organization_id="org-vec-001",
        meeting_id=meeting.id,
        transcript_id=transcript.id,
        chunk_index=0,
        text="We discussed deploying Kubernetes cluster and migrating database.",
        token_count=10,
        start_time=0.0,
        end_time=5.0,
        segment_ids="[]",
        language_codes="[\"en\"]",
        dominant_language="en",
        is_split_utterance=False,
        model_name="all-MiniLM-L6-v2",
        dimension=embedding_dim,
        embedding=vector_blob,
        created_at=now,
        updated_at=now,
    )
    db.add(chunk)
    db.commit()

    # Query chunk and verify vector BLOB unpacking
    queried_chunk = db.get(TranscriptChunkRecord, "chunk-vec-001")
    assert queried_chunk is not None
    assert queried_chunk.embedding is not None
    assert len(queried_chunk.embedding) == embedding_dim * 4

    unpacked_vector = list(struct.unpack(f"{embedding_dim}f", queried_chunk.embedding))
    assert len(unpacked_vector) == embedding_dim
    assert abs(unpacked_vector[10] - (0.05 * 10)) < 1e-5

    # Check FTS5 availability on SQLite engine
    with engine.connect() as conn:
        try:
            conn.exec_driver_sql("CREATE VIRTUAL TABLE IF NOT EXISTS test_fts USING fts5(content);")
            conn.exec_driver_sql("INSERT INTO test_fts(content) VALUES ('Kubernetes deployment in production');")
            result = conn.exec_driver_sql("SELECT content FROM test_fts WHERE test_fts MATCH 'Kubernetes';").fetchall()
            assert len(result) == 1
            assert "Kubernetes" in result[0][0]
        except Exception as e:
            pytest.fail(f"FTS5 virtual table error: {e}")
    db.close()


def test_tenant_isolation_preserved_under_concurrency(temp_db):
    """Test 8: Verify tenant isolation is strictly preserved across concurrent sessions."""
    SessionFactory = temp_db["SessionFactory"]

    # Seed Org A and Org B
    db = SessionFactory()
    org_a = Organization(id="org-tenant-a", name="Tenant A", slug="tenant-a", is_active=True)
    org_b = Organization(id="org-tenant-b", name="Tenant B", slug="tenant-b", is_active=True)
    db.add_all([org_a, org_b])
    db.commit()

    meeting_a = Meeting(
        id="meeting-tenant-a",
        organization_id="org-tenant-a",
        title="Confidential Strategy Org A",
        status=MeetingStatus.COMPLETED.value,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    meeting_b = Meeting(
        id="meeting-tenant-b",
        organization_id="org-tenant-b",
        title="Confidential Strategy Org B",
        status=MeetingStatus.COMPLETED.value,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    db.add_all([meeting_a, meeting_b])
    db.commit()
    db.close()

    isolation_failures = []

    def query_org_a():
        for _ in range(20):
            session = SessionFactory()
            try:
                results = session.query(Meeting).filter(Meeting.organization_id == "org-tenant-a").all()
                for r in results:
                    if r.organization_id != "org-tenant-a" or "Org B" in r.title:
                        isolation_failures.append(f"Leak into Org A query: {r.id}, {r.title}")
            finally:
                session.close()

    def query_org_b():
        for _ in range(20):
            session = SessionFactory()
            try:
                results = session.query(Meeting).filter(Meeting.organization_id == "org-tenant-b").all()
                for r in results:
                    if r.organization_id != "org-tenant-b" or "Org A" in r.title:
                        isolation_failures.append(f"Leak into Org B query: {r.id}, {r.title}")
            finally:
                session.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        f_a1 = executor.submit(query_org_a)
        f_a2 = executor.submit(query_org_a)
        f_b1 = executor.submit(query_org_b)
        f_b2 = executor.submit(query_org_b)

        f_a1.result()
        f_a2.result()
        f_b1.result()
        f_b2.result()

    assert isolation_failures == [], f"Tenant isolation breaches detected: {isolation_failures}"


def test_network_filesystem_detection():
    """Test 9: Verify network filesystem detection identifies UNC paths and avoids inappropriate WAL."""
    assert is_network_filesystem("\\\\fileserver\\share\\meeting.db") is True
    assert is_network_filesystem("//fileserver/share/meeting.db") is True
    assert is_network_filesystem(":memory:") is False
    assert is_network_filesystem("./meeting_intelligence.db") is False
    assert is_network_filesystem("C:\\local\\data\\meeting.db") is False
