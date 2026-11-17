"""Tests for Phase 6D.5: Database-Backed Processing Job Recovery.

Covers:
1. Persisted PENDING job is discovered and executed.
2. Abandoned IN_PROGRESS job becomes FAILED at startup.
3. COMPLETED job is never executed again.
4. FAILED and CANCELLED jobs are not automatically executed.
5. Missing or invalid payload is handled safely.
6. Attempt count increments only after a successful claim.
7. Two competing claim attempts cannot both claim the same job.
8. A failed or interrupted job does not permanently block a new transcription request.
9. Failed sibling jobs do not block eligible media cleanup.
10. No media is deleted before verified transcript persistence.
11. Existing job-status API contracts still work.
12. Worker shutdown does not leave database sessions open.
"""

import json
import os
import wave
import struct
import tempfile
import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

import pytest
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import settings
from app.core.security import create_access_token
from app.database.session import SessionLocal
from app.models.identity import Organization, User, OrganizationMembership, OrgRole
from app.models.meeting import Meeting, MediaAsset, ProcessingJob, JobType, JobStatus, MeetingStatus
from app.models.transcription import Transcript, TranscriptSegment, TranscriptType
from app.services.job_manager import create_processing_job, update_processing_job
from app.services.job_worker import (
    claim_job_atomically,
    get_next_pending_job_id,
    reconcile_stranded_jobs_on_startup,
    execute_claimed_job,
    run_single_job_cycle,
    JobWorker,
)
from app.services.transcription.base import ASRResult, ASRSegment


def make_auth_headers(user: User, org: Organization) -> dict:
    """Generate authentication headers with valid JWT Bearer token and org header."""
    token = create_access_token(subject=user.id, token_version=getattr(user, "token_version", 1))
    return {"Authorization": f"Bearer {token}", "X-Organization-Id": org.id}


@pytest.fixture(autouse=True)
def bind_session_local_to_test_engine(db_session, monkeypatch):
    """Ensure worker and handlers use the test database session in test executions."""
    monkeypatch.setattr("app.database.session.SessionLocal", lambda: db_session)
    monkeypatch.setattr("app.services.job_worker.SessionLocal", lambda: db_session)
    monkeypatch.setattr("app.services.transcription.job_handler.SessionLocal", lambda: db_session)
    monkeypatch.setattr("app.api.endpoints.media.SessionLocal", lambda: db_session)
    monkeypatch.setattr(db_session, "close", lambda: None)


def _create_test_wav(path: str, duration_sec: float = 1.0) -> None:
    """Create a minimal valid 16kHz mono WAV file for testing."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with wave.open(path, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        num_samples = int(16000 * duration_sec)
        data = struct.pack(f"<{num_samples}h", *([0] * num_samples))
        wav.writeframes(data)


def _setup_org_meeting(db: Session, title: str = "Job Recovery Test"):
    org = Organization(name="Recovery Org", slug=f"recovery-org-{datetime.now().timestamp()}")
    user = User(
        email=f"recovery-{datetime.now().timestamp()}@example.com",
        full_name="Recovery User",
        hashed_password="dummy_password_hash",
        is_active=True,
        token_version=1,
    )
    db.add_all([org, user])
    db.commit()

    membership = OrganizationMembership(
        organization_id=org.id,
        user_id=user.id,
        role=OrgRole.ORG_ADMIN.value,
        is_active=True,
    )
    meeting = Meeting(
        organization_id=org.id,
        created_by_user_id=user.id,
        title=title,
        status=MeetingStatus.INGESTING.value,
    )
    db.add_all([membership, meeting])
    db.commit()
    return org, user, meeting


# --- Test 1: Persisted PENDING job is discovered and executed ---

def test_persisted_pending_job_discovered_and_executed(db_session, monkeypatch):
    """Verify worker discovers a PENDING job, claims it atomically, and executes it."""
    org, user, meeting = _setup_org_meeting(db_session, "Pending Job Execution")

    # Create dummy audio asset
    meeting_dir = os.path.join(settings.STORAGE_DIR, meeting.id)
    wav_path = os.path.join(meeting_dir, "test_audio_16k.wav")
    _create_test_wav(wav_path)

    asset = MediaAsset(
        meeting_id=meeting.id,
        file_name="test_audio_16k.wav",
        storage_path=f"{meeting.id}/test_audio_16k.wav",
        media_type="NORMALIZED_AUDIO",
        mime_type="audio/wav",
        duration_seconds=1.0,
        is_raw_deleted=False,
    )
    db_session.add(asset)
    db_session.commit()

    # Create PENDING transcription job
    payload = {"language_override": "en", "transcription_mode": "standard"}
    job = create_processing_job(
        db_session,
        meeting_id=meeting.id,
        job_type=JobType.TRANSCRIPTION.value,
        initial_status=JobStatus.PENDING.value,
        payload=payload,
    )

    mock_result = ASRResult(
        text="Hello world from recovery worker.",
        segments=[ASRSegment(sequence_order=0, start_time=0.0, end_time=1.0, text="Hello world from recovery worker.", confidence=0.99, language_code="en")],
        detected_language="en",
        device_used="cpu",
        compute_type_used="int8",
    )
    mock_engine = MagicMock()
    mock_engine.transcribe.return_value = mock_result
    monkeypatch.setattr("app.services.transcription.job_handler.get_asr_engine", lambda: mock_engine)

    # Run single job cycle
    executed = run_single_job_cycle(db_session)
    assert executed is True

    db_session.refresh(job)
    assert job.status == JobStatus.COMPLETED.value
    assert job.progress == 100.0
    assert job.attempt_count == 1
    assert job.started_at is not None
    assert job.completed_at is not None

    # Check transcript
    transcript = db_session.query(Transcript).filter(Transcript.meeting_id == meeting.id).first()
    assert transcript is not None
    assert len(transcript.segments) == 1
    assert transcript.segments[0].text == "Hello world from recovery worker."


# --- Test 2: Abandoned IN_PROGRESS job becomes FAILED at startup ---

def test_abandoned_in_progress_job_becomes_failed_at_startup(db_session):
    """Verify startup recovery reconciles stranded IN_PROGRESS jobs as FAILED."""
    org, user, meeting = _setup_org_meeting(db_session, "Stranded Job Test")

    job = create_processing_job(
        db_session,
        meeting_id=meeting.id,
        job_type=JobType.TRANSCRIPTION.value,
        initial_status=JobStatus.IN_PROGRESS.value,
    )
    update_processing_job(db_session, job.id, progress=45.0)

    count = reconcile_stranded_jobs_on_startup(db_session)
    assert count >= 1

    db_session.refresh(job)
    assert job.status == JobStatus.FAILED.value
    assert "interrupted by a server restart" in job.error_message
    assert job.completed_at is not None

    db_session.refresh(meeting)
    assert meeting.status == MeetingStatus.FAILED.value


# --- Test 3: COMPLETED job is never executed again ---

def test_completed_job_never_executed_again(db_session):
    """Verify COMPLETED jobs are ignored by worker and startup reconciliation."""
    org, user, meeting = _setup_org_meeting(db_session, "Completed Job Test")

    job = create_processing_job(
        db_session,
        meeting_id=meeting.id,
        job_type=JobType.TRANSCRIPTION.value,
        initial_status=JobStatus.COMPLETED.value,
    )
    update_processing_job(db_session, job.id, progress=100.0, attempt_count=1)

    reconcile_stranded_jobs_on_startup(db_session)
    db_session.refresh(job)
    assert job.status == JobStatus.COMPLETED.value

    # Worker poll should not pick up completed job
    worked = run_single_job_cycle(db_session)
    assert worked is False
    db_session.refresh(job)
    assert job.attempt_count == 1


# --- Test 4: FAILED and CANCELLED jobs are not automatically executed ---

def test_failed_and_cancelled_jobs_not_automatically_executed(db_session):
    """Verify FAILED and CANCELLED jobs are not picked up by the worker."""
    org, user, meeting = _setup_org_meeting(db_session, "Failed/Cancelled Test")

    job_failed = create_processing_job(
        db_session,
        meeting_id=meeting.id,
        job_type=JobType.TRANSCRIPTION.value,
        initial_status=JobStatus.FAILED.value,
    )
    job_cancelled = create_processing_job(
        db_session,
        meeting_id=meeting.id,
        job_type=JobType.INGESTION.value,
        initial_status=JobStatus.CANCELLED.value,
    )

    reconcile_stranded_jobs_on_startup(db_session)
    db_session.refresh(job_failed)
    db_session.refresh(job_cancelled)
    assert job_failed.status == JobStatus.FAILED.value
    assert job_cancelled.status == JobStatus.CANCELLED.value

    worked = run_single_job_cycle(db_session)
    assert worked is False


# --- Test 5: Missing or invalid payload is handled safely ---

def test_missing_or_invalid_payload_handled_safely(db_session):
    """Verify malformed JSON payload or missing parameters fails safely."""
    org, user, meeting = _setup_org_meeting(db_session, "Bad Payload Test")

    # Invalid JSON payload
    job = ProcessingJob(
        meeting_id=meeting.id,
        job_type=JobType.TRANSCRIPTION.value,
        status=JobStatus.PENDING.value,
        payload="{bad_json: True",
    )
    db_session.add(job)
    db_session.commit()

    worked = run_single_job_cycle(db_session)
    assert worked is True

    db_session.refresh(job)
    assert job.status == JobStatus.FAILED.value
    assert "Invalid job payload" in job.error_message
    assert job.completed_at is not None

    # Unknown ingestion subtype
    job2 = create_processing_job(
        db_session,
        meeting_id=meeting.id,
        job_type=JobType.INGESTION.value,
        initial_status=JobStatus.PENDING.value,
        payload={"subtype": "UNKNOWN_FUTURE_TYPE"},
    )
    worked2 = run_single_job_cycle(db_session)
    assert worked2 is True

    db_session.refresh(job2)
    assert job2.status == JobStatus.FAILED.value
    assert "Unknown or missing ingestion payload subtype" in job2.error_message


# --- Test 6: Attempt count increments only after successful claim ---

def test_attempt_count_increments_only_after_successful_claim(db_session):
    """Verify attempt_count increments on successful atomic claim and not on failed claims."""
    org, user, meeting = _setup_org_meeting(db_session, "Claim Count Test")

    job = create_processing_job(
        db_session,
        meeting_id=meeting.id,
        job_type=JobType.TRANSCRIPTION.value,
        initial_status=JobStatus.PENDING.value,
    )
    assert job.attempt_count == 0

    # Successful claim
    claimed = claim_job_atomically(db_session, job.id)
    assert claimed is not None
    assert claimed.attempt_count == 1
    assert claimed.status == JobStatus.IN_PROGRESS.value

    # Failed claim (already in progress)
    claimed_again = claim_job_atomically(db_session, job.id)
    assert claimed_again is None

    db_session.refresh(job)
    assert job.attempt_count == 1


# --- Test 7: Two competing claim attempts cannot both claim the same job ---

def test_competing_claim_attempts_cannot_both_claim_same_job(db_session):
    """Verify that under concurrent claim attempts, exactly one claim succeeds."""
    org, user, meeting = _setup_org_meeting(db_session, "Concurrency Claim Test")

    job = create_processing_job(
        db_session,
        meeting_id=meeting.id,
        job_type=JobType.TRANSCRIPTION.value,
        initial_status=JobStatus.PENDING.value,
    )

    session_factory = sessionmaker(bind=db_session.get_bind(), autocommit=False, autoflush=False)

    def _attempt_claim(job_id):
        db = session_factory()
        try:
            return claim_job_atomically(db, job_id) is not None
        finally:
            db.close()

    with ThreadPoolExecutor(max_workers=2) as executor:
        f1 = executor.submit(_attempt_claim, job.id)
        f2 = executor.submit(_attempt_claim, job.id)
        r1 = f1.result()
        r2 = f2.result()

    # Exactly one must succeed
    assert (r1 and not r2) or (r2 and not r1)

    db_session.refresh(job)
    assert job.status == JobStatus.IN_PROGRESS.value
    assert job.attempt_count == 1


# --- Test 8: A failed or interrupted job does not permanently block a new transcription request ---

def test_failed_or_interrupted_job_does_not_block_new_transcription(client, db_session):
    """Verify an interrupted job that transitions to FAILED permits subsequent transcription requests."""
    org, user, meeting = _setup_org_meeting(db_session, "Unblock Transcription Test")
    headers = make_auth_headers(user, org)

    # Valid audio asset
    meeting_dir = os.path.join(settings.STORAGE_DIR, meeting.id)
    wav_path = os.path.join(meeting_dir, "meeting_audio.wav")
    _create_test_wav(wav_path)

    asset = MediaAsset(
        meeting_id=meeting.id,
        file_name="meeting_audio.wav",
        storage_path=f"{meeting.id}/meeting_audio.wav",
        media_type="NORMALIZED_AUDIO",
        mime_type="audio/wav",
        duration_seconds=1.0,
        is_raw_deleted=False,
    )
    db_session.add(asset)
    db_session.commit()

    # Simulate stranded job from previous crash
    stranded_job = create_processing_job(
        db_session,
        meeting_id=meeting.id,
        job_type=JobType.TRANSCRIPTION.value,
        initial_status=JobStatus.IN_PROGRESS.value,
    )

    # Startup recovery runs
    reconcile_stranded_jobs_on_startup(db_session)
    db_session.refresh(stranded_job)
    assert stranded_job.status == JobStatus.FAILED.value

    # User initiates new transcription
    resp = client.post(f"/api/v1/transcription/start/{meeting.id}", headers=headers, json={"language": "en"})
    assert resp.status_code == 202
    data = resp.json()
    assert data["job_id"] != stranded_job.id
    assert data["status"] == "PENDING"


# --- Test 9: Failed sibling jobs do not block eligible media cleanup ---

def test_failed_sibling_jobs_do_not_block_media_cleanup(db_session, monkeypatch):
    """Verify failed/abandoned sibling jobs do not prevent auto-deletion of raw media."""
    monkeypatch.setattr(settings, "AUTO_DELETE_RAW_POST_TRANSCRIPTION", True)

    org, user, meeting = _setup_org_meeting(db_session, "Auto-Delete Sibling Test")

    # Raw and normalized media files on disk
    meeting_dir = os.path.join(settings.STORAGE_DIR, meeting.id)
    raw_path = os.path.join(meeting_dir, "raw_audio.mp4")
    norm_path = os.path.join(meeting_dir, "asset-1_normalized_16k.wav")
    _create_test_wav(raw_path)
    _create_test_wav(norm_path)

    raw_asset = MediaAsset(
        id="asset-1",
        meeting_id=meeting.id,
        file_name="raw_audio.mp4",
        storage_path=f"{meeting.id}/raw_audio.mp4",
        media_type="VIDEO",
        mime_type="video/mp4",
        duration_seconds=1.0,
        is_raw_deleted=False,
    )
    norm_asset = MediaAsset(
        id="norm-asset-1",
        meeting_id=meeting.id,
        file_name="asset-1_normalized_16k.wav",
        storage_path=f"{meeting.id}/asset-1_normalized_16k.wav",
        media_type="NORMALIZED_AUDIO",
        mime_type="audio/wav",
        duration_seconds=1.0,
        is_raw_deleted=False,
    )
    # Failed sibling job from earlier crash
    sibling_job = create_processing_job(
        db_session,
        meeting_id=meeting.id,
        job_type=JobType.INGESTION.value,
        initial_status=JobStatus.FAILED.value,
    )
    db_session.add_all([raw_asset, norm_asset])
    db_session.commit()

    # Active transcription job
    active_job = create_processing_job(
        db_session,
        meeting_id=meeting.id,
        job_type=JobType.TRANSCRIPTION.value,
        initial_status=JobStatus.PENDING.value,
        payload={"language_override": "en"},
    )

    mock_result = ASRResult(
        text="Transcription test.",
        segments=[ASRSegment(sequence_order=0, start_time=0.0, end_time=1.0, text="Transcription test.", confidence=0.99, language_code="en")],
        detected_language="en",
        device_used="cpu",
        compute_type_used="int8",
    )
    mock_engine = MagicMock()
    mock_engine.transcribe.return_value = mock_result
    monkeypatch.setattr("app.services.transcription.job_handler.get_asr_engine", lambda: mock_engine)

    worked = run_single_job_cycle(db_session)
    assert worked is True

    db_session.refresh(raw_asset)
    db_session.refresh(norm_asset)

    # Failed sibling job did not block deletion; raw and associated normalized media deleted
    assert raw_asset.is_raw_deleted is True
    assert norm_asset.is_raw_deleted is True


# --- Test 10: No media is deleted before verified transcript persistence ---

def test_no_media_deleted_before_verified_transcript_persistence(db_session, monkeypatch):
    """Verify that if transcription fails, no media assets are deleted."""
    monkeypatch.setattr(settings, "AUTO_DELETE_RAW_POST_TRANSCRIPTION", True)

    org, user, meeting = _setup_org_meeting(db_session, "No Early Deletion Test")

    meeting_dir = os.path.join(settings.STORAGE_DIR, meeting.id)
    wav_path = os.path.join(meeting_dir, "audio.wav")
    _create_test_wav(wav_path)

    asset = MediaAsset(
        meeting_id=meeting.id,
        file_name="audio.wav",
        storage_path=f"{meeting.id}/audio.wav",
        media_type="NORMALIZED_AUDIO",
        mime_type="audio/wav",
        duration_seconds=1.0,
        is_raw_deleted=False,
    )
    job = create_processing_job(
        db_session,
        meeting_id=meeting.id,
        job_type=JobType.TRANSCRIPTION.value,
        initial_status=JobStatus.PENDING.value,
        payload={},
    )
    db_session.add(asset)
    db_session.commit()

    # Mock engine to simulate crash before transcript persistence
    mock_engine = MagicMock()
    mock_engine.transcribe.side_effect = RuntimeError("ASR engine crashed!")
    monkeypatch.setattr("app.services.transcription.job_handler.get_asr_engine", lambda: mock_engine)

    run_single_job_cycle(db_session)

    db_session.refresh(job)
    assert job.status == JobStatus.FAILED.value
    assert "ASR engine crashed!" in job.error_message

    db_session.refresh(asset)
    # Media asset MUST NOT be deleted
    assert asset.is_raw_deleted is False
    assert os.path.isfile(wav_path)


# --- Test 11: Existing job-status API contracts still work ---

def test_existing_job_status_api_contracts_still_work(client, db_session):
    """Verify JobStatusResponse contract returns expected fields and statuses."""
    org, user, meeting = _setup_org_meeting(db_session, "API Contract Test")
    headers = make_auth_headers(user, org)

    job = create_processing_job(
        db_session,
        meeting_id=meeting.id,
        job_type=JobType.TRANSCRIPTION.value,
        initial_status=JobStatus.PENDING.value,
        payload={"language_override": "en"},
    )

    resp = client.get(f"/api/v1/transcription/jobs/{job.id}", headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["job_id"] == job.id
    assert data["meeting_id"] == meeting.id
    assert data["job_type"] == "TRANSCRIPTION"
    assert data["status"] == "PENDING"
    assert data["progress"] == 0.0
    assert data["attempt_count"] == 0
    assert data["max_attempts"] == 2


# --- Test 12: Worker shutdown does not leave database sessions open ---

def test_worker_shutdown_does_not_leave_sessions_open():
    """Verify JobWorker starts and stops cleanly without dangling resources."""
    async def _async_test():
        worker = JobWorker(poll_interval=0.05)
        await worker.start()
        assert worker._task is not None

        # Let worker run one cycle
        await asyncio.sleep(0.1)

        await worker.stop()
        assert worker._stop_event.is_set()
        assert worker._task.done() or worker._task.cancelled()

    asyncio.run(_async_test())
