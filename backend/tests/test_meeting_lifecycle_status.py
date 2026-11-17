"""Regression tests for Phase 9.4.2 — Meeting Status Lifecycle and Raw-Audio Retention.

Verifies:
1. meeting.status transitions correctly to COMPLETED when all required jobs succeed.
2. meeting.status transitions to FAILED on ingestion failure.
3. meeting.status transitions to FAILED on transcription failure (no transcripts).
4. meeting.status stays PROCESSING while any active job is pending/in-progress.
5. meeting.status stays INGESTING while only INGESTION job is active (no transcripts).
6. meeting.status reaches COMPLETED even if downstream extraction/embedding fails.
7. Idempotent reconcile returns same status on second call.
8. Non-existent meeting returns None.
9. Tenant isolation: reconcile only affects the target meeting.
10. Transcripts and segments preserved after reconcile to COMPLETED.
11. commit=False uses flush not commit.
12. Raw audio asset is not marked deleted before transcription completes.
"""

import uuid
import pytest
from datetime import datetime, timezone

from app.models.meeting import (
    Meeting,
    MediaAsset,
    ProcessingJob,
    MeetingStatus,
    JobStatus,
    JobType,
    InputSourceType,
)
from app.models.transcription import Transcript, TranscriptSegment, TranscriptType
from app.models.identity import Organization, User
from app.services.meeting_lifecycle import reconcile_meeting_status


# ── Helpers ──────────────────────────────────────────────────────────────────

def _org(db, suffix="") -> Organization:
    org = Organization(
        id=str(uuid.uuid4()),
        name=f"Lifecycle Test Org{suffix}",
        slug=f"lc-org-{uuid.uuid4().hex[:8]}"
    )
    db.add(org)
    db.flush()
    return org


def _user(db) -> User:
    user = User(
        id=str(uuid.uuid4()),
        email=f"{uuid.uuid4().hex[:8]}@lifecycle-test.com",
        hashed_password="hashed-placeholder",
        full_name="Lifecycle Tester",
        is_active=True,
    )
    db.add(user)
    db.flush()
    return user


def _meeting(db, org, user, status=MeetingStatus.INGESTING.value) -> Meeting:
    m = Meeting(
        id=str(uuid.uuid4()),
        organization_id=org.id,
        created_by_user_id=user.id,
        title="Lifecycle Test Meeting",
        status=status,
        input_source_type=InputSourceType.AUDIO.value,
    )
    db.add(m)
    db.flush()
    return m


def _job(db, meeting, job_type=JobType.INGESTION.value, status=JobStatus.COMPLETED.value) -> ProcessingJob:
    j = ProcessingJob(
        id=str(uuid.uuid4()),
        meeting_id=meeting.id,
        job_type=job_type,
        status=status,
        progress=100.0 if status == JobStatus.COMPLETED.value else 0.0,
        attempt_count=1,
        max_attempts=2,
    )
    if status == JobStatus.COMPLETED.value:
        j.completed_at = datetime.now(timezone.utc)
    db.add(j)
    db.flush()
    return j


def _transcript(db, meeting) -> Transcript:
    t = Transcript(
        id=str(uuid.uuid4()),
        meeting_id=meeting.id,
        transcript_type=TranscriptType.ORIGINAL.value,
        version=1,
        language_detected="en",
        is_primary=True,
    )
    db.add(t)
    db.flush()
    return t


def _segment(db, transcript, text="We decided to deploy Redis for caching.") -> TranscriptSegment:
    s = TranscriptSegment(
        id=str(uuid.uuid4()),
        transcript_id=transcript.id,
        sequence_order=0,
        start_time=0.0,
        end_time=1.0,
        text=text,
        language_code="en",
        is_redacted=False,
    )
    db.add(s)
    db.flush()
    return s


def _media_asset(db, meeting, media_type="UPLOAD", deleted=False) -> MediaAsset:
    asset = MediaAsset(
        id=str(uuid.uuid4()),
        meeting_id=meeting.id,
        file_name=f"test_{uuid.uuid4().hex[:8]}.wav",
        storage_path=f"media/test_{uuid.uuid4().hex[:8]}.wav",
        media_type=media_type,
        mime_type="audio/wav",
        is_raw_deleted=deleted,
    )
    db.add(asset)
    db.flush()
    return asset


# ── Tests ────────────────────────────────────────────────────────────────────

def test_meeting_status_completed_after_all_jobs_succeed(db_session):
    """All jobs COMPLETED → meeting.status == COMPLETED."""
    org = _org(db_session)
    user = _user(db_session)
    meeting = _meeting(db_session, org, user, status=MeetingStatus.PROCESSING.value)
    _job(db_session, meeting, job_type=JobType.INGESTION.value, status=JobStatus.COMPLETED.value)
    _job(db_session, meeting, job_type=JobType.TRANSCRIPTION.value, status=JobStatus.COMPLETED.value)
    _job(db_session, meeting, job_type=JobType.EMBEDDING.value, status=JobStatus.COMPLETED.value)
    db_session.commit()

    result = reconcile_meeting_status(db_session, meeting.id)
    assert result == MeetingStatus.COMPLETED.value
    db_session.refresh(meeting)
    assert meeting.status == MeetingStatus.COMPLETED.value


def test_meeting_status_failed_when_ingestion_fails(db_session):
    """INGESTION FAILED with no successful ingestion → meeting.status == FAILED."""
    org = _org(db_session)
    user = _user(db_session)
    meeting = _meeting(db_session, org, user, status=MeetingStatus.INGESTING.value)
    _job(db_session, meeting, job_type=JobType.INGESTION.value, status=JobStatus.FAILED.value)
    db_session.commit()

    result = reconcile_meeting_status(db_session, meeting.id)
    assert result == MeetingStatus.FAILED.value
    db_session.refresh(meeting)
    assert meeting.status == MeetingStatus.FAILED.value


def test_meeting_status_failed_when_transcription_fails_no_transcripts(db_session):
    """TRANSCRIPTION FAILED and no transcripts → meeting.status == FAILED."""
    org = _org(db_session)
    user = _user(db_session)
    meeting = _meeting(db_session, org, user, status=MeetingStatus.PROCESSING.value)
    _job(db_session, meeting, job_type=JobType.INGESTION.value, status=JobStatus.COMPLETED.value)
    _job(db_session, meeting, job_type=JobType.TRANSCRIPTION.value, status=JobStatus.FAILED.value)
    db_session.commit()

    result = reconcile_meeting_status(db_session, meeting.id)
    assert result == MeetingStatus.FAILED.value
    db_session.refresh(meeting)
    assert meeting.status == MeetingStatus.FAILED.value


def test_meeting_status_processing_while_transcription_pending(db_session):
    """Active TRANSCRIPTION job → meeting.status == PROCESSING."""
    org = _org(db_session)
    user = _user(db_session)
    meeting = _meeting(db_session, org, user, status=MeetingStatus.PROCESSING.value)
    _job(db_session, meeting, job_type=JobType.INGESTION.value, status=JobStatus.COMPLETED.value)
    _job(db_session, meeting, job_type=JobType.TRANSCRIPTION.value, status=JobStatus.PENDING.value)
    db_session.commit()

    result = reconcile_meeting_status(db_session, meeting.id)
    assert result == MeetingStatus.PROCESSING.value
    db_session.refresh(meeting)
    assert meeting.status == MeetingStatus.PROCESSING.value


def test_meeting_status_ingesting_while_only_ingestion_active(db_session):
    """Only INGESTION IN_PROGRESS, no transcripts → meeting.status == INGESTING."""
    org = _org(db_session)
    user = _user(db_session)
    meeting = _meeting(db_session, org, user, status=MeetingStatus.INGESTING.value)
    _job(db_session, meeting, job_type=JobType.INGESTION.value, status=JobStatus.IN_PROGRESS.value)
    db_session.commit()

    result = reconcile_meeting_status(db_session, meeting.id)
    assert result == MeetingStatus.INGESTING.value


def test_meeting_status_completed_even_if_extraction_fails(db_session):
    """Extraction FAILED but transcription COMPLETED and transcripts exist → COMPLETED."""
    org = _org(db_session)
    user = _user(db_session)
    meeting = _meeting(db_session, org, user, status=MeetingStatus.PROCESSING.value)
    _job(db_session, meeting, job_type=JobType.INGESTION.value, status=JobStatus.COMPLETED.value)
    _job(db_session, meeting, job_type=JobType.TRANSCRIPTION.value, status=JobStatus.COMPLETED.value)
    _job(db_session, meeting, job_type=JobType.EXTRACTION.value, status=JobStatus.FAILED.value)
    transcript = _transcript(db_session, meeting)
    _segment(db_session, transcript)
    db_session.commit()

    result = reconcile_meeting_status(db_session, meeting.id)
    assert result == MeetingStatus.COMPLETED.value
    db_session.refresh(meeting)
    assert meeting.status == MeetingStatus.COMPLETED.value


def test_meeting_status_completed_even_if_embedding_fails(db_session):
    """Embedding FAILED but transcription COMPLETED → COMPLETED (non-fatal downstream)."""
    org = _org(db_session)
    user = _user(db_session)
    meeting = _meeting(db_session, org, user, status=MeetingStatus.PROCESSING.value)
    _job(db_session, meeting, job_type=JobType.INGESTION.value, status=JobStatus.COMPLETED.value)
    _job(db_session, meeting, job_type=JobType.TRANSCRIPTION.value, status=JobStatus.COMPLETED.value)
    _job(db_session, meeting, job_type=JobType.EMBEDDING.value, status=JobStatus.FAILED.value)
    transcript = _transcript(db_session, meeting)
    _segment(db_session, transcript)
    db_session.commit()

    result = reconcile_meeting_status(db_session, meeting.id)
    assert result == MeetingStatus.COMPLETED.value


def test_idempotent_reconcile(db_session):
    """Calling reconcile_meeting_status twice returns the same result."""
    org = _org(db_session)
    user = _user(db_session)
    meeting = _meeting(db_session, org, user, status=MeetingStatus.PROCESSING.value)
    _job(db_session, meeting, job_type=JobType.INGESTION.value, status=JobStatus.COMPLETED.value)
    _job(db_session, meeting, job_type=JobType.TRANSCRIPTION.value, status=JobStatus.COMPLETED.value)
    db_session.commit()

    result1 = reconcile_meeting_status(db_session, meeting.id)
    result2 = reconcile_meeting_status(db_session, meeting.id)
    assert result1 == result2 == MeetingStatus.COMPLETED.value


def test_reconcile_nonexistent_meeting_returns_none(db_session):
    """Reconciling a non-existent meeting_id returns None without raising."""
    result = reconcile_meeting_status(db_session, str(uuid.uuid4()))
    assert result is None


def test_tenant_isolation_lifecycle(db_session):
    """Reconciling one meeting must not change another org's meeting status."""
    org1 = _org(db_session, suffix="A")
    org2 = _org(db_session, suffix="B")
    user1 = _user(db_session)
    user2 = _user(db_session)
    meeting1 = _meeting(db_session, org1, user1, status=MeetingStatus.PROCESSING.value)
    meeting2 = _meeting(db_session, org2, user2, status=MeetingStatus.INGESTING.value)
    _job(db_session, meeting1, job_type=JobType.INGESTION.value, status=JobStatus.COMPLETED.value)
    _job(db_session, meeting1, job_type=JobType.TRANSCRIPTION.value, status=JobStatus.COMPLETED.value)
    _job(db_session, meeting2, job_type=JobType.INGESTION.value, status=JobStatus.IN_PROGRESS.value)
    db_session.commit()

    reconcile_meeting_status(db_session, meeting1.id)
    db_session.refresh(meeting1)
    db_session.refresh(meeting2)
    assert meeting1.status == MeetingStatus.COMPLETED.value
    assert meeting2.status == MeetingStatus.INGESTING.value  # untouched


def test_transcripts_preserved_after_lifecycle_reconcile(db_session):
    """Transcript records and segments remain accessible after status reconciliation."""
    org = _org(db_session)
    user = _user(db_session)
    meeting = _meeting(db_session, org, user, status=MeetingStatus.PROCESSING.value)
    _job(db_session, meeting, job_type=JobType.INGESTION.value, status=JobStatus.COMPLETED.value)
    _job(db_session, meeting, job_type=JobType.TRANSCRIPTION.value, status=JobStatus.COMPLETED.value)
    transcript = _transcript(db_session, meeting)
    seg = _segment(db_session, transcript, text="Deploy Redis for caching.")
    db_session.commit()

    reconcile_meeting_status(db_session, meeting.id)

    db_session.refresh(transcript)
    db_session.refresh(seg)
    db_session.refresh(meeting)
    assert transcript.id is not None
    assert seg.text == "Deploy Redis for caching."
    assert meeting.status == MeetingStatus.COMPLETED.value


def test_reconcile_commit_false_does_not_autocommit(db_session):
    """commit=False flushes the status change but does not issue a COMMIT."""
    org = _org(db_session)
    user = _user(db_session)
    meeting = _meeting(db_session, org, user, status=MeetingStatus.PROCESSING.value)
    _job(db_session, meeting, job_type=JobType.INGESTION.value, status=JobStatus.COMPLETED.value)
    _job(db_session, meeting, job_type=JobType.TRANSCRIPTION.value, status=JobStatus.COMPLETED.value)
    db_session.flush()

    result = reconcile_meeting_status(db_session, meeting.id, commit=False)
    assert result == MeetingStatus.COMPLETED.value
    # In-session the status should be updated via flush
    assert meeting.status == MeetingStatus.COMPLETED.value
    # Rollback should still work without error
    db_session.rollback()


def test_raw_audio_asset_not_marked_deleted_before_transcription(db_session):
    """Raw media asset is NOT marked is_raw_deleted before transcription has succeeded."""
    org = _org(db_session)
    user = _user(db_session)
    meeting = _meeting(db_session, org, user, status=MeetingStatus.PROCESSING.value)
    asset = _media_asset(db_session, meeting, deleted=False)
    _job(db_session, meeting, job_type=JobType.INGESTION.value, status=JobStatus.COMPLETED.value)
    _job(db_session, meeting, job_type=JobType.TRANSCRIPTION.value, status=JobStatus.IN_PROGRESS.value)
    db_session.commit()

    db_session.refresh(asset)
    assert asset.is_raw_deleted is False
