"""Tests for Phase 6D.6: Automated Media Retention and Expired Asset Cleanup.

Covers:
1. Expired assets are identified correctly.
2. Assets inside the retention period are preserved.
3. Eligible expired raw audio is deleted.
4. Associated normalized audio is handled safely.
5. Active sibling jobs prevent deletion.
6. Uncertain asset associations are retained.
7. Already-deleted assets are handled idempotently.
8. A failed deletion does not prevent other assets from being processed.
9. Scheduler runs during startup and stops during shutdown.
10. Existing transcript and meeting intelligence records remain untouched.
"""

import os
import json
import uuid
import wave
import struct
import asyncio
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.identity import Organization, User, OrganizationMembership, OrgRole
from app.models.meeting import Meeting, MediaAsset, ProcessingJob, JobType, JobStatus, MeetingStatus
from app.models.memory import AuditLog
from app.models.transcription import Transcript, TranscriptSegment, TranscriptType
from app.models.intelligence import MeetingSummary, Topic, Decision, ActionItem
from app.services.lifecycle import (
    find_expired_media_assets,
    cleanup_expired_media_assets,
    delete_raw_media_asset,
    is_associated_normalized_asset,
)
from app.services.retention_scheduler import RetentionScheduler


@pytest.fixture(autouse=True)
def bind_session_local_to_test_engine(db_session, monkeypatch):
    """Ensure worker and handlers use the test database session in test executions."""
    monkeypatch.setattr("app.database.session.SessionLocal", lambda: db_session)
    monkeypatch.setattr("app.services.job_worker.SessionLocal", lambda: db_session)
    monkeypatch.setattr("app.services.retention_scheduler.SessionLocal", lambda: db_session)
    monkeypatch.setattr("app.services.transcription.job_handler.SessionLocal", lambda: db_session)
    monkeypatch.setattr("app.api.endpoints.media.SessionLocal", lambda: db_session)
    monkeypatch.setattr(db_session, "close", lambda: None)


def _create_synthetic_wav(path: str, duration_sec: float = 1.0) -> None:
    """Create a minimal valid 16kHz mono WAV file for testing."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with wave.open(path, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        num_samples = int(16000 * duration_sec)
        data = struct.pack(f"<{num_samples}h", *([0] * num_samples))
        wav.writeframes(data)


def _setup_org_meeting(db: Session, title: str = "Retention Test Meeting"):
    org = Organization(name="Retention Org", slug=f"retention-org-{uuid.uuid4().hex[:8]}")
    user = User(
        email=f"retention-{uuid.uuid4().hex[:8]}@example.com",
        full_name="Retention User",
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
        status=MeetingStatus.COMPLETED.value,
    )
    db.add_all([membership, meeting])
    db.commit()
    return org, user, meeting


# --- Test 1: Expired assets are identified correctly ---

def test_expired_assets_identified_correctly(db_session):
    """Verify find_expired_media_assets finds only assets older than retention threshold."""
    org, user, meeting = _setup_org_meeting(db_session, "Expired Identification Test")
    now = datetime.now(timezone.utc)

    # 1. Expired asset (35 days old)
    expired_asset_1 = MediaAsset(
        id=str(uuid.uuid4()),
        meeting_id=meeting.id,
        file_name="old_audio_1.wav",
        storage_path=f"{meeting.id}/old_audio_1.wav",
        media_type="AUDIO",
        mime_type="audio/wav",
        created_at=now - timedelta(days=35),
        is_raw_deleted=False,
    )
    # 2. Expired asset (40 days old)
    expired_asset_2 = MediaAsset(
        id=str(uuid.uuid4()),
        meeting_id=meeting.id,
        file_name="old_video_2.mp4",
        storage_path=f"{meeting.id}/old_video_2.mp4",
        media_type="VIDEO",
        mime_type="video/mp4",
        created_at=now - timedelta(days=40),
        is_raw_deleted=False,
    )
    # 3. Non-expired asset (10 days old)
    fresh_asset = MediaAsset(
        id=str(uuid.uuid4()),
        meeting_id=meeting.id,
        file_name="recent_audio.wav",
        storage_path=f"{meeting.id}/recent_audio.wav",
        media_type="AUDIO",
        mime_type="audio/wav",
        created_at=now - timedelta(days=10),
        is_raw_deleted=False,
    )
    # 4. Old asset but already deleted
    already_deleted_asset = MediaAsset(
        id=str(uuid.uuid4()),
        meeting_id=meeting.id,
        file_name="already_deleted.wav",
        storage_path=f"{meeting.id}/already_deleted.wav",
        media_type="AUDIO",
        mime_type="audio/wav",
        created_at=now - timedelta(days=50),
        is_raw_deleted=True,
    )
    db_session.add_all([expired_asset_1, expired_asset_2, fresh_asset, already_deleted_asset])
    db_session.commit()

    expired = find_expired_media_assets(db_session, retention_days=30)
    expired_ids = [a.id for a in expired]

    assert expired_asset_1.id in expired_ids
    assert expired_asset_2.id in expired_ids
    assert fresh_asset.id not in expired_ids
    assert already_deleted_asset.id not in expired_ids
    # Oldest first ordering
    assert expired_ids.index(expired_asset_2.id) < expired_ids.index(expired_asset_1.id)


# --- Test 2: Assets inside the retention period are preserved ---

def test_assets_inside_retention_period_preserved(db_session, tmp_path, monkeypatch):
    """Verify cleanup_expired_media_assets preserves assets created within retention period."""
    monkeypatch.setattr(settings, "STORAGE_DIR", str(tmp_path))
    org, user, meeting = _setup_org_meeting(db_session, "Preservation Test")
    now = datetime.now(timezone.utc)

    # File on disk
    file_path = tmp_path / meeting.id / "active_audio.wav"
    _create_synthetic_wav(str(file_path))

    asset = MediaAsset(
        id=str(uuid.uuid4()),
        meeting_id=meeting.id,
        file_name="active_audio.wav",
        storage_path=f"{meeting.id}/active_audio.wav",
        media_type="AUDIO",
        mime_type="audio/wav",
        created_at=now - timedelta(days=29),  # 29 days < 30 days
        is_raw_deleted=False,
    )
    db_session.add(asset)
    db_session.commit()

    stats = cleanup_expired_media_assets(db_session, retention_days=30)

    db_session.refresh(asset)
    assert asset.is_raw_deleted is False
    assert os.path.isfile(file_path)
    assert stats["deleted_raw_count"] == 0
    assert stats["examined_count"] == 0


# --- Test 3: Eligible expired raw audio is deleted ---

def test_eligible_expired_raw_audio_deleted(db_session, tmp_path, monkeypatch):
    """Verify expired raw audio is physically unlinked, marked deleted, and audited."""
    monkeypatch.setattr(settings, "STORAGE_DIR", str(tmp_path))
    org, user, meeting = _setup_org_meeting(db_session, "Expired Raw Cleanup")
    now = datetime.now(timezone.utc)

    file_path = tmp_path / meeting.id / "expired_upload.wav"
    _create_synthetic_wav(str(file_path))

    raw_asset = MediaAsset(
        id="raw-expired-1",
        meeting_id=meeting.id,
        file_name="expired_upload.wav",
        storage_path=f"{meeting.id}/expired_upload.wav",
        media_type="AUDIO",
        mime_type="audio/wav",
        created_at=now - timedelta(days=32),
        is_raw_deleted=False,
    )
    db_session.add(raw_asset)
    db_session.commit()

    stats = cleanup_expired_media_assets(db_session, retention_days=30)

    db_session.refresh(raw_asset)
    assert raw_asset.is_raw_deleted is True
    assert not os.path.exists(file_path)
    assert stats["deleted_raw_count"] == 1

    # Check AuditLog
    audit = (
        db_session.query(AuditLog)
        .filter(AuditLog.entity_id == "raw-expired-1", AuditLog.action == "DELETE_RAW_AUDIO")
        .first()
    )
    assert audit is not None
    assert audit.organization_id == org.id
    details = json.loads(audit.details)
    assert details["status"] == "deleted"
    assert details["deletion_type"] == "application_unlink"


# --- Test 4: Associated normalized audio is handled safely ---

def test_associated_normalized_audio_handled_safely(db_session, tmp_path, monkeypatch):
    """Verify deleting expired raw audio also cleans up its explicitly associated normalized audio."""
    monkeypatch.setattr(settings, "STORAGE_DIR", str(tmp_path))
    org, user, meeting = _setup_org_meeting(db_session, "Raw and Normalized Cleanup")
    now = datetime.now(timezone.utc)

    raw_path = tmp_path / meeting.id / "raw_video.mp4"
    norm_path = tmp_path / meeting.id / "raw-asset-4_normalized_16k.wav"
    _create_synthetic_wav(str(raw_path))
    _create_synthetic_wav(str(norm_path))

    raw_asset = MediaAsset(
        id="raw-asset-4",
        meeting_id=meeting.id,
        file_name="raw_video.mp4",
        storage_path=f"{meeting.id}/raw_video.mp4",
        media_type="VIDEO",
        mime_type="video/mp4",
        created_at=now - timedelta(days=31),
        is_raw_deleted=False,
    )
    norm_asset = MediaAsset(
        id="norm-asset-4",
        meeting_id=meeting.id,
        file_name="raw-asset-4_normalized_16k.wav",
        storage_path=f"{meeting.id}/raw-asset-4_normalized_16k.wav",
        media_type="NORMALIZED_AUDIO",
        mime_type="audio/wav",
        created_at=now - timedelta(days=31),
        is_raw_deleted=False,
    )
    db_session.add_all([raw_asset, norm_asset])
    db_session.commit()

    stats = cleanup_expired_media_assets(db_session, retention_days=30)

    db_session.refresh(raw_asset)
    db_session.refresh(norm_asset)

    # Both must be deleted physically and marked in DB
    assert raw_asset.is_raw_deleted is True
    assert norm_asset.is_raw_deleted is True
    assert not os.path.exists(raw_path)
    assert not os.path.exists(norm_path)
    assert stats["deleted_raw_count"] == 1
    assert stats["deleted_normalized_count"] == 1

    # Distinct audit logs for both
    raw_audit = db_session.query(AuditLog).filter(AuditLog.entity_id == "raw-asset-4").first()
    norm_audit = db_session.query(AuditLog).filter(AuditLog.entity_id == "norm-asset-4").first()
    assert raw_audit is not None
    assert norm_audit is not None
    assert raw_audit.action == "DELETE_RAW_AUDIO"
    assert norm_audit.action == "DELETE_NORMALIZED_AUDIO"


# --- Test 5: Active sibling jobs prevent deletion ---

def test_active_sibling_jobs_prevent_deletion(db_session, tmp_path, monkeypatch):
    """Verify active media-consuming jobs (PENDING/IN_PROGRESS) block retention deletion."""
    monkeypatch.setattr(settings, "STORAGE_DIR", str(tmp_path))
    org, user, meeting = _setup_org_meeting(db_session, "Active Sibling Job Test")
    now = datetime.now(timezone.utc)

    file_path = tmp_path / meeting.id / "in_use_audio.wav"
    _create_synthetic_wav(str(file_path))

    asset = MediaAsset(
        id="asset-in-use",
        meeting_id=meeting.id,
        file_name="in_use_audio.wav",
        storage_path=f"{meeting.id}/in_use_audio.wav",
        media_type="AUDIO",
        mime_type="audio/wav",
        created_at=now - timedelta(days=35),
        is_raw_deleted=False,
    )
    # Active transcription job in progress
    active_job = ProcessingJob(
        id=str(uuid.uuid4()),
        meeting_id=meeting.id,
        job_type=JobType.TRANSCRIPTION.value,
        status=JobStatus.IN_PROGRESS.value,
        progress=50.0,
    )
    db_session.add_all([asset, active_job])
    db_session.commit()

    stats = cleanup_expired_media_assets(db_session, retention_days=30)

    db_session.refresh(asset)
    # Media asset MUST NOT be deleted while job is active
    assert asset.is_raw_deleted is False
    assert os.path.exists(file_path)
    assert stats["skipped_active_jobs_count"] == 1
    assert stats["deleted_raw_count"] == 0

    # Once job completes, subsequent retention cleanup deletes the expired asset
    active_job.status = JobStatus.COMPLETED.value
    db_session.commit()

    stats2 = cleanup_expired_media_assets(db_session, retention_days=30)
    db_session.refresh(asset)
    assert asset.is_raw_deleted is True
    assert not os.path.exists(file_path)
    assert stats2["deleted_raw_count"] == 1


# --- Test 6: Uncertain asset associations are retained ---

def test_uncertain_asset_associations_retained(db_session, tmp_path, monkeypatch):
    """Verify normalized audio with ambiguous/uncertain association in multi-raw meeting is NOT deleted."""
    monkeypatch.setattr(settings, "STORAGE_DIR", str(tmp_path))
    org, user, meeting = _setup_org_meeting(db_session, "Uncertain Association Test")
    now = datetime.now(timezone.utc)

    # Multi-raw meeting: two raw assets already deleted
    raw1 = MediaAsset(
        id="raw-1",
        meeting_id=meeting.id,
        file_name="raw1.mp4",
        storage_path=f"{meeting.id}/raw1.mp4",
        media_type="VIDEO",
        mime_type="video/mp4",
        created_at=now - timedelta(days=40),
        is_raw_deleted=True,
    )
    raw2 = MediaAsset(
        id="raw-2",
        meeting_id=meeting.id,
        file_name="raw2.mp4",
        storage_path=f"{meeting.id}/raw2.mp4",
        media_type="VIDEO",
        mime_type="video/mp4",
        created_at=now - timedelta(days=40),
        is_raw_deleted=True,
    )
    # Ambiguous normalized asset: does not explicitly match raw-1 or raw-2 token prefix
    norm_path = tmp_path / meeting.id / "ambiguous_normalized.wav"
    _create_synthetic_wav(str(norm_path))

    ambiguous_norm = MediaAsset(
        id="norm-ambiguous",
        meeting_id=meeting.id,
        file_name="ambiguous_normalized.wav",
        storage_path=f"{meeting.id}/ambiguous_normalized.wav",
        media_type="NORMALIZED_AUDIO",
        mime_type="audio/wav",
        created_at=now - timedelta(days=40),
        is_raw_deleted=False,
    )
    db_session.add_all([raw1, raw2, ambiguous_norm])
    db_session.commit()

    stats = cleanup_expired_media_assets(db_session, retention_days=30)

    db_session.refresh(ambiguous_norm)
    # Ambiguous normalized asset must be preserved
    assert ambiguous_norm.is_raw_deleted is False
    assert os.path.exists(norm_path)
    assert stats["skipped_uncertain_association_count"] == 1


# --- Test 7: Already-deleted assets are handled idempotently ---

def test_already_deleted_assets_handled_idempotently(db_session, tmp_path, monkeypatch):
    """Verify running retention cleanup on already-deleted assets is completely idempotent."""
    monkeypatch.setattr(settings, "STORAGE_DIR", str(tmp_path))
    org, user, meeting = _setup_org_meeting(db_session, "Idempotency Test")
    now = datetime.now(timezone.utc)

    file_path = tmp_path / meeting.id / "deleted_file.wav"
    _create_synthetic_wav(str(file_path))

    asset = MediaAsset(
        id="asset-idempotent",
        meeting_id=meeting.id,
        file_name="deleted_file.wav",
        storage_path=f"{meeting.id}/deleted_file.wav",
        media_type="AUDIO",
        mime_type="audio/wav",
        created_at=now - timedelta(days=35),
        is_raw_deleted=False,
    )
    db_session.add(asset)
    db_session.commit()

    # First cleanup run
    stats1 = cleanup_expired_media_assets(db_session, retention_days=30)
    assert stats1["deleted_raw_count"] == 1

    audit_count_1 = db_session.query(AuditLog).filter(AuditLog.entity_id == "asset-idempotent").count()
    assert audit_count_1 == 1

    # Second cleanup run (idempotent repeat)
    stats2 = cleanup_expired_media_assets(db_session, retention_days=30)
    assert stats2["deleted_raw_count"] == 0
    assert stats2["examined_count"] == 0  # find_expired filters out is_raw_deleted=True

    audit_count_2 = db_session.query(AuditLog).filter(AuditLog.entity_id == "asset-idempotent").count()
    assert audit_count_2 == 1  # No duplicate audit logs


# --- Test 8: A failed deletion does not prevent other assets from being processed ---

def test_failed_deletion_does_not_block_other_assets(db_session, tmp_path, monkeypatch):
    """Verify an error during deletion of one asset is isolated and does not abort remaining assets."""
    monkeypatch.setattr(settings, "STORAGE_DIR", str(tmp_path))
    org, user, meeting = _setup_org_meeting(db_session, "Failure Isolation Test")
    now = datetime.now(timezone.utc)

    file1 = tmp_path / meeting.id / "asset1.wav"
    file2 = tmp_path / meeting.id / "asset2.wav"
    _create_synthetic_wav(str(file1))
    _create_synthetic_wav(str(file2))

    asset1 = MediaAsset(
        id="asset-fail-1",
        meeting_id=meeting.id,
        file_name="asset1.wav",
        storage_path=f"{meeting.id}/asset1.wav",
        media_type="AUDIO",
        mime_type="audio/wav",
        created_at=now - timedelta(days=35),
        is_raw_deleted=False,
    )
    asset2 = MediaAsset(
        id="asset-succeed-2",
        meeting_id=meeting.id,
        file_name="asset2.wav",
        storage_path=f"{meeting.id}/asset2.wav",
        media_type="AUDIO",
        mime_type="audio/wav",
        created_at=now - timedelta(days=34),
        is_raw_deleted=False,
    )
    db_session.add_all([asset1, asset2])
    db_session.commit()

    original_unlink = Path.unlink

    def faulty_unlink(path_self, *args, **kwargs):
        if "asset1.wav" in str(path_self):
            raise OSError("Simulated permission / I/O error on asset 1")
        return original_unlink(path_self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", faulty_unlink)

    stats = cleanup_expired_media_assets(db_session, retention_days=30)

    db_session.refresh(asset1)
    db_session.refresh(asset2)

    # Asset 1 failed: not marked deleted, DB rolled back for asset 1
    assert asset1.is_raw_deleted is False
    assert os.path.exists(file1)

    # Asset 2 succeeded: marked deleted, unlinked
    assert asset2.is_raw_deleted is True
    assert not os.path.exists(file2)

    # Stats reflect error isolation
    assert stats["error_count"] == 1
    assert stats["deleted_raw_count"] == 1
    assert len(stats["errors"]) == 1
    assert stats["errors"][0]["asset_id"] == "asset-fail-1"


# --- Test 9: Scheduler runs during startup and stops during shutdown ---

def test_scheduler_runs_at_startup_and_stops_cleanly():
    """Verify RetentionScheduler initializes, runs its initial cycle, and terminates cleanly."""
    async def _async_test():
        scheduler = RetentionScheduler(cleanup_interval=0.05, retention_days=30)
        assert scheduler._is_running is False

        await scheduler.start()
        assert scheduler._is_running is True
        assert scheduler._task is not None

        # Allow initial cycle and background loop to run briefly
        await asyncio.sleep(0.1)

        await scheduler.stop()
        assert scheduler._is_running is False
        assert scheduler._stop_event.is_set()
        assert scheduler._task.done() or scheduler._task.cancelled()

    asyncio.run(_async_test())


# --- Test 10: Existing transcript and meeting intelligence records remain untouched ---

def test_transcripts_and_intelligence_remain_untouched(db_session, tmp_path, monkeypatch):
    """Verify deleting expired media leaves all transcripts, summaries, topics, and decisions intact."""
    monkeypatch.setattr(settings, "STORAGE_DIR", str(tmp_path))
    org, user, meeting = _setup_org_meeting(db_session, "Intelligence Preservation Test")
    now = datetime.now(timezone.utc)

    file_path = tmp_path / meeting.id / "meeting_recording.wav"
    _create_synthetic_wav(str(file_path))

    asset = MediaAsset(
        id="asset-with-intel",
        meeting_id=meeting.id,
        file_name="meeting_recording.wav",
        storage_path=f"{meeting.id}/meeting_recording.wav",
        media_type="AUDIO",
        mime_type="audio/wav",
        created_at=now - timedelta(days=35),
        is_raw_deleted=False,
    )
    transcript = Transcript(
        id=str(uuid.uuid4()),
        meeting_id=meeting.id,
        transcript_type=TranscriptType.ORIGINAL.value,
        version=1,
        language_detected="en",
        is_primary=True,
    )
    segment = TranscriptSegment(
        id=str(uuid.uuid4()),
        transcript_id=transcript.id,
        sequence_order=0,
        start_time=0.0,
        end_time=1.0,
        text="All decisions are permanently recorded.",
        language_code="en",
        confidence=0.99,
        is_redacted=False,
    )
    summary = MeetingSummary(
        id=str(uuid.uuid4()),
        meeting_id=meeting.id,
        executive_summary="Executive summary must survive media cleanup.",
        version=1,
    )
    topic = Topic(
        id=str(uuid.uuid4()),
        meeting_id=meeting.id,
        title="Architecture Discussion",
        description="Review of retention policy",
    )
    decision = Decision(
        id=str(uuid.uuid4()),
        meeting_id=meeting.id,
        description="Retain transcripts while removing raw audio.",
    )
    action_item = ActionItem(
        id=str(uuid.uuid4()),
        meeting_id=meeting.id,
        description="Verify retention policy.",
    )
    db_session.add_all([asset, transcript, segment, summary, topic, decision, action_item])
    db_session.commit()

    stats = cleanup_expired_media_assets(db_session, retention_days=30)

    db_session.refresh(asset)
    assert asset.is_raw_deleted is True
    assert not os.path.exists(file_path)
    assert stats["deleted_raw_count"] == 1

    # Verify transcripts and all meeting intelligence are untouched
    db_transcript = db_session.get(Transcript, transcript.id)
    db_segment = db_session.get(TranscriptSegment, segment.id)
    db_summary = db_session.get(MeetingSummary, summary.id)
    db_topic = db_session.get(Topic, topic.id)
    db_decision = db_session.get(Decision, decision.id)
    db_action = db_session.get(ActionItem, action_item.id)

    assert db_transcript is not None
    assert db_segment is not None
    assert db_segment.text == "All decisions are permanently recorded."
    assert db_summary is not None
    assert db_summary.executive_summary == "Executive summary must survive media cleanup."
    assert db_topic is not None
    assert db_decision is not None
    assert db_action is not None
