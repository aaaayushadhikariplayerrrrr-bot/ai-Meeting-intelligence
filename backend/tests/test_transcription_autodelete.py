"""Regression and verification tests for Complete Raw + Normalized Audio Lifecycle Cleanup.

Phase 6D.1 Step 2:
- Verifies post-transcription auto-deletion removes both original upload and associated normalized WAV.
- Verifies auto-deletion disabled preserves both assets.
- Verifies normalized asset belonging to another meeting is NEVER deleted (cross-meeting isolation).
- Verifies unrelated normalized asset in the same meeting is NOT deleted (intra-meeting isolation).
- Verifies missing files on disk are handled safely and updated in DB without error.
- Verifies repeated deletion is safe and idempotent.
- Verifies failed transcript persistence/verification aborts lifecycle deletion, retaining both assets.
- Verifies distinct, accurate AuditLog records for both raw and normalized deletions.
- Verifies manual DELETE /api/v1/media/assets/{id} deletes both raw and associated normalized copies with tenant authorization.
"""

import os
import json
import wave
import struct
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from app.core.config import settings
from app.models.identity import Organization, User, OrganizationMembership, OrgRole
from app.models.meeting import Meeting, MediaAsset, ProcessingJob, JobType, JobStatus
from app.models.memory import AuditLog
from app.models.transcription import Transcript, TranscriptSegment
from app.services.transcription.base import ASRSegment, ASRResult
from app.services.transcription.job_handler import process_transcription_job_sync
from app.services.lifecycle import delete_raw_media_asset, is_associated_normalized_asset
from app.services.media_storage import (
    get_meeting_storage_dir,
    to_safe_relative_path,
    resolve_safe_storage_path,
)


def _create_synthetic_wav(path: Path, duration_sec: float = 1.0) -> None:
    """Create a valid 16 kHz 16-bit mono PCM WAV for testing."""
    sample_rate = 16000
    num_samples = int(sample_rate * duration_sec)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        data = struct.pack(f"<{num_samples}h", *([0] * num_samples))
        wav.writeframes(data)


def _setup_mock_asr(monkeypatch, text="Auto deletion test transcript"):
    """Helper to mock ASR engine output."""
    mock_result = ASRResult(
        segments=[
            ASRSegment(
                sequence_order=0,
                start_time=0.0,
                end_time=1.0,
                text=text,
                language_code="en",
                confidence=0.95,
            )
        ],
        full_text=text,
        detected_language="en",
        language_probability=0.99,
        duration_seconds=1.0,
        device_used="cpu",
        compute_type_used="int8",
        model_name="mock-model",
        engine_name="faster-whisper",
    )

    class MockASREngine:
        def transcribe(self, *args, **kwargs):
            return mock_result

    monkeypatch.setattr("app.services.transcription.job_handler.get_asr_engine", lambda: MockASREngine())


def test_autodelete_enabled_deletes_raw_and_associated_normalized(db_session, monkeypatch):
    """When AUTO_DELETE_RAW_POST_TRANSCRIPTION=True:
    1. Raw media asset is physically unlinked and marked is_raw_deleted=True.
    2. Associated normalized audio asset is ALSO unlinked and marked is_raw_deleted=True.
    3. Distinct AuditLog records are created for both raw and normalized assets.
    4. Job completes with status COMPLETED and 100% progress.
    """
    monkeypatch.setattr(settings, "AUTO_DELETE_RAW_POST_TRANSCRIPTION", True)
    _setup_mock_asr(monkeypatch)
    monkeypatch.setattr("app.services.transcription.job_handler.SessionLocal", lambda: db_session)
    monkeypatch.setattr(db_session, "close", lambda: None)

    org = Organization(name="AutoDelete Dual Org", slug="autodelete-dual-org")
    user = User(email="owner@dualdelete.com", full_name="Dual Owner")
    db_session.add_all([org, user])
    db_session.commit()

    meeting = Meeting(
        organization_id=org.id,
        created_by_user_id=user.id,
        title="Privacy Dual Auto-Delete Meeting",
    )
    db_session.add(meeting)
    db_session.commit()

    # Stage raw file and associated normalized WAV
    meeting_dir = get_meeting_storage_dir(meeting.id)
    raw_file = meeting_dir / "raw_recording.mp4"
    raw_file.write_bytes(b"DUMMY_RAW_VIDEO_PAYLOAD_FOR_TESTING")

    raw_asset = MediaAsset(
        meeting_id=meeting.id,
        file_name="raw_recording.mp4",
        storage_path=to_safe_relative_path(raw_file),
        media_type="VIDEO",
        mime_type="video/mp4",
        file_size_bytes=raw_file.stat().st_size,
        duration_seconds=1.0,
        is_raw_deleted=False,
    )
    db_session.add(raw_asset)
    db_session.commit()

    # Create associated normalized WAV referencing raw_asset.id
    norm_filename = f"{raw_asset.id}_normalized_16k.wav"
    norm_file = meeting_dir / norm_filename
    _create_synthetic_wav(norm_file, duration_sec=1.0)

    norm_asset = MediaAsset(
        meeting_id=meeting.id,
        file_name=norm_filename,
        storage_path=to_safe_relative_path(norm_file),
        media_type="NORMALIZED_AUDIO",
        mime_type="audio/wav",
        file_size_bytes=norm_file.stat().st_size,
        duration_seconds=1.0,
        is_raw_deleted=False,
    )
    db_session.add(norm_asset)
    db_session.commit()

    job = ProcessingJob(
        meeting_id=meeting.id,
        job_type=JobType.TRANSCRIPTION.value,
        status=JobStatus.PENDING.value,
    )
    db_session.add(job)
    db_session.commit()

    # Run transcription
    process_transcription_job_sync(job.id)

    db_session.refresh(job)
    assert job.status == JobStatus.COMPLETED.value
    assert job.progress == 100.0

    # Assert raw asset is deleted on disk and flagged in DB
    db_session.refresh(raw_asset)
    assert raw_asset.is_raw_deleted is True
    assert not raw_file.exists()

    # Assert normalized asset is ALSO deleted on disk and flagged in DB
    db_session.refresh(norm_asset)
    assert norm_asset.is_raw_deleted is True
    assert not norm_file.exists()

    # Assert distinct AuditLog for raw asset
    raw_audits = (
        db_session.query(AuditLog)
        .filter(AuditLog.entity_id == raw_asset.id, AuditLog.action == "DELETE_RAW_AUDIO")
        .all()
    )
    assert len(raw_audits) == 1
    assert raw_audits[0].user_id == user.id
    raw_details = json.loads(raw_audits[0].details)
    assert raw_details["asset_role"] == "ORIGINAL_RAW"
    assert raw_details["status"] == "deleted"

    # Assert distinct AuditLog for normalized asset
    norm_audits = (
        db_session.query(AuditLog)
        .filter(AuditLog.entity_id == norm_asset.id, AuditLog.action == "DELETE_NORMALIZED_AUDIO")
        .all()
    )
    assert len(norm_audits) == 1
    assert norm_audits[0].user_id == user.id
    norm_details = json.loads(norm_audits[0].details)
    assert norm_details["asset_role"] == "NORMALIZED_AUDIO"
    assert norm_details["source_raw_asset_id"] == raw_asset.id
    assert norm_details["status"] == "deleted"


def test_autodelete_disabled_preserves_both_assets(db_session, monkeypatch):
    """When AUTO_DELETE_RAW_POST_TRANSCRIPTION=False:
    1. Both raw upload and normalized WAV are preserved on disk.
    2. Both DB records keep is_raw_deleted=False.
    3. No deletion audit logs are written.
    4. Job completes with status COMPLETED.
    """
    monkeypatch.setattr(settings, "AUTO_DELETE_RAW_POST_TRANSCRIPTION", False)
    _setup_mock_asr(monkeypatch)
    monkeypatch.setattr("app.services.transcription.job_handler.SessionLocal", lambda: db_session)
    monkeypatch.setattr(db_session, "close", lambda: None)

    org = Organization(name="Preserve Org", slug="preserve-org")
    user = User(email="owner@preserve.com", full_name="Preserve Owner")
    db_session.add_all([org, user])
    db_session.commit()

    meeting = Meeting(
        organization_id=org.id,
        created_by_user_id=user.id,
        title="Preserve Both Meeting",
    )
    db_session.add(meeting)
    db_session.commit()

    meeting_dir = get_meeting_storage_dir(meeting.id)
    raw_file = meeting_dir / "keep_raw.mp4"
    raw_file.write_bytes(b"KEEP_RAW_DATA")

    norm_file = meeting_dir / "keep_norm.wav"
    _create_synthetic_wav(norm_file, duration_sec=1.0)

    raw_asset = MediaAsset(
        meeting_id=meeting.id,
        file_name="keep_raw.mp4",
        storage_path=to_safe_relative_path(raw_file),
        media_type="VIDEO",
        mime_type="video/mp4",
        file_size_bytes=raw_file.stat().st_size,
        duration_seconds=1.0,
        is_raw_deleted=False,
    )
    norm_asset = MediaAsset(
        meeting_id=meeting.id,
        file_name="keep_norm.wav",
        storage_path=to_safe_relative_path(norm_file),
        media_type="NORMALIZED_AUDIO",
        mime_type="audio/wav",
        file_size_bytes=norm_file.stat().st_size,
        duration_seconds=1.0,
        is_raw_deleted=False,
    )
    db_session.add_all([raw_asset, norm_asset])
    db_session.commit()

    job = ProcessingJob(
        meeting_id=meeting.id,
        job_type=JobType.TRANSCRIPTION.value,
        status=JobStatus.PENDING.value,
    )
    db_session.add(job)
    db_session.commit()

    process_transcription_job_sync(job.id)

    db_session.refresh(job)
    assert job.status == JobStatus.COMPLETED.value

    db_session.refresh(raw_asset)
    assert raw_asset.is_raw_deleted is False
    assert raw_file.exists()

    db_session.refresh(norm_asset)
    assert norm_asset.is_raw_deleted is False
    assert norm_file.exists()

    audits = (
        db_session.query(AuditLog)
        .filter(AuditLog.action.in_(["DELETE_RAW_AUDIO", "DELETE_NORMALIZED_AUDIO"]))
        .all()
    )
    assert len(audits) == 0

    # Cleanup
    if raw_file.exists():
        raw_file.unlink()
    if norm_file.exists():
        norm_file.unlink()


def test_cross_meeting_normalized_asset_never_deleted(db_session):
    """A normalized asset belonging to Meeting B must NEVER be deleted when deleting Meeting A's raw asset."""
    org = Organization(name="Cross Meet Org", slug="cross-meet-org")
    user = User(email="cross@meet.com", full_name="Cross User")
    meeting_a = Meeting(organization_id=org.id, title="Meeting A")
    meeting_b = Meeting(organization_id=org.id, title="Meeting B")
    db_session.add_all([org, user, meeting_a, meeting_b])
    db_session.commit()

    dir_a = get_meeting_storage_dir(meeting_a.id)
    raw_a_file = dir_a / "meeting_a_raw.mp4"
    raw_a_file.write_bytes(b"MEETING_A_RAW_CONTENT")

    raw_a = MediaAsset(
        meeting_id=meeting_a.id,
        file_name="meeting_a_raw.mp4",
        storage_path=to_safe_relative_path(raw_a_file),
        media_type="VIDEO",
        mime_type="video/mp4",
        is_raw_deleted=False,
    )
    db_session.add(raw_a)
    db_session.commit()

    # Meeting B normalized asset
    dir_b = get_meeting_storage_dir(meeting_b.id)
    norm_b_file = dir_b / "meeting_b_norm.wav"
    _create_synthetic_wav(norm_b_file, duration_sec=1.0)

    norm_b = MediaAsset(
        meeting_id=meeting_b.id,
        file_name="meeting_b_norm.wav",
        storage_path=to_safe_relative_path(norm_b_file),
        media_type="NORMALIZED_AUDIO",
        mime_type="audio/wav",
        is_raw_deleted=False,
    )
    db_session.add(norm_b)
    db_session.commit()

    # Delete Meeting A's raw asset
    delete_raw_media_asset(db_session, raw_a.id, user_id=user.id, delete_normalized=True)

    # Meeting A raw asset is deleted
    db_session.refresh(raw_a)
    assert raw_a.is_raw_deleted is True
    assert not raw_a_file.exists()

    # Meeting B normalized asset MUST REMAIN completely untouched
    db_session.refresh(norm_b)
    assert norm_b.is_raw_deleted is False
    assert norm_b_file.exists()

    # No audit log for Meeting B's asset
    b_audits = db_session.query(AuditLog).filter(AuditLog.entity_id == norm_b.id).all()
    assert len(b_audits) == 0

    # Cleanup
    if norm_b_file.exists():
        norm_b_file.unlink()


def test_unrelated_normalized_asset_in_same_meeting_not_deleted(db_session):
    """When a meeting has multiple raw assets, deleting raw asset 1 only deletes
    its own associated normalized asset, leaving raw asset 2's normalized asset intact.
    """
    org = Organization(name="Intra Meet Org", slug="intra-meet-org")
    user = User(email="intra@meet.com", full_name="Intra User")
    meeting = Meeting(organization_id=org.id, title="Multi-Upload Meeting")
    db_session.add_all([org, user, meeting])
    db_session.commit()

    meeting_dir = get_meeting_storage_dir(meeting.id)

    # Upload 1
    raw_1_file = meeting_dir / "raw_1.mp4"
    raw_1_file.write_bytes(b"RAW_1_DATA")
    raw_1 = MediaAsset(
        meeting_id=meeting.id,
        file_name="raw_1.mp4",
        storage_path=to_safe_relative_path(raw_1_file),
        media_type="VIDEO",
        mime_type="video/mp4",
        is_raw_deleted=False,
    )
    db_session.add(raw_1)
    db_session.commit()

    norm_1_file = meeting_dir / f"{raw_1.id}_normalized_16k.wav"
    _create_synthetic_wav(norm_1_file, duration_sec=1.0)
    norm_1 = MediaAsset(
        meeting_id=meeting.id,
        file_name=f"{raw_1.id}_normalized_16k.wav",
        storage_path=to_safe_relative_path(norm_1_file),
        media_type="NORMALIZED_AUDIO",
        mime_type="audio/wav",
        is_raw_deleted=False,
    )

    # Upload 2
    raw_2_file = meeting_dir / "raw_2.mp4"
    raw_2_file.write_bytes(b"RAW_2_DATA")
    raw_2 = MediaAsset(
        meeting_id=meeting.id,
        file_name="raw_2.mp4",
        storage_path=to_safe_relative_path(raw_2_file),
        media_type="VIDEO",
        mime_type="video/mp4",
        is_raw_deleted=False,
    )
    db_session.add_all([norm_1, raw_2])
    db_session.commit()

    norm_2_file = meeting_dir / f"{raw_2.id}_normalized_16k.wav"
    _create_synthetic_wav(norm_2_file, duration_sec=1.0)
    norm_2 = MediaAsset(
        meeting_id=meeting.id,
        file_name=f"{raw_2.id}_normalized_16k.wav",
        storage_path=to_safe_relative_path(norm_2_file),
        media_type="NORMALIZED_AUDIO",
        mime_type="audio/wav",
        is_raw_deleted=False,
    )
    db_session.add(norm_2)
    db_session.commit()

    # Delete raw asset 1
    delete_raw_media_asset(db_session, raw_1.id, user_id=user.id, delete_normalized=True)

    # Raw 1 and Norm 1 are deleted
    db_session.refresh(raw_1)
    db_session.refresh(norm_1)
    assert raw_1.is_raw_deleted is True
    assert not raw_1_file.exists()
    assert norm_1.is_raw_deleted is True
    assert not norm_1_file.exists()

    # Raw 2 and Norm 2 are PRESERVED
    db_session.refresh(raw_2)
    db_session.refresh(norm_2)
    assert raw_2.is_raw_deleted is False
    assert raw_2_file.exists()
    assert norm_2.is_raw_deleted is False
    assert norm_2_file.exists()

    # Cleanup
    if raw_2_file.exists():
        raw_2_file.unlink()
    if norm_2_file.exists():
        norm_2_file.unlink()


def test_missing_files_handled_safely(db_session):
    """If physical files are already missing from disk, deletion completes safely without error,
    updates is_raw_deleted=True, and records status='file_not_found_marked_deleted' in AuditLog details.
    """
    org = Organization(name="Missing Files Org", slug="missing-files-org")
    user = User(email="missing@files.com", full_name="Missing User")
    meeting = Meeting(organization_id=org.id, title="Missing Files Meeting")
    db_session.add_all([org, user, meeting])
    db_session.commit()

    # Create DB assets without creating physical files on disk
    raw_asset = MediaAsset(
        meeting_id=meeting.id,
        file_name="phantom_raw.mp4",
        storage_path=f"storage/{meeting.id}/phantom_raw.mp4",
        media_type="VIDEO",
        mime_type="video/mp4",
        is_raw_deleted=False,
    )
    db_session.add(raw_asset)
    db_session.commit()

    norm_asset = MediaAsset(
        meeting_id=meeting.id,
        file_name=f"{raw_asset.id}_phantom_norm.wav",
        storage_path=f"storage/{meeting.id}/{raw_asset.id}_phantom_norm.wav",
        media_type="NORMALIZED_AUDIO",
        mime_type="audio/wav",
        is_raw_deleted=False,
    )
    db_session.add(norm_asset)
    db_session.commit()

    # Deletion should not crash
    result = delete_raw_media_asset(db_session, raw_asset.id, user_id=user.id, delete_normalized=True)
    assert result.is_raw_deleted is True

    db_session.refresh(raw_asset)
    db_session.refresh(norm_asset)
    assert raw_asset.is_raw_deleted is True
    assert norm_asset.is_raw_deleted is True

    # Audit log details note file_not_found_marked_deleted
    raw_audit = db_session.query(AuditLog).filter(AuditLog.entity_id == raw_asset.id).first()
    assert raw_audit is not None
    assert "file_not_found_marked_deleted" in raw_audit.details

    norm_audit = db_session.query(AuditLog).filter(AuditLog.entity_id == norm_asset.id).first()
    assert norm_audit is not None
    assert "file_not_found_marked_deleted" in norm_audit.details


def test_repeated_deletion_is_idempotent(db_session):
    """Calling delete_raw_media_asset repeatedly on already-deleted assets returns safely
    without error and does not generate duplicate audit log rows.
    """
    org = Organization(name="Idempotency Org", slug="idempotency-org")
    user = User(email="idem@potency.com", full_name="Idem User")
    meeting = Meeting(organization_id=org.id, title="Idempotent Meeting")
    db_session.add_all([org, user, meeting])
    db_session.commit()

    meeting_dir = get_meeting_storage_dir(meeting.id)
    raw_file = meeting_dir / "idem_raw.wav"
    _create_synthetic_wav(raw_file, duration_sec=1.0)

    raw_asset = MediaAsset(
        meeting_id=meeting.id,
        file_name="idem_raw.wav",
        storage_path=to_safe_relative_path(raw_file),
        media_type="AUDIO",
        mime_type="audio/wav",
        is_raw_deleted=False,
    )
    db_session.add(raw_asset)
    db_session.commit()

    # 1. First deletion
    first_res = delete_raw_media_asset(db_session, raw_asset.id, user_id=user.id, delete_normalized=True)
    assert first_res.is_raw_deleted is True

    initial_audit_count = db_session.query(AuditLog).filter(AuditLog.entity_id == raw_asset.id).count()
    assert initial_audit_count == 1

    # 2. Second deletion (idempotent call)
    second_res = delete_raw_media_asset(db_session, raw_asset.id, user_id=user.id, delete_normalized=True)
    assert second_res.is_raw_deleted is True

    second_audit_count = db_session.query(AuditLog).filter(AuditLog.entity_id == raw_asset.id).count()
    assert second_audit_count == 1  # No duplicate audit log entry


def test_failed_transcript_persistence_retains_assets(db_session, monkeypatch):
    """If transcript persistence verification fails, the job fails and NEITHER raw nor
    normalized assets are deleted.
    """
    monkeypatch.setattr(settings, "AUTO_DELETE_RAW_POST_TRANSCRIPTION", True)
    _setup_mock_asr(monkeypatch, text="This will fail verification")
    monkeypatch.setattr("app.services.transcription.job_handler.SessionLocal", lambda: db_session)
    monkeypatch.setattr(db_session, "close", lambda: None)

    org = Organization(name="Fail Verification Org", slug="fail-verify-org")
    user = User(email="fail@verify.com", full_name="Fail User")
    db_session.add_all([org, user])
    db_session.commit()

    meeting = Meeting(
        organization_id=org.id,
        created_by_user_id=user.id,
        title="Failed Verification Meeting",
    )
    db_session.add(meeting)
    db_session.commit()

    meeting_dir = get_meeting_storage_dir(meeting.id)
    raw_file = meeting_dir / "verify_raw.mp4"
    raw_file.write_bytes(b"RETAIN_DUE_TO_FAILURE")

    norm_file = meeting_dir / "verify_norm.wav"
    _create_synthetic_wav(norm_file, duration_sec=1.0)

    raw_asset = MediaAsset(
        meeting_id=meeting.id,
        file_name="verify_raw.mp4",
        storage_path=to_safe_relative_path(raw_file),
        media_type="VIDEO",
        mime_type="video/mp4",
        is_raw_deleted=False,
    )
    norm_asset = MediaAsset(
        meeting_id=meeting.id,
        file_name="verify_norm.wav",
        storage_path=to_safe_relative_path(norm_file),
        media_type="NORMALIZED_AUDIO",
        mime_type="audio/wav",
        is_raw_deleted=False,
    )
    db_session.add_all([raw_asset, norm_asset])
    db_session.commit()

    job = ProcessingJob(
        meeting_id=meeting.id,
        job_type=JobType.TRANSCRIPTION.value,
        status=JobStatus.PENDING.value,
    )
    db_session.add(job)
    db_session.commit()

    # Simulate persistence verification failure by making count() return 999 instead of 1
    orig_query = db_session.query

    def failing_query(*args, **kwargs):
        query_obj = orig_query(*args, **kwargs)
        if args and args[0] is TranscriptSegment:
            mock_q = MagicMock(wraps=query_obj)
            mock_filter = MagicMock()
            mock_filter.count.return_value = 999  # Mismatch with len(segments) == 1
            mock_q.filter.return_value = mock_filter
            return mock_q
        return query_obj

    monkeypatch.setattr(db_session, "query", failing_query)

    # Run job
    process_transcription_job_sync(job.id)

    # Restore query for assertions
    monkeypatch.setattr(db_session, "query", orig_query)

    db_session.refresh(job)
    assert job.status == JobStatus.FAILED.value
    assert "persistence mismatch" in (job.error_message or "").lower()

    # Both assets MUST be preserved
    db_session.refresh(raw_asset)
    db_session.refresh(norm_asset)
    assert raw_asset.is_raw_deleted is False
    assert raw_file.exists()
    assert norm_asset.is_raw_deleted is False
    assert norm_file.exists()

    # Zero deletion audit logs
    audits = db_session.query(AuditLog).filter(AuditLog.action.in_(["DELETE_RAW_AUDIO", "DELETE_NORMALIZED_AUDIO"])).all()
    assert len(audits) == 0

    # Cleanup
    if raw_file.exists():
        raw_file.unlink()
    if norm_file.exists():
        norm_file.unlink()


def test_manual_deletion_endpoint_deletes_raw_and_associated_normalized(client, db_session, org_admin_headers):
    """When an ORG_ADMIN manually calls DELETE /api/v1/media/assets/{raw_asset.id}:
    1. Both raw upload and associated normalized WAV are unlinked from disk.
    2. Both DB records are updated to is_raw_deleted=True.
    3. Distinct AuditLog records are created for both assets.
    4. HTTP 200 OK is returned with the MediaAssetResponse.
    """
    # 1. Resolve org and user from org_admin_headers
    org = db_session.query(Organization).first()
    assert org is not None

    meeting = Meeting(organization_id=org.id, title="Manual Deletion Meeting")
    db_session.add(meeting)
    db_session.commit()

    meeting_dir = get_meeting_storage_dir(meeting.id)
    raw_file = meeting_dir / "manual_raw.mp4"
    raw_file.write_bytes(b"MANUAL_RAW_DATA")

    raw_asset = MediaAsset(
        meeting_id=meeting.id,
        file_name="manual_raw.mp4",
        storage_path=to_safe_relative_path(raw_file),
        media_type="VIDEO",
        mime_type="video/mp4",
        is_raw_deleted=False,
    )
    db_session.add(raw_asset)
    db_session.commit()

    norm_filename = f"{raw_asset.id}_normalized_16k.wav"
    norm_file = meeting_dir / norm_filename
    _create_synthetic_wav(norm_file, duration_sec=1.0)

    norm_asset = MediaAsset(
        meeting_id=meeting.id,
        file_name=norm_filename,
        storage_path=to_safe_relative_path(norm_file),
        media_type="NORMALIZED_AUDIO",
        mime_type="audio/wav",
        is_raw_deleted=False,
    )
    db_session.add(norm_asset)
    db_session.commit()

    # Call manual deletion endpoint
    resp = client.delete(f"/api/v1/media/assets/{raw_asset.id}", headers=org_admin_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == raw_asset.id
    assert data["is_raw_deleted"] is True

    # Both files unlinked
    assert not raw_file.exists()
    assert not norm_file.exists()

    # Both marked in DB
    db_session.refresh(raw_asset)
    db_session.refresh(norm_asset)
    assert raw_asset.is_raw_deleted is True
    assert norm_asset.is_raw_deleted is True

    # Both have distinct audit logs
    raw_audits = db_session.query(AuditLog).filter(AuditLog.entity_id == raw_asset.id, AuditLog.action == "DELETE_RAW_AUDIO").all()
    assert len(raw_audits) == 1

    norm_audits = db_session.query(AuditLog).filter(AuditLog.entity_id == norm_asset.id, AuditLog.action == "DELETE_NORMALIZED_AUDIO").all()
    assert len(norm_audits) == 1


def test_two_youtube_assets_in_one_meeting_delete_independently(db_session):
    """When a meeting contains two YouTube raw assets, deleting one deletes only its own
    associated normalized WAV, leaving the other YouTube raw and normalized WAV intact.
    """
    org = Organization(name="YouTube Dual Org", slug="yt-dual-org")
    user = User(email="yt@dual.com", full_name="YT Dual User")
    meeting = Meeting(organization_id=org.id, title="Dual YouTube Meeting")
    db_session.add_all([org, user, meeting])
    db_session.commit()

    meeting_dir = get_meeting_storage_dir(meeting.id)

    # YouTube Stream 1
    raw_yt_1_file = meeting_dir / "yt_vid1_raw.mp4"
    raw_yt_1_file.write_bytes(b"YT_VIDEO_1_RAW")
    raw_yt_1 = MediaAsset(
        meeting_id=meeting.id,
        file_name="yt_vid1_raw.mp4",
        storage_path=to_safe_relative_path(raw_yt_1_file),
        media_type="YOUTUBE_RAW",
        mime_type="audio/mp4",
        is_raw_deleted=False,
    )
    db_session.add(raw_yt_1)
    db_session.commit()

    norm_yt_1_file = meeting_dir / f"{raw_yt_1.id}_yt_vid1_normalized_16k.wav"
    _create_synthetic_wav(norm_yt_1_file, duration_sec=1.0)
    norm_yt_1 = MediaAsset(
        meeting_id=meeting.id,
        file_name=norm_yt_1_file.name,
        storage_path=to_safe_relative_path(norm_yt_1_file),
        media_type="NORMALIZED_AUDIO",
        mime_type="audio/wav",
        is_raw_deleted=False,
    )

    # YouTube Stream 2
    raw_yt_2_file = meeting_dir / "yt_vid2_raw.mp4"
    raw_yt_2_file.write_bytes(b"YT_VIDEO_2_RAW")
    raw_yt_2 = MediaAsset(
        meeting_id=meeting.id,
        file_name="yt_vid2_raw.mp4",
        storage_path=to_safe_relative_path(raw_yt_2_file),
        media_type="YOUTUBE_RAW",
        mime_type="audio/mp4",
        is_raw_deleted=False,
    )
    db_session.add_all([norm_yt_1, raw_yt_2])
    db_session.commit()

    norm_yt_2_file = meeting_dir / f"{raw_yt_2.id}_yt_vid2_normalized_16k.wav"
    _create_synthetic_wav(norm_yt_2_file, duration_sec=1.0)
    norm_yt_2 = MediaAsset(
        meeting_id=meeting.id,
        file_name=norm_yt_2_file.name,
        storage_path=to_safe_relative_path(norm_yt_2_file),
        media_type="NORMALIZED_AUDIO",
        mime_type="audio/wav",
        is_raw_deleted=False,
    )
    db_session.add(norm_yt_2)
    db_session.commit()

    # Delete YouTube Stream 1
    delete_raw_media_asset(db_session, raw_yt_1.id, user_id=user.id, delete_normalized=True)

    # Stream 1 raw and normalized are deleted
    db_session.refresh(raw_yt_1)
    db_session.refresh(norm_yt_1)
    assert raw_yt_1.is_raw_deleted is True
    assert not raw_yt_1_file.exists()
    assert norm_yt_1.is_raw_deleted is True
    assert not norm_yt_1_file.exists()

    # Stream 2 raw and normalized are PRESERVED
    db_session.refresh(raw_yt_2)
    db_session.refresh(norm_yt_2)
    assert raw_yt_2.is_raw_deleted is False
    assert raw_yt_2_file.exists()
    assert norm_yt_2.is_raw_deleted is False
    assert norm_yt_2_file.exists()

    # Audit records distinct
    yt_1_raw_audit = db_session.query(AuditLog).filter(AuditLog.entity_id == raw_yt_1.id).all()
    assert len(yt_1_raw_audit) == 1
    yt_1_norm_audit = db_session.query(AuditLog).filter(AuditLog.entity_id == norm_yt_1.id).all()
    assert len(yt_1_norm_audit) == 1

    yt_2_raw_audit = db_session.query(AuditLog).filter(AuditLog.entity_id == raw_yt_2.id).all()
    assert len(yt_2_raw_audit) == 0
    yt_2_norm_audit = db_session.query(AuditLog).filter(AuditLog.entity_id == norm_yt_2.id).all()
    assert len(yt_2_norm_audit) == 0

    # Cleanup
    if raw_yt_2_file.exists():
        raw_yt_2_file.unlink()
    if norm_yt_2_file.exists():
        norm_yt_2_file.unlink()


def test_exact_token_association_and_overlap_safety(db_session):
    """Verify exact token matching:
    1. asset-1 does NOT collide with asset-10_normalized_16k.wav.
    2. asset-1 matches asset-1_normalized_16k.wav.
    3. Valid exact UUID matches.
    4. Wrong UUID does not match.
    5. Unlinked normalized asset in multi-raw meeting is retained (conservative).
    """
    import uuid
    org = Organization(name="Token Safety Org", slug="token-safety-org")
    meeting = Meeting(organization_id=org.id, title="Token Safety Meeting")
    db_session.add_all([org, meeting])
    db_session.commit()

    # 1. Overlapping prefix test: asset-1 vs asset-10
    raw_1 = MediaAsset(id="asset-1", meeting_id=meeting.id, file_name="raw1.wav", storage_path="storage/raw1.wav", media_type="AUDIO", mime_type="audio/wav")
    raw_10 = MediaAsset(id="asset-10", meeting_id=meeting.id, file_name="raw10.wav", storage_path="storage/raw10.wav", media_type="AUDIO", mime_type="audio/wav")
    norm_1 = MediaAsset(id="norm-1", meeting_id=meeting.id, file_name="asset-1_normalized_16k.wav", storage_path="storage/asset-1_normalized_16k.wav", media_type="NORMALIZED_AUDIO", mime_type="audio/wav")
    norm_10 = MediaAsset(id="norm-10", meeting_id=meeting.id, file_name="asset-10_normalized_16k.wav", storage_path="storage/asset-10_normalized_16k.wav", media_type="NORMALIZED_AUDIO", mime_type="audio/wav")
    db_session.add_all([raw_1, raw_10, norm_1, norm_10])
    db_session.commit()

    # asset-1 MUST NOT match asset-10's normalized file
    assert is_associated_normalized_asset(raw_1, norm_10, db_session) is False
    # asset-1 MUST match asset-1's normalized file
    assert is_associated_normalized_asset(raw_1, norm_1, db_session) is True
    # asset-10 MUST match asset-10's normalized file
    assert is_associated_normalized_asset(raw_10, norm_10, db_session) is True
    # asset-10 MUST NOT match asset-1's normalized file
    assert is_associated_normalized_asset(raw_10, norm_1, db_session) is False

    # 2. Exact valid UUID vs wrong UUID
    u1 = str(uuid.uuid4())
    u2 = str(uuid.uuid4())
    raw_u1 = MediaAsset(id=u1, meeting_id=meeting.id, file_name="u1.mp4", storage_path="storage/u1.mp4", media_type="VIDEO", mime_type="video/mp4")
    norm_u1 = MediaAsset(id=str(uuid.uuid4()), meeting_id=meeting.id, file_name=f"{u1}_normalized_16k.wav", storage_path=f"storage/{u1}_normalized_16k.wav", media_type="NORMALIZED_AUDIO", mime_type="audio/wav")
    norm_u2 = MediaAsset(id=str(uuid.uuid4()), meeting_id=meeting.id, file_name=f"{u2}_normalized_16k.wav", storage_path=f"storage/{u2}_normalized_16k.wav", media_type="NORMALIZED_AUDIO", mime_type="audio/wav")
    db_session.add_all([raw_u1, norm_u1, norm_u2])
    db_session.commit()

    assert is_associated_normalized_asset(raw_u1, norm_u1, db_session) is True
    assert is_associated_normalized_asset(raw_u1, norm_u2, db_session) is False

    # 3. Unlinked candidate in multi-raw-asset meeting
    norm_unlinked = MediaAsset(id=str(uuid.uuid4()), meeting_id=meeting.id, file_name="arbitrary_unlinked.wav", storage_path="storage/arbitrary_unlinked.wav", media_type="NORMALIZED_AUDIO", mime_type="audio/wav")
    db_session.add(norm_unlinked)
    db_session.commit()
    # In a meeting with multiple raw assets, unlinked asset CANNOT be attributed to raw_u1
    assert is_associated_normalized_asset(raw_u1, norm_unlinked, db_session) is False


def test_active_sibling_job_defers_deletion_until_completion(db_session, monkeypatch):
    """When a sibling media-consuming job (TRANSCRIPTION or DIARIZATION) is PENDING or IN_PROGRESS:
    1. Automatic media cleanup is deferred.
    2. Assets and files remain intact.
    3. Once the sibling job completes, deferred cleanup can proceed.
    """
    monkeypatch.setattr(settings, "AUTO_DELETE_RAW_POST_TRANSCRIPTION", True)
    _setup_mock_asr(monkeypatch)
    monkeypatch.setattr("app.services.transcription.job_handler.SessionLocal", lambda: db_session)
    monkeypatch.setattr(db_session, "close", lambda: None)

    org = Organization(name="Sibling Job Org", slug="sibling-job-org")
    user = User(email="sibling@job.com", full_name="Sibling User")
    meeting = Meeting(organization_id=org.id, created_by_user_id=user.id, title="Sibling Meeting")
    db_session.add_all([org, user, meeting])
    db_session.commit()

    meeting_dir = get_meeting_storage_dir(meeting.id)
    raw_file = meeting_dir / "sibling_raw.mp4"
    raw_file.write_bytes(b"SIBLING_RAW_PAYLOAD")

    raw_asset = MediaAsset(
        meeting_id=meeting.id,
        file_name="sibling_raw.mp4",
        storage_path=to_safe_relative_path(raw_file),
        media_type="VIDEO",
        mime_type="video/mp4",
        is_raw_deleted=False,
    )
    db_session.add(raw_asset)
    db_session.commit()

    norm_file = meeting_dir / f"{raw_asset.id}_normalized_16k.wav"
    _create_synthetic_wav(norm_file, duration_sec=1.0)
    norm_asset = MediaAsset(
        meeting_id=meeting.id,
        file_name=norm_file.name,
        storage_path=to_safe_relative_path(norm_file),
        media_type="NORMALIZED_AUDIO",
        mime_type="audio/wav",
        is_raw_deleted=False,
    )
    db_session.add(norm_asset)
    db_session.commit()

    # Job 1: Transcription job that will finish
    job_1 = ProcessingJob(
        meeting_id=meeting.id,
        job_type=JobType.TRANSCRIPTION.value,
        status=JobStatus.PENDING.value,
    )
    # Job 2: Concurrent sibling media-consuming job (DIARIZATION) still IN_PROGRESS
    job_2 = ProcessingJob(
        meeting_id=meeting.id,
        job_type=JobType.DIARIZATION.value,
        status=JobStatus.IN_PROGRESS.value,
    )
    db_session.add_all([job_1, job_2])
    db_session.commit()

    # Execute Job 1
    process_transcription_job_sync(job_1.id)

    db_session.refresh(job_1)
    assert job_1.status == JobStatus.COMPLETED.value

    # Because Job 2 is still IN_PROGRESS, media cleanup was DEFERRED
    db_session.refresh(raw_asset)
    db_session.refresh(norm_asset)
    assert raw_asset.is_raw_deleted is False
    assert raw_file.exists()
    assert norm_asset.is_raw_deleted is False
    assert norm_file.exists()

    # Zero deletion audit logs recorded so far
    audits = db_session.query(AuditLog).filter(AuditLog.action.in_(["DELETE_RAW_AUDIO", "DELETE_NORMALIZED_AUDIO"])).all()
    assert len(audits) == 0

    # Now Job 2 completes
    job_2.status = JobStatus.COMPLETED.value
    db_session.commit()

    # Job 3: A subsequent transcription job (or retry) completes when no sibling jobs remain
    job_3 = ProcessingJob(
        meeting_id=meeting.id,
        job_type=JobType.TRANSCRIPTION.value,
        status=JobStatus.PENDING.value,
    )
    db_session.add(job_3)
    db_session.commit()

    process_transcription_job_sync(job_3.id)

    db_session.refresh(job_3)
    assert job_3.status == JobStatus.COMPLETED.value

    # Now deferred cleanup proceeds: both assets are unlinked and marked deleted
    db_session.refresh(raw_asset)
    db_session.refresh(norm_asset)
    assert raw_asset.is_raw_deleted is True
    assert not raw_file.exists()
    assert norm_asset.is_raw_deleted is True
    assert not norm_file.exists()


def test_unrelated_job_type_does_not_block_deletion(db_session, monkeypatch):
    """An unrelated active job (e.g. SUMMARIZATION or EXTRACTION) that does NOT consume
    audio media files must NOT block post-transcription auto-deletion.
    """
    monkeypatch.setattr(settings, "AUTO_DELETE_RAW_POST_TRANSCRIPTION", True)
    _setup_mock_asr(monkeypatch)
    monkeypatch.setattr("app.services.transcription.job_handler.SessionLocal", lambda: db_session)
    monkeypatch.setattr(db_session, "close", lambda: None)

    org = Organization(name="Unrelated Job Org", slug="unrelated-job-org")
    user = User(email="unrelated@job.com", full_name="Unrelated User")
    meeting = Meeting(organization_id=org.id, created_by_user_id=user.id, title="Unrelated Meeting")
    db_session.add_all([org, user, meeting])
    db_session.commit()

    meeting_dir = get_meeting_storage_dir(meeting.id)
    raw_file = meeting_dir / "unrelated_raw.mp4"
    raw_file.write_bytes(b"UNRELATED_RAW_PAYLOAD")

    raw_asset = MediaAsset(
        meeting_id=meeting.id,
        file_name="unrelated_raw.mp4",
        storage_path=to_safe_relative_path(raw_file),
        media_type="VIDEO",
        mime_type="video/mp4",
        is_raw_deleted=False,
    )
    db_session.add(raw_asset)
    db_session.commit()

    norm_file = meeting_dir / f"{raw_asset.id}_normalized_16k.wav"
    _create_synthetic_wav(norm_file, duration_sec=1.0)
    norm_asset = MediaAsset(
        meeting_id=meeting.id,
        file_name=norm_file.name,
        storage_path=to_safe_relative_path(norm_file),
        media_type="NORMALIZED_AUDIO",
        mime_type="audio/wav",
        is_raw_deleted=False,
    )
    db_session.add(norm_asset)
    db_session.commit()

    job_1 = ProcessingJob(
        meeting_id=meeting.id,
        job_type=JobType.TRANSCRIPTION.value,
        status=JobStatus.PENDING.value,
    )
    # Active SUMMARIZATION job (does not consume audio files, consumes transcript text)
    job_summary = ProcessingJob(
        meeting_id=meeting.id,
        job_type=JobType.SUMMARIZATION.value,
        status=JobStatus.IN_PROGRESS.value,
    )
    db_session.add_all([job_1, job_summary])
    db_session.commit()

    process_transcription_job_sync(job_1.id)

    db_session.refresh(job_1)
    assert job_1.status == JobStatus.COMPLETED.value

    # Deletion proceeds immediately because SUMMARIZATION does not require media files
    db_session.refresh(raw_asset)
    db_session.refresh(norm_asset)
    assert raw_asset.is_raw_deleted is True
    assert not raw_file.exists()
    assert norm_asset.is_raw_deleted is True
    assert not norm_file.exists()
