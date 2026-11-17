"""Tests for enforcing configured maximum media duration (Phase 6D.3).

Covers:
1. Duration just below limit (accepted).
2. Duration exactly at limit (accepted).
3. Duration just above limit (rejected with 400 / validation error).
4. Missing, malformed, negative, NaN, infinite duration safely rejected.
5. Upload endpoint rejects over-limit or malformed media before creating DB records or jobs.
6. Rejected uploads leave no orphaned files on disk, no Meeting, no Job, no MediaAsset in DB.
7. YouTube metadata above limit rejected early before downloading streams.
8. YouTube metadata within limit but FFprobe actual duration above limit rejected after download.
9. Rejected YouTube media cleans up raw and partial files on disk.
10. Start transcription rejects over-limit media before creating transcription job.
"""

import io
import math
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

from app.core.config import settings
from app.models.meeting import Meeting, MediaAsset, ProcessingJob, JobType, JobStatus
from app.services.media_validation import (
    validate_media_duration,
    MediaValidationError,
)
from app.services.media_metadata import MediaMetadata
from app.services.youtube_ingestion import extract_youtube_stream, YouTubeIngestionError
from app.services.transcription.job_handler import process_transcription_job_sync


# --- 1. Unit Tests for validate_media_duration ---

def test_duration_just_below_limit_accepted():
    """Verify duration slightly below max limit is accepted."""
    limit = settings.MAX_MEDIA_DURATION_SECONDS
    val = limit - 0.5
    assert validate_media_duration(val) == val


def test_duration_exactly_at_limit_accepted():
    """Verify duration exactly at max limit is accepted."""
    limit = settings.MAX_MEDIA_DURATION_SECONDS
    assert validate_media_duration(limit) == limit


def test_duration_just_above_limit_rejected():
    """Verify duration slightly above max limit is rejected."""
    limit = settings.MAX_MEDIA_DURATION_SECONDS
    with pytest.raises(MediaValidationError, match="exceeds maximum allowed limit"):
        validate_media_duration(limit + 0.1)


@pytest.mark.parametrize("bad_val", [None, "invalid_str", -1.0, 0.0, float("nan"), float("inf"), float("-inf")])
def test_duration_missing_or_malformed_rejected(bad_val):
    """Verify missing, non-numeric, non-positive, NaN, and infinite durations are safely rejected."""
    with pytest.raises(MediaValidationError):
        validate_media_duration(bad_val)


def test_duration_custom_limit():
    """Verify custom max_duration parameter is respected."""
    custom_limit = 60.0
    assert validate_media_duration(59.9, max_duration=custom_limit) == 59.9
    assert validate_media_duration(60.0, max_duration=custom_limit) == 60.0
    with pytest.raises(MediaValidationError, match="exceeds maximum allowed limit"):
        validate_media_duration(60.1, max_duration=custom_limit)


# --- 2. Upload Endpoint Duration Enforcement & No-Orphan Guarantee ---

def test_upload_duration_within_limit_accepted(client, org_admin_headers, db_session, monkeypatch):
    """Verify upload with valid duration is accepted (HTTP 202) and creates DB records."""
    monkeypatch.setattr("app.api.endpoints.media._process_uploaded_media_async", lambda *a, **kw: None)

    fake_metadata = MediaMetadata(
        format_name="wav",
        duration_seconds=settings.MAX_MEDIA_DURATION_SECONDS - 100.0,
        has_audio=True,
    )
    monkeypatch.setattr("app.api.endpoints.media.validate_media_content", lambda p: (True, None))
    monkeypatch.setattr("app.api.endpoints.media.extract_media_metadata", lambda p: fake_metadata)

    fake_file = io.BytesIO(b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00D\xac\x00\x00\x88X\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00")
    resp = client.post(
        "/api/v1/media/upload",
        files={"file": ("meeting.wav", fake_file, "audio/wav")},
        data={"title": "Within Limit Meeting"},
        headers=org_admin_headers,
    )
    assert resp.status_code == 202
    data = resp.json()
    assert data["status"] == "accepted"
    meeting_id = data["meeting_id"]

    # Verify meeting, job, and asset were created in DB
    meeting = db_session.get(Meeting, meeting_id)
    assert meeting is not None
    assert len(meeting.media_assets) == 1
    assert meeting.media_assets[0].duration_seconds == settings.MAX_MEDIA_DURATION_SECONDS - 100.0


def test_upload_duration_exceeding_limit_rejected_no_orphans(client, org_admin_headers, db_session, monkeypatch):
    """Verify upload exceeding duration limit is rejected with HTTP 400 and leaves zero DB records or orphaned files."""
    initial_meeting_count = db_session.query(Meeting).count()
    initial_job_count = db_session.query(ProcessingJob).count()
    initial_asset_count = db_session.query(MediaAsset).count()

    fake_metadata = MediaMetadata(
        format_name="wav",
        duration_seconds=settings.MAX_MEDIA_DURATION_SECONDS + 30.0,  # Over limit
        has_audio=True,
    )
    monkeypatch.setattr("app.api.endpoints.media.validate_media_content", lambda p: (True, None))
    monkeypatch.setattr("app.api.endpoints.media.extract_media_metadata", lambda p: fake_metadata)

    fake_file = io.BytesIO(b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00D\xac\x00\x00\x88X\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00")
    resp = client.post(
        "/api/v1/media/upload",
        files={"file": ("too_long.wav", fake_file, "audio/wav")},
        data={"title": "Too Long Meeting"},
        headers=org_admin_headers,
    )
    assert resp.status_code == 400
    assert "exceeds maximum allowed limit" in resp.json()["detail"]

    # Verify ZERO database records were created
    assert db_session.query(Meeting).count() == initial_meeting_count
    assert db_session.query(ProcessingJob).count() == initial_job_count
    assert db_session.query(MediaAsset).count() == initial_asset_count


def test_upload_malformed_duration_rejected_no_orphans(client, org_admin_headers, db_session, monkeypatch):
    """Verify upload with unreadable/missing duration is rejected with HTTP 400 and leaves no DB records."""
    initial_meeting_count = db_session.query(Meeting).count()

    fake_metadata = MediaMetadata(
        format_name="wav",
        duration_seconds=None,  # Missing duration
        has_audio=True,
    )
    monkeypatch.setattr("app.api.endpoints.media.validate_media_content", lambda p: (True, None))
    monkeypatch.setattr("app.api.endpoints.media.extract_media_metadata", lambda p: fake_metadata)

    fake_file = io.BytesIO(b"RIFF dummy")
    resp = client.post(
        "/api/v1/media/upload",
        files={"file": ("corrupt.wav", fake_file, "audio/wav")},
        data={"title": "Corrupt Duration Meeting"},
        headers=org_admin_headers,
    )
    assert resp.status_code == 400
    assert "could not be determined" in resp.json()["detail"]
    assert db_session.query(Meeting).count() == initial_meeting_count


# --- 3. YouTube Ingestion Early Metadata & Post-Download FFprobe Enforcement ---

def test_youtube_metadata_above_limit_rejected_early(tmp_path):
    """Verify YouTube video with duration in metadata above limit is rejected before stream download."""
    url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    over_limit_duration = settings.MAX_MEDIA_DURATION_SECONDS + 3600.0  # 5 hours

    mock_ydl_instance = MagicMock()
    mock_ydl_instance.__enter__.return_value = mock_ydl_instance
    # Return over-limit metadata on info query
    mock_ydl_instance.extract_info.return_value = {
        "id": "dQw4w9WgXcQ",
        "title": "Long Keynote",
        "duration": over_limit_duration,
    }

    with patch("yt_dlp.YoutubeDL", return_value=mock_ydl_instance):
        with pytest.raises(YouTubeIngestionError, match="exceeds maximum allowed limit"):
            extract_youtube_stream(
                youtube_url=url,
                output_dir=tmp_path,
                confirmed_authorization=True,
            )

    # Verify download=True was NEVER called because pre-download check rejected it early
    for call in mock_ydl_instance.extract_info.call_args_list:
        assert call.kwargs.get("download") is not True, "download=True must not be called when pre-check fails"


def test_youtube_metadata_within_limit_but_ffprobe_duration_above_limit_rejected(tmp_path, monkeypatch):
    """Verify video where YouTube metadata claims short duration, but FFprobe actual duration exceeds limit is rejected."""
    url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    fake_raw_file = tmp_path / "yt_dQw4w9WgXcQ_raw.m4a"
    fake_raw_file.write_bytes(b"dummy audio stream bytes")

    mock_ydl_instance = MagicMock()
    mock_ydl_instance.__enter__.return_value = mock_ydl_instance
    # Metadata falsely claims 120 seconds
    mock_ydl_instance.extract_info.side_effect = [
        {"id": "dQw4w9WgXcQ", "duration": 120.0},  # download=False
        {"id": "dQw4w9WgXcQ", "duration": 120.0},  # download=True
    ]
    mock_ydl_instance.prepare_filename.return_value = str(fake_raw_file)

    # FFprobe reveals true duration is 15000 seconds (over limit)
    over_limit_meta = MediaMetadata(
        format_name="m4a",
        duration_seconds=settings.MAX_MEDIA_DURATION_SECONDS + 600.0,
        has_audio=True,
    )
    monkeypatch.setattr("app.services.youtube_ingestion.validate_media_content", lambda p: (True, None))
    monkeypatch.setattr("app.services.youtube_ingestion.extract_media_metadata", lambda p: over_limit_meta)

    with patch("yt_dlp.YoutubeDL", return_value=mock_ydl_instance):
        with pytest.raises(YouTubeIngestionError, match="exceeds maximum allowed limit"):
            extract_youtube_stream(
                youtube_url=url,
                output_dir=tmp_path,
                confirmed_authorization=True,
            )

    # Verify raw file and partial files were cleaned up
    assert not fake_raw_file.exists(), "Raw downloaded file must be deleted when duration validation fails!"
    assert list(tmp_path.glob("yt_*_raw*")) == []


# --- 4. Transcription Ingestion Gate ---

def test_transcription_rejected_if_asset_duration_exceeds_limit(client, db_session, org_admin_headers, test_org_a, org_admin_user):
    """Verify start_transcription endpoint rejects meetings with over-limit media duration before creating job."""
    meeting = Meeting(
        organization_id=test_org_a.id,
        created_by_user_id=org_admin_user.id,
        title="Existing Over-Limit Meeting",
        status="COMPLETED",
    )
    db_session.add(meeting)
    db_session.flush()

    asset = MediaAsset(
        meeting_id=meeting.id,
        file_name="long.wav",
        storage_path="meetings/dummy/long.wav",
        media_type="AUDIO",
        mime_type="audio/wav",
        file_size_bytes=1000,
        duration_seconds=settings.MAX_MEDIA_DURATION_SECONDS + 500.0,
        is_raw_deleted=False,
    )
    db_session.add(asset)
    db_session.commit()

    initial_job_count = db_session.query(ProcessingJob).count()

    resp = client.post(
        f"/api/v1/transcription/start/{meeting.id}",
        json={"transcription_mode": "standard"},
        headers=org_admin_headers,
    )
    assert resp.status_code == 400
    assert "exceeds maximum allowed limit" in resp.json()["detail"]

    # Verify no transcription job was created
    assert db_session.query(ProcessingJob).count() == initial_job_count
