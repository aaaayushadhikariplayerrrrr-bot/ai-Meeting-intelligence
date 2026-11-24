"""Tests for Safe Job Retry and Synchronized Audio Streaming with Privacy Policy."""
import os
from datetime import datetime, timezone
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.models.identity import User, Organization, OrganizationMembership, OrgRole
from app.models.meeting import Meeting, MediaAsset, ProcessingJob, InputSourceType, MeetingStatus, JobType, JobStatus
from app.core.security import hash_password, create_access_token
from app.services.media_storage import get_meeting_storage_dir


@pytest.fixture
def retry_test_setup(db_session: Session):
    org = Organization(name="Pipeline Org", slug="pipeline-org", is_active=True)
    db_session.add(org)
    db_session.commit()

    user = User(
        email="dev_pipeline@org.com",
        full_name="Pipeline Dev",
        hashed_password=hash_password("Password123!"),
        is_active=True,
    )
    db_session.add(user)
    db_session.commit()

    mem = OrganizationMembership(
        organization_id=org.id,
        user_id=user.id,
        role=OrgRole.ORG_ADMIN.value,
        is_active=True,
    )
    db_session.add(mem)
    db_session.commit()

    token = create_access_token(subject=user.id, token_version=user.token_version)
    headers = {
        "Authorization": f"Bearer {token}",
        "X-Organization-Id": org.id,
    }
    return {"org": org, "user": user, "headers": headers}


def test_safe_job_retry_validation_and_reset(client: TestClient, retry_test_setup, db_session: Session):
    headers = retry_test_setup["headers"]
    org = retry_test_setup["org"]

    m = Meeting(
        organization_id=org.id,
        title="Failed Ingestion Test",
        status=MeetingStatus.FAILED.value,
        input_source_type=InputSourceType.AUDIO.value,
    )
    db_session.add(m)
    db_session.commit()
    db_session.refresh(m)

    # 1. Non-failed job cannot be retried
    job_completed = ProcessingJob(
        meeting_id=m.id,
        job_type=JobType.TRANSCRIPTION.value,
        status=JobStatus.COMPLETED.value,
        progress=100.0,
        attempt_count=1,
        max_attempts=3,
    )
    db_session.add(job_completed)
    db_session.commit()

    res_not_failed = client.post(f"/api/v1/media/jobs/{job_completed.id}/retry", headers=headers)
    assert res_not_failed.status_code == 400
    assert "Only FAILED jobs are eligible" in res_not_failed.json()["detail"]

    # 2. Failed job where raw media was deleted per privacy policy cannot be retried for ASR
    job_failed_deleted = ProcessingJob(
        meeting_id=m.id,
        job_type=JobType.TRANSCRIPTION.value,
        status=JobStatus.FAILED.value,
        progress=25.0,
        attempt_count=1,
        max_attempts=3,
        error_message="Worker timeout during ASR",
    )
    deleted_asset = MediaAsset(
        meeting_id=m.id,
        file_name="deleted_audio.wav",
        storage_path="mock/path/deleted_audio.wav",
        media_type="AUDIO",
        mime_type="audio/wav",
        is_raw_deleted=True,
    )
    db_session.add_all([job_failed_deleted, deleted_asset])
    db_session.commit()

    res_privacy_block = client.post(f"/api/v1/media/jobs/{job_failed_deleted.id}/retry", headers=headers)
    assert res_privacy_block.status_code == 400
    assert "privacy policy" in res_privacy_block.json()["detail"]

    # 3. Eligible failed job with available media succeeds
    meeting_dir = get_meeting_storage_dir(m.id)
    os.makedirs(meeting_dir, exist_ok=True)
    real_file = meeting_dir / "valid_audio.wav"
    real_file.write_text("valid audio bits")

    active_asset = MediaAsset(
        meeting_id=m.id,
        file_name="valid_audio.wav",
        storage_path=str(real_file),
        media_type="AUDIO",
        mime_type="audio/wav",
        is_raw_deleted=False,
    )
    job_eligible = ProcessingJob(
        meeting_id=m.id,
        job_type=JobType.TRANSCRIPTION.value,
        status=JobStatus.FAILED.value,
        progress=40.0,
        attempt_count=1,
        max_attempts=3,
        error_message="Temporary CUDA OOM",
    )
    db_session.add_all([active_asset, job_eligible])
    db_session.commit()

    res_retry_success = client.post(f"/api/v1/media/jobs/{job_eligible.id}/retry", headers=headers)
    assert res_retry_success.status_code == 200
    retried_job = res_retry_success.json()
    assert retried_job["status"] == JobStatus.PENDING.value
    assert retried_job["progress"] == 0.0
    assert retried_job["error_message"] is None


def test_audio_streaming_respects_privacy_policy(client: TestClient, retry_test_setup, db_session: Session):
    headers = retry_test_setup["headers"]
    org = retry_test_setup["org"]

    m = Meeting(
        organization_id=org.id,
        title="Privacy Safe Audio Stream Check",
        status=MeetingStatus.COMPLETED.value,
        input_source_type=InputSourceType.AUDIO.value,
    )
    db_session.add(m)
    db_session.commit()
    db_session.refresh(m)

    # 1. When media asset has is_raw_deleted == True, returns 410 Gone
    deleted_asset = MediaAsset(
        meeting_id=m.id,
        file_name="auto_deleted.wav",
        storage_path="mock/auto_deleted.wav",
        media_type="AUDIO",
        mime_type="audio/wav",
        is_raw_deleted=True,
    )
    db_session.add(deleted_asset)
    db_session.commit()

    res_gone = client.get(f"/api/v1/media/meetings/{m.id}/audio", headers=headers)
    assert res_gone.status_code == 410
    assert "deleted per the organization's privacy and retention policy" in res_gone.json()["detail"]
