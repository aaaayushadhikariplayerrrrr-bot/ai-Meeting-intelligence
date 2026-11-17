"""Tests for Phase 7.4: Background Job Integration for Structured Meeting Intelligence.

Covers:
1. Extraction job dispatch via run_single_job_cycle.
2. Valid and invalid job payloads (missing, malformed JSON, schema violation).
3. Meeting ID and organization consistency validation.
4. Privacy-safe transcript enforcement through the worker.
5. Failed extraction preserves previously persisted intelligence (transaction rollback).
6. Reprocessing without duplicate records.
7. Job lifecycle timestamps (started_at, completed_at).
8. Startup recovery of interrupted IN_PROGRESS extraction jobs.
9. Completed extraction jobs skipped on re-execution.
10. Mocked Ollama failure fallback transparency through the worker.
11. Existing transcription job behavior remaining unchanged.
"""

import json
from datetime import datetime, timezone
from uuid import uuid4
from unittest.mock import patch, MagicMock

import httpx
import pytest
from sqlalchemy.orm import Session

from app.models.identity import Organization, User
from app.models.meeting import Meeting, ProcessingJob, JobType, JobStatus, MeetingStatus
from app.models.transcription import Transcript, TranscriptSegment, TranscriptType
from app.models.intelligence import MeetingSummary, Topic, Decision, ActionItem, Risk, UnresolvedQuestion
from app.services.job_manager import create_processing_job
from app.services.job_worker import (
    claim_job_atomically,
    get_next_pending_job_id,
    reconcile_stranded_jobs_on_startup,
    execute_claimed_job,
    run_single_job_cycle,
)
from app.services.intelligence.job_handler import process_extraction_job_sync
from app.services.intelligence.ollama_adapter import OllamaExtractor


@pytest.fixture(autouse=True)
def bind_session_local_to_test_engine(db_session, monkeypatch):
    """Ensure worker and handlers use the test database session."""
    monkeypatch.setattr("app.database.session.SessionLocal", lambda: db_session)
    monkeypatch.setattr("app.services.job_worker.SessionLocal", lambda: db_session)
    monkeypatch.setattr("app.services.intelligence.job_handler.SessionLocal", lambda: db_session)
    monkeypatch.setattr(db_session, "close", lambda: None)


def _setup_test_meeting(db: Session, title: str = "Worker Intelligence Test"):
    """Helper to populate an org, meeting, and privacy-safe transcript with segments."""
    org = Organization(name="Intelligence Worker Org", slug=f"intel-org-{uuid4().hex[:8]}")
    meeting = Meeting(organization_id=org.id, title=title, status=MeetingStatus.COMPLETED.value)
    db.add_all([org, meeting])
    db.flush()

    t_red = Transcript(
        meeting_id=meeting.id,
        transcript_type=TranscriptType.CLEANED_REDACTED.value,
        version=1,
    )
    db.add(t_red)
    db.flush()

    segs = [
        TranscriptSegment(
            transcript_id=t_red.id,
            sequence_order=0,
            start_time=0.0,
            end_time=10.0,
            text="We decided to deploy the structured intelligence worker.",
            language_code="en",
        ),
        TranscriptSegment(
            transcript_id=t_red.id,
            sequence_order=1,
            start_time=10.5,
            end_time=20.0,
            text="I will test the background job dispatch by Friday.",
            language_code="en",
        ),
        TranscriptSegment(
            transcript_id=t_red.id,
            sequence_order=2,
            start_time=20.5,
            end_time=30.0,
            text="Risk is potential payload mismatch between worker and engine.",
            language_code="en",
        ),
        TranscriptSegment(
            transcript_id=t_red.id,
            sequence_order=3,
            start_time=30.5,
            end_time=40.0,
            text="How will we verify startup recovery?",
            language_code="en",
        ),
    ]
    db.add_all(segs)
    db.commit()
    return org, meeting, t_red


def test_extraction_job_dispatch_successful_cycle(db_session):
    """Verify an EXTRACTION job is claimed, executed, completed, and intelligence persisted."""
    org, meeting, t_red = _setup_test_meeting(db_session)

    job = create_processing_job(
        db=db_session,
        meeting_id=meeting.id,
        job_type=JobType.EXTRACTION.value,
        payload={
            "meeting_id": meeting.id,
            "provider_override": "deterministic",
            "output_language": "en",
        },
    )
    assert job.status == JobStatus.PENDING.value
    assert job.started_at is None
    assert job.completed_at is None

    # Execute single job cycle
    handled = run_single_job_cycle(db_session)
    assert handled is True

    db_session.refresh(job)
    assert job.status == JobStatus.COMPLETED.value
    assert job.progress == 100.0
    assert job.started_at is not None
    assert job.completed_at is not None
    assert job.completed_at >= job.started_at
    assert job.error_message is None
    assert job.attempt_count == 1

    # Verify intelligence entities were persisted
    summary = db_session.query(MeetingSummary).filter(MeetingSummary.meeting_id == meeting.id).first()
    assert summary is not None
    assert summary.version == 1
    assert "worker intelligence test" in summary.executive_summary.lower()

    decisions = db_session.query(Decision).filter(Decision.meeting_id == meeting.id).all()
    assert len(decisions) >= 1
    assert decisions[0].timestamp_seconds == 0.0

    actions = db_session.query(ActionItem).filter(ActionItem.meeting_id == meeting.id).all()
    assert len(actions) >= 1

    risks = db_session.query(Risk).filter(Risk.meeting_id == meeting.id).all()
    assert len(risks) >= 1

    questions = db_session.query(UnresolvedQuestion).filter(UnresolvedQuestion.meeting_id == meeting.id).all()
    assert len(questions) >= 1


def test_extraction_job_invalid_payload_missing(db_session):
    """Verify a job with missing/None payload fails with MissingPayloadError."""
    org, meeting, _ = _setup_test_meeting(db_session)

    job = create_processing_job(
        db=db_session,
        meeting_id=meeting.id,
        job_type=JobType.EXTRACTION.value,
        payload=None,
    )

    handled = run_single_job_cycle(db_session)
    assert handled is True

    db_session.refresh(job)
    assert job.status == JobStatus.FAILED.value
    assert job.error_details == "MissingPayloadError"
    assert "missing or empty" in job.error_message
    assert job.completed_at is not None


def test_extraction_job_invalid_payload_malformed_json(db_session):
    """Verify a job with non-JSON payload string fails safely."""
    org, meeting, _ = _setup_test_meeting(db_session)

    job = ProcessingJob(
        meeting_id=meeting.id,
        job_type=JobType.EXTRACTION.value,
        status=JobStatus.PENDING.value,
        payload="{not valid json",
    )
    db_session.add(job)
    db_session.commit()

    handled = run_single_job_cycle(db_session)
    assert handled is True

    db_session.refresh(job)
    assert job.status == JobStatus.FAILED.value
    assert "Invalid job payload JSON" in job.error_message
    assert job.completed_at is not None


def test_extraction_job_invalid_payload_schema_validation(db_session):
    """Verify a job with invalid fields fails schema validation."""
    org, meeting, _ = _setup_test_meeting(db_session)

    # Missing required meeting_id
    job = create_processing_job(
        db=db_session,
        meeting_id=meeting.id,
        job_type=JobType.EXTRACTION.value,
        payload={"output_language": "en"},
    )

    handled = run_single_job_cycle(db_session)
    assert handled is True

    db_session.refresh(job)
    assert job.status == JobStatus.FAILED.value
    assert job.error_details == "PayloadValidationError"
    assert "meeting_id" in job.error_message


def test_extraction_job_invalid_provider_override_rejected(db_session):
    """Verify invalid provider override string is rejected by validator."""
    org, meeting, _ = _setup_test_meeting(db_session)

    job = create_processing_job(
        db=db_session,
        meeting_id=meeting.id,
        job_type=JobType.EXTRACTION.value,
        payload={"meeting_id": meeting.id, "provider_override": "unknown_ai_provider"},
    )

    handled = run_single_job_cycle(db_session)
    assert handled is True

    db_session.refresh(job)
    assert job.status == JobStatus.FAILED.value
    assert job.error_details == "PayloadValidationError"
    assert "Invalid provider_override" in job.error_message


def test_extraction_job_meeting_id_mismatch(db_session):
    """Verify job payload meeting_id differing from job row meeting_id is rejected."""
    org, meeting, _ = _setup_test_meeting(db_session)
    spoofed_meeting_id = str(uuid4())

    job = create_processing_job(
        db=db_session,
        meeting_id=meeting.id,
        job_type=JobType.EXTRACTION.value,
        payload={"meeting_id": spoofed_meeting_id},
    )

    handled = run_single_job_cycle(db_session)
    assert handled is True

    db_session.refresh(job)
    assert job.status == JobStatus.FAILED.value
    assert job.error_details == "MeetingMismatchError"
    assert "does not match" in job.error_message


def test_extraction_job_organization_mismatch(db_session):
    """Verify payload specifying a different organization than meeting is rejected."""
    org, meeting, _ = _setup_test_meeting(db_session)
    different_org_id = str(uuid4())

    job = create_processing_job(
        db=db_session,
        meeting_id=meeting.id,
        job_type=JobType.EXTRACTION.value,
        payload={
            "meeting_id": meeting.id,
            "organization_id": different_org_id,
        },
    )

    handled = run_single_job_cycle(db_session)
    assert handled is True

    db_session.refresh(job)
    assert job.status == JobStatus.FAILED.value
    assert job.error_details == "OrganizationMismatchError"
    assert "Organization mismatch" in job.error_message


def test_extraction_job_rejects_unredacted_transcript_only(db_session):
    """Verify extraction fails cleanly when meeting only has unredacted ORIGINAL transcript."""
    org = Organization(name="Privacy Org", slug=f"priv-org-{uuid4().hex[:8]}")
    meeting = Meeting(organization_id=org.id, title="Unredacted Meeting", status=MeetingStatus.COMPLETED.value)
    db_session.add_all([org, meeting])
    db_session.flush()

    # Add only ORIGINAL transcript (no REDACTED or CLEANED_REDACTED)
    t_orig = Transcript(
        meeting_id=meeting.id,
        transcript_type=TranscriptType.ORIGINAL.value,
        version=1,
    )
    db_session.add(t_orig)
    db_session.flush()

    seg = TranscriptSegment(
        transcript_id=t_orig.id,
        sequence_order=0,
        start_time=0.0,
        end_time=5.0,
        text="Aadhaar 1234 5678 9012 should not be processed.",
    )
    db_session.add(seg)
    db_session.commit()

    job = create_processing_job(
        db=db_session,
        meeting_id=meeting.id,
        job_type=JobType.EXTRACTION.value,
        payload={"meeting_id": meeting.id},
    )

    handled = run_single_job_cycle(db_session)
    assert handled is True

    db_session.refresh(job)
    assert job.status == JobStatus.FAILED.value
    assert job.error_details == "SafeTranscriptRequiredError"
    assert "privacy-safe" in job.error_message.lower()
    # Ensure raw text or PII from transcript is NOT in error message
    assert "1234 5678 9012" not in job.error_message


def test_extraction_job_explicit_unredacted_transcript_rejected(db_session):
    """Verify explicitly passing an unredacted transcript ID fails even if redacted exists."""
    org, meeting, t_red = _setup_test_meeting(db_session)

    # Add an ORIGINAL transcript as well
    t_orig = Transcript(
        meeting_id=meeting.id,
        transcript_type=TranscriptType.ORIGINAL.value,
        version=2,
    )
    db_session.add(t_orig)
    db_session.commit()

    job = create_processing_job(
        db=db_session,
        meeting_id=meeting.id,
        job_type=JobType.EXTRACTION.value,
        payload={
            "meeting_id": meeting.id,
            "source_transcript_id": t_orig.id,
        },
    )

    handled = run_single_job_cycle(db_session)
    assert handled is True

    db_session.refresh(job)
    assert job.status == JobStatus.FAILED.value
    assert job.error_details == "SafeTranscriptRequiredError"
    assert "ORIGINAL" in job.error_message


def test_failed_extraction_preserves_previously_persisted_intelligence(db_session):
    """Verify when re-extraction fails, the previous version of intelligence remains intact."""
    org, meeting, t_red = _setup_test_meeting(db_session)

    # 1. First run succeeds
    job1 = create_processing_job(
        db=db_session,
        meeting_id=meeting.id,
        job_type=JobType.EXTRACTION.value,
        payload={"meeting_id": meeting.id, "provider_override": "deterministic"},
    )
    run_single_job_cycle(db_session)
    db_session.refresh(job1)
    assert job1.status == JobStatus.COMPLETED.value

    # Check baseline intelligence records
    summary_v1 = db_session.query(MeetingSummary).filter(MeetingSummary.meeting_id == meeting.id).first()
    assert summary_v1 is not None
    assert summary_v1.version == 1
    decisions_v1 = db_session.query(Decision).filter(Decision.meeting_id == meeting.id).all()
    assert len(decisions_v1) >= 1
    d_id = decisions_v1[0].id

    # 2. Second run fails during extraction (e.g. simulated engine error)
    job2 = create_processing_job(
        db=db_session,
        meeting_id=meeting.id,
        job_type=JobType.EXTRACTION.value,
        payload={"meeting_id": meeting.id, "provider_override": "deterministic"},
    )

    with patch(
        "app.services.intelligence.job_handler.extract_meeting_intelligence",
        side_effect=RuntimeError("Simulated LLM engine crash"),
    ):
        run_single_job_cycle(db_session)

    db_session.refresh(job2)
    assert job2.status == JobStatus.FAILED.value
    assert "RuntimeError" in job2.error_details

    # Verify previous intelligence records are completely intact!
    summary_after = db_session.query(MeetingSummary).filter(MeetingSummary.meeting_id == meeting.id).first()
    assert summary_after is not None
    assert summary_after.version == 1  # Version did NOT increment
    decisions_after = db_session.query(Decision).filter(Decision.meeting_id == meeting.id).all()
    assert len(decisions_after) == len(decisions_v1)
    assert decisions_after[0].id == d_id


def test_reprocessing_without_duplicate_records(db_session):
    """Verify successive extractions cleanly replace items and increment summary version."""
    org, meeting, t_red = _setup_test_meeting(db_session)

    # Job 1
    job1 = create_processing_job(
        db=db_session,
        meeting_id=meeting.id,
        job_type=JobType.EXTRACTION.value,
        payload={"meeting_id": meeting.id, "provider_override": "deterministic"},
    )
    run_single_job_cycle(db_session)
    db_session.refresh(job1)
    assert job1.status == JobStatus.COMPLETED.value

    assert db_session.query(MeetingSummary).filter(MeetingSummary.meeting_id == meeting.id).count() == 1
    assert db_session.query(Decision).filter(Decision.meeting_id == meeting.id).count() == 1
    assert db_session.query(ActionItem).filter(ActionItem.meeting_id == meeting.id).count() == 1

    # Job 2 (reprocessing)
    job2 = create_processing_job(
        db=db_session,
        meeting_id=meeting.id,
        job_type=JobType.EXTRACTION.value,
        payload={"meeting_id": meeting.id, "provider_override": "deterministic"},
    )
    run_single_job_cycle(db_session)
    db_session.refresh(job2)
    assert job2.status == JobStatus.COMPLETED.value

    # Counts must NOT duplicate!
    assert db_session.query(MeetingSummary).filter(MeetingSummary.meeting_id == meeting.id).count() == 1
    assert db_session.query(Decision).filter(Decision.meeting_id == meeting.id).count() == 1
    assert db_session.query(ActionItem).filter(ActionItem.meeting_id == meeting.id).count() == 1

    # Summary version must have incremented to 2
    summary = db_session.query(MeetingSummary).filter(MeetingSummary.meeting_id == meeting.id).first()
    assert summary.version == 2


def test_extraction_job_startup_recovery_interrupted(db_session):
    """Verify abandoned IN_PROGRESS extraction jobs are safely reconciled to FAILED at startup."""
    org, meeting, _ = _setup_test_meeting(db_session)

    job = ProcessingJob(
        meeting_id=meeting.id,
        job_type=JobType.EXTRACTION.value,
        status=JobStatus.IN_PROGRESS.value,
        attempt_count=1,
        started_at=datetime.now(timezone.utc),
        payload=json.dumps({"meeting_id": meeting.id}),
    )
    db_session.add(job)
    db_session.commit()

    reconciled_count = reconcile_stranded_jobs_on_startup(db_session)
    assert reconciled_count == 1

    db_session.refresh(job)
    assert job.status == JobStatus.FAILED.value
    assert job.error_details == "ServerRestartInterruption"
    assert "interrupted by a server restart" in job.error_message
    assert job.completed_at is not None


def test_completed_extraction_job_never_executed_again(db_session):
    """Verify an already COMPLETED job is immediately skipped by the worker."""
    org, meeting, _ = _setup_test_meeting(db_session)

    job = ProcessingJob(
        meeting_id=meeting.id,
        job_type=JobType.EXTRACTION.value,
        status=JobStatus.COMPLETED.value,
        attempt_count=1,
        started_at=datetime.now(timezone.utc),
        completed_at=datetime.now(timezone.utc),
        payload=json.dumps({"meeting_id": meeting.id}),
    )
    db_session.add(job)
    db_session.commit()

    # Calling process_extraction_job_sync directly
    with patch("app.services.intelligence.job_handler.extract_meeting_intelligence") as mock_extract:
        process_extraction_job_sync(job.id)
        mock_extract.assert_not_called()

    db_session.refresh(job)
    assert job.status == JobStatus.COMPLETED.value


def test_worker_handles_ollama_fallback_transparently(db_session):
    """Verify worker transparently degrades to deterministic extractor when Ollama fails."""
    org, meeting, t_red = _setup_test_meeting(db_session)

    job = create_processing_job(
        db=db_session,
        meeting_id=meeting.id,
        job_type=JobType.EXTRACTION.value,
        payload={
            "meeting_id": meeting.id,
            "provider_override": "ollama",
            "output_language": "en",
        },
    )

    # Mock failing Ollama client
    def mock_fail_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Ollama daemon unreachable")

    failing_transport = httpx.MockTransport(mock_fail_handler)
    failing_client = httpx.Client(transport=failing_transport)
    failing_ollama = OllamaExtractor(client=failing_client)

    # Process job with failing Ollama extractor
    with patch("app.services.intelligence.engine.OllamaExtractor", return_value=failing_ollama):
        handled = run_single_job_cycle(db_session)
        assert handled is True

    db_session.refresh(job)
    # The job SUCCEEDS because fallback was permitted and executed
    assert job.status == JobStatus.COMPLETED.value
    assert job.progress == 100.0

    # Entities were successfully persisted
    summary = db_session.query(MeetingSummary).filter(MeetingSummary.meeting_id == meeting.id).first()
    assert summary is not None
    assert summary.version == 1


def test_existing_transcription_job_unaffected(db_session):
    """Verify TRANSCRIPTION job routing is preserved and unaffected by EXTRACTION job support."""
    org, meeting, _ = _setup_test_meeting(db_session)

    job = create_processing_job(
        db=db_session,
        meeting_id=meeting.id,
        job_type=JobType.TRANSCRIPTION.value,
        payload={"language_override": "en", "transcription_mode": "standard"},
    )

    with patch("app.services.transcription.job_handler.process_transcription_job_sync") as mock_transcribe:
        handled = run_single_job_cycle(db_session)
        assert handled is True
        mock_transcribe.assert_called_once_with(
            job_id=job.id,
            language_override="en",
            transcription_mode="standard",
        )
