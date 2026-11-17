"""Meeting Lifecycle and Status Reconciliation Service.

Provides centralized, authoritative determination of overall MeetingStatus
across all ingestion, transcription, intelligence extraction, and memory indexing jobs.
"""

import logging
from typing import Optional, List
from sqlalchemy.orm import Session

from app.models.meeting import Meeting, ProcessingJob, MeetingStatus, JobStatus, JobType
from app.models.transcription import Transcript

logger = logging.getLogger(__name__)


def reconcile_meeting_status(db: Session, meeting_id: str, commit: bool = True) -> Optional[str]:
    """Reconcile and update the overall MeetingStatus based on all associated processing jobs.

    Rules:
    1. If meeting does not exist, return None.
    2. If meeting is in SCHEDULED or RECORDING and has NO processing jobs, leave unchanged.
    3. Active Jobs check:
       - If any INGESTION job is PENDING or IN_PROGRESS (and no transcript exists):
         meeting.status = INGESTING.
       - If any TRANSCRIPTION, EXTRACTION, EMBEDDING, or DIARIZATION job is PENDING or IN_PROGRESS:
         meeting.status = PROCESSING.
    4. Terminal State (no active jobs remaining):
       - If the latest INGESTION job FAILED and no completed INGESTION job exists:
         meeting.status = FAILED.
       - If the latest TRANSCRIPTION job FAILED and no valid Transcript exists:
         meeting.status = FAILED.
       - If all attempted jobs are COMPLETED (or CANCELLED) and at least one core job completed:
         meeting.status = COMPLETED.
       - If downstream intelligence (EXTRACTION) or memory indexing (EMBEDDING) failed,
         but transcription succeeded and transcripts exist:
         meeting.status = COMPLETED (downstream non-fatal failure does not invalidate meeting).
    """
    meeting = db.get(Meeting, meeting_id)
    if not meeting:
        logger.warning(f"reconcile_meeting_status: Meeting '{meeting_id}' not found.")
        return None

    jobs: List[ProcessingJob] = (
        db.query(ProcessingJob)
        .filter(ProcessingJob.meeting_id == meeting_id)
        .order_by(ProcessingJob.created_at.asc())
        .all()
    )

    if not jobs:
        # Preserve SCHEDULED or RECORDING if no jobs exist
        if meeting.status in (MeetingStatus.SCHEDULED.value, MeetingStatus.RECORDING.value):
            return meeting.status
        return meeting.status

    active_jobs = [
        j for j in jobs
        if j.status in (JobStatus.PENDING.value, JobStatus.IN_PROGRESS.value)
    ]

    new_status = meeting.status

    if active_jobs:
        active_types = {j.job_type for j in active_jobs}
        if active_types == {JobType.INGESTION.value}:
            transcripts_count = (
                db.query(Transcript)
                .filter(Transcript.meeting_id == meeting_id)
                .count()
            )
            if transcripts_count == 0:
                new_status = MeetingStatus.INGESTING.value
            else:
                new_status = MeetingStatus.PROCESSING.value
        else:
            new_status = MeetingStatus.PROCESSING.value
    else:
        # All jobs have reached terminal states (COMPLETED, FAILED, CANCELLED)
        has_transcripts = (
            db.query(Transcript)
            .filter(Transcript.meeting_id == meeting_id)
            .count()
        ) > 0

        ingestion_jobs = [j for j in jobs if j.job_type == JobType.INGESTION.value]
        transcription_jobs = [j for j in jobs if j.job_type == JobType.TRANSCRIPTION.value]

        latest_ingestion = ingestion_jobs[-1] if ingestion_jobs else None
        latest_transcription = transcription_jobs[-1] if transcription_jobs else None

        has_successful_ingestion = any(j.status == JobStatus.COMPLETED.value for j in ingestion_jobs)
        has_successful_transcription = any(j.status == JobStatus.COMPLETED.value for j in transcription_jobs)

        if latest_ingestion and latest_ingestion.status == JobStatus.FAILED.value and not has_successful_ingestion:
            new_status = MeetingStatus.FAILED.value
        elif (
            latest_transcription
            and latest_transcription.status == JobStatus.FAILED.value
            and not has_successful_transcription
            and not has_transcripts
        ):
            new_status = MeetingStatus.FAILED.value
        else:
            # Core processing succeeded (or no core jobs failed)
            new_status = MeetingStatus.COMPLETED.value

    if meeting.status != new_status:
        logger.info(
            f"Meeting '{meeting_id}' status transitioned: {meeting.status} -> {new_status}"
        )
        meeting.status = new_status
        if commit:
            db.commit()
        else:
            db.flush()

    return new_status
