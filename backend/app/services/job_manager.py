import json
from datetime import datetime
from typing import Optional, Dict, Any
from sqlalchemy.orm import Session

from app.models.meeting import ProcessingJob, JobStatus, JobType
from app.services.meeting_lifecycle import reconcile_meeting_status


def create_processing_job(
    db: Session,
    meeting_id: str,
    job_type: str = JobType.INGESTION.value,
    initial_status: str = JobStatus.PENDING.value,
    payload: Optional[Dict[str, Any]] = None,
    max_attempts: int = 2,
) -> ProcessingJob:
    """Create and commit a new ProcessingJob entry with optional execution payload."""
    payload_str = json.dumps(payload) if payload is not None else None
    job = ProcessingJob(
        meeting_id=meeting_id,
        job_type=job_type,
        status=initial_status,
        progress=0.0,
        payload=payload_str,
        max_attempts=max_attempts,
        attempt_count=0,
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    # Reconcile overall meeting status to reflect the queued/in-progress job
    try:
        reconcile_meeting_status(db, meeting_id)
    except Exception:
        pass

    return job


def update_processing_job(
    db: Session,
    job_id: str,
    status: Optional[str] = None,
    progress: Optional[float] = None,
    error_message: Optional[str] = None,
    error_details: Optional[str] = None,
    started_at: Optional[datetime] = None,
    completed_at: Optional[datetime] = None,
    attempt_count: Optional[int] = None,
) -> Optional[ProcessingJob]:
    """Update job status, progress percentage, error reporting, and lifecycle timestamps."""
    job = db.get(ProcessingJob, job_id)
    if not job:
        return None

    if status is not None:
        job.status = status
    if progress is not None:
        job.progress = max(0.0, min(100.0, float(progress)))
    if error_message is not None:
        job.error_message = error_message
    if error_details is not None:
        job.error_details = error_details
    if started_at is not None:
        job.started_at = started_at
    if completed_at is not None:
        job.completed_at = completed_at
    if attempt_count is not None:
        job.attempt_count = attempt_count

    db.commit()
    db.refresh(job)

    # Reconcile overall meeting status whenever a job reaches a terminal state
    if status in (JobStatus.COMPLETED.value, JobStatus.FAILED.value):
        try:
            reconcile_meeting_status(db, job.meeting_id)
        except Exception:
            pass

    return job
