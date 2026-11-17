"""Database-backed Job Worker and Recovery Engine.

Provides:
- Atomic compare-and-swap job claiming via conditional SQL updates, preventing competing claims.
- Single-worker asyncio polling loop that offloads synchronous execution (FFmpeg/CTranslate2/ASR)
  to a worker thread (asyncio.to_thread).
- Conservative startup recovery:
  - Leaves PENDING jobs with valid persisted payloads available for the worker.
  - Marks abandoned IN_PROGRESS jobs as FAILED with a clear interruption reason so they do not
    permanently block new user-initiated requests or eligible media retention cleanups.
  - Preserves COMPLETED, CANCELLED, and existing FAILED jobs.
- Clean execution dispatch for INGESTION (file upload & YouTube) and TRANSCRIPTION.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any
from sqlalchemy import update, text
from sqlalchemy.orm import Session

from app.core.config import settings
from app.database.session import SessionLocal
from app.models.meeting import Meeting, ProcessingJob, JobStatus, JobType, MeetingStatus
from app.services.meeting_lifecycle import reconcile_meeting_status

logger = logging.getLogger(__name__)


def claim_job_atomically(db: Session, job_id: str) -> Optional[ProcessingJob]:
    """Atomically transition a job from PENDING to IN_PROGRESS.

    Uses an atomic conditional SQL UPDATE statement.
    Returns the updated ProcessingJob instance if successfully claimed,
    or None if the job was already claimed, cancelled, or not found.
    """
    now = datetime.now(timezone.utc)
    stmt = (
        update(ProcessingJob)
        .where(
            ProcessingJob.id == job_id,
            ProcessingJob.status == JobStatus.PENDING.value
        )
        .values(
            status=JobStatus.IN_PROGRESS.value,
            attempt_count=ProcessingJob.attempt_count + 1,
            started_at=now,
            updated_at=now,
        )
    )
    res = db.execute(stmt)
    db.commit()
    if res.rowcount == 0:
        return None
    return db.get(ProcessingJob, job_id)


def get_next_pending_job_id(db: Session) -> Optional[str]:
    """Find the oldest PENDING job in FIFO order."""
    job_row = (
        db.query(ProcessingJob.id)
        .filter(ProcessingJob.status == JobStatus.PENDING.value)
        .order_by(ProcessingJob.created_at.asc())
        .first()
    )
    return job_row[0] if job_row else None


def reconcile_stranded_jobs_on_startup(db: Session) -> int:
    """Identify and reconcile abandoned IN_PROGRESS jobs upon application startup.

    CONSERVATIVE POLICY:
    - Leaves PENDING jobs with valid persisted payloads available for the worker.
    - Marks abandoned IN_PROGRESS jobs as FAILED with a clear interruption reason.
    - Does NOT automatically retry interrupted transcription or ingestion jobs.
    - Does NOT modify COMPLETED, CANCELLED or already FAILED jobs.
    - Ensures abandoned jobs no longer permanently block new user requests or media retention checks.
    """
    stranded_jobs = (
        db.query(ProcessingJob)
        .filter(ProcessingJob.status == JobStatus.IN_PROGRESS.value)
        .all()
    )
    count = 0
    now = datetime.now(timezone.utc)
    for job in stranded_jobs:
        job.status = JobStatus.FAILED.value
        job.error_message = "Job was interrupted by a server restart or crash."
        job.error_details = "ServerRestartInterruption"
        job.completed_at = now
        count += 1

        # Use centralized reconciler: correctly handles downstream-only failures
        # (EXTRACTION/EMBEDDING failures do not mark meeting FAILED if transcription succeeded)
        try:
            reconcile_meeting_status(db, job.meeting_id, commit=False)
        except Exception as re_exc:
            logger.warning(f"reconcile_meeting_status failed during startup reconcile: {re_exc}")

    if count > 0:
        db.commit()
        logger.warning(f"Reconciled {count} abandoned IN_PROGRESS job(s) as FAILED on startup.")
    return count


def execute_claimed_job(job_id: str) -> None:
    """Execute a claimed job in a worker thread without blocking the asyncio event loop."""
    db: Session = SessionLocal()
    job_type: str = ""
    meeting_id: str = ""
    payload: Dict[str, Any] = {}
    try:
        job = db.get(ProcessingJob, job_id)
        if not job or job.status != JobStatus.IN_PROGRESS.value:
            return

        job_type = job.job_type
        meeting_id = job.meeting_id

        if job.payload:
            try:
                payload = json.loads(job.payload)
            except Exception as pe:
                job.status = JobStatus.FAILED.value
                job.error_message = f"Invalid job payload JSON: {pe}"
                job.completed_at = datetime.now(timezone.utc)
                db.commit()
                return
    finally:
        db.close()

    try:
        # Route by job type
        if job_type == JobType.TRANSCRIPTION.value:
            from app.services.transcription.job_handler import process_transcription_job_sync
            process_transcription_job_sync(
                job_id=job_id,
                language_override=payload.get("language_override"),
                transcription_mode=payload.get("transcription_mode", "standard"),
            )

        elif job_type == JobType.INGESTION.value:
            subtype = payload.get("subtype")
            if subtype == "UPLOAD":
                from app.api.endpoints.media import _process_uploaded_media_async
                _process_uploaded_media_async(
                    meeting_id=meeting_id,
                    job_id=job_id,
                    raw_asset_id=payload.get("raw_asset_id"),
                    raw_file_path_str=payload.get("raw_file_path"),
                    normalized_filename=payload.get("normalized_filename"),
                )
            elif subtype == "YOUTUBE":
                from app.api.endpoints.media import _process_youtube_media_async
                _process_youtube_media_async(
                    meeting_id=meeting_id,
                    job_id=job_id,
                    youtube_url=payload.get("youtube_url"),
                    meeting_dir_str=payload.get("meeting_dir"),
                    raw_asset_id=payload.get("raw_asset_id"),
                )
            else:
                db_err = SessionLocal()
                try:
                    job = db_err.get(ProcessingJob, job_id)
                    if job:
                        job.status = JobStatus.FAILED.value
                        job.error_message = f"Unknown or missing ingestion payload subtype: {subtype}"
                        job.completed_at = datetime.now(timezone.utc)
                        db_err.commit()
                finally:
                    db_err.close()

        elif job_type == JobType.EXTRACTION.value:
            from app.services.intelligence.job_handler import process_extraction_job_sync
            process_extraction_job_sync(job_id=job_id)

        elif job_type == JobType.EMBEDDING.value:
            from app.services.memory.job_handler import process_indexing_job_sync
            process_indexing_job_sync(job_id=job_id)

        else:
            db_err = SessionLocal()
            try:
                job = db_err.get(ProcessingJob, job_id)
                if job:
                    job.status = JobStatus.FAILED.value
                    job.error_message = f"Unsupported job type: {job_type}"
                    job.completed_at = datetime.now(timezone.utc)
                    db_err.commit()
            finally:
                db_err.close()

        # Update completed_at if job reached terminal status without setting it
        db_term = SessionLocal()
        try:
            job = db_term.get(ProcessingJob, job_id)
            if job and job.status in (JobStatus.COMPLETED.value, JobStatus.FAILED.value) and not job.completed_at:
                job.completed_at = datetime.now(timezone.utc)
                db_term.commit()
        finally:
            db_term.close()

    except Exception as e:
        logger.exception(f"Unhandled error while executing job {job_id}: {e}")
        try:
            db_exc = SessionLocal()
            try:
                job = db_exc.get(ProcessingJob, job_id)
                if job:
                    job.status = JobStatus.FAILED.value
                    job.error_message = f"Worker execution error: {str(e)}"
                    job.completed_at = datetime.now(timezone.utc)
                    db_exc.commit()
            finally:
                db_exc.close()
        except Exception:
            pass


def run_single_job_cycle(db: Session) -> bool:
    """Synchronously discover, atomically claim, and execute a single pending job.

    Used by deterministic unit and integration tests without sleep loops.
    Returns True if a job was found, claimed, and executed; False otherwise.
    """
    job_id = get_next_pending_job_id(db)
    if not job_id:
        return False

    claimed = claim_job_atomically(db, job_id)
    if not claimed:
        return False

    execute_claimed_job(job_id)
    return True


class JobWorker:
    """Single-process asynchronous polling worker for database-backed ProcessingJobs."""

    def __init__(self, poll_interval: Optional[float] = None):
        self.poll_interval = poll_interval or getattr(settings, "JOB_POLL_INTERVAL_SECONDS", 2.0)
        self._stop_event = asyncio.Event()
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start the background polling loop."""
        self._stop_event.clear()
        self._task = asyncio.create_task(self._run_loop())
        logger.info(f"Background JobWorker started (poll interval: {self.poll_interval}s).")

    async def stop(self) -> None:
        """Signal worker to stop polling and await current task completion."""
        self._stop_event.set()
        if self._task:
            try:
                await asyncio.wait_for(asyncio.shield(self._task), timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass
        logger.info("Background JobWorker stopped.")

    async def poll_once(self) -> bool:
        """Perform a single poll cycle: claim and offload execution to a worker thread."""
        db: Session = SessionLocal()
        job_id = None
        try:
            job_id = get_next_pending_job_id(db)
            if not job_id:
                return False

            claimed = claim_job_atomically(db, job_id)
            if not claimed:
                return False
        finally:
            db.close()

        # Offload synchronous handler to worker thread to avoid blocking asyncio event loop
        await asyncio.to_thread(execute_claimed_job, job_id)
        return True

    async def _run_loop(self) -> None:
        """Main polling loop."""
        while not self._stop_event.is_set():
            try:
                processed = await self.poll_once()
                if not processed:
                    try:
                        await asyncio.wait_for(self._stop_event.wait(), timeout=self.poll_interval)
                    except asyncio.TimeoutError:
                        pass
            except Exception as e:
                logger.error(f"Error in JobWorker loop: {e}", exc_info=True)
                try:
                    await asyncio.wait_for(self._stop_event.wait(), timeout=self.poll_interval)
                except asyncio.TimeoutError:
                    pass
