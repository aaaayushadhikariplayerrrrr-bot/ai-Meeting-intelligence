"""Background Job Handler for Structured Meeting Intelligence Extraction."""

import json
import logging
from datetime import datetime, timezone
from typing import Optional, Any
from sqlalchemy.orm import Session

from app.database.session import SessionLocal
from app.models.meeting import Meeting, ProcessingJob, JobStatus
from app.services.meeting_lifecycle import reconcile_meeting_status
from app.schemas.intelligence import ExtractionJobPayload
from app.services.intelligence.base import (
    IntelligenceError,
    SafeTranscriptRequiredError,
    ProviderExecutionError,
    IntelligenceDisabledError,
)
from app.services.intelligence.engine import extract_meeting_intelligence

logger = logging.getLogger(__name__)


def process_extraction_job_sync(
    job_id: str,
    ollama_extractor: Optional[Any] = None,
    deterministic_extractor: Optional[Any] = None,
) -> None:
    """Synchronous background handler executing an intelligence extraction job."""
    db: Session = SessionLocal()
    try:
        job = db.get(ProcessingJob, job_id)
        if not job:
            logger.error(f"ProcessingJob '{job_id}' not found.")
            return

        if job.status == JobStatus.COMPLETED.value:
            logger.info(f"ProcessingJob '{job_id}' is already completed; skipping execution.")
            return

        # 1. Transition to IN_PROGRESS
        job.status = JobStatus.IN_PROGRESS.value
        job.progress = 10.0
        now = datetime.now(timezone.utc)
        if not job.started_at:
            job.started_at = now
        db.commit()

        # 2. Validate payload existence and JSON structure
        if not job.payload:
            job.status = JobStatus.FAILED.value
            job.error_message = "Malformed extraction job: payload is missing or empty."
            job.error_details = "MissingPayloadError"
            job.completed_at = datetime.now(timezone.utc)
            db.commit()
            return

        try:
            raw_payload = json.loads(job.payload)
            if not isinstance(raw_payload, dict):
                raise ValueError("Payload must be a JSON object.")
        except Exception as pe:
            job.status = JobStatus.FAILED.value
            job.error_message = f"Malformed extraction job payload JSON: {str(pe)}"
            job.error_details = "InvalidPayloadJSONError"
            job.completed_at = datetime.now(timezone.utc)
            db.commit()
            return

        # 3. Validate payload schema
        try:
            payload_dto = ExtractionJobPayload.model_validate(raw_payload)
        except Exception as ve:
            job.status = JobStatus.FAILED.value
            job.error_message = f"Invalid extraction job payload: {str(ve)}"
            job.error_details = "PayloadValidationError"
            job.completed_at = datetime.now(timezone.utc)
            db.commit()
            return

        # 4. Validate meeting and tenant consistency
        if payload_dto.meeting_id != job.meeting_id:
            job.status = JobStatus.FAILED.value
            job.error_message = (
                f"Payload meeting_id '{payload_dto.meeting_id}' does not match job meeting_id '{job.meeting_id}'."
            )
            job.error_details = "MeetingMismatchError"
            job.completed_at = datetime.now(timezone.utc)
            db.commit()
            return

        meeting = db.get(Meeting, job.meeting_id)
        if not meeting:
            job.status = JobStatus.FAILED.value
            job.error_message = f"Meeting '{job.meeting_id}' not found."
            job.error_details = "MeetingNotFoundError"
            job.completed_at = datetime.now(timezone.utc)
            db.commit()
            return

        # Reject payload if it claims an organization different from the meeting
        if payload_dto.organization_id and payload_dto.organization_id != meeting.organization_id:
            job.status = JobStatus.FAILED.value
            job.error_message = (
                f"Organization mismatch: payload specifies '{payload_dto.organization_id}' "
                f"but meeting belongs to '{meeting.organization_id}'."
            )
            job.error_details = "OrganizationMismatchError"
            job.completed_at = datetime.now(timezone.utc)
            db.commit()
            return

        # 5. Progress update before model extraction
        job.progress = 30.0
        db.commit()

        # 6. Dispatch extraction to engine
        try:
            result = extract_meeting_intelligence(
                meeting_id=meeting.id,
                organization_id=meeting.organization_id,
                db=db,
                source_transcript_id=payload_dto.source_transcript_id,
                provider_override=payload_dto.provider_override,
                output_language=payload_dto.output_language,
                allow_fallback=True,
                persist=True,
                ollama_extractor=ollama_extractor,
                deterministic_extractor=deterministic_extractor,
            )
        except SafeTranscriptRequiredError as stre:
            db.rollback()
            job.status = JobStatus.FAILED.value
            job.error_message = str(stre)
            job.error_details = "SafeTranscriptRequiredError"
            job.completed_at = datetime.now(timezone.utc)
            db.commit()
            return
        except (IntelligenceDisabledError, ProviderExecutionError, IntelligenceError) as ie:
            db.rollback()
            job.status = JobStatus.FAILED.value
            job.error_message = str(ie)
            job.error_details = type(ie).__name__
            job.completed_at = datetime.now(timezone.utc)
            db.commit()
            try:
                reconcile_meeting_status(db, job.meeting_id)
            except Exception:
                pass
            return
        except Exception as e:
            logger.exception(f"Unhandled extraction error in job '{job_id}': {e}")
            db.rollback()
            job.status = JobStatus.FAILED.value
            job.error_message = f"Extraction failed: {type(e).__name__}"
            job.error_details = type(e).__name__
            job.completed_at = datetime.now(timezone.utc)
            db.commit()
            try:
                reconcile_meeting_status(db, job.meeting_id)
            except Exception:
                pass
            return

        # 7. Complete job successfully
        job.status = JobStatus.COMPLETED.value
        job.progress = 100.0
        job.error_message = None
        job.error_details = None
        job.completed_at = datetime.now(timezone.utc)
        db.commit()
        logger.info(
            f"Extraction job '{job_id}' completed successfully for meeting '{meeting.id}' "
            f"using provider '{result.provider_used}' (fallback={result.fallback_warning is not None})."
        )
        # Downstream extraction completion must not change COMPLETED meetings; reconcile handles it
        try:
            reconcile_meeting_status(db, meeting.id)
        except Exception as re_exc:
            logger.warning(f"reconcile_meeting_status failed after extraction success: {re_exc}")

    except Exception as e:
        logger.exception(f"Fatal worker error in extraction job '{job_id}': {e}")
        try:
            db.rollback()
            job = db.get(ProcessingJob, job_id)
            if job:
                job.status = JobStatus.FAILED.value
                job.error_message = f"Worker execution error: {type(e).__name__}"
                job.error_details = type(e).__name__
                job.completed_at = datetime.now(timezone.utc)
                db.commit()
                try:
                    reconcile_meeting_status(db, job.meeting_id)
                except Exception:
                    pass
        except Exception:
            pass
    finally:
        db.close()
