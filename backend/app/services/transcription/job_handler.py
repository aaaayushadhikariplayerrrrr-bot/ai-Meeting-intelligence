"""Background Job Processor for Multilingual Speech Recognition.

Orchestrates:
- Audio asset resolution (normalized 16 kHz mono WAV)
- Version-controlled Transcript & TranscriptSegment persistence
- Progress tracking and error capturing in ProcessingJob
- Safe raw audio lifecycle hook triggering only after verified persistence
"""

import os
import uuid
import logging
from datetime import datetime, timezone
from typing import Optional
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.meeting import Meeting, MediaAsset, ProcessingJob, JobStatus, MeetingStatus, JobType
from app.services.meeting_lifecycle import reconcile_meeting_status
from app.models.transcription import Transcript, TranscriptSegment, TranscriptType
from app.services.transcription.manager import get_asr_engine
from app.services.transcription.base import TranscriptionOptions
from app.services.audio_extraction import normalize_audio_for_asr
from app.services.lifecycle import delete_raw_media_asset
from app.services.media_storage import resolve_safe_storage_path
from app.database.session import SessionLocal

logger = logging.getLogger(__name__)


def process_transcription_job_sync(
    job_id: str,
    language_override: Optional[str] = None,
    transcription_mode: Optional[str] = "standard",
    options: Optional[TranscriptionOptions] = None,
) -> None:
    """Synchronous worker function designed to run in background tasks."""
    db: Session = SessionLocal()
    try:
        job = db.query(ProcessingJob).filter(ProcessingJob.id == job_id).first()
        if not job:
            logger.error(f"ProcessingJob {job_id} not found.")
            return

        if job.status == JobStatus.COMPLETED.value:
            logger.info(f"ProcessingJob {job_id} is already completed; skipping execution.")
            return

        job.status = JobStatus.IN_PROGRESS.value
        job.progress = 5.0
        if not job.started_at:
            job.started_at = datetime.now(timezone.utc)
        db.commit()

        meeting = db.query(Meeting).filter(Meeting.id == job.meeting_id).first()
        if not meeting:
            job.status = JobStatus.FAILED.value
            job.error_message = f"Meeting {job.meeting_id} not found."
            db.commit()
            return

        # 1. Resolve normalized audio asset
        audio_asset = None
        # First preference: normalized WAV asset (media_type == "AUDIO" and mime_type == "audio/wav")
        for asset in meeting.media_assets:
            if not asset.is_raw_deleted and asset.mime_type == "audio/wav":
                audio_asset = asset
                break

        # Fallback preference: any non-deleted media asset
        if not audio_asset:
            for asset in meeting.media_assets:
                if not asset.is_raw_deleted:
                    audio_asset = asset
                    break

        if not audio_asset:
            job.status = JobStatus.FAILED.value
            job.error_message = "No valid media asset found for transcription."
            db.commit()
            return

        target_audio_path = os.path.join(settings.STORAGE_DIR, audio_asset.storage_path)
        if not os.path.isfile(target_audio_path):
            try:
                resolved = resolve_safe_storage_path(audio_asset.storage_path)
                if resolved.is_file():
                    target_audio_path = str(resolved)
            except Exception:
                pass

        if not os.path.isfile(target_audio_path):
            job.status = JobStatus.FAILED.value
            job.error_message = f"Media file does not exist on disk."
            db.commit()
            return

        if audio_asset.duration_seconds is not None and audio_asset.duration_seconds > settings.MAX_MEDIA_DURATION_SECONDS:
            job.status = JobStatus.FAILED.value
            max_hrs = settings.MAX_MEDIA_DURATION_SECONDS / 3600.0
            job.error_message = (
                f"Media asset duration ({audio_asset.duration_seconds:.1f}s) exceeds maximum allowed limit of "
                f"{settings.MAX_MEDIA_DURATION_SECONDS:.1f}s ({max_hrs:.1f} hours)."
            )
            db.commit()
            return

        # Ensure normalized 16 kHz mono WAV
        ephemeral_normalized_path: Optional[str] = None
        if not target_audio_path.lower().endswith(".wav"):
            job.progress = 10.0
            db.commit()
            logger.info(f"Normalizing audio for meeting {meeting.id} to 16 kHz mono WAV")
            normalized_path = os.path.join(
                settings.STORAGE_DIR,
                meeting.id,
                f"{uuid.uuid4().hex}_normalized.wav"
            )
            normalize_audio_for_asr(target_audio_path, normalized_path)
            target_audio_path = normalized_path
            ephemeral_normalized_path = normalized_path

        # 2. Determine next transcript version to avoid overwriting existing original transcripts
        existing_orig = (
            db.query(Transcript)
            .filter(
                Transcript.meeting_id == meeting.id,
                Transcript.transcript_type == TranscriptType.ORIGINAL.value
            )
            .order_by(Transcript.version.desc())
            .first()
        )
        if existing_orig and existing_orig.created_at and job.created_at and existing_orig.created_at >= job.created_at:
            if len(existing_orig.segments) > 0:
                logger.info(
                    f"Transcript version {existing_orig.version} already persisted for meeting {meeting.id}; "
                    f"skipping duplicate transcription."
                )
                job.status = JobStatus.COMPLETED.value
                job.progress = 100.0
                job.completed_at = datetime.now(timezone.utc)
                db.commit()
                try:
                    reconcile_meeting_status(db, meeting.id)
                except Exception as re_exc:
                    logger.warning(f"reconcile_meeting_status failed on duplicate-transcript exit: {re_exc}")
                return

        new_version = (existing_orig.version + 1) if existing_orig else 1
        is_first_transcript = (existing_orig is None)

        # 3. Setup progress callback
        def on_progress(percent: float) -> None:
            try:
                # Map 0-100% to 15-95%
                job.progress = round(15.0 + (percent * 0.8), 1)
                db.commit()
            except Exception as pe:
                logger.warning(f"Failed to update progress: {pe}")

        # 4. Execute ASR transcription
        job.progress = 15.0
        db.commit()

        asr_options = options or TranscriptionOptions(
            language=language_override,
            transcription_mode=transcription_mode or "standard",
            chunk_length_seconds=settings.ASR_CHUNK_LENGTH_SECONDS,
            chunk_overlap_seconds=settings.ASR_CHUNK_OVERLAP_SECONDS,
            vad_filter=settings.ASR_VAD_FILTER,
            beam_size=settings.ASR_BEAM_SIZE,
            temperature=settings.ASR_TEMPERATURE,
        )

        engine = get_asr_engine()
        result = engine.transcribe(
            audio_path=target_audio_path,
            options=asr_options,
            progress_callback=on_progress,
        )

        # 5. Persist Transcript container
        transcript = Transcript(
            id=str(uuid.uuid4()),
            meeting_id=meeting.id,
            transcript_type=TranscriptType.ORIGINAL.value,
            version=new_version,
            language_detected=result.detected_language or "en",
            is_primary=is_first_transcript,
        )
        db.add(transcript)
        db.flush()

        # 6. Persist TranscriptSegments
        for seg in result.segments:
            db_segment = TranscriptSegment(
                id=str(uuid.uuid4()),
                transcript_id=transcript.id,
                speaker_id=None,  # Not invented; genuinely diarized when available
                sequence_order=seg.sequence_order,
                start_time=seg.start_time,
                end_time=seg.end_time,
                text=seg.text,
                language_code=seg.language_code or (result.detected_language or "en"),
                confidence=seg.confidence,
                is_redacted=False,
            )
            db.add(db_segment)

        # 7. Verification: check segments count
        db.flush()
        persisted_count = (
            db.query(TranscriptSegment)
            .filter(TranscriptSegment.transcript_id == transcript.id)
            .count()
        )
        if persisted_count != len(result.segments):
            raise RuntimeError(
                f"Transcript persistence mismatch: expected {len(result.segments)} segments, got {persisted_count}"
            )

        # 8. Complete Job & Meeting status
        job.status = JobStatus.COMPLETED.value
        job.progress = 100.0
        job.error_message = None
        job.completed_at = datetime.now(timezone.utc)
        db.commit()
        # Reconcile overall meeting status now that transcription has completed
        try:
            reconcile_meeting_status(db, meeting.id)
        except Exception as re_exc:
            logger.warning(f"reconcile_meeting_status failed after transcription success: {re_exc}")
        logger.info(
            f"Transcription job {job_id} completed successfully for meeting {meeting.id}. "
            f"Persisted {persisted_count} segments on device {result.device_used} ({result.compute_type_used})."
        )

        # 9. Privacy Lifecycle: Safe post-transcription raw + normalized media deletion (if configured)
        if settings.AUTO_DELETE_RAW_POST_TRANSCRIPTION:
            try:
                # Check for other active or pending media-consuming jobs in the same meeting
                media_consuming_job_types = [
                    JobType.TRANSCRIPTION.value,
                    JobType.INGESTION.value,
                    JobType.DIARIZATION.value,
                ]
                active_sibling_jobs = (
                    db.query(ProcessingJob)
                    .filter(
                        ProcessingJob.meeting_id == meeting.id,
                        ProcessingJob.id != job_id,
                        ProcessingJob.job_type.in_(media_consuming_job_types),
                        ProcessingJob.status.in_([JobStatus.PENDING.value, JobStatus.IN_PROGRESS.value]),
                    )
                    .count()
                )

                if active_sibling_jobs > 0:
                    logger.info(
                        f"Deferring post-transcription media auto-deletion for meeting {meeting.id}: "
                        f"{active_sibling_jobs} sibling media-consuming job(s) are still active."
                    )
                else:
                    raw_assets = [
                        asset for asset in meeting.media_assets
                        if not asset.is_raw_deleted and asset.media_type != "NORMALIZED_AUDIO"
                    ]
                    if raw_assets:
                        for target_raw in raw_assets:
                            try:
                                logger.info(
                                    f"Auto-deleting raw media asset {target_raw.id} and associated normalized audio "
                                    f"according to privacy retention policy."
                                )
                                delete_raw_media_asset(
                                    db=db,
                                    media_asset_id=target_raw.id,
                                    user_id=meeting.created_by_user_id,
                                    delete_normalized=settings.AUTO_DELETE_NORMALIZED_POST_TRANSCRIPTION
                                )
                            except Exception as raw_del_err:
                                logger.error(
                                    f"Auto-deletion of raw media asset {target_raw.id} failed: {raw_del_err}"
                                )
                    else:
                        # If only normalized assets exist, clean them up as well
                        normalized_assets = [
                            asset for asset in meeting.media_assets
                            if not asset.is_raw_deleted and asset.media_type == "NORMALIZED_AUDIO"
                        ]
                        for target_norm in normalized_assets:
                            try:
                                logger.info(
                                    f"Auto-deleting normalized audio asset {target_norm.id} according to privacy retention policy."
                                )
                                delete_raw_media_asset(
                                    db=db,
                                    media_asset_id=target_norm.id,
                                    user_id=meeting.created_by_user_id,
                                    delete_normalized=False
                                )
                            except Exception as norm_del_err:
                                logger.error(
                                    f"Auto-deletion of normalized audio asset {target_norm.id} failed: {norm_del_err}"
                                )

                    if ephemeral_normalized_path and os.path.isfile(ephemeral_normalized_path):
                        try:
                            os.remove(ephemeral_normalized_path)
                            logger.info(f"Cleaned up ephemeral normalized WAV: {ephemeral_normalized_path}")
                        except OSError as oe:
                            logger.warning(f"Could not remove ephemeral normalized WAV: {oe}")
            except Exception as auto_del_err:
                logger.error(
                    f"Post-transcription media auto-deletion encountered an unhandled error for meeting {meeting.id}: {auto_del_err}"
                )

    except Exception as e:
        logger.exception(f"Transcription job {job_id} failed: {e}")
        db.rollback()
        try:
            job = db.query(ProcessingJob).filter(ProcessingJob.id == job_id).first()
            if job:
                job.status = JobStatus.FAILED.value
                job.error_message = str(e)
                job.error_details = str(type(e).__name__)
                job.completed_at = datetime.now(timezone.utc)
                db.commit()
                try:
                    reconcile_meeting_status(db, job.meeting_id)
                except Exception:
                    pass
        except Exception as inner_e:
            logger.error(f"Failed to update failed job status: {inner_e}")
    finally:
        db.close()
