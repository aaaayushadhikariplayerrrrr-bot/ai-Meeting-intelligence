"""Transcript Processing and Derivation Orchestration Service.

Coordinates:
- Transcript cleaning (readability, stutter reduction, filler removal)
- Privacy redaction (PII, Indian identifiers, API keys, credentials)
- Combined cleaning + redaction
- Safe derived transcript versioning without modifying original ASR records
- Explicit segment-to-source linking and immutable audit logging
"""

import json
import uuid
import logging
from typing import Optional, Dict, Any, List, Tuple, Set
from sqlalchemy.orm import Session
from fastapi import HTTPException, status

from app.models.transcription import Transcript, TranscriptSegment, TranscriptType
from app.models.meeting import Meeting
from app.models.memory import AuditLog
from app.services.transcript_cleaning.cleaner import clean_segment_text
from app.services.privacy_redaction.redactor import redact_text

logger = logging.getLogger(__name__)


def derive_processed_transcript(
    meeting_id: str,
    organization_id: str,
    target_type: TranscriptType,
    db: Session,
    source_transcript_id: Optional[str] = None,
    user_id: Optional[str] = None,
    ip_address: Optional[str] = None,
) -> Transcript:
    """Generate a new derived transcript version (CLEANED, REDACTED, or CLEANED_REDACTED) from an existing transcript.
    
    CRITICAL INVARIANTS:
    - Never overwrites or modifies the original transcript or its segments.
    - Increments version number for new derived transcripts.
    - Sets parent_transcript_id on the new Transcript container.
    - Sets source_segment_id on each derived segment pointing to its source.
    - Preserves timestamps, sequence order, speaker attribution, and language code.
    - Records immutable audit trail without printing or persisting raw sensitive PII.
    """
    # 1. Validate meeting and tenant isolation
    meeting = (
        db.query(Meeting)
        .filter(Meeting.id == meeting_id, Meeting.organization_id == organization_id)
        .first()
    )
    if not meeting:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Meeting {meeting_id} not found in this organization."
        )

    # 2. Resolve source transcript
    if source_transcript_id:
        source_transcript = (
            db.query(Transcript)
            .filter(
                Transcript.id == source_transcript_id,
                Transcript.meeting_id == meeting.id
            )
            .first()
        )
    else:
        # Default: Latest ORIGINAL transcript
        source_transcript = (
            db.query(Transcript)
            .filter(
                Transcript.meeting_id == meeting.id,
                Transcript.transcript_type == TranscriptType.ORIGINAL.value
            )
            .order_by(Transcript.version.desc())
            .first()
        )

    if not source_transcript:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No source transcript found for this meeting to process. Run ASR transcription first."
        )

    source_segments: List[TranscriptSegment] = list(source_transcript.segments)
    if not source_segments:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Source transcript has no segments to process."
        )

    # 3. Determine next version for target_type to avoid overwriting existing derived versions
    existing_max_version = (
        db.query(Transcript.version)
        .filter(
            Transcript.meeting_id == meeting.id,
            Transcript.transcript_type == target_type.value
        )
        .order_by(Transcript.version.desc())
        .first()
    )
    new_version = (existing_max_version[0] + 1) if existing_max_version else 1

    # 4. Create new derived Transcript container
    derived_transcript = Transcript(
        id=str(uuid.uuid4()),
        meeting_id=meeting.id,
        transcript_type=target_type.value,
        version=new_version,
        language_detected=source_transcript.language_detected,
        is_primary=False,
        parent_transcript_id=source_transcript.id,
    )
    db.add(derived_transcript)
    db.flush()

    # 5. Process segments
    all_categories: Set[str] = set()
    total_redaction_count = 0
    total_cleaned_segments = 0

    should_clean = target_type in (TranscriptType.CLEANED, TranscriptType.CLEANED_REDACTED)
    should_redact = target_type in (TranscriptType.REDACTED, TranscriptType.CLEANED_REDACTED)

    derived_segments = []
    for src_seg in source_segments:
        current_text = src_seg.text
        segment_categories = []
        is_segment_redacted = False

        # Cleaning stage
        if should_clean:
            cleaning_res = clean_segment_text(current_text, language_code=src_seg.language_code)
            current_text = cleaning_res.cleaned_text
            if cleaning_res.has_changed:
                total_cleaned_segments += 1

        # Redaction stage
        if should_redact:
            redaction_res = redact_text(current_text)
            current_text = redaction_res.redacted_text
            segment_categories = redaction_res.categories_detected
            is_segment_redacted = redaction_res.is_redacted
            if redaction_res.is_redacted:
                total_redaction_count += redaction_res.redaction_count
                all_categories.update(segment_categories)

        derived_seg = TranscriptSegment(
            id=str(uuid.uuid4()),
            transcript_id=derived_transcript.id,
            speaker_id=src_seg.speaker_id,
            sequence_order=src_seg.sequence_order,
            start_time=src_seg.start_time,
            end_time=src_seg.end_time,
            text=current_text,
            language_code=src_seg.language_code,
            confidence=src_seg.confidence,
            is_redacted=is_segment_redacted,
            source_segment_id=src_seg.id,
            redaction_metadata=json.dumps(segment_categories) if segment_categories else None,
        )
        derived_segments.append(derived_seg)
        db.add(derived_seg)

    db.flush()

    # Verification check
    persisted_count = (
        db.query(TranscriptSegment)
        .filter(TranscriptSegment.transcript_id == derived_transcript.id)
        .count()
    )
    if persisted_count != len(source_segments):
        raise RuntimeError(
            f"Derived transcript segment count mismatch: expected {len(source_segments)}, got {persisted_count}"
        )

    # 6. Record immutable AuditLog entry (NO sensitive values logged)
    audit_details = {
        "meeting_id": meeting.id,
        "source_transcript_id": source_transcript.id,
        "derived_transcript_id": derived_transcript.id,
        "transcript_type": target_type.value,
        "version": new_version,
        "segment_count": persisted_count,
        "cleaned_segments_count": total_cleaned_segments if should_clean else 0,
        "redactions_count": total_redaction_count if should_redact else 0,
        "redaction_categories": sorted(list(all_categories)) if should_redact else [],
    }

    audit_entry = AuditLog(
        id=str(uuid.uuid4()),
        organization_id=organization_id,
        user_id=user_id,
        action=f"TRANSCRIPT_{target_type.value}",
        entity_type="Transcript",
        entity_id=derived_transcript.id,
        details=json.dumps(audit_details),
        ip_address=ip_address,
    )
    db.add(audit_entry)
    db.commit()
    db.refresh(derived_transcript)

    logger.info(
        f"Derived transcript {derived_transcript.id} (type: {target_type.value}, v{new_version}) "
        f"successfully generated from source {source_transcript.id} with {persisted_count} segments."
    )

    # Automatic Privacy-Safe Memory Indexing Trigger:
    # Durable background indexing job is scheduled ONLY after verified persistence of a privacy-safe transcript
    if target_type in (TranscriptType.REDACTED, TranscriptType.CLEANED_REDACTED):
        try:
            from app.services.memory.job_handler import schedule_indexing_job
            schedule_indexing_job(
                db=db,
                meeting_id=meeting.id,
                organization_id=organization_id,
                transcript_id=derived_transcript.id,
            )
        except Exception as schedule_err:
            logger.warning(
                f"Failed to auto-schedule memory indexing job for transcript '{derived_transcript.id}': {schedule_err}"
            )

    return derived_transcript
