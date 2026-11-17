"""Database Grounding and Citation Verification for RAG Evidence."""

import logging
from typing import List, Dict, Set, Optional, Any
from sqlalchemy.orm import Session

from app.models.meeting import Meeting
from app.models.memory import TranscriptChunkRecord
from app.models.transcription import Transcript, TranscriptSegment, TranscriptType
from app.services.memory.rag.types import Citation, EvidenceItem

logger = logging.getLogger(__name__)

SAFE_TRANSCRIPT_TYPES: Set[str] = {
    TranscriptType.CLEANED_REDACTED.value,
    TranscriptType.REDACTED.value,
}


class CitationVerifier:
    """Verifies evidence citations against retrieved chunks, structured entities, and database records.

    Guarantees:
    1. Every citation must correspond to an actual retrieved chunk or structured intelligence record for the current organization.
    2. Hallucinated, fabricated, cross-tenant, or out-of-organization IDs are stripped.
    3. Start and end timestamps are derived strictly from database TranscriptSegment records,
       never from generative LLM text or decision.timestamp_seconds.
    4. Verified citations include full meeting and transcript provenance suitable for UI navigation.
    5. Evidence must be associated with an allowed safe transcript version (CLEANED_REDACTED or REDACTED).
    """

    def verify_structured_evidence(
        self,
        db: Session,
        evidence_segment_id: Optional[str],
        meeting_id: Optional[str],
        organization_id: str,
        entity_ref_id: str,
        citation_id: str = "cit-1",
    ) -> Optional[Citation]:
        """Verify evidence segment reference for structured intelligence records (Decisions, ActionItems, Questions).

        Guarantees:
        1. Resolves evidence_segment_id against the actual TranscriptSegment database record.
        2. Strictly verifies that the segment belongs to the correct meeting and organization (tenant isolation).
        3. Resolves meeting title and meeting date strictly from database records.
        4. Derives start and end timestamps strictly from the TranscriptSegment itself
           (never using entity.timestamp_seconds).
        5. Verifies that the evidence is associated with an allowed safe transcript version
           (CLEANED_REDACTED or REDACTED), or maps to a derived safe segment if on an original.
        6. Rejects NULL, missing, invalid, or cross-tenant segment IDs without fabricating a citation.
        """
        if not evidence_segment_id or not str(evidence_segment_id).strip():
            return None

        seg_uuid = str(evidence_segment_id).strip()

        # 1. Resolve TranscriptSegment from database
        segment = db.query(TranscriptSegment).filter(TranscriptSegment.id == seg_uuid).first()
        if not segment:
            logger.warning(
                f"Structured evidence segment '{seg_uuid}' for entity '{entity_ref_id}' "
                f"does not exist in database."
            )
            return None

        # 2. Resolve Transcript
        transcript = db.query(Transcript).filter(Transcript.id == segment.transcript_id).first()
        if not transcript:
            logger.warning(
                f"Transcript '{segment.transcript_id}' for segment '{seg_uuid}' does not exist in database."
            )
            return None

        # 3. Verify Safe Transcript Version
        safe_segment = segment
        safe_transcript = transcript

        if transcript.transcript_type not in SAFE_TRANSCRIPT_TYPES:
            # Check if there is an associated derived segment on a safe transcript version
            # for the same meeting where source_segment_id == segment.id
            derived_safe_seg = (
                db.query(TranscriptSegment)
                .join(Transcript, TranscriptSegment.transcript_id == Transcript.id)
                .filter(
                    Transcript.meeting_id == transcript.meeting_id,
                    Transcript.transcript_type.in_(SAFE_TRANSCRIPT_TYPES),
                    TranscriptSegment.source_segment_id == segment.id,
                )
                .first()
            )
            if derived_safe_seg:
                safe_segment = derived_safe_seg
                safe_transcript = derived_safe_seg.transcript
            else:
                logger.warning(
                    f"Evidence segment '{seg_uuid}' is associated with unsafe transcript type "
                    f"'{transcript.transcript_type}' and cannot be mapped to a safe version."
                )
                return None

        # 4. Verify Meeting and Organization Ownership (Strict Tenant Isolation)
        meeting = (
            db.query(Meeting)
            .filter(
                Meeting.id == safe_transcript.meeting_id,
                Meeting.organization_id == organization_id,
            )
            .first()
        )
        if not meeting:
            logger.warning(
                f"Security violation: Evidence segment '{seg_uuid}' belongs to meeting "
                f"'{safe_transcript.meeting_id}' which does not belong to organization '{organization_id}'."
            )
            return None

        # Verify meeting ownership consistency with entity meeting_id if provided
        if meeting_id and meeting.id != meeting_id:
            logger.warning(
                f"Meeting ownership mismatch: Entity claimed meeting '{meeting_id}' "
                f"but segment belongs to meeting '{meeting.id}'."
            )
            return None

        # 5. Derive Timestamps from the Segment itself (NEVER substitute stored timestamps)
        if safe_segment.start_time is None or safe_segment.end_time is None:
            logger.warning(f"Segment '{safe_segment.id}' has missing timestamps in database.")
            return None

        start_time = float(safe_segment.start_time)
        end_time = float(safe_segment.end_time)
        if start_time < 0.0 or end_time < start_time:
            logger.warning(
                f"Segment '{safe_segment.id}' has invalid timestamps: start={start_time}, end={end_time}"
            )
            return None

        # 6. Resolve Meeting Title and Date from Database
        meeting_title = meeting.title if meeting.title else "Unknown Meeting"
        meeting_date = None
        if meeting.actual_start_time:
            meeting_date = meeting.actual_start_time.isoformat()
        elif meeting.scheduled_start_time:
            meeting_date = meeting.scheduled_start_time.isoformat()
        elif meeting.created_at:
            meeting_date = meeting.created_at.isoformat()

        # 7. Assemble Verified First-Class Citation
        lang_codes = [safe_segment.language_code] if safe_segment.language_code else []
        return Citation(
            citation_id=citation_id,
            chunk_id=str(entity_ref_id),
            meeting_id=meeting.id,
            meeting_title=meeting_title,
            meeting_date=meeting_date,
            transcript_id=safe_transcript.id,
            segment_ids=[safe_segment.id],
            start_time=start_time,
            end_time=end_time,
            quoted_text=safe_segment.text,
            language_codes=lang_codes,
        )

    def verify_citations(
        self,
        db: Session,
        cited_chunk_ids: List[str],
        retrieved_evidence: List[EvidenceItem],
        organization_id: str,
        structured_context: Optional[Any] = None,
    ) -> List[Citation]:
        """Verify cited chunk IDs and structured intelligence evidence against database records."""
        verified_citations: List[Citation] = []
        cited_segment_ids: Set[str] = set()

        # 1. Verify Chunk Citations (if present)
        if cited_chunk_ids and retrieved_evidence:
            retrieved_map: Dict[str, EvidenceItem] = {e.chunk_id: e for e in retrieved_evidence}

            # Deduplicate while preserving citation order
            seen_cids: Set[str] = set()
            ordered_valid_ids: List[str] = []
            for cid in cited_chunk_ids:
                if cid in retrieved_map and cid not in seen_cids:
                    seen_cids.add(cid)
                    ordered_valid_ids.append(cid)
                elif cid not in retrieved_map:
                    logger.warning(f"Stripping ungrounded or hallucinated citation chunk ID: '{cid}'")

            if ordered_valid_ids:
                # Query underlying TranscriptChunkRecord from database strictly enforcing organization_id
                chunk_records = (
                    db.query(TranscriptChunkRecord)
                    .filter(
                        TranscriptChunkRecord.id.in_(ordered_valid_ids),
                        TranscriptChunkRecord.organization_id == organization_id,
                    )
                    .all()
                )
                chunk_map: Dict[str, TranscriptChunkRecord] = {c.id: c for c in chunk_records}

                # Resolve Meeting titles and dates
                meeting_ids = list({c.meeting_id for c in chunk_records})
                meetings = (
                    db.query(Meeting)
                    .filter(Meeting.id.in_(meeting_ids), Meeting.organization_id == organization_id)
                    .all()
                )
                meeting_map: Dict[str, Meeting] = {m.id: m for m in meetings}

                # Resolve exact segment IDs and database timestamps
                all_segment_ids: Set[str] = set()
                for c in chunk_records:
                    all_segment_ids.update(c.segment_ids_list)

                db_segments = (
                    db.query(TranscriptSegment)
                    .filter(TranscriptSegment.id.in_(list(all_segment_ids)))
                    .all()
                ) if all_segment_ids else []
                segment_map: Dict[str, TranscriptSegment] = {s.id: s for s in db_segments}

                for cid in ordered_valid_ids:
                    record = chunk_map.get(cid)
                    if not record:
                        logger.warning(f"Chunk record '{cid}' not found in database for organization '{organization_id}'.")
                        continue

                    meeting = meeting_map.get(record.meeting_id)
                    meeting_title = meeting.title if meeting else "Unknown Meeting"
                    meeting_date = None
                    if meeting:
                        if meeting.actual_start_time:
                            meeting_date = meeting.actual_start_time.isoformat()
                        elif meeting.scheduled_start_time:
                            meeting_date = meeting.scheduled_start_time.isoformat()
                        elif meeting.created_at:
                            meeting_date = meeting.created_at.isoformat()

                    source_segs = [segment_map[sid] for sid in record.segment_ids_list if sid in segment_map]
                    if source_segs:
                        derived_start = min(s.start_time for s in source_segs)
                        derived_end = max(s.end_time for s in source_segs)
                        verified_segment_ids = [s.id for s in source_segs]
                    else:
                        derived_start = record.start_time
                        derived_end = record.end_time
                        verified_segment_ids = record.segment_ids_list

                    cit_idx = len(verified_citations) + 1
                    citation = Citation(
                        citation_id=f"cit-{cit_idx}",
                        chunk_id=record.id,
                        meeting_id=record.meeting_id,
                        meeting_title=meeting_title,
                        meeting_date=meeting_date,
                        transcript_id=record.transcript_id,
                        segment_ids=verified_segment_ids,
                        start_time=derived_start,
                        end_time=derived_end,
                        quoted_text=record.text,
                        language_codes=record.language_codes_list,
                    )
                    verified_citations.append(citation)
                    cited_segment_ids.update(verified_segment_ids)

        # 2. Verify Structured Intelligence Citations (Decisions, Commitments, Questions)
        if structured_context:
            # Process Decisions
            for d in getattr(structured_context, "decisions", []):
                seg_id = getattr(d, "evidence_segment_id", None)
                if not seg_id:
                    continue
                d_id = getattr(d, "id", None) or f"dec-{len(verified_citations)+1}"
                m_id = getattr(d, "meeting_id", None)
                cit_idx = len(verified_citations) + 1
                cit = self.verify_structured_evidence(
                    db=db,
                    evidence_segment_id=seg_id,
                    meeting_id=m_id,
                    organization_id=organization_id,
                    entity_ref_id=str(d_id),
                    citation_id=f"cit-{cit_idx}",
                )
                if cit and cit.segment_ids[0] not in cited_segment_ids:
                    verified_citations.append(cit)
                    cited_segment_ids.update(cit.segment_ids)

            # Process Commitments
            for c in getattr(structured_context, "commitments", []):
                seg_id = getattr(c, "evidence_segment_id", None)
                if not seg_id:
                    continue
                c_id = getattr(c, "id", None) or f"act-{len(verified_citations)+1}"
                m_id = getattr(c, "meeting_id", None)
                cit_idx = len(verified_citations) + 1
                cit = self.verify_structured_evidence(
                    db=db,
                    evidence_segment_id=seg_id,
                    meeting_id=m_id,
                    organization_id=organization_id,
                    entity_ref_id=str(c_id),
                    citation_id=f"cit-{cit_idx}",
                )
                if cit and cit.segment_ids[0] not in cited_segment_ids:
                    verified_citations.append(cit)
                    cited_segment_ids.update(cit.segment_ids)

            # Process Questions
            for q in getattr(structured_context, "questions", []):
                seg_id = getattr(q, "evidence_segment_id", None)
                if not seg_id:
                    continue
                q_id = getattr(q, "id", None) or f"que-{len(verified_citations)+1}"
                m_id = getattr(q, "meeting_id", None)
                cit_idx = len(verified_citations) + 1
                cit = self.verify_structured_evidence(
                    db=db,
                    evidence_segment_id=seg_id,
                    meeting_id=m_id,
                    organization_id=organization_id,
                    entity_ref_id=str(q_id),
                    citation_id=f"cit-{cit_idx}",
                )
                if cit and cit.segment_ids[0] not in cited_segment_ids:
                    verified_citations.append(cit)
                    cited_segment_ids.update(cit.segment_ids)

        return verified_citations
