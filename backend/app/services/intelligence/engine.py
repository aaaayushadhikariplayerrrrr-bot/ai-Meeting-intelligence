"""Core Orchestrator for Structured Meeting Intelligence Extraction."""

import json
import logging
from typing import Optional, List, Dict, Any, Tuple
from uuid import uuid4
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from fastapi import HTTPException, status

from app.core.config import settings
from app.models.meeting import Meeting
from app.models.transcription import Transcript, TranscriptSegment, TranscriptType
from app.models.intelligence import (
    MeetingSummary,
    Topic,
    Decision,
    ActionItem,
    Risk,
    UnresolvedQuestion,
    ActionItemStatus,
    RiskSeverity,
    QuestionStatus,
)
from app.schemas.intelligence import (
    MeetingSummaryResponse,
    TopicResponse,
    DecisionResponse,
    ActionItemResponse,
    RiskResponse,
    UnresolvedQuestionResponse,
    StructuredExtractionPayload,
)
from app.services.intelligence.base import (
    ProviderType,
    IntelligenceError,
    SafeTranscriptRequiredError,
    IntelligenceDisabledError,
    ProviderExecutionError,
    IntelligenceExtractionResult,
)
from app.services.intelligence.chunking import chunk_transcript_segments, merge_extracted_payloads
from app.services.intelligence.evidence import ground_and_verify_evidence
from app.services.intelligence.deterministic import DeterministicExtractor
from app.services.intelligence.ollama_adapter import OllamaExtractor

logger = logging.getLogger(__name__)


def resolve_privacy_safe_transcript(
    meeting_id: str,
    db: Session,
    requested_transcript_id: Optional[str] = None,
) -> Transcript:
    """Resolve and enforce privacy-safe transcript input.

    STRICT PRIVACY RULES:
    1. Only REDACTED or CLEANED_REDACTED transcripts may be processed for intelligence.
    2. Explicit requests for ORIGINAL or CLEANED are rejected immediately with SafeTranscriptRequiredError.
    3. If no transcript ID is specified, resolves the latest CLEANED_REDACTED (preferred) or REDACTED version.
    4. If no privacy-safe transcript exists, raises SafeTranscriptRequiredError.
    """
    if requested_transcript_id:
        transcript = (
            db.query(Transcript)
            .filter(Transcript.id == requested_transcript_id, Transcript.meeting_id == meeting_id)
            .first()
        )
        if not transcript:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Specified transcript '{requested_transcript_id}' not found for meeting '{meeting_id}'."
            )

        # Enforce privacy-safe constraint
        if transcript.transcript_type not in (TranscriptType.REDACTED.value, TranscriptType.CLEANED_REDACTED.value):
            raise SafeTranscriptRequiredError(
                f"Transcript '{transcript.id}' is of type '{transcript.transcript_type}'. "
                "Intelligence extraction strictly requires privacy-safe transcripts (REDACTED or CLEANED_REDACTED). "
                "Unredacted transcripts (ORIGINAL, CLEANED) cannot be processed for intelligence."
            )
        return transcript

    # Auto-resolve best privacy-safe transcript
    privacy_safe_transcripts = (
        db.query(Transcript)
        .filter(
            Transcript.meeting_id == meeting_id,
            Transcript.transcript_type.in_([
                TranscriptType.CLEANED_REDACTED.value,
                TranscriptType.REDACTED.value,
            ])
        )
        .all()
    )

    if not privacy_safe_transcripts:
        raise SafeTranscriptRequiredError(
            f"No privacy-safe (redacted) transcript available for meeting '{meeting_id}'. "
            "Please generate a REDACTED or CLEANED_REDACTED transcript version before running intelligence extraction."
        )

    # Priority: CLEANED_REDACTED (0) > REDACTED (1), picking the latest version
    priority_map = {
        TranscriptType.CLEANED_REDACTED.value: 0,
        TranscriptType.REDACTED.value: 1,
    }
    selected = sorted(
        privacy_safe_transcripts,
        key=lambda t: (priority_map.get(t.transcript_type, 99), -t.version)
    )[0]

    return selected


def extract_meeting_intelligence(
    meeting_id: str,
    organization_id: str,
    db: Session,
    source_transcript_id: Optional[str] = None,
    provider_override: Optional[str] = None,
    output_language: Optional[str] = None,
    allow_fallback: bool = True,
    persist: bool = True,
    ollama_extractor: Optional[OllamaExtractor] = None,
    deterministic_extractor: Optional[DeterministicExtractor] = None,
) -> IntelligenceExtractionResult:
    """Orchestrate end-to-end structured intelligence extraction with evidence grounding and privacy validation."""
    # 1. Validate meeting and tenant boundary
    meeting = (
        db.query(Meeting)
        .filter(Meeting.id == meeting_id, Meeting.organization_id == organization_id)
        .first()
    )
    if not meeting:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Meeting '{meeting_id}' not found in this organization."
        )

    # 2. Enforce privacy-safe transcript resolution
    transcript = resolve_privacy_safe_transcript(
        meeting_id=meeting.id,
        db=db,
        requested_transcript_id=source_transcript_id,
    )

    segments = (
        db.query(TranscriptSegment)
        .filter(TranscriptSegment.transcript_id == transcript.id)
        .order_by(TranscriptSegment.sequence_order.asc())
        .all()
    )
    if not segments:
        raise IntelligenceError(f"Transcript '{transcript.id}' has no utterances or segments to process.")

    # 3. Provider selection
    raw_provider = (provider_override or getattr(settings, "INTELLIGENCE_PROVIDER", "deterministic")).strip().lower()
    if raw_provider == ProviderType.DISABLED.value:
        raise IntelligenceDisabledError(
            "Structured meeting intelligence extraction is disabled in configuration."
        )
    if raw_provider not in {ProviderType.OLLAMA.value, ProviderType.DETERMINISTIC.value}:
        raise IntelligenceError(f"Unknown intelligence provider '{raw_provider}'.")

    target_lang = output_language or getattr(settings, "INTELLIGENCE_OUTPUT_LANGUAGE", "en")

    # 4. Chunking and budget limits
    max_chars = getattr(settings, "INTELLIGENCE_MAX_TRANSCRIPT_CHARS", 40000)
    max_chunks = getattr(settings, "INTELLIGENCE_MAX_CHUNKS", 50)
    chunks, is_partial, partial_reason = chunk_transcript_segments(
        segments=segments,
        max_chars=max_chars,
        max_chunks=max_chunks,
    )

    # Instantiate extractors
    ollama = ollama_extractor or OllamaExtractor()
    deterministic = deterministic_extractor or DeterministicExtractor()

    provider_used = raw_provider
    fallback_warning: Optional[str] = None
    chunk_payloads: List[StructuredExtractionPayload] = []

    # 5. Execute extraction per chunk
    for chunk in chunks:
        if raw_provider == ProviderType.OLLAMA.value:
            try:
                payload = ollama.extract(
                    segments=chunk,
                    meeting_title=meeting.title,
                    output_language=target_lang,
                )
            except ProviderExecutionError as pe:
                if allow_fallback:
                    logger.warning(f"Ollama extraction failed: {pe}. Falling back to deterministic extractor.")
                    payload = deterministic.extract(
                        segments=chunk,
                        meeting_title=meeting.title,
                        output_language=target_lang,
                    )
                    provider_used = ProviderType.DETERMINISTIC.value
                    fallback_warning = f"Ollama execution failed ({str(pe)}). Deterministic extractor was used as fallback."
                else:
                    raise
        else:
            payload = deterministic.extract(
                segments=chunk,
                meeting_title=meeting.title,
                output_language=target_lang,
            )

        chunk_payloads.append(payload)

    # 6. Merge multi-chunk results
    merged_payload = merge_extracted_payloads(chunk_payloads)

    # 7. Ground evidence against verified transcript segments
    grounded_payload = ground_and_verify_evidence(
        payload=merged_payload,
        segments=segments,
    )

    # 8. Database persistence (if persist=True)
    summary_resp: Optional[MeetingSummaryResponse] = None
    topics_resp: List[TopicResponse] = []
    decisions_resp: List[DecisionResponse] = []
    actions_resp: List[ActionItemResponse] = []
    risks_resp: List[RiskResponse] = []
    questions_resp: List[UnresolvedQuestionResponse] = []

    if persist:
        try:
            # Atomic clean of existing extracted entities for this meeting to prevent duplicates
            db.query(Topic).filter(Topic.meeting_id == meeting.id).delete()
            db.query(Decision).filter(Decision.meeting_id == meeting.id).delete()
            db.query(ActionItem).filter(ActionItem.meeting_id == meeting.id).delete()
            db.query(Risk).filter(Risk.meeting_id == meeting.id).delete()
            db.query(UnresolvedQuestion).filter(UnresolvedQuestion.meeting_id == meeting.id).delete()

            # Update existing summary in place or create new
            existing_summary = (
                db.query(MeetingSummary)
                .filter(MeetingSummary.meeting_id == meeting.id)
                .first()
            )
            takeaways_str = json.dumps(grounded_payload.key_takeaways) if grounded_payload.key_takeaways else None

            if existing_summary:
                existing_summary.transcript_id = transcript.id
                existing_summary.executive_summary = grounded_payload.executive_summary
                existing_summary.detailed_summary = grounded_payload.detailed_summary
                existing_summary.key_takeaways = takeaways_str
                existing_summary.version = existing_summary.version + 1
                existing_summary.provider_used = provider_used
                existing_summary.is_heuristic = (provider_used == ProviderType.DETERMINISTIC.value)
                existing_summary.fallback_warning = fallback_warning
                summary_orm = existing_summary
            else:
                summary_orm = MeetingSummary(
                    meeting_id=meeting.id,
                    transcript_id=transcript.id,
                    executive_summary=grounded_payload.executive_summary,
                    detailed_summary=grounded_payload.detailed_summary,
                    key_takeaways=takeaways_str,
                    version=1,
                    provider_used=provider_used,
                    is_heuristic=(provider_used == ProviderType.DETERMINISTIC.value),
                    fallback_warning=fallback_warning,
                )
                db.add(summary_orm)
            db.flush()

            # Persist Topics
            topic_orms = [
                Topic(
                    meeting_id=meeting.id,
                    title=t.title,
                    description=t.description,
                    start_time=t.start_time,
                    end_time=t.end_time,
                )
                for t in grounded_payload.topics
            ]
            db.add_all(topic_orms)

            # Persist Decisions
            decision_orms = [
                Decision(
                    meeting_id=meeting.id,
                    description=d.description,
                    rationale_or_context=d.rationale_or_context,
                    decided_by=d.decided_by,
                    evidence_segment_id=d.evidence_segment_id,
                    timestamp_seconds=d.timestamp_seconds,
                )
                for d in grounded_payload.decisions
            ]
            db.add_all(decision_orms)

            # Persist Action Items
            action_orms = [
                ActionItem(
                    meeting_id=meeting.id,
                    description=a.description,
                    owner_name=a.owner_name,
                    owner_user_id=a.owner_user_id,
                    deadline=a.deadline,
                    status=a.status.value if isinstance(a.status, ActionItemStatus) else str(a.status),
                    evidence_segment_id=a.evidence_segment_id,
                    timestamp_seconds=a.timestamp_seconds,
                )
                for a in grounded_payload.action_items
            ]
            db.add_all(action_orms)

            # Persist Risks
            risk_orms = [
                Risk(
                    meeting_id=meeting.id,
                    description=r.description,
                    severity=r.severity.value if isinstance(r.severity, RiskSeverity) else str(r.severity),
                    mitigation_suggestion=r.mitigation_suggestion,
                    evidence_segment_id=r.evidence_segment_id,
                )
                for r in grounded_payload.risks
            ]
            db.add_all(risk_orms)

            # Persist Unresolved Questions
            question_orms = [
                UnresolvedQuestion(
                    meeting_id=meeting.id,
                    question_text=q.question_text,
                    raised_by=q.raised_by,
                    status=q.status.value if isinstance(q.status, QuestionStatus) else str(q.status),
                    evidence_segment_id=q.evidence_segment_id,
                )
                for q in grounded_payload.unresolved_questions
            ]
            db.add_all(question_orms)

            db.commit()
            db.refresh(summary_orm)

            summary_resp = MeetingSummaryResponse.model_validate(summary_orm)
            topics_resp = [TopicResponse.model_validate(t) for t in topic_orms]
            decisions_resp = [DecisionResponse.model_validate(d) for d in decision_orms]
            actions_resp = [ActionItemResponse.model_validate(a) for a in action_orms]
            risks_resp = [RiskResponse.model_validate(r) for r in risk_orms]
            questions_resp = [UnresolvedQuestionResponse.model_validate(q) for q in question_orms]
        except Exception:
            db.rollback()
            raise
    else:
        # Construct ephemeral in-memory response DTOs when persist=False
        now = datetime.now(timezone.utc)
        takeaways_str = json.dumps(grounded_payload.key_takeaways) if grounded_payload.key_takeaways else None
        summary_resp = MeetingSummaryResponse(
            id=str(uuid4()),
            meeting_id=meeting.id,
            transcript_id=transcript.id,
            executive_summary=grounded_payload.executive_summary,
            detailed_summary=grounded_payload.detailed_summary,
            key_takeaways=takeaways_str,
            version=1,
            provider_used=provider_used,
            is_heuristic=(provider_used == ProviderType.DETERMINISTIC.value),
            fallback_warning=fallback_warning,
            created_at=now,
            updated_at=now,
        )
        topics_resp = [
            TopicResponse(
                id=str(uuid4()),
                meeting_id=meeting.id,
                title=t.title,
                description=t.description,
                start_time=t.start_time,
                end_time=t.end_time,
                created_at=now,
                updated_at=now,
            )
            for t in grounded_payload.topics
        ]
        decisions_resp = [
            DecisionResponse(
                id=str(uuid4()),
                meeting_id=meeting.id,
                description=d.description,
                rationale_or_context=d.rationale_or_context,
                decided_by=d.decided_by,
                evidence_segment_id=d.evidence_segment_id,
                timestamp_seconds=d.timestamp_seconds,
                created_at=now,
                updated_at=now,
            )
            for d in grounded_payload.decisions
        ]
        actions_resp = [
            ActionItemResponse(
                id=str(uuid4()),
                meeting_id=meeting.id,
                description=a.description,
                owner_name=a.owner_name,
                owner_user_id=a.owner_user_id,
                deadline=a.deadline,
                status=a.status.value if isinstance(a.status, ActionItemStatus) else str(a.status),
                evidence_segment_id=a.evidence_segment_id,
                timestamp_seconds=a.timestamp_seconds,
                created_at=now,
                updated_at=now,
            )
            for a in grounded_payload.action_items
        ]
        risks_resp = [
            RiskResponse(
                id=str(uuid4()),
                meeting_id=meeting.id,
                description=r.description,
                severity=r.severity.value if isinstance(r.severity, RiskSeverity) else str(r.severity),
                mitigation_suggestion=r.mitigation_suggestion,
                evidence_segment_id=r.evidence_segment_id,
                created_at=now,
                updated_at=now,
            )
            for r in grounded_payload.risks
        ]
        questions_resp = [
            UnresolvedQuestionResponse(
                id=str(uuid4()),
                meeting_id=meeting.id,
                question_text=q.question_text,
                raised_by=q.raised_by,
                status=q.status.value if isinstance(q.status, QuestionStatus) else str(q.status),
                evidence_segment_id=q.evidence_segment_id,
                created_at=now,
                updated_at=now,
            )
            for q in grounded_payload.unresolved_questions
        ]

    return IntelligenceExtractionResult(
        meeting_id=meeting.id,
        source_transcript_id=transcript.id,
        summary=summary_resp,
        topics=topics_resp,
        decisions=decisions_resp,
        action_items=actions_resp,
        risks=risks_resp,
        unresolved_questions=questions_resp,
        provider_used=provider_used,
        is_heuristic=(provider_used == ProviderType.DETERMINISTIC.value),
        fallback_warning=fallback_warning,
        is_partial=is_partial,
        partial_reason=partial_reason,
        raw_payload=grounded_payload,
    )
