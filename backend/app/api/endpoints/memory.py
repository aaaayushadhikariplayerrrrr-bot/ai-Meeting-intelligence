"""Secure REST API Endpoints for Organizational Memory.

Provides:
- POST /api/v1/memory/query: Ask Organizational AI (Cross-meeting RAG question answering).
- POST /api/v1/memory/search: Hybrid keyword and semantic search across privacy-safe meeting chunks.

Mandatory Security & Privacy Controls:
1. Organization context is resolved exclusively from the authenticated user's active tenant membership (TenantContext).
2. Client-supplied organization IDs are never trusted and strictly overridden by the authenticated context.
3. Scoped meeting_id is verified against the database ensuring it belongs to the authenticated tenant (HTTP 404 on mismatch/not found).
4. Unredacted (ORIGINAL/CLEANED) transcripts are never exposed; only privacy-safe chunks are indexed and searchable.
5. All database lookups and hybrid retrieval operations remain tenant-scoped.
6. Rate limiting is enforced per client IP via InMemoryRateLimiter.
7. Engine abstentions (NO_EVIDENCE, INSUFFICIENT_EVIDENCE, etc.) return HTTP 200 with structured status rather than server errors.
"""

from typing import List, Dict, Optional, Set
import logging
from sqlalchemy import func
from sqlalchemy.orm import Session
from fastapi import APIRouter, Depends, HTTPException, Query, status, Request

from app.core.config import settings
from app.core.rate_limit import memory_rate_limiter
from app.database.session import get_db
from app.api.deps import get_current_tenant_membership, TenantContext
from app.models.meeting import Meeting
from app.models.memory import TranscriptChunkRecord
from app.models.intelligence import Decision
from app.models.transcription import TranscriptSegment
from app.schemas.memory import (
    MemoryQueryApiRequest,
    MemorySearchApiRequest,
    MemorySearchResultItem,
    MemorySearchResponse,
    RAGQueryRequest,
    RAGQueryResponse,
    AbstentionStatus,
    RetrievalDiagnostics,
    DecisionConflictItem,
)
from app.services.memory.embedding import get_embedding_service
from app.services.memory.conflict_detector import ConflictDetector
from app.services.memory.exceptions import (
    ModelNotCachedError,
    EmbeddingModelLoadError,
    TenantSecurityError,
)
from app.services.memory.retrieval import search_hybrid_rrf, search_keyword
from app.services.memory.rag.engine import RAGEngine
from app.services.memory.types import HybridSearchResult

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/query",
    response_model=RAGQueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Ask Organizational AI",
    description="Cross-meeting question answering grounded in verified privacy-safe meeting memory with database citations.",
)
def query_memory(
    request: Request,
    payload: MemoryQueryApiRequest,
    tenant: TenantContext = Depends(get_current_tenant_membership),
    db: Session = Depends(get_db),
    _rate_limit: None = Depends(memory_rate_limiter),
) -> RAGQueryResponse:
    """Ask natural-language questions across organization's meetings with verified citations.

    Security & Privacy Rules:
    - Organization is derived exclusively from active tenant membership (never client input).
    - If meeting_id is supplied, ownership is validated against tenant organization.
    - Answers are grounded strictly in database-verified transcript chunks.
    - Honest provider attribution is preserved (never labels deterministic output as LLM).
    - Valid abstentions (NO_EVIDENCE, INSUFFICIENT_EVIDENCE, etc.) return HTTP 200 with is_abstained=True.
    """
    org_id = tenant.organization.id

    # 1. Input Validation
    clean_question = payload.question.strip()
    if not clean_question:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty or whitespace-only.",
        )

    if payload.start_date and payload.end_date and payload.start_date > payload.end_date:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="start_date cannot be later than end_date.",
        )

    # 2. Meeting Ownership Check (if meeting_id is scoped)
    if payload.meeting_id:
        meeting = (
            db.query(Meeting)
            .filter(Meeting.id == payload.meeting_id, Meeting.organization_id == org_id)
            .first()
        )
        if not meeting:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Meeting not found or access denied.",
            )

    # 3. Construct Engine Request using strictly validated tenant org_id
    rag_request = RAGQueryRequest(
        question=clean_question,
        organization_id=org_id,
        meeting_id=payload.meeting_id,
        start_date=payload.start_date,
        end_date=payload.end_date,
        max_evidence_chunks=payload.max_evidence_chunks,
        output_language=payload.output_language,
    )

    engine = RAGEngine()
    try:
        response = engine.query(db=db, request=rag_request)
        return response
    except Exception as exc:
        logger.exception(f"Unexpected RAG query engine execution failure: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while processing organizational memory query.",
        )


@router.post(
    "/search",
    response_model=MemorySearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Search Organizational Memory",
    description="Hybrid keyword and semantic search across privacy-safe meeting chunks.",
)
def search_memory(
    request: Request,
    payload: MemorySearchApiRequest,
    tenant: TenantContext = Depends(get_current_tenant_membership),
    db: Session = Depends(get_db),
    _rate_limit: None = Depends(memory_rate_limiter),
) -> MemorySearchResponse:
    """Execute tenant-isolated hybrid search across privacy-safe meeting transcript chunks.

    Security & Privacy Rules:
    - Organization is derived exclusively from active tenant membership (never client input).
    - If meeting_id is supplied, ownership is validated against tenant organization.
    - Dates are pre-filtered on meeting timestamps.
    - Returns only privacy-safe chunks; unredacted transcript content is never returned.
    - Timestamps and segment IDs are derived directly from SQLite database records.
    """
    org_id = tenant.organization.id

    # 1. Input Validation
    clean_query = payload.query.strip()
    if not clean_query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query cannot be empty or whitespace-only.",
        )

    if payload.start_date and payload.end_date and payload.start_date > payload.end_date:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="start_date cannot be later than end_date.",
        )

    # 2. Meeting Ownership Check (if meeting_id is scoped)
    if payload.meeting_id:
        meeting = (
            db.query(Meeting)
            .filter(Meeting.id == payload.meeting_id, Meeting.organization_id == org_id)
            .first()
        )
        if not meeting:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Meeting not found or access denied.",
            )

    # 3. Date Range Pre-filter on Meetings
    target_meeting_id = payload.meeting_id
    matching_meeting_ids: Optional[Set[str]] = None
    if payload.start_date or payload.end_date:
        meeting_date_col = func.coalesce(Meeting.actual_start_time, Meeting.scheduled_start_time, Meeting.created_at)
        m_query = db.query(Meeting.id).filter(Meeting.organization_id == org_id)
        if payload.start_date:
            m_query = m_query.filter(meeting_date_col >= payload.start_date)
        if payload.end_date:
            m_query = m_query.filter(meeting_date_col <= payload.end_date)
        matching_meeting_ids = {r[0] for r in m_query.all()}
        if not matching_meeting_ids or (target_meeting_id and target_meeting_id not in matching_meeting_ids):
            return MemorySearchResponse(query=clean_query, total_results=0, results=[])

    # 4. Perform Hybrid Search with Fallback to Keyword Search
    candidates: List[HybridSearchResult] = []
    service = get_embedding_service()
    try:
        query_vector = service.embed_query(clean_query)
        candidates = search_hybrid_rrf(
            db=db,
            query=clean_query,
            query_vector=query_vector,
            organization_id=org_id,
            meeting_id=target_meeting_id,
            limit=payload.limit * 2,
            expected_dim=service.dimension,
        )
    except (ModelNotCachedError, EmbeddingModelLoadError):
        logger.info("Embedding model not cached; falling back to FTS5 keyword search.")
        kw_candidates = search_keyword(
            db=db,
            query=clean_query,
            organization_id=org_id,
            meeting_id=target_meeting_id,
            limit=payload.limit * 2,
        )
        candidates = [
            HybridSearchResult(
                chunk_id=c.chunk_id,
                organization_id=c.organization_id,
                meeting_id=c.meeting_id,
                transcript_id=c.transcript_id,
                chunk_index=c.chunk_index,
                segment_ids=c.segment_ids,
                text=c.text,
                start_time=c.start_time,
                end_time=c.end_time,
                language_codes=c.language_codes,
                dominant_language=c.dominant_language,
                token_count=c.token_count,
                rrf_score=0.0,
                keyword_rank=c.rank,
                semantic_rank=None,
            )
            for c in kw_candidates
        ]
    except Exception as exc:
        logger.exception(f"Memory search execution failure: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while searching organizational memory.",
        )

    # 5. Apply Date Filtering to Candidates (if applicable)
    if matching_meeting_ids is not None and not target_meeting_id:
        candidates = [c for c in candidates if c.meeting_id in matching_meeting_ids]

    # Verify candidate relevance: if no candidate has an FTS5 keyword rank and none contains any query keyword,
    # the candidates are low-similarity nearest neighbor artifacts with NO actual matches in the tenant.
    from app.services.memory.rag.providers import DeterministicAnswerProvider
    provider = DeterministicAnswerProvider()
    keywords = provider._extract_query_keywords(clean_query)
    if keywords and candidates:
        has_relevant = any(
            c.keyword_rank is not None or any(kw in c.text.lower() for kw in keywords)
            for c in candidates
        )
        if not has_relevant:
            candidates = []

    # Limit candidates to requested limit
    candidates = candidates[:payload.limit]
    if not candidates:
        return MemorySearchResponse(query=clean_query, total_results=0, results=[])

    # 6. Resolve Meeting titles and dates
    meeting_ids = list({c.meeting_id for c in candidates})
    meetings = db.query(Meeting).filter(Meeting.id.in_(meeting_ids), Meeting.organization_id == org_id).all()
    meeting_map: Dict[str, Meeting] = {m.id: m for m in meetings}

    # 7. Resolve Database Segment IDs and exact timestamps
    all_segment_ids: Set[str] = set()
    for c in candidates:
        all_segment_ids.update(c.segment_ids)

    db_segments = (
        db.query(TranscriptSegment)
        .filter(TranscriptSegment.id.in_(list(all_segment_ids)))
        .all()
    ) if all_segment_ids else []
    segment_map: Dict[str, TranscriptSegment] = {s.id: s for s in db_segments}

    # 8. Assemble Verified Results
    results: List[MemorySearchResultItem] = []
    for cand in candidates:
        m = meeting_map.get(cand.meeting_id)
        m_title = m.title if m else "Unknown Meeting"
        m_date = None
        if m:
            if m.actual_start_time:
                m_date = m.actual_start_time.isoformat()
            elif m.scheduled_start_time:
                m_date = m.scheduled_start_time.isoformat()
            elif m.created_at:
                m_date = m.created_at.isoformat()

        # Derive start/end timestamps from verified database segments
        source_segs = [segment_map[sid] for sid in cand.segment_ids if sid in segment_map]
        if source_segs:
            derived_start = min(s.start_time for s in source_segs)
            derived_end = max(s.end_time for s in source_segs)
            verified_segment_ids = [s.id for s in source_segs]
        else:
            derived_start = cand.start_time
            derived_end = cand.end_time
            verified_segment_ids = cand.segment_ids

        results.append(
            MemorySearchResultItem(
                chunk_id=cand.chunk_id,
                meeting_id=cand.meeting_id,
                meeting_title=m_title,
                meeting_date=m_date,
                transcript_id=cand.transcript_id,
                segment_ids=verified_segment_ids,
                start_time=derived_start,
                end_time=derived_end,
                text=cand.text,
                language_codes=cand.language_codes,
                dominant_language=cand.dominant_language,
                rrf_score=cand.rrf_score,
                keyword_rank=cand.keyword_rank,
                semantic_rank=cand.semantic_rank,
            )
        )

    return MemorySearchResponse(
        query=clean_query,
        total_results=len(results),
        results=results,
    )


@router.get(
    "/conflicts",
    response_model=List[DecisionConflictItem],
    status_code=status.HTTP_200_OK,
    summary="Detect potential contradictions across active decisions in organizational memory",
    description="Identify potential contradictory stances or mutually exclusive technologies among active decisions.",
)
def get_memory_conflicts(
    subject: Optional[str] = Query(None, description="Optional keyword or subject filter"),
    meeting_id: Optional[str] = Query(None, description="Optional meeting UUID filter"),
    tenant: TenantContext = Depends(get_current_tenant_membership),
    db: Session = Depends(get_db),
    _rate_limit: None = Depends(memory_rate_limiter),
) -> List[DecisionConflictItem]:
    """Detect heuristic conflicts across active decisions for the authenticated organization."""
    org_id = str(tenant.organization.id)
    if meeting_id:
        m = db.query(Meeting).filter(Meeting.id == meeting_id, Meeting.organization_id == org_id).first()
        if not m:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Meeting not found or access denied.",
            )

    detected = ConflictDetector.detect_conflicts(
        db=db,
        organization_id=org_id,
        subject_filter=subject.strip() if subject else None,
    )

    if meeting_id:
        m_dec_ids = {
            d[0]
            for d in db.query(Decision.id).filter(Decision.meeting_id == meeting_id).all()
        }
        detected = [
            c for c in detected
            if c.decision_a_id in m_dec_ids or c.decision_b_id in m_dec_ids
        ]

    return detected
