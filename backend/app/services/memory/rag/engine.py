"""Master Cross-Meeting RAG Question-Answering Engine."""

import logging
import re
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from sqlalchemy import or_, func
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.identity import Organization
from app.models.meeting import Meeting
from app.services.memory.embedding import (
    MultilingualE5EmbeddingService,
    get_embedding_service,
)
from app.services.memory.exceptions import (
    ModelNotCachedError,
    EmbeddingModelLoadError,
    TenantSecurityError,
    RetrievalError,
)
from app.services.memory.rag.context import ContextBuilder
from app.services.memory.rag.providers import (
    AnswerProvider,
    DeterministicAnswerProvider,
    OllamaAnswerProvider,
    GeminiAnswerProvider,
)
from app.services.memory.structured_retrieval import (
    StructuredMemoryRetriever,
    StructuredMemoryContext,
)
from app.services.memory.rag.types import (
    RAGQueryRequest,
    RAGQueryResponse,
    Citation,
    EvidenceItem,
    RetrievalDiagnostics,
    AbstentionStatus,
    RawAnswerResult,
)
from app.services.memory.rag.verification import CitationVerifier
from app.services.memory.retrieval import search_hybrid_rrf
from app.services.memory.types import HybridSearchResult
from app.services.memory.rag.external_knowledge import (
    ExternalKnowledgeService,
    get_external_knowledge_service,
)

logger = logging.getLogger(__name__)


class RAGEngine:
    """Orchestrates the 9-stage cross-meeting RAG question-answering workflow."""

    def __init__(
        self,
        embedding_service: Optional[MultilingualE5EmbeddingService] = None,
        primary_provider: Optional[AnswerProvider] = None,
        context_builder: Optional[ContextBuilder] = None,
        citation_verifier: Optional[CitationVerifier] = None,
        structured_retriever: Optional[StructuredMemoryRetriever] = None,
        external_knowledge_service: Optional[ExternalKnowledgeService] = None,
    ):
        self.embedding_service = embedding_service
        self.primary_provider = primary_provider
        self.fallback_provider = DeterministicAnswerProvider()
        self.context_builder = context_builder or ContextBuilder()
        self.citation_verifier = citation_verifier or CitationVerifier()
        self.structured_retriever = structured_retriever or StructuredMemoryRetriever()
        self.external_knowledge_service = external_knowledge_service or get_external_knowledge_service()

    def query(
        self,
        db: Session,
        request: RAGQueryRequest,
    ) -> RAGQueryResponse:
        """Execute cross-meeting retrieval-augmented question answering."""
        # 1. Validate Question and Tenant Organization Context
        question = request.question.strip()
        if not question:
            raise ValueError("Query question cannot be empty.")

        org_id = request.organization_id.strip()
        if not org_id:
            raise TenantSecurityError("A valid organization_id is required for RAG queries.")

        org = db.query(Organization).filter(Organization.id == org_id).first()
        if not org:
            raise TenantSecurityError(f"Organization '{org_id}' does not exist.")

        # If meeting_id is supplied, verify tenant ownership
        if request.meeting_id:
            meeting = (
                db.query(Meeting)
                .filter(Meeting.id == request.meeting_id, Meeting.organization_id == org_id)
                .first()
            )
            if not meeting:
                # Meeting does not belong to organization -> strictly refuse cross-tenant access
                return RAGQueryResponse(
                    question=question,
                    answer="Meeting not found within your organization.",
                    citations=[],
                    abstention_status=AbstentionStatus.NO_EVIDENCE,
                    is_abstained=True,
                    diagnostics=RetrievalDiagnostics(
                        query=question,
                        provider_used="deterministic",
                        is_heuristic=True,
                    ),
                )

        # 2. Date Range Filtering Pre-computation (if requested)
        target_meeting_id = request.meeting_id
        matching_ids = set()
        if request.start_date or request.end_date:
            meeting_date_col = func.coalesce(Meeting.actual_start_time, Meeting.scheduled_start_time, Meeting.created_at)
            m_query = db.query(Meeting.id).filter(Meeting.organization_id == org_id)
            if request.start_date:
                m_query = m_query.filter(meeting_date_col >= request.start_date)
            if request.end_date:
                m_query = m_query.filter(meeting_date_col <= request.end_date)
            matching_ids = {r[0] for r in m_query.all()}
            if not matching_ids or (request.meeting_id and request.meeting_id not in matching_ids):
                return RAGQueryResponse(
                    question=question,
                    answer="No meetings found within the specified date range.",
                    citations=[],
                    abstention_status=AbstentionStatus.NO_EVIDENCE,
                    is_abstained=True,
                    diagnostics=RetrievalDiagnostics(
                        query=question,
                        provider_used="deterministic",
                        is_heuristic=True,
                    ),
                )

        # 3. Generate Query Vector using Multilingual E5
        service = self.embedding_service or get_embedding_service()
        try:
            query_vector = service.embed_query(question)
        except (ModelNotCachedError, EmbeddingModelLoadError) as mnc:
            logger.warning(f"Embedding model unavailable for RAG query: {mnc}")
            return RAGQueryResponse(
                question=question,
                answer="The multilingual embedding service is currently unavailable because model weights are not cached locally.",
                citations=[],
                abstention_status=AbstentionStatus.MODEL_UNAVAILABLE,
                is_abstained=True,
                diagnostics=RetrievalDiagnostics(
                    query=question,
                    provider_used="none",
                    is_heuristic=True,
                    fallback_warning="MODEL_NOT_CACHED",
                ),
            )
        except Exception as emb_exc:
            logger.exception(f"Unexpected query embedding failure: {emb_exc}")
            return RAGQueryResponse(
                question=question,
                answer="An error occurred while processing the question embedding.",
                citations=[],
                abstention_status=AbstentionStatus.MODEL_UNAVAILABLE,
                is_abstained=True,
                diagnostics=RetrievalDiagnostics(
                    query=question,
                    provider_used="none",
                    is_heuristic=True,
                    fallback_warning=str(type(emb_exc).__name__),
                ),
            )

        # 4. Hybrid Retrieval (FTS5 BM25 + NumPy Vector Cosine + RRF)
        try:
            candidates: List[HybridSearchResult] = search_hybrid_rrf(
                db=db,
                query=question,
                query_vector=query_vector,
                organization_id=org_id,
                meeting_id=target_meeting_id,
                project_id=request.project_id,
                limit=request.max_evidence_chunks * 2,
                k=60,
                expected_dim=service.dimension,
            )
        except Exception as ret_exc:
            logger.exception(f"Hybrid retrieval failed: {ret_exc}")
            return RAGQueryResponse(
                question=question,
                answer="An error occurred during retrieval from organizational memory.",
                citations=[],
                abstention_status=AbstentionStatus.RETRIEVAL_UNAVAILABLE,
                is_abstained=True,
                diagnostics=RetrievalDiagnostics(
                    query=question,
                    provider_used="none",
                    is_heuristic=True,
                    fallback_warning=str(type(ret_exc).__name__),
                ),
            )

        # Apply date range filtering to candidates if specified
        if (request.start_date or request.end_date) and not target_meeting_id:
            candidates = [c for c in candidates if c.meeting_id in matching_ids]

        # Verify candidate relevance:
        # 1. Lexical confirmation: if any candidate has an FTS5 keyword rank or contains query keywords in text.
        # 2. Cross-lingual queries: when searching in Indic scripts (Devanagari, Gujarati) against English text,
        #    literal string overlap is impossible; retain dense semantic candidates if semantic retrieval is confident.
        # 3. For queries with zero lexical overlap and low semantic confidence (e.g. completely out-of-domain
        #    questions or cross-tenant queries where the organization has no evidence), abstain with NO_EVIDENCE.
        keywords = self.fallback_provider._extract_query_keywords(question)
        if keywords and candidates:
            # Check for substantive word matches or stem matches across candidates
            has_lexical_match = any(
                c.keyword_rank is not None or self.fallback_provider._matches_keyword_or_stem(c.text, keywords)
                for c in candidates
            )
            if not has_lexical_match:
                # Detect cross-script Indic query via Unicode scripts (Devanagari \u0900-\u097F, Gujarati \u0A80-\u0AFF)
                is_indic_script = bool(re.search(r'[\u0900-\u097F\u0A80-\u0AFF]', question))
                top_semantic_score = max(
                    (c.semantic_score for c in candidates if c.semantic_score is not None),
                    default=0.0
                )
                # For cross-script Indic queries on Latin evidence, min_threshold is 0.75.
                # For Latin-script queries (English, Hinglish) with zero lexical overlap,
                # min_threshold is 0.78 to reject out-of-domain queries (which score up to 0.777).
                min_threshold = 0.75 if is_indic_script else 0.78
                if top_semantic_score >= min_threshold:
                    candidates = candidates[:request.max_evidence_chunks]
                else:
                    candidates = []

        # 4b. Structured Memory Retrieval (Decisions, Commitments, Questions, and Conflicts)
        structured_context: StructuredMemoryContext = self.structured_retriever.search_structured_memory(
            db=db,
            organization_id=org_id,
            query=question,
            meeting_id=target_meeting_id,
            project_id=request.project_id,
            start_date=request.start_date,
            end_date=request.end_date,
        )

        # 5. Handle Empty Retrieval
        if not candidates and not structured_context.has_structured_evidence:
            return RAGQueryResponse(
                question=question,
                answer="No relevant meeting discussions or decisions were found in your organization's recorded meetings for this question.",
                citations=[],
                abstention_status=AbstentionStatus.NO_EVIDENCE,
                is_abstained=True,
                diagnostics=RetrievalDiagnostics(
                    query=question,
                    total_candidates_retrieved=0,
                    evidence_items_count=0,
                    provider_used="deterministic",
                    is_heuristic=True,
                    structured_decisions_count=0,
                    structured_commitments_count=0,
                    structured_questions_count=0,
                    conflicts_detected_count=0,
                ),
            )

        # 6. Assemble Bounded Evidence Context
        evidence_items: List[EvidenceItem] = self.context_builder.assemble_evidence_items(
            db=db,
            candidates=candidates,
            organization_id=org_id,
            max_chunks=request.max_evidence_chunks,
        )

        if not evidence_items and not structured_context.has_structured_evidence:
            return RAGQueryResponse(
                question=question,
                answer="No valid meeting evidence could be assembled for this question.",
                citations=[],
                abstention_status=AbstentionStatus.NO_EVIDENCE,
                is_abstained=True,
                diagnostics=RetrievalDiagnostics(
                    query=question,
                    total_candidates_retrieved=len(candidates),
                    evidence_items_count=0,
                    provider_used="deterministic",
                    is_heuristic=True,
                    structured_decisions_count=0,
                    structured_commitments_count=0,
                    structured_questions_count=0,
                    conflicts_detected_count=0,
                ),
            )

        # Detect if query or retrieved meeting evidence mentions an external subject needing background context
        ext_subject = self.external_knowledge_service.extract_external_subject(question, evidence_items)
        external_knowledge = (
            self.external_knowledge_service.retrieve_external_knowledge(ext_subject)
            if ext_subject
            else None
        )

        # 7. Generate Answer via Answer Provider with Automatic Fallback
        provider = self.primary_provider
        fallback_warning: Optional[str] = None
        raw_result: Optional[RawAnswerResult] = None

        if provider:
            try:
                raw_result = provider.generate_answer(
                    question=question,
                    evidence_items=evidence_items,
                    output_language=request.output_language,
                    structured_context=structured_context,
                    external_knowledge=external_knowledge,
                )
            except Exception as prov_exc:
                logger.warning(f"Primary answer provider failed, falling back to deterministic: {prov_exc}")
                fallback_warning = f"Primary provider failed ({type(prov_exc).__name__}); deterministic fallback used."

        if not raw_result:
            raw_result = self.fallback_provider.generate_answer(
                question=question,
                evidence_items=evidence_items,
                output_language=request.output_language,
                structured_context=structured_context,
                external_knowledge=external_knowledge,
            )
            if fallback_warning:
                raw_result.fallback_warning = fallback_warning

        # 8. Verify Citations against Database Records
        # Citations are suppressed if the answer abstains with NO_EVIDENCE, INSUFFICIENT_EVIDENCE,
        # MODEL_UNAVAILABLE, or RETRIEVAL_UNAVAILABLE. Conflicting evidence and supported answers retain citations.
        verified_citations: List[Citation] = []
        is_strictly_abstained = raw_result.abstention_status in (
            AbstentionStatus.NO_EVIDENCE,
            AbstentionStatus.INSUFFICIENT_EVIDENCE,
            AbstentionStatus.MODEL_UNAVAILABLE,
            AbstentionStatus.RETRIEVAL_UNAVAILABLE,
        )

        if not is_strictly_abstained:
            verified_citations = self.citation_verifier.verify_citations(
                db=db,
                cited_chunk_ids=raw_result.cited_chunk_ids,
                retrieved_evidence=evidence_items,
                organization_id=org_id,
                structured_context=structured_context,
            )

        # 9. Formulate Final Response with Transparency Diagnostics
        is_abstained = raw_result.abstention_status in (
            AbstentionStatus.NO_EVIDENCE,
            AbstentionStatus.INSUFFICIENT_EVIDENCE,
            AbstentionStatus.CONFLICTING_EVIDENCE,
            AbstentionStatus.MODEL_UNAVAILABLE,
            AbstentionStatus.RETRIEVAL_UNAVAILABLE,
        )

        meetings_represented = sorted(list(
            {e.meeting_id for e in evidence_items}
            | {c.meeting_id for c in verified_citations}
            | {d.meeting_id for d in structured_context.decisions if getattr(d, "meeting_id", None)}
        ))
        diagnostics = RetrievalDiagnostics(
            query=question,
            total_candidates_retrieved=len(candidates),
            evidence_items_count=len(evidence_items),
            meetings_represented=meetings_represented,
            provider_used=raw_result.provider_used,
            is_heuristic=raw_result.is_heuristic,
            fallback_warning=raw_result.fallback_warning,
            structured_decisions_count=len(structured_context.decisions),
            structured_commitments_count=len(structured_context.commitments),
            structured_questions_count=len(structured_context.questions),
            conflicts_detected_count=len(structured_context.conflicts) + len(raw_result.transcript_conflicts),
        )

        # 9. Determine Source Classification (Meeting Evidence, External Knowledge, Combined, Insufficient Evidence)
        final_ext_knowledge = raw_result.external_knowledge or external_knowledge
        has_external = bool(final_ext_knowledge and final_ext_knowledge.is_available)
        has_meeting_evidence = bool(
            verified_citations
            or (structured_context and (structured_context.decisions or structured_context.commitments))
            or (evidence_items and not is_strictly_abstained)
        )

        if is_strictly_abstained or (not has_meeting_evidence and not has_external):
            answer_source_type = "INSUFFICIENT_EVIDENCE"
        elif has_meeting_evidence and has_external:
            answer_source_type = "COMBINED"
        elif has_external and not has_meeting_evidence:
            answer_source_type = "EXTERNAL_KNOWLEDGE"
        else:
            answer_source_type = "MEETING_EVIDENCE"

        return RAGQueryResponse(
            question=question,
            answer=raw_result.answer_text,
            citations=verified_citations,
            abstention_status=raw_result.abstention_status,
            is_abstained=is_abstained,
            diagnostics=diagnostics,
            structured_decisions=[d.model_dump() for d in structured_context.decisions],
            structured_commitments=[c.model_dump() for c in structured_context.commitments],
            structured_questions=[q.model_dump() for q in structured_context.questions],
            detected_conflicts=structured_context.conflicts,
            transcript_conflicts=raw_result.transcript_conflicts,
            external_knowledge=final_ext_knowledge,
            answer_source_type=answer_source_type,
        )
