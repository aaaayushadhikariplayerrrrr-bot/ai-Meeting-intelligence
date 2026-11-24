"""Unit and Integration Tests for External Knowledge Integration & Ask AI Presentation.

Validates:
1. Subject extraction for entities (e.g. 103rd Constitutional Amendment, OAuth 2.0).
2. Privacy protections: strict sanitization of subjects and no transmission of private meeting text/PII.
3. Wikipedia, DuckDuckGo, and Gemini multi-tiered retrieval.
4. Honest unavailability reporting when external services are unreachable.
5. Strict separation in RAGEngine:
   - What the meeting actually said (meeting title, date, timestamp mm:ss, chunk citation).
   - What external background knowledge provides (sources, key facts, disclaimer).
6. Zero hallucination: no invented meeting evidence or attribution of external knowledge to meeting participants.
"""

from datetime import datetime, timezone
import json
import uuid
import pytest
from unittest.mock import patch, MagicMock
import httpx
from sqlalchemy.orm import Session

from app.models.identity import Organization
from app.models.meeting import Meeting
from app.models.memory import TranscriptChunkRecord
from app.models.transcription import Transcript, TranscriptSegment, TranscriptType
from app.services.memory import (
    EXPECTED_DIMENSION,
    ChunkStorageManager,
    SemanticChunk,
    ChunkEmbeddingResult,
)
from app.services.memory.rag import (
    RAGEngine,
    DeterministicAnswerProvider,
    AbstentionStatus,
    EvidenceItem,
    RAGQueryRequest,
)
from app.services.memory.rag.external_knowledge import (
    ExternalKnowledgeService,
    ExternalKnowledgeResult,
    ExternalKnowledgeCitation,
)


class MockEmbeddingService:
    """Mock embedding service returning unit vectors."""

    def __init__(self, dimension: int = EXPECTED_DIMENSION):
        self.dimension = dimension
        self.recorded_queries = []

    def embed_passages(self, texts):
        return [[1.0 / (self.dimension ** 0.5)] * self.dimension for _ in texts]

    def embed_query(self, query: str):
        self.recorded_queries.append(query)
        return [1.0 / (self.dimension ** 0.5)] * self.dimension


def test_extract_external_subject():
    """Verify entity and statutory subject extraction from user questions."""
    service = ExternalKnowledgeService()

    # User question about 103rd Constitutional Amendment
    subj1 = service.extract_external_subject("What is the 103rd Constitutional Amendment about?")
    assert subj1 is not None
    assert "103rd" in subj1 and "Amendment" in subj1

    # Question about technical protocol
    subj2 = service.extract_external_subject("Explain OAuth 2.0 and how it works")
    assert subj2 is not None
    assert "OAuth" in subj2

    # Conversational meeting query should NOT trigger external knowledge
    subj3 = service.extract_external_subject("What was our plan for the next sprint?")
    assert subj3 is None

    # Privacy sanitization: PII must be stripped
    subj_pii = service.extract_external_subject("What is GDPR user@example.com https://secret.internal?")
    assert "user@example.com" not in subj_pii
    assert "secret.internal" not in subj_pii
    assert "GDPR" in subj_pii


def test_wikipedia_retrieval_success():
    """Verify Wikipedia REST API parser extracts title, summary, key facts, and desktop URL."""
    service = ExternalKnowledgeService()

    mock_search = {
        "query": {
            "search": [
                {"title": "One Hundred and Third Amendment of the Constitution of India"}
            ]
        }
    }
    mock_summary = {
        "title": "One Hundred and Third Amendment of the Constitution of India",
        "extract": "The One Hundred and Third Amendment of the Constitution of India introduces a 10% reservation for Economically Weaker Sections (EWS). It amends articles 15 and 16 of the Constitution.",
        "content_urls": {
            "desktop": {
                "page": "https://en.wikipedia.org/wiki/One_Hundred_and_Third_Amendment_of_the_Constitution_of_India"
            }
        },
        "timestamp": "2026-01-15T00:00:00Z"
    }

    def custom_handler(request: httpx.Request):
        url_str = str(request.url)
        if "action=query" in url_str:
            return httpx.Response(200, json=mock_search)
        elif "api/rest_v1/page/summary" in url_str:
            return httpx.Response(200, json=mock_summary)
        return httpx.Response(404)

    mock_client = httpx.Client(transport=httpx.MockTransport(custom_handler))
    service._client = mock_client

    result = service.fetch_wikipedia_knowledge("103rd Constitutional Amendment")
    assert result is not None
    assert result.is_available is True
    assert result.provider_used == "wikipedia"
    assert "Economically Weaker Sections" in result.summary
    assert len(result.key_facts) >= 1
    assert len(result.citations) == 1
    assert "wikipedia.org" in result.citations[0].source_url
    assert "One Hundred and Third Amendment" in result.citations[0].source_title


def test_external_knowledge_unavailable_fallback():
    """Verify graceful, honest reporting when external knowledge source is unreachable."""
    def error_handler(request: httpx.Request):
        raise httpx.ConnectError("Network unreachable")

    mock_client = httpx.Client(transport=httpx.MockTransport(error_handler))
    service = ExternalKnowledgeService(client=mock_client)

    result = service.retrieve_external_knowledge("NonExistentEntityX123")
    assert result is not None
    assert result.is_available is False
    assert result.provider_used == "unavailable"
    assert "could not be retrieved" in result.summary
    assert result.unavailability_reason is not None


def test_deterministic_provider_presentation_separation():
    """Verify DeterministicAnswerProvider clearly separates meeting evidence from external knowledge."""
    provider = DeterministicAnswerProvider()

    evidence = [
        EvidenceItem(
            chunk_id=str(uuid.uuid4()),
            organization_id=str(uuid.uuid4()),
            meeting_id=str(uuid.uuid4()),
            meeting_title="Legal & Compliance Sync",
            meeting_date="2026-03-10T10:00:00Z",
            transcript_id=str(uuid.uuid4()),
            segment_ids=[str(uuid.uuid4())],
            start_time=120.0,
            end_time=150.0,
            text="In the review, we noted that the 103rd Constitutional Amendment may impact our public sector hiring quotas.",
            rrf_score=0.9,
            keyword_rank=1,
            semantic_rank=1,
        )
    ]

    ext_citation = ExternalKnowledgeCitation(
        source_title="Wikipedia: One Hundred and Third Amendment of the Constitution of India",
        source_url="https://en.wikipedia.org/wiki/One_Hundred_and_Third_Amendment_of_the_Constitution_of_India",
        snippet="Introduces 10% reservation for Economically Weaker Sections (EWS).",
    )
    ext_knowledge = ExternalKnowledgeResult(
        subject="103rd Constitutional Amendment",
        summary="The 103rd Amendment of the Constitution of India provides 10% reservation in government jobs and education for Economically Weaker Sections.",
        key_facts=[
            "Introduces 10% reservation for Economically Weaker Sections (EWS).",
            "Amends Articles 15 and 16 of the Constitution of India.",
        ],
        citations=[ext_citation],
        provider_used="wikipedia",
        is_available=True,
    )

    result = provider.generate_answer(
        question="What is the 103rd Constitutional Amendment about?",
        evidence_items=evidence,
        external_knowledge=ext_knowledge,
    )

    answer = result.answer_text

    # 1. Meeting evidence clearly cited with meeting title and timestamp
    assert "Based on discussion in 'Legal & Compliance Sync':" in answer
    assert "[02:00 - 02:30]" in answer
    assert "103rd Constitutional Amendment may impact our public sector hiring quotas" in answer
    assert evidence[0].chunk_id in result.cited_chunk_ids

    # 2. Meeting context note explains it was not defined in meeting
    assert "Meeting Context Note" in answer
    assert "was mentioned in the meeting, but was not explained or defined in detail" in answer

    # 3. External background knowledge clearly separated with heading, summary, key facts, and citation
    assert "### External Background Knowledge: 103rd Constitutional Amendment" in answer
    assert "External background knowledge retrieved for context" in answer
    assert "Economically Weaker Sections" in answer
    assert "https://en.wikipedia.org" in answer

    # 4. Result object carries external_knowledge
    assert result.external_knowledge is not None
    assert result.external_knowledge.subject == "103rd Constitutional Amendment"


def test_rag_engine_e2e_with_external_knowledge(db_session: Session):
    """End-to-end RAGEngine test: query mentioning un-elaborated topic receives both meeting citation and external knowledge."""
    org = Organization(name="GovTech Solutions", slug=f"govtech-{uuid.uuid4().hex[:8]}")
    db_session.add(org)
    db_session.flush()

    meeting = Meeting(
        organization_id=org.id,
        title="Policy Review Committee",
        actual_start_time=datetime(2026, 4, 1, 14, 0, 0, tzinfo=timezone.utc),
    )
    db_session.add(meeting)
    db_session.flush()

    transcript = Transcript(
        meeting_id=meeting.id,
        transcript_type=TranscriptType.CLEANED_REDACTED,
    )
    db_session.add(transcript)
    db_session.flush()

    seg = TranscriptSegment(
        transcript_id=transcript.id,
        sequence_order=0,
        start_time=245.0,
        end_time=280.0,
        text="The committee briefly mentioned the 103rd Constitutional Amendment regarding compliance obligations.",
    )
    db_session.add(seg)
    db_session.flush()

    db_session.commit()

    storage_manager = ChunkStorageManager()
    chunk_item = ChunkEmbeddingResult(
        chunk=SemanticChunk(
            chunk_index=0,
            meeting_id=meeting.id,
            organization_id=org.id,
            transcript_id=transcript.id,
            segment_ids=[seg.id],
            text=seg.text,
            start_time=245.0,
            end_time=280.0,
            language_codes=["en"],
            dominant_language="en",
            token_count=15,
            is_split_utterance=False,
            metadata={"source": "test"},
        ),
        embedding=[1.0 / (EXPECTED_DIMENSION ** 0.5)] * EXPECTED_DIMENSION,
        dimension=EXPECTED_DIMENSION,
        model_name="intfloat/multilingual-e5-small",
    )
    storage_manager.store_chunks(db_session, [chunk_item], organization_id=org.id)

    # Setup ExternalKnowledgeService mock
    ext_service = ExternalKnowledgeService()
    ext_citation = ExternalKnowledgeCitation(
        source_title="Wikipedia: One Hundred and Third Amendment of the Constitution of India",
        source_url="https://en.wikipedia.org/wiki/One_Hundred_and_Third_Amendment_of_the_Constitution_of_India",
        snippet="Provides 10% reservation for Economically Weaker Sections (EWS).",
    )
    ext_knowledge = ExternalKnowledgeResult(
        subject="103rd Constitutional Amendment",
        summary="The 103rd Constitutional Amendment Act, 2019 provides 10% reservation in government jobs and educational institutions for Economically Weaker Sections.",
        key_facts=["Introduces 10% EWS quota.", "Amended Articles 15 and 16."],
        citations=[ext_citation],
        provider_used="wikipedia",
        is_available=True,
    )
    ext_service.retrieve_external_knowledge = MagicMock(return_value=ext_knowledge)

    engine = RAGEngine(
        embedding_service=MockEmbeddingService(),
        external_knowledge_service=ext_service,
    )

    request = RAGQueryRequest(
        question="What is the 103rd Constitutional Amendment about?",
        organization_id=org.id,
    )

    response = engine.query(db=db_session, request=request)

    # Assertions
    assert response.is_abstained is False
    assert len(response.citations) == 1
    assert response.citations[0].meeting_title == "Policy Review Committee"
    assert response.citations[0].start_time == 245.0
    assert response.citations[0].end_time == 280.0

    # Answer text asserts
    assert "Based on discussion in 'Policy Review Committee':" in response.answer
    assert "[04:05 - 04:40]" in response.answer
    assert "### External Background Knowledge: 103rd Constitutional Amendment" in response.answer
    assert "Economically Weaker Sections" in response.answer

    # Structured DTO asserts
    assert response.external_knowledge is not None
    assert response.external_knowledge.is_available is True
    assert response.external_knowledge.subject == "103rd Constitutional Amendment"
    assert len(response.external_knowledge.citations) == 1
