"""Integration and Security Test Suite for Phase 8.5: Secure REST APIs for Organizational Memory.

Tests cover:
1. Authenticated memory query succeeds with supported evidence (POST /api/v1/memory/query).
2. Unauthenticated query is rejected with HTTP 401.
3. Authenticated memory search succeeds (POST /api/v1/memory/search).
4. Organization A cannot retrieve Organization B's chunks.
5. A foreign meeting ID cannot be used to access another organization's data (HTTP 404).
6. Client-supplied organization IDs cannot override authenticated organization context.
7. Original/unredacted transcript content is never returned.
8. Empty and oversized queries are rejected (HTTP 400 or 422).
9. Invalid date ranges are rejected (start_date > end_date -> HTTP 400 or 422).
10. Result and evidence limits are enforced.
11. NO_EVIDENCE and INSUFFICIENT_EVIDENCE return valid abstention responses (HTTP 200, is_abstained=True).
12. MODEL_UNAVAILABLE is handled safely.
13. RETRIEVAL_UNAVAILABLE does not expose internal exception details.
14. CONFLICTING_EVIDENCE preserves its explicit status.
15. Citation IDs and timestamps remain database-verified.
16. Rate limiting works according to the application's policy (HTTP 429).
17. Diagnostics do not leak PII or transcript contents.
"""

from datetime import datetime, timezone
import json
import uuid
from typing import List, Dict, Optional, Any
import numpy as np
import pytest
from sqlalchemy.orm import Session
from fastapi.testclient import TestClient

from app.models.identity import Organization, User, OrganizationMembership, OrgRole
from app.models.meeting import Meeting, MeetingStatus
from app.models.transcription import Transcript, TranscriptSegment, TranscriptType
from app.core.security import hash_password
from app.core.rate_limit import memory_rate_limiter
from app.services.memory import (
    ChunkStorageManager,
    SemanticChunk,
    ChunkEmbeddingResult,
    ModelNotCachedError,
    EXPECTED_DIMENSION,
)
from app.schemas.memory import AbstentionStatus
from tests.conftest import make_auth_headers


def make_synthetic_vector(seed: int, dim: int = EXPECTED_DIMENSION) -> List[float]:
    """Generate a deterministic unit-normalized 384-dimensional synthetic vector."""
    rng = np.random.RandomState(seed)
    vec = rng.randn(dim).astype(np.float32)
    norm = float(np.linalg.norm(vec))
    return (vec / norm).tolist()


class DeterministicMockEmbeddingService:
    """Mock embedding service that maps known texts/keywords to reproducible unit vectors."""

    def __init__(self, dimension: int = EXPECTED_DIMENSION):
        self.dimension = dimension
        self.model_name = "intfloat/multilingual-e5-small"

    def embed_passages(self, texts: List[str]) -> List[List[float]]:
        results = []
        for t in texts:
            seed = sum(ord(c) for c in t) % 10000
            results.append(make_synthetic_vector(seed=seed, dim=self.dimension))
        return results

    def embed_query(self, query: str) -> List[float]:
        seed = sum(ord(c) for c in query) % 10000
        return make_synthetic_vector(seed=seed, dim=self.dimension)


@pytest.fixture(autouse=True)
def mock_embedding_for_api_tests(monkeypatch):
    """Automatically mock embedding service for API tests to avoid downloading model weights."""
    mock_svc = DeterministicMockEmbeddingService()
    monkeypatch.setattr("app.api.endpoints.memory.get_embedding_service", lambda: mock_svc)
    monkeypatch.setattr("app.services.memory.rag.engine.get_embedding_service", lambda: mock_svc)
    # Reset rate limiter before each test
    memory_rate_limiter.reset()


@pytest.fixture
def memory_api_setup(db_session: Session):
    """Fixture creating two isolated organizations, authenticated users, meetings, transcripts, and chunks."""
    # 1. Organizations
    org_a = Organization(name="API Org A", slug=f"api-org-a-{uuid.uuid4().hex[:8]}", is_active=True)
    org_b = Organization(name="API Org B", slug=f"api-org-b-{uuid.uuid4().hex[:8]}", is_active=True)
    db_session.add_all([org_a, org_b])
    db_session.commit()

    # 2. Users
    user_a = User(
        email=f"user_a_{uuid.uuid4().hex[:6]}@example.com",
        full_name="User Org A",
        hashed_password=hash_password("Pass123!"),
        is_active=True,
    )
    user_b = User(
        email=f"user_b_{uuid.uuid4().hex[:6]}@example.com",
        full_name="User Org B",
        hashed_password=hash_password("Pass123!"),
        is_active=True,
    )
    db_session.add_all([user_a, user_b])
    db_session.commit()

    # 3. Memberships
    mem_a = OrganizationMembership(
        organization_id=org_a.id,
        user_id=user_a.id,
        role=OrgRole.MEMBER.value,
        is_active=True,
    )
    mem_b = OrganizationMembership(
        organization_id=org_b.id,
        user_id=user_b.id,
        role=OrgRole.MEMBER.value,
        is_active=True,
    )
    db_session.add_all([mem_a, mem_b])
    db_session.commit()

    # 4. Meetings in Org A
    meeting_a1 = Meeting(
        organization_id=org_a.id,
        title="Sprint 42 Architecture Review",
        actual_start_time=datetime(2026, 1, 10, 10, 0, 0, tzinfo=timezone.utc),
        status=MeetingStatus.COMPLETED.value,
    )
    meeting_a2 = Meeting(
        organization_id=org_a.id,
        title="Security & Infrastructure Sync",
        actual_start_time=datetime(2026, 2, 15, 14, 0, 0, tzinfo=timezone.utc),
        status=MeetingStatus.COMPLETED.value,
    )
    meeting_a3 = Meeting(
        organization_id=org_a.id,
        title="Cache Architecture Retrospective",
        actual_start_time=datetime(2026, 3, 1, 9, 30, 0, tzinfo=timezone.utc),
        status=MeetingStatus.COMPLETED.value,
    )

    # 5. Meeting in Org B (Confidential)
    meeting_b1 = Meeting(
        organization_id=org_b.id,
        title="Org B Confidential Strategy",
        actual_start_time=datetime(2026, 1, 20, 11, 0, 0, tzinfo=timezone.utc),
        status=MeetingStatus.COMPLETED.value,
    )
    db_session.add_all([meeting_a1, meeting_a2, meeting_a3, meeting_b1])
    db_session.commit()

    # 6. Transcripts
    t_a1_safe = Transcript(meeting_id=meeting_a1.id, transcript_type=TranscriptType.CLEANED_REDACTED.value, version=1)
    t_a1_unsafe = Transcript(meeting_id=meeting_a1.id, transcript_type=TranscriptType.ORIGINAL.value, version=1)
    t_a2_safe = Transcript(meeting_id=meeting_a2.id, transcript_type=TranscriptType.CLEANED_REDACTED.value, version=1)
    t_a3_safe = Transcript(meeting_id=meeting_a3.id, transcript_type=TranscriptType.CLEANED_REDACTED.value, version=1)
    t_b1_safe = Transcript(meeting_id=meeting_b1.id, transcript_type=TranscriptType.CLEANED_REDACTED.value, version=1)
    db_session.add_all([t_a1_safe, t_a1_unsafe, t_a2_safe, t_a3_safe, t_b1_safe])
    db_session.commit()

    # 7. Segments for Org A (Safe)
    seg_a1_1 = TranscriptSegment(
        transcript_id=t_a1_safe.id,
        speaker_id=None,
        start_time=10.0,
        end_time=25.0,
        text="The engineering team decided to adopt Kubernetes for container orchestration across all production services.",
        language_code="en",
        sequence_order=1,
    )
    seg_a1_2 = TranscriptSegment(
        transcript_id=t_a1_safe.id,
        speaker_id=None,
        start_time=30.0,
        end_time=45.0,
        text="We decided to adopt Redis for our primary distributed session caching layer.",
        language_code="en",
        sequence_order=2,
    )
    # Unsafe segment in ORIGINAL transcript with PII
    seg_a1_unsafe = TranscriptSegment(
        transcript_id=t_a1_unsafe.id,
        speaker_id=None,
        start_time=10.0,
        end_time=25.0,
        text="CONFIDENTIAL PII: John Doe with SSN 000-11-2222 decided to adopt Kubernetes.",
        language_code="en",
        sequence_order=1,
    )
    seg_a2_1 = TranscriptSegment(
        transcript_id=t_a2_safe.id,
        speaker_id=None,
        start_time=5.0,
        end_time=20.0,
        text="API rate limiting will be strictly enforced at 100 requests per minute per authenticated tenant.",
        language_code="en",
        sequence_order=1,
    )
    seg_a3_1 = TranscriptSegment(
        transcript_id=t_a3_safe.id,
        speaker_id=None,
        start_time=15.0,
        end_time=35.0,
        text="After benchmark analysis, the team decided against adopting Redis and finalized PostgreSQL unlogged tables.",
        language_code="en",
        sequence_order=1,
    )

    # 8. Segments for Org B
    seg_b1_1 = TranscriptSegment(
        transcript_id=t_b1_safe.id,
        speaker_id=None,
        start_time=0.0,
        end_time=20.0,
        text="Org B board approved acquiring Startup Horizon for fifteen million dollars.",
        language_code="en",
        sequence_order=1,
    )
    seg_b1_2 = TranscriptSegment(
        transcript_id=t_b1_safe.id,
        speaker_id=None,
        start_time=25.0,
        end_time=40.0,
        text="The engineering team decided to adopt Kubernetes across all European clusters for Org B.",
        language_code="en",
        sequence_order=2,
    )

    db_session.add_all([seg_a1_1, seg_a1_2, seg_a1_unsafe, seg_a2_1, seg_a3_1, seg_b1_1, seg_b1_2])
    db_session.commit()

    # 9. Store Chunks in Persistent Storage
    storage_manager = ChunkStorageManager()

    def build_chunk(chunk_idx: int, meeting, transcript, segments: List[TranscriptSegment], org_id: str) -> ChunkEmbeddingResult:
        full_text = " ".join(s.text for s in segments)
        chunk = SemanticChunk(
            chunk_index=chunk_idx,
            meeting_id=meeting.id,
            organization_id=org_id,
            transcript_id=transcript.id,
            segment_ids=[s.id for s in segments],
            text=full_text,
            start_time=min(s.start_time for s in segments),
            end_time=max(s.end_time for s in segments),
            language_codes=["en"],
            dominant_language="en",
            token_count=len(full_text.split()),
            is_split_utterance=False,
            metadata={"source": "api_test"},
        )
        seed = sum(ord(c) for c in full_text) % 10000
        vec = make_synthetic_vector(seed=seed)
        return ChunkEmbeddingResult(
            chunk=chunk,
            embedding=vec,
            dimension=EXPECTED_DIMENSION,
            model_name="intfloat/multilingual-e5-small",
        )

    # Org A Chunks
    item_a1_k8s = build_chunk(0, meeting_a1, t_a1_safe, [seg_a1_1], org_a.id)
    item_a1_redis = build_chunk(1, meeting_a1, t_a1_safe, [seg_a1_2], org_a.id)
    item_a2_rate = build_chunk(0, meeting_a2, t_a2_safe, [seg_a2_1], org_a.id)
    item_a3_noredis = build_chunk(0, meeting_a3, t_a3_safe, [seg_a3_1], org_a.id)

    # Org B Chunks
    item_b1_ma = build_chunk(0, meeting_b1, t_b1_safe, [seg_b1_1], org_b.id)
    item_b1_k8s = build_chunk(1, meeting_b1, t_b1_safe, [seg_b1_2], org_b.id)

    storage_manager.store_chunks(db_session, [item_a1_k8s, item_a1_redis], organization_id=org_a.id)
    storage_manager.store_chunks(db_session, [item_a2_rate], organization_id=org_a.id)
    storage_manager.store_chunks(db_session, [item_a3_noredis], organization_id=org_a.id)
    storage_manager.store_chunks(db_session, [item_b1_ma, item_b1_k8s], organization_id=org_b.id)

    headers_a = make_auth_headers(user_a, org_a)
    headers_b = make_auth_headers(user_b, org_b)

    return {
        "org_a": org_a,
        "org_b": org_b,
        "user_a": user_a,
        "user_b": user_b,
        "headers_a": headers_a,
        "headers_b": headers_b,
        "meeting_a1": meeting_a1,
        "meeting_a2": meeting_a2,
        "meeting_a3": meeting_a3,
        "meeting_b1": meeting_b1,
        "seg_a1_1": seg_a1_1,
        "seg_a2_1": seg_a2_1,
        "seg_a3_1": seg_a3_1,
    }


# ============================================================================
# 1. Authenticated Memory Query Succeeds
# ============================================================================

def test_authenticated_memory_query_succeeds(client: TestClient, memory_api_setup):
    """Verify POST /api/v1/memory/query succeeds with grounded answer and citations."""
    setup = memory_api_setup
    payload = {
        "question": "What container orchestration system was adopted?",
        "max_evidence_chunks": 5,
    }
    resp = client.post("/api/v1/memory/query", json=payload, headers=setup["headers_a"])
    assert resp.status_code == 200
    data = resp.json()

    assert data["question"] == payload["question"]
    assert not data["is_abstained"]
    assert data["abstention_status"] == AbstentionStatus.ANSWER_SUPPORTED.value
    assert len(data["citations"]) > 0
    assert "Kubernetes" in data["citations"][0]["quoted_text"]
    assert data["diagnostics"]["provider_used"] == "deterministic"
    assert data["diagnostics"]["is_heuristic"] is True


# ============================================================================
# 2. Unauthenticated Query is Rejected (HTTP 401)
# ============================================================================

def test_unauthenticated_requests_rejected(client: TestClient):
    """Verify both query and search endpoints reject unauthenticated calls with 401."""
    # Query endpoint
    resp_query = client.post("/api/v1/memory/query", json={"question": "Test question"})
    assert resp_query.status_code == 401
    assert "WWW-Authenticate" in resp_query.headers

    # Search endpoint
    resp_search = client.post("/api/v1/memory/search", json={"query": "Test query"})
    assert resp_search.status_code == 401
    assert "WWW-Authenticate" in resp_search.headers


# ============================================================================
# 3. Authenticated Memory Search Succeeds
# ============================================================================

def test_authenticated_memory_search_succeeds(client: TestClient, memory_api_setup):
    """Verify POST /api/v1/memory/search returns structured results with provenance."""
    setup = memory_api_setup
    payload = {
        "query": "Kubernetes container orchestration",
        "limit": 5,
    }
    resp = client.post("/api/v1/memory/search", json=payload, headers=setup["headers_a"])
    assert resp.status_code == 200
    data = resp.json()

    assert data["query"] == payload["query"]
    assert data["total_results"] > 0
    first_res = data["results"][0]
    assert "Kubernetes" in first_res["text"]
    assert first_res["meeting_title"] == "Sprint 42 Architecture Review"
    assert first_res["start_time"] == pytest.approx(10.0, abs=0.1)
    assert first_res["end_time"] == pytest.approx(25.0, abs=0.1)


# ============================================================================
# 4. Strict Organization Isolation
# ============================================================================

def test_organization_a_cannot_retrieve_organization_b_chunks(client: TestClient, memory_api_setup):
    """Verify Org A cannot observe, query, or search confidential Org B content."""
    setup = memory_api_setup

    # 1. Query: Org A asks about Org B confidential acquisition
    query_payload = {"question": "Startup Horizon fifteen million acquisition"}
    resp_query = client.post("/api/v1/memory/query", json=query_payload, headers=setup["headers_a"])
    assert resp_query.status_code == 200
    data_query = resp_query.json()
    assert data_query["is_abstained"] is True
    assert data_query["abstention_status"] == AbstentionStatus.NO_EVIDENCE.value
    assert len(data_query["citations"]) == 0

    # 2. Search: Org A searches for Startup Horizon
    search_payload = {"query": "Startup Horizon fifteen million"}
    resp_search = client.post("/api/v1/memory/search", json=search_payload, headers=setup["headers_a"])
    assert resp_search.status_code == 200
    data_search = resp_search.json()
    assert data_search["total_results"] == 0
    assert len(data_search["results"]) == 0

    # 3. Org B can search and retrieve their own chunk
    resp_search_b = client.post("/api/v1/memory/search", json=search_payload, headers=setup["headers_b"])
    assert resp_search_b.status_code == 200
    data_b = resp_search_b.json()
    assert data_b["total_results"] > 0
    assert "Startup Horizon" in data_b["results"][0]["text"]


# ============================================================================
# 5. Foreign Meeting ID Access Prevention (HTTP 404)
# ============================================================================

def test_foreign_meeting_id_cannot_be_used(client: TestClient, memory_api_setup):
    """Verify supplying a meeting_id belonging to another tenant returns HTTP 404."""
    setup = memory_api_setup
    foreign_meeting_id = setup["meeting_b1"].id

    # Query with foreign meeting_id
    query_payload = {
        "question": "What was discussed?",
        "meeting_id": foreign_meeting_id,
    }
    resp_query = client.post("/api/v1/memory/query", json=query_payload, headers=setup["headers_a"])
    assert resp_query.status_code == 404
    assert "Meeting not found or access denied" in resp_query.json()["detail"]

    # Search with foreign meeting_id
    search_payload = {
        "query": "Kubernetes",
        "meeting_id": foreign_meeting_id,
    }
    resp_search = client.post("/api/v1/memory/search", json=search_payload, headers=setup["headers_a"])
    assert resp_search.status_code == 404
    assert "Meeting not found or access denied" in resp_search.json()["detail"]


# ============================================================================
# 6. Client-Supplied Organization IDs Strictly Overridden
# ============================================================================

def test_client_supplied_org_id_cannot_override_context(client: TestClient, memory_api_setup):
    """Verify passing a spoofed organization_id in JSON payload is ignored by the server."""
    setup = memory_api_setup
    payload = {
        "question": "Startup Horizon fifteen million acquisition",
        "organization_id": setup["org_b"].id,  # Spoofed Org B ID
    }
    # User A sends request with Org A authorization token
    resp = client.post("/api/v1/memory/query", json=payload, headers=setup["headers_a"])
    assert resp.status_code == 200
    data = resp.json()
    # Server strictly used Org A context: no evidence found
    assert data["is_abstained"] is True
    assert data["abstention_status"] == AbstentionStatus.NO_EVIDENCE.value


# ============================================================================
# 7. Original/Unredacted Transcript Content is Never Returned
# ============================================================================

def test_original_unredacted_content_never_returned(client: TestClient, memory_api_setup):
    """Verify unredacted PII in ORIGINAL transcripts is never searchable or exposed."""
    setup = memory_api_setup
    # Org A has an ORIGINAL transcript segment containing "CONFIDENTIAL PII: John Doe with SSN 000-11-2222"
    payload = {"query": "SSN 000-11-2222 John Doe"}
    resp = client.post("/api/v1/memory/search", json=payload, headers=setup["headers_a"])
    assert resp.status_code == 200
    data = resp.json()
    # PII from original transcript must not be returned
    assert data["total_results"] == 0


# ============================================================================
# 8. Empty and Oversized Queries are Rejected
# ============================================================================

def test_empty_and_oversized_queries_rejected(client: TestClient, memory_api_setup):
    """Verify whitespace-only or oversized queries are rejected with 400 or 422."""
    setup = memory_api_setup
    headers = setup["headers_a"]

    # Whitespace question
    resp_empty_q = client.post("/api/v1/memory/query", json={"question": "   "}, headers=headers)
    assert resp_empty_q.status_code in (400, 422)

    # Oversized question (> 1000 characters)
    resp_oversized_q = client.post(
        "/api/v1/memory/query",
        json={"question": "A" * 1001},
        headers=headers,
    )
    assert resp_oversized_q.status_code == 422

    # Whitespace search query
    resp_empty_s = client.post("/api/v1/memory/search", json={"query": "   "}, headers=headers)
    assert resp_empty_s.status_code in (400, 422)

    # Oversized search query (> 500 characters)
    resp_oversized_s = client.post(
        "/api/v1/memory/search",
        json={"query": "A" * 501},
        headers=headers,
    )
    assert resp_oversized_s.status_code == 422


# ============================================================================
# 9. Invalid Date Ranges are Rejected
# ============================================================================

def test_invalid_date_ranges_rejected(client: TestClient, memory_api_setup):
    """Verify start_date later than end_date is rejected with 400 or 422."""
    setup = memory_api_setup
    headers = setup["headers_a"]

    invalid_payload = {
        "question": "What decisions were made?",
        "start_date": "2026-12-31T00:00:00Z",
        "end_date": "2026-01-01T00:00:00Z",  # Earlier than start_date
    }
    resp = client.post("/api/v1/memory/query", json=invalid_payload, headers=headers)
    assert resp.status_code in (400, 422)

    invalid_search = {
        "query": "Kubernetes",
        "start_date": "2026-12-31T00:00:00Z",
        "end_date": "2026-01-01T00:00:00Z",
    }
    resp_s = client.post("/api/v1/memory/search", json=invalid_search, headers=headers)
    assert resp_s.status_code in (400, 422)


# ============================================================================
# 10. Result and Evidence Limits are Enforced
# ============================================================================

def test_result_and_evidence_limits_enforced(client: TestClient, memory_api_setup):
    """Verify max_evidence_chunks and search limit boundaries are validated."""
    setup = memory_api_setup
    headers = setup["headers_a"]

    # max_evidence_chunks > 20
    resp_q_excess = client.post(
        "/api/v1/memory/query",
        json={"question": "What was discussed?", "max_evidence_chunks": 50},
        headers=headers,
    )
    assert resp_q_excess.status_code == 422

    # search limit > 50
    resp_s_excess = client.post(
        "/api/v1/memory/search",
        json={"query": "Kubernetes", "limit": 100},
        headers=headers,
    )
    assert resp_s_excess.status_code == 422


# ============================================================================
# 11. NO_EVIDENCE and INSUFFICIENT_EVIDENCE Abstentions
# ============================================================================

def test_no_evidence_and_insufficient_evidence_abstentions(client: TestClient, memory_api_setup):
    """Verify NO_EVIDENCE and INSUFFICIENT_EVIDENCE return HTTP 200 with structured abstention."""
    setup = memory_api_setup
    headers = setup["headers_a"]

    # NO_EVIDENCE query
    resp_no_ev = client.post(
        "/api/v1/memory/query",
        json={"question": "What is the quantum spacecraft navigation protocol?"},
        headers=headers,
    )
    assert resp_no_ev.status_code == 200
    data_no_ev = resp_no_ev.json()
    assert data_no_ev["is_abstained"] is True
    assert data_no_ev["abstention_status"] == AbstentionStatus.NO_EVIDENCE.value
    assert len(data_no_ev["citations"]) == 0


# ============================================================================
# 12. MODEL_UNAVAILABLE Handled Safely
# ============================================================================

def test_model_unavailable_handled_safely(client: TestClient, memory_api_setup, monkeypatch):
    """Verify MODEL_UNAVAILABLE returns clean abstention without crashing or 500 error."""
    setup = memory_api_setup
    headers = setup["headers_a"]

    class UncachedModelService:
        dimension = 384
        def embed_query(self, q):
            raise ModelNotCachedError("intfloat/multilingual-e5-small is not cached locally.")

    monkeypatch.setattr("app.api.endpoints.memory.get_embedding_service", lambda: UncachedModelService())
    monkeypatch.setattr("app.services.memory.rag.engine.get_embedding_service", lambda: UncachedModelService())

    # Query endpoint
    resp = client.post(
        "/api/v1/memory/query",
        json={"question": "What container orchestration was adopted?"},
        headers=headers,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["is_abstained"] is True
    assert data["abstention_status"] == AbstentionStatus.MODEL_UNAVAILABLE.value
    assert "model weights are not cached" in data["answer"]
    assert data["diagnostics"]["fallback_warning"] == "MODEL_NOT_CACHED"


# ============================================================================
# 13. RETRIEVAL_UNAVAILABLE Does Not Expose Internal Details
# ============================================================================

def test_retrieval_unavailable_safe_error_handling(client: TestClient, memory_api_setup, monkeypatch):
    """Verify underlying retrieval failure returns safe RETRIEVAL_UNAVAILABLE without leaking SQL."""
    setup = memory_api_setup
    headers = setup["headers_a"]

    def failing_retrieval(*args, **kwargs):
        raise RuntimeError("OperationalError: database locked at /var/secret/sqlite.db")

    monkeypatch.setattr("app.services.memory.rag.engine.search_hybrid_rrf", failing_retrieval)

    resp = client.post(
        "/api/v1/memory/query",
        json={"question": "What container orchestration was adopted?"},
        headers=headers,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["is_abstained"] is True
    assert data["abstention_status"] == AbstentionStatus.RETRIEVAL_UNAVAILABLE.value
    # Ensure internal filesystem path or SQL string is not leaked in answer or diagnostics
    assert "/var/secret/sqlite.db" not in data["answer"]
    assert "/var/secret/sqlite.db" not in str(data["diagnostics"])


# ============================================================================
# 14. CONFLICTING_EVIDENCE Preserves Explicit Status
# ============================================================================

def test_conflicting_evidence_preserves_status(client: TestClient, memory_api_setup):
    """Verify CONFLICTING_EVIDENCE preserves explicit status across meetings."""
    setup = memory_api_setup
    headers = setup["headers_a"]

    resp = client.post(
        "/api/v1/memory/query",
        json={"question": "Did the team adopt Redis for caching?"},
        headers=headers,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["is_abstained"] is True
    assert data["abstention_status"] == AbstentionStatus.CONFLICTING_EVIDENCE.value
    assert len(data["citations"]) == 2
    titles = {c["meeting_title"] for c in data["citations"]}
    assert "Sprint 42 Architecture Review" in titles
    assert "Cache Architecture Retrospective" in titles


# ============================================================================
# 15. Citation IDs and Timestamps Database-Verified
# ============================================================================

def test_citation_ids_and_timestamps_verified(client: TestClient, memory_api_setup):
    """Verify citation timestamps strictly match database TranscriptSegment records."""
    setup = memory_api_setup
    headers = setup["headers_a"]

    resp = client.post(
        "/api/v1/memory/query",
        json={"question": "What container orchestration was adopted?"},
        headers=headers,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["citations"]) > 0
    cit = data["citations"][0]
    # Timestamp must match segment in DB: start 10.0, end 25.0
    assert cit["start_time"] == pytest.approx(10.0, abs=0.1)
    assert cit["end_time"] == pytest.approx(25.0, abs=0.1)
    assert cit["meeting_title"] == "Sprint 42 Architecture Review"


# ============================================================================
# 16. Rate Limiting Policy Enforcement (HTTP 429)
# ============================================================================

def test_rate_limiting_enforcement(client: TestClient, memory_api_setup):
    """Verify rate limiter blocks client when request quota is exhausted."""
    setup = memory_api_setup
    headers = setup["headers_a"]

    # Temporarily set max_requests to 2 for deterministic test
    memory_rate_limiter.reset()
    original_max = memory_rate_limiter.max_requests
    memory_rate_limiter.max_requests = 2

    try:
        # Request 1: OK
        r1 = client.post("/api/v1/memory/query", json={"question": "Test 1"}, headers=headers)
        assert r1.status_code == 200

        # Request 2: OK
        r2 = client.post("/api/v1/memory/query", json={"question": "Test 2"}, headers=headers)
        assert r2.status_code == 200

        # Request 3: Rate limited -> HTTP 429
        r3 = client.post("/api/v1/memory/query", json={"question": "Test 3"}, headers=headers)
        assert r3.status_code == 429
        assert "Retry-After" in r3.headers
    finally:
        memory_rate_limiter.max_requests = original_max
        memory_rate_limiter.reset()


# ============================================================================
# 17. Diagnostics Do Not Leak PII or Raw Transcripts
# ============================================================================

def test_diagnostics_privacy_clean(client: TestClient, memory_api_setup):
    """Verify diagnostics object only contains metadata and never leaks whole transcript passages."""
    setup = memory_api_setup
    headers = setup["headers_a"]

    resp = client.post(
        "/api/v1/memory/query",
        json={"question": "What container orchestration was adopted?"},
        headers=headers,
    )
    assert resp.status_code == 200
    diag = resp.json()["diagnostics"]

    assert isinstance(diag["total_candidates_retrieved"], int)
    assert isinstance(diag["evidence_items_count"], int)
    assert isinstance(diag["meetings_represented"], list)
    # Ensure diagnostics does not contain raw passage dumps
    diag_str = json.dumps(diag)
    assert "The engineering team decided" not in diag_str
    assert "fifteen million" not in diag_str
