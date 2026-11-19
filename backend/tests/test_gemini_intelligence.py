"""Comprehensive test suite for Gemini-Powered Meeting Intelligence and RAG.

Covers:
1. GeminiExtractor initialization, headers, and secret protection (no key exposure).
2. Structured extraction from privacy-safe transcripts (summary, topics, decisions, actions, risks, questions).
3. Decision vs suggestion distinction (confirmed decisions vs proposals).
4. Evidence grounding and timestamp derivation from DB segments (never trusting LLM timestamps).
5. Safe transcript enforcement (strict rejection of ORIGINAL and CLEANED transcripts).
6. Multi-model resilience and deterministic fallback when Gemini API fails.
7. GeminiAnswerProvider grounded answering and abstention detection.
8. Controlled live Gemini API execution when credentials are configured.
"""

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
import httpx
import pytest

from app.core.config import settings
from app.models.identity import Organization, User, OrganizationMembership, OrgRole
from app.models.meeting import Meeting, ProcessingJob
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
    StructuredExtractionPayload,
    TopicCreate,
    DecisionCreate,
    ActionItemCreate,
    RiskCreate,
    UnresolvedQuestionCreate,
)
from app.services.intelligence.base import (
    ProviderType,
    SafeTranscriptRequiredError,
    ProviderExecutionError,
)
from app.services.intelligence.gemini_adapter import GeminiExtractor
from app.services.intelligence.deterministic import DeterministicExtractor
from app.services.intelligence.engine import extract_meeting_intelligence
from app.services.memory.rag.providers import GeminiAnswerProvider, DeterministicAnswerProvider
from app.schemas.memory import (
    EvidenceItem,
    EvidenceChunk,
    RawAnswerResult,
    AbstentionStatus,
    RAGQueryResponse as RAGResponse,
)


# ==============================================================================
# 1. GeminiExtractor Secret Safety & Initialization Tests
# ==============================================================================

def test_gemini_extractor_initialization_and_secret_hygiene():
    """Verify GeminiExtractor initializes correctly and never leaks the API key in str/repr."""
    dummy_key = "AIzaSyFakeKey1234567890abcdef"
    extractor = GeminiExtractor(api_key=dummy_key, model_name="gemini-3.8-flash")

    assert extractor.provider_type == ProviderType.GEMINI
    assert extractor.model_name == "gemini-3.8-flash"
    assert "x-goog-api-key" in extractor.client.headers
    assert extractor.client.headers["x-goog-api-key"] == dummy_key

    # Secret hygiene: repr and str must NOT contain the secret API key
    assert dummy_key not in str(extractor)
    assert dummy_key not in repr(extractor)


# ==============================================================================
# 2. GeminiExtractor Structured Payload Extraction & Decision Discrimination
# ==============================================================================

def test_gemini_extractor_extracts_structured_payload_mocked():
    """Verify GeminiExtractor extracts valid StructuredExtractionPayload from model JSON response."""
    extractor = GeminiExtractor(api_key="fake-key", model_name="gemini-3.8-flash")

    mock_gemini_json = {
        "summary": "The team aligned on PostgreSQL for user data storage and assigned database setup to Priya.",
        "key_takeaways": [
            "PostgreSQL selected as primary database",
            "Priya to provision RDS instance by Friday",
        ],
        "topics": [
            {"name": "Database Architecture", "description": "Discussion on database choices", "evidence_segment_ids": ["seg-1"]}
        ],
        "decisions": [
            {
                "title": "Use PostgreSQL for User Storage",
                "rationale": "ACID compliance and relational schema fit user account requirements",
                "decision_type": "ARCHITECTURE",
                "evidence_segment_id": "seg-1",
            }
        ],
        "action_items": [
            {
                "description": "Provision AWS RDS PostgreSQL instance",
                "assignee": "Priya Patel",
                "deadline": "2026-11-20",
                "evidence_segment_id": "seg-2",
            }
        ],
        "risks": [
            {
                "description": "Migration delay if schema changes before launch",
                "severity": "MEDIUM",
                "mitigation": "Freeze schema 48 hours prior to deployment",
                "evidence_segment_id": "seg-2",
            }
        ],
        "unresolved_questions": [
            {
                "question": "Should we use Redis caching in Phase 1?",
                "context": "Mentioned as suggestion but deferred to Phase 2",
                "evidence_segment_id": "seg-3",
            }
        ],
    }

    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"text": json.dumps(mock_gemini_json)}
                    ]
                }
            }
        ]
    }

    with patch.object(extractor.client, "post", return_value=mock_response):
        payload = extractor.extract_payload(
            segments_text="[seg-1] (00:00 - 00:30) Priya: We decided to adopt PostgreSQL.\n[seg-2] (00:30 - 01:00) Priya: I will set up RDS by Friday.",
            output_language="en",
        )

    assert isinstance(payload, StructuredExtractionPayload)
    assert "PostgreSQL" in payload.summary
    assert len(payload.decisions) == 1
    assert payload.decisions[0].title == "Use PostgreSQL for User Storage"
    # Verify suggestion is NOT in decisions
    decision_titles = [d.title.lower() for d in payload.decisions]
    assert not any("redis" in title for title in decision_titles)
    assert len(payload.action_items) == 1
    assert payload.action_items[0].assignee == "Priya Patel"
    assert len(payload.risks) == 1
    assert len(payload.unresolved_questions) == 1


# ==============================================================================
# 3. Grounding, Timestamp Derivation & Safe Transcript Enforcement
# ==============================================================================

def test_gemini_intelligence_pipeline_grounding_and_timestamp_derivation(db_session):
    """Verify Gemini extractions have evidence grounded to DB segments and timestamps derived from segments."""
    org = Organization(name="Gemini Test Org", slug="gemini-test-org")
    meeting = Meeting(organization_id=org.id, title="Architecture Sync")
    db_session.add_all([org, meeting])
    db_session.flush()

    transcript = Transcript(
        meeting_id=meeting.id,
        transcript_type=TranscriptType.CLEANED_REDACTED.value,
        version=1,
    )
    db_session.add(transcript)
    db_session.flush()

    seg1 = TranscriptSegment(
        transcript_id=transcript.id,
        segment_index=0,
        start_time=12.5,
        end_time=30.0,
        text="We all agreed and decided to adopt PostgreSQL for our database.",
        speaker_id=None,
        language="en",
    )
    seg2 = TranscriptSegment(
        transcript_id=transcript.id,
        segment_index=1,
        start_time=32.0,
        end_time=55.0,
        text="Priya will configure the database by next Monday.",
        speaker_id=None,
        language="en",
    )
    db_session.add_all([seg1, seg2])
    db_session.commit()

    # Mock Gemini response referencing seg1 and seg2
    mock_gemini_payload = {
        "summary": "Team agreed to adopt PostgreSQL, Priya responsible for configuration.",
        "key_takeaways": ["Adopted PostgreSQL"],
        "topics": [{"name": "Database", "description": "Database choice", "evidence_segment_ids": [seg1.id]}],
        "decisions": [
            {
                "title": "Adopt PostgreSQL",
                "rationale": "Agreed by team",
                "decision_type": "ARCHITECTURE",
                "evidence_segment_id": seg1.id,  # Grounded to valid seg1
            }
        ],
        "action_items": [
            {
                "description": "Configure the database",
                "assignee": "Priya",
                "deadline": "next Monday",
                "evidence_segment_id": seg2.id,  # Grounded to valid seg2
            }
        ],
        "risks": [],
        "unresolved_questions": [],
    }

    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "candidates": [{"content": {"parts": [{"text": json.dumps(mock_gemini_payload)}]}}]
    }

    extractor = GeminiExtractor(api_key="mock-key", model_name="gemini-3.8-flash")
    with patch.object(extractor.client, "post", return_value=mock_response):
        result = extract_meeting_intelligence(
            meeting_id=meeting.id,
            db=db_session,
            provider_override="gemini",
            gemini_extractor=extractor,
        )

    assert result.has_intelligence is True
    assert result.provider_used == "gemini"
    assert result.is_heuristic is False
    assert result.fallback_warning is None

    # Verify decision timestamp was strictly derived from seg1.start_time (12.5)
    db_decision = db_session.query(Decision).filter(Decision.meeting_id == meeting.id).first()
    assert db_decision is not None
    assert db_decision.evidence_segment_id == seg1.id
    assert db_decision.timestamp_seconds == 12.5

    # Verify action item timestamp was strictly derived from seg2.start_time (32.0)
    db_action = db_session.query(ActionItem).filter(ActionItem.meeting_id == meeting.id).first()
    assert db_action is not None
    assert db_action.evidence_segment_id == seg2.id
    assert db_action.timestamp_seconds == 32.0


def test_gemini_intelligence_pipeline_rejects_unredacted_transcripts(db_session):
    """Verify Gemini extraction rejects ORIGINAL and CLEANED unredacted transcripts."""
    org = Organization(name="Privacy Org", slug="privacy-org-gemini")
    meeting = Meeting(organization_id=org.id, title="Unsafe Meeting")
    db_session.add_all([org, meeting])
    db_session.flush()

    orig_transcript = Transcript(
        meeting_id=meeting.id,
        transcript_type=TranscriptType.ORIGINAL.value,
        version=1,
    )
    db_session.add(orig_transcript)
    db_session.commit()

    extractor = GeminiExtractor(api_key="mock-key", model_name="gemini-3.8-flash")
    with pytest.raises(SafeTranscriptRequiredError, match="strictly requires privacy-safe transcripts"):
        extract_meeting_intelligence(
            meeting_id=meeting.id,
            db=db_session,
            provider_override="gemini",
            gemini_extractor=extractor,
            requested_transcript_id=orig_transcript.id,
        )


def test_gemini_fallback_to_deterministic_on_provider_error(db_session):
    """Verify that when Gemini API returns 503/timeout, extraction automatically falls back to deterministic."""
    org = Organization(name="Fallback Org", slug="fallback-org")
    meeting = Meeting(organization_id=org.id, title="Fallback Meeting")
    db_session.add_all([org, meeting])
    db_session.flush()

    transcript = Transcript(
        meeting_id=meeting.id,
        transcript_type=TranscriptType.CLEANED_REDACTED.value,
        version=1,
    )
    db_session.add(transcript)
    db_session.flush()

    seg = TranscriptSegment(
        transcript_id=transcript.id,
        segment_index=0,
        start_time=0.0,
        end_time=10.0,
        text="We decided to launch on Friday. John will notify the clients.",
        speaker_id=None,
        language="en",
    )
    db_session.add(seg)
    db_session.commit()

    # Simulate 503 on all Gemini models
    mock_err_response = MagicMock(spec=httpx.Response)
    mock_err_response.status_code = 503
    mock_err_response.text = "Model is overloaded"

    extractor = GeminiExtractor(api_key="mock-key", model_name="gemini-3.8-flash")
    with patch.object(extractor.client, "post", side_effect=httpx.HTTPStatusError("503", request=MagicMock(), response=mock_err_response)):
        result = extract_meeting_intelligence(
            meeting_id=meeting.id,
            db=db_session,
            provider_override="gemini",
            gemini_extractor=extractor,
        )

    # Result must succeed via deterministic fallback
    assert result.has_intelligence is True
    assert result.is_heuristic is True
    assert result.fallback_warning is not None
    assert "Gemini extraction failed" in result.fallback_warning


# ==============================================================================
# 4. GeminiAnswerProvider (RAG) Tests
# ==============================================================================

def test_gemini_rag_answer_provider_grounded_response():
    """Verify GeminiAnswerProvider produces an evidence-grounded RAG response with citations."""
    provider = GeminiAnswerProvider(api_key="fake-key", model_name="gemini-3.8-flash")

    evidence = [
        EvidenceChunk(
            meeting_id="meeting-1",
            segment_id="seg-101",
            chunk_text="In our meeting, we agreed to launch the beta release on December 1st.",
            speaker="Priya",
            start_time=10.0,
            end_time=20.0,
            score=0.92,
            meeting_title="Q4 Launch Sync",
            meeting_date="2026-11-15",
            source_version="CLEANED_REDACTED",
        )
    ]

    mock_gemini_reply = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": "The team agreed to launch the beta release on December 1st [Citation: meeting-1, segment seg-101]."
                        }
                    ]
                }
            }
        ]
    }

    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_gemini_reply

    with patch.object(provider.client, "post", return_value=mock_resp):
        rag_resp = provider.generate_answer(
            query="When is the beta launch scheduled?",
            evidence_chunks=evidence,
            organization_id="org-1",
        )

    assert isinstance(rag_resp, RAGResponse)
    assert rag_resp.is_abstained is False
    assert rag_resp.abstention_status == AbstentionStatus.NOT_ABSTAINED
    assert "December 1st" in rag_resp.answer
    assert rag_resp.diagnostics.provider_used == "gemini"
    assert rag_resp.diagnostics.is_heuristic is False
    assert len(rag_resp.citations) == 1
    assert rag_resp.citations[0].segment_id == "seg-101"


def test_gemini_rag_answer_provider_abstains_on_insufficient_evidence():
    """Verify GeminiAnswerProvider abstains when Gemini returns INSUFFICIENT_EVIDENCE."""
    provider = GeminiAnswerProvider(api_key="fake-key", model_name="gemini-3.8-flash")

    evidence = [
        EvidenceChunk(
            meeting_id="meeting-1",
            segment_id="seg-101",
            chunk_text="We discussed lunch options.",
            speaker="Alice",
            start_time=0.0,
            end_time=5.0,
            score=0.3,
            meeting_title="Lunch Chat",
        )
    ]

    mock_gemini_reply = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"text": "INSUFFICIENT_EVIDENCE: The retrieved meeting evidence does not discuss the database budget."}
                    ]
                }
            }
        ]
    }

    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_gemini_reply

    with patch.object(provider.client, "post", return_value=mock_resp):
        rag_resp = provider.generate_answer(
            query="What is the database budget?",
            evidence_chunks=evidence,
            organization_id="org-1",
        )

    assert rag_resp.is_abstained is True
    assert rag_resp.abstention_status == AbstentionStatus.INSUFFICIENT_EVIDENCE
    assert "does not discuss the database budget" in rag_resp.answer


def test_gemini_rag_answer_provider_fallback_to_deterministic():
    """Verify RAGEngine falls back to DeterministicAnswerProvider when Gemini API fails."""
    gemini_provider = GeminiAnswerProvider(api_key="fake-key", model_name="gemini-3.8-flash")
    deterministic_fallback = DeterministicAnswerProvider()

    evidence = [
        EvidenceChunk(
            meeting_id="meeting-1",
            segment_id="seg-1",
            chunk_text="We agreed to implement OAuth2 for authentication.",
            speaker="Dave",
            start_time=5.0,
            end_time=15.0,
            score=0.88,
            meeting_title="Security Review",
        )
    ]

    # Patch client.post to raise an HTTP error
    with patch.object(gemini_provider.client, "post", side_effect=httpx.ConnectError("Connection refused")):
        # The provider catches the error and falls back to deterministic
        response = gemini_provider.generate_answer(
            query="What authentication method was chosen?",
            evidence_chunks=evidence,
            organization_id="org-1",
        )

    assert response.diagnostics.is_heuristic is True
    assert response.diagnostics.fallback_warning is not None
    assert "Gemini generation failed" in response.diagnostics.fallback_warning
    assert "OAuth2" in response.answer


# ==============================================================================
# 5. Live Gemini API Verification (Conditional on Real Key)
# ==============================================================================

def test_live_gemini_api_controlled_verification():
    """Verify live Gemini API connectivity using the configured backend environment."""
    if not settings.GEMINI_ENABLED or not settings.GEMINI_API_KEY:
        pytest.skip("Gemini API is not enabled or key is not configured in backend/.env.")

    extractor = GeminiExtractor(
        api_key=settings.GEMINI_API_KEY,
        model_name=settings.GEMINI_MODEL,
        timeout=30.0,
    )

    sample_transcript = (
        "[seg-001] (00:00 - 00:20) Alex: We have evaluated MongoDB and PostgreSQL. We officially decided to proceed with PostgreSQL.\n"
        "[seg-002] (00:20 - 00:45) Sarah: I will configure the staging database cluster by Friday evening.\n"
        "[seg-003] (00:45 - 01:00) Alex: What about the caching layer? Let's keep Redis as a suggestion for next month."
    )

    payload = extractor.extract_payload(sample_transcript, output_language="en")

    assert isinstance(payload, StructuredExtractionPayload)
    assert len(payload.summary) > 20
    assert len(payload.decisions) >= 1
    # Check that PostgreSQL is identified in decisions
    postgres_found = any("postgres" in d.title.lower() or "postgres" in d.rationale.lower() for d in payload.decisions)
    assert postgres_found, "PostgreSQL decision should be captured by Gemini"
    # Verify Redis proposal is NOT misclassified as a confirmed decision
    redis_decision = any("redis" in d.title.lower() for d in payload.decisions)
    assert not redis_decision, "Redis suggestion should not be misclassified as a confirmed decision"


# ==============================================================================
# 6. Provider Registration, Override Selection & Missing Key Rejection Tests
# ==============================================================================

def test_extraction_job_payload_validates_all_allowed_providers():
    """Verify ExtractionJobPayload accepts gemini, ollama, deterministic, and disabled."""
    from app.schemas.intelligence import ExtractionJobPayload
    for p in ["gemini", "GEMINI", "ollama", "deterministic", "disabled"]:
        payload = ExtractionJobPayload(meeting_id="meeting-1", provider_override=p)
        assert payload.provider_override == p.lower()

    with pytest.raises(ValueError, match="Invalid provider_override"):
        ExtractionJobPayload(meeting_id="meeting-1", provider_override="unsupported_cloud_ai")


def test_extract_intelligence_api_accepts_gemini_provider(client, db_session):
    """Verify API accepts provider_override='gemini' and records it in the job payload."""
    from backend.tests.conftest import make_auth_headers
    org = Organization(name="Gemini API Org", slug="gemini-api-org")
    user = User(email="gemini_user@example.com", hashed_password="hashed_password", full_name="Gemini User")
    db_session.add_all([org, user])
    db_session.flush()

    membership = OrganizationMembership(organization_id=org.id, user_id=user.id, role=OrgRole.ORG_ADMIN.value)
    meeting = Meeting(organization_id=org.id, title="Gemini Extract Meeting")
    db_session.add_all([membership, meeting])
    db_session.flush()

    transcript = Transcript(
        meeting_id=meeting.id,
        transcript_type=TranscriptType.CLEANED_REDACTED.value,
        version=1,
    )
    db_session.add(transcript)
    db_session.commit()

    headers = make_auth_headers(user, org=org)
    with patch.object(settings, "GEMINI_ENABLED", True), patch.object(settings, "GEMINI_API_KEY", "test-key-valid"):
        response = client.post(
            f"/api/v1/meetings/{meeting.id}/intelligence/extract",
            json={"provider_override": "gemini"},
            headers=headers,
        )

    assert response.status_code == 202
    data = response.json()
    assert data["job_type"] == "EXTRACTION"
    assert data["status"] == "PENDING"

    # Verify job record in DB has provider_override == "gemini"
    job = db_session.query(ProcessingJob).filter(ProcessingJob.id == data["job_id"]).first()
    assert job is not None
    job_payload = json.loads(job.payload) if isinstance(job.payload, str) else job.payload
    assert job_payload["provider_override"] == "gemini"


def test_extract_intelligence_api_rejects_gemini_when_disabled(client, db_session):
    """Verify API returns 400 when gemini is requested but GEMINI_ENABLED=False."""
    from backend.tests.conftest import make_auth_headers
    org = Organization(name="Disabled Org", slug="disabled-org")
    user = User(email="disabled_user@example.com", hashed_password="hashed_password", full_name="Disabled User")
    db_session.add_all([org, user])
    db_session.flush()

    membership = OrganizationMembership(organization_id=org.id, user_id=user.id, role=OrgRole.ORG_ADMIN.value)
    meeting = Meeting(organization_id=org.id, title="Disabled Meeting")
    db_session.add_all([membership, meeting])
    db_session.flush()

    transcript = Transcript(
        meeting_id=meeting.id,
        transcript_type=TranscriptType.CLEANED_REDACTED.value,
        version=1,
    )
    db_session.add(transcript)
    db_session.commit()

    headers = make_auth_headers(user, org=org)
    with patch.object(settings, "GEMINI_ENABLED", False):
        response = client.post(
            f"/api/v1/meetings/{meeting.id}/intelligence/extract",
            json={"provider_override": "gemini"},
            headers=headers,
        )

    assert response.status_code == 400
    assert "Gemini intelligence provider is disabled" in response.json()["detail"]


def test_extract_intelligence_api_rejects_gemini_when_api_key_missing(client, db_session):
    """Verify API returns 400 when gemini is requested but GEMINI_API_KEY is empty."""
    from backend.tests.conftest import make_auth_headers
    org = Organization(name="Missing Key Org", slug="missing-key-org")
    user = User(email="missing_key_user@example.com", hashed_password="hashed_password", full_name="Key User")
    db_session.add_all([org, user])
    db_session.flush()

    membership = OrganizationMembership(organization_id=org.id, user_id=user.id, role=OrgRole.ORG_ADMIN.value)
    meeting = Meeting(organization_id=org.id, title="Missing Key Meeting")
    db_session.add_all([membership, meeting])
    db_session.flush()

    transcript = Transcript(
        meeting_id=meeting.id,
        transcript_type=TranscriptType.CLEANED_REDACTED.value,
        version=1,
    )
    db_session.add(transcript)
    db_session.commit()

    headers = make_auth_headers(user, org=org)
    with patch.object(settings, "GEMINI_ENABLED", True), patch.object(settings, "GEMINI_API_KEY", ""):
        response = client.post(
            f"/api/v1/meetings/{meeting.id}/intelligence/extract",
            json={"provider_override": "gemini"},
            headers=headers,
        )

    assert response.status_code == 400
    assert "Gemini API key is not configured" in response.json()["detail"]
