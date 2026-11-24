"""Automated verification test suite for Ollama Qwen 3 8B Fallback and 3-Tier Resilience.

Tests:
1. Direct Ollama extraction with qwen3:8b (live or mocked).
2. Gemini success: Ollama should not be called.
3. Gemini quota failure (429): Ollama should be attempted and succeed.
4. Gemini timeout: Ollama should be attempted and succeed.
5. Ollama unavailable: deterministic fallback should run.
6. Ollama malformed JSON: deterministic fallback should run.
7. Explicit Ollama provider override (provider_override="ollama").
8. Explicit Gemini provider override (provider_override="gemini").
9. Multilingual transcript extraction (Hindi, English, and Hinglish).
10. Decisions versus proposals discrimination.
11. Missing responsible persons safely handled (owner_name=None).
12. Preservation of original evidence and timestamps.
"""

import json
from unittest.mock import MagicMock, patch
import httpx
import pytest
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.identity import Organization, User
from app.models.meeting import Meeting
from app.models.transcription import Transcript, TranscriptSegment, TranscriptType
from app.models.intelligence import Decision, ActionItem
from app.schemas.intelligence import (
    StructuredExtractionPayload,
    TopicCreate,
    DecisionCreate,
    ActionItemCreate,
    RiskCreate,
    UnresolvedQuestionCreate,
)
from app.services.intelligence.base import ProviderExecutionError, ProviderType
from app.services.intelligence.gemini_adapter import GeminiExtractor
from app.services.intelligence.ollama_adapter import OllamaExtractor
from app.services.intelligence.engine import extract_meeting_intelligence


# ==============================================================================
# Helpers & Fixtures
# ==============================================================================

def create_test_meeting(db: Session, title: str = "Test Meeting"):
    org = Organization(name="Ollama Test Org", slug=f"ollama-org-{title.lower().replace(' ', '-')}")
    user = User(email=f"user_{title.lower().replace(' ', '_')}@example.com", hashed_password="hashed_pwd", full_name="Test User")
    db.add_all([org, user])
    db.flush()

    meeting = Meeting(organization_id=org.id, title=title)
    db.add(meeting)
    db.flush()

    transcript = Transcript(
        meeting_id=meeting.id,
        transcript_type=TranscriptType.CLEANED_REDACTED.value,
        version=1,
    )
    db.add(transcript)
    db.flush()

    seg1 = TranscriptSegment(
        id="seg-101",
        transcript_id=transcript.id,
        sequence_order=1,
        start_time=10.0,
        end_time=25.0,
        text="Aarav: We officially decide to deploy the model on AWS. Priya will set up the cluster by Friday.",
        language_code="en",
    )
    db.add(seg1)
    db.commit()
    return meeting, transcript, seg1


def make_mock_ollama_payload():
    return {
        "executive_summary": "The team officially decided to deploy the model on AWS with cluster setup scheduled for Friday.",
        "detailed_summary": "### Cloud Deployment\nTeam reviewed architecture options and finalized AWS infrastructure.",
        "key_takeaways": ["AWS chosen for model deployment", "Friday cluster completion target"],
        "topics": [{"title": "AWS Deployment", "description": "Cloud hosting discussion", "start_time": 10.0, "end_time": 25.0}],
        "decisions": [{"description": "Deploy the model on AWS", "rationale_or_context": "Infrastructure scale", "decided_by": "Aarav", "evidence_segment_id": "seg-101"}],
        "action_items": [{"description": "Set up the cluster", "owner_name": "Priya", "deadline": "Friday", "status": "PENDING", "evidence_segment_id": "seg-101"}],
        "risks": [{"description": "Cluster provisioning latency", "severity": "MEDIUM", "mitigation_suggestion": "Pre-warm instances", "evidence_segment_id": "seg-101"}],
        "unresolved_questions": [{"question_text": "Will AWS quotas support peak load?", "raised_by": "Aarav", "status": "OPEN", "evidence_segment_id": "seg-101"}],
    }


# ==============================================================================
# Test Cases 1 - 12
# ==============================================================================

def test_1_direct_ollama_extraction_qwen3():
    """1. Direct Ollama extraction with qwen3:8b and schema validation."""
    mock_payload = make_mock_ollama_payload()

    def mock_handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/generate"
        body = json.loads(request.content)
        assert body["model"] in {"qwen3:8b", getattr(settings, "LLM_MODEL_NAME", "qwen3:8b")}
        assert body["format"] == "json"
        return httpx.Response(200, json={"response": json.dumps(mock_payload)})

    client = httpx.Client(transport=httpx.MockTransport(mock_handler))
    adapter = OllamaExtractor(model_name="qwen3:8b", client=client)

    seg = TranscriptSegment(id="seg-101", start_time=10.0, end_time=25.0, text="Aarav: We decide to deploy on AWS.", language_code="en")
    payload = adapter.extract([seg], meeting_title="Architecture Review")

    assert isinstance(payload, StructuredExtractionPayload)
    assert "AWS" in payload.executive_summary
    assert len(payload.decisions) == 1
    assert payload.decisions[0].decided_by == "Aarav"
    assert len(payload.action_items) == 1
    assert payload.action_items[0].owner_name == "Priya"


def test_2_gemini_success_ollama_not_called(db_session: Session):
    """2. Gemini success: Ollama should NOT be called."""
    meeting, _, _ = create_test_meeting(db_session, "Gemini Success Meeting")

    mock_gemini_extractor = MagicMock(spec=GeminiExtractor)
    mock_gemini_extractor.model_name = "gemini-3.8-flash"
    mock_gemini_extractor.extract.return_value = StructuredExtractionPayload(
        executive_summary="Gemini extraction completed successfully.",
        detailed_summary="Gemini detailed content.",
        key_takeaways=["Takeaway 1"],
        topics=[TopicCreate(title="Topic 1", description="Description 1", start_time=10.0, end_time=25.0)],
        decisions=[DecisionCreate(description="Decision 1", rationale_or_context="Rationale", decided_by="Aarav", evidence_segment_id="seg-101", timestamp_seconds=10.0)],
        action_items=[ActionItemCreate(description="Action 1", owner_name="Priya", deadline=None, status="PENDING", evidence_segment_id="seg-101", timestamp_seconds=10.0)],
        risks=[],
        unresolved_questions=[],
    )

    mock_ollama_extractor = MagicMock(spec=OllamaExtractor)

    result = extract_meeting_intelligence(
        meeting_id=meeting.id,
        db=db_session,
        provider_override="gemini",
        gemini_extractor=mock_gemini_extractor,
        ollama_extractor=mock_ollama_extractor,
        allow_fallback=True,
    )

    assert result.provider_used == "gemini"
    assert result.is_heuristic is False
    assert result.fallback_warning is None
    mock_gemini_extractor.extract.assert_called_once()
    mock_ollama_extractor.extract.assert_not_called()


def test_3_gemini_quota_failure_ollama_attempted(db_session: Session):
    """3. Gemini quota failure (HTTP 429): Ollama should be attempted and succeed."""
    meeting, _, _ = create_test_meeting(db_session, "Gemini Quota Fallback Meeting")

    mock_gemini_extractor = MagicMock(spec=GeminiExtractor)
    mock_gemini_extractor.model_name = "gemini-3.8-flash"
    mock_gemini_extractor.extract.side_effect = ProviderExecutionError("Google Gemini API quota or rate limit exceeded.")

    mock_ollama_extractor = MagicMock(spec=OllamaExtractor)
    mock_ollama_extractor.model_name = "qwen3:8b"
    mock_ollama_extractor.extract.return_value = StructuredExtractionPayload(
        executive_summary="Ollama fallback extracted successfully after Gemini quota exhaustion.",
        detailed_summary="Ollama details.",
        key_takeaways=["Ollama fallback succeeded"],
        topics=[],
        decisions=[DecisionCreate(description="Deploy on AWS", rationale_or_context="Cost effective", decided_by="Aarav", evidence_segment_id="seg-101", timestamp_seconds=10.0)],
        action_items=[ActionItemCreate(description="Set up AWS cluster", owner_name="Priya", deadline=None, status="PENDING", evidence_segment_id="seg-101", timestamp_seconds=10.0)],
        risks=[],
        unresolved_questions=[],
    )

    result = extract_meeting_intelligence(
        meeting_id=meeting.id,
        db=db_session,
        provider_override="gemini",
        gemini_extractor=mock_gemini_extractor,
        ollama_extractor=mock_ollama_extractor,
        allow_fallback=True,
    )

    assert result.provider_used == "ollama"
    assert result.is_heuristic is False
    assert result.fallback_warning is not None
    assert "Gemini extraction failed" in result.fallback_warning
    assert "quota or rate limit" in result.fallback_warning
    assert "Local Ollama (qwen3:8b)" in result.fallback_warning
    mock_gemini_extractor.extract.assert_called_once()
    mock_ollama_extractor.extract.assert_called_once()


def test_4_gemini_timeout_ollama_attempted(db_session: Session):
    """4. Gemini timeout: Ollama should be attempted and succeed."""
    meeting, _, _ = create_test_meeting(db_session, "Gemini Timeout Fallback Meeting")

    mock_gemini_extractor = MagicMock(spec=GeminiExtractor)
    mock_gemini_extractor.model_name = "gemini-3.8-flash"
    mock_gemini_extractor.extract.side_effect = ProviderExecutionError("Google Gemini API request timed out after 120.0s.")

    mock_ollama_extractor = MagicMock(spec=OllamaExtractor)
    mock_ollama_extractor.model_name = "qwen3:8b"
    mock_ollama_extractor.extract.return_value = StructuredExtractionPayload(
        executive_summary="Ollama extracted successfully after Gemini timeout.",
        detailed_summary="Ollama details.",
        key_takeaways=["Ollama recovered after timeout"],
        topics=[],
        decisions=[DecisionCreate(description="Deploy on AWS", rationale_or_context="Reliability", decided_by="Aarav", evidence_segment_id="seg-101", timestamp_seconds=10.0)],
        action_items=[],
        risks=[],
        unresolved_questions=[],
    )

    result = extract_meeting_intelligence(
        meeting_id=meeting.id,
        db=db_session,
        provider_override="gemini",
        gemini_extractor=mock_gemini_extractor,
        ollama_extractor=mock_ollama_extractor,
        allow_fallback=True,
    )

    assert result.provider_used == "ollama"
    assert result.is_heuristic is False
    assert result.fallback_warning is not None
    assert "timed out" in result.fallback_warning
    assert "Local Ollama (qwen3:8b)" in result.fallback_warning
    mock_ollama_extractor.extract.assert_called_once()


def test_5_gemini_and_ollama_unavailable_deterministic_fallback(db_session: Session):
    """5. Ollama unavailable: deterministic fallback should run seamlessly."""
    meeting, _, _ = create_test_meeting(db_session, "Dual Failure Deterministic Fallback Meeting")

    mock_gemini_extractor = MagicMock(spec=GeminiExtractor)
    mock_gemini_extractor.extract.side_effect = ProviderExecutionError("Google Gemini API request timed out after 120.0s.")

    mock_ollama_extractor = MagicMock(spec=OllamaExtractor)
    mock_ollama_extractor.extract.side_effect = ProviderExecutionError("Ollama server is not running or unreachable at 'http://127.0.0.1:11434'.")

    result = extract_meeting_intelligence(
        meeting_id=meeting.id,
        db=db_session,
        provider_override="gemini",
        gemini_extractor=mock_gemini_extractor,
        ollama_extractor=mock_ollama_extractor,
        allow_fallback=True,
    )

    assert result.provider_used == "deterministic"
    assert result.is_heuristic is True
    assert result.fallback_warning is not None
    assert "Gemini extraction failed" in result.fallback_warning
    assert "Local Ollama fallback failed" in result.fallback_warning
    assert "Deterministic extractor was used as fallback." in result.fallback_warning


def test_6_gemini_and_ollama_malformed_json_deterministic_fallback(db_session: Session):
    """6. Ollama malformed JSON: deterministic fallback should run safely."""
    meeting, _, _ = create_test_meeting(db_session, "Ollama Malformed Output Meeting")

    mock_gemini_extractor = MagicMock(spec=GeminiExtractor)
    mock_gemini_extractor.extract.side_effect = ProviderExecutionError("Gemini 503 Service Unavailable")

    mock_ollama_extractor = MagicMock(spec=OllamaExtractor)
    mock_ollama_extractor.extract.side_effect = ProviderExecutionError("Ollama output was not valid JSON: Expecting value: line 1 column 1 (char 0)")

    result = extract_meeting_intelligence(
        meeting_id=meeting.id,
        db=db_session,
        provider_override="gemini",
        gemini_extractor=mock_gemini_extractor,
        ollama_extractor=mock_ollama_extractor,
        allow_fallback=True,
    )

    assert result.provider_used == "deterministic"
    assert result.is_heuristic is True
    assert result.fallback_warning is not None
    assert "not valid JSON" in result.fallback_warning


def test_7_explicit_ollama_provider_override(db_session: Session):
    """7. Explicit Ollama provider override (provider_override='ollama')."""
    meeting, _, _ = create_test_meeting(db_session, "Explicit Ollama Override Meeting")

    mock_gemini_extractor = MagicMock(spec=GeminiExtractor)
    mock_ollama_extractor = MagicMock(spec=OllamaExtractor)
    mock_ollama_extractor.model_name = "qwen3:8b"
    mock_ollama_extractor.extract.return_value = StructuredExtractionPayload(
        executive_summary="Direct Ollama override execution completed.",
        detailed_summary=None,
        key_takeaways=[],
        topics=[],
        decisions=[],
        action_items=[],
        risks=[],
        unresolved_questions=[],
    )

    result = extract_meeting_intelligence(
        meeting_id=meeting.id,
        db=db_session,
        provider_override="ollama",
        gemini_extractor=mock_gemini_extractor,
        ollama_extractor=mock_ollama_extractor,
    )

    assert result.provider_used == "ollama"
    assert result.fallback_warning is None
    mock_gemini_extractor.extract.assert_not_called()
    mock_ollama_extractor.extract.assert_called_once()


def test_8_explicit_gemini_provider_override(db_session: Session):
    """8. Explicit Gemini provider override (provider_override='gemini')."""
    meeting, _, _ = create_test_meeting(db_session, "Explicit Gemini Override Meeting")

    mock_gemini_extractor = MagicMock(spec=GeminiExtractor)
    mock_gemini_extractor.model_name = "gemini-3.8-flash"
    mock_gemini_extractor.extract.return_value = StructuredExtractionPayload(
        executive_summary="Direct Gemini override execution completed.",
        detailed_summary=None,
        key_takeaways=[],
        topics=[],
        decisions=[],
        action_items=[],
        risks=[],
        unresolved_questions=[],
    )
    mock_ollama_extractor = MagicMock(spec=OllamaExtractor)

    result = extract_meeting_intelligence(
        meeting_id=meeting.id,
        db=db_session,
        provider_override="gemini",
        gemini_extractor=mock_gemini_extractor,
        ollama_extractor=mock_ollama_extractor,
    )

    assert result.provider_used == "gemini"
    assert result.fallback_warning is None
    mock_gemini_extractor.extract.assert_called_once()
    mock_ollama_extractor.extract.assert_not_called()


def test_9_multilingual_transcript_extraction_hindi_english_hinglish():
    """9. Multilingual extraction with Hindi, English, and Hinglish."""
    # Test Devanagari Hindi payload parsing
    hindi_data = {
        "executive_summary": "बैठक में शुक्रवार को उत्पादन में नया संस्करण तैनात करने का निर्णय लिया गया।",
        "detailed_summary": "विस्तृत चर्चा के बाद टीम ने नई सुविधाओं को मंजूरी दी।",
        "key_takeaways": ["शुक्रवार को रिलीज़", "सुरक्षा ऑडिट पूरा"],
        "topics": [{"title": "उत्पादन विमोचन", "description": "रिलीज़ योजना", "start_time": 0.0, "end_time": 15.0}],
        "decisions": [{"description": "शुक्रवार को रिलीज़ करेंगे", "rationale_or_context": "ऑडिट सफल रहा", "decided_by": "राजेश", "evidence_segment_id": "seg-h1"}],
        "action_items": [{"description": "दस्तावेज़ अपडेट करें", "owner_name": "प्रिया", "deadline": "कल शाम", "status": "PENDING", "evidence_segment_id": "seg-h1"}],
        "risks": [],
        "unresolved_questions": [],
    }

    adapter = OllamaExtractor()
    payload = adapter._sanitize_and_validate_payload(hindi_data)

    assert "शुक्रवार" in payload.executive_summary
    assert len(payload.decisions) == 1
    assert payload.decisions[0].decided_by == "राजेश"
    assert payload.action_items[0].owner_name == "प्रिया"

    # Test Hinglish code-switching payload parsing
    hinglish_data = {
        "executive_summary": "Team ne decide kiya ki hum cloud migration AWS par complete karenge and Rahul testing lead karega.",
        "detailed_summary": {"topics": [{"title": "Migration Sync", "description": "Budget aur timeline finalize kiya."}]},
        "key_takeaways": ["AWS migration finalized", "Testing starts Monday"],
        "topics": [{"title": "Cloud Migration", "description": "Timeline sync", "start_time": 0.0, "end_time": 20.0}],
        "decisions": [{"description": "AWS par migrate karenge", "rationale_or_context": "Reliability better hai", "decided_by": "Vikas", "evidence_segment_id": "seg-hg1"}],
        "action_items": [{"description": "Lead the testing", "owner_name": "Rahul", "deadline": "Monday", "status": "PENDING", "evidence_segment_id": "seg-hg1"}],
        "risks": [],
        "unresolved_questions": [],
    }

    hinglish_payload = adapter._sanitize_and_validate_payload(hinglish_data)
    assert "Rahul" in hinglish_payload.executive_summary
    assert "Migration Sync" in hinglish_payload.detailed_summary  # Losslessly normalized from dict to markdown
    assert len(hinglish_payload.decisions) == 1
    assert hinglish_payload.action_items[0].owner_name == "Rahul"


def test_10_decisions_vs_proposals_discrimination():
    """10. Ensure unconfirmed proposals and suggestions are not treated as firm decisions."""
    adapter = OllamaExtractor()

    raw_data = {
        "executive_summary": "Team evaluated database options.",
        "detailed_summary": None,
        "key_takeaways": [],
        "topics": [],
        "decisions": [
            {"description": "We officially decide to adopt PostgreSQL 16 on Aurora.", "rationale_or_context": "ACID compliance", "decided_by": "Rajesh"},
            {"description": "SUGGESTION: We could maybe consider MongoDB later for logs.", "rationale_or_context": "Just an idea", "decided_by": "Intern"},
            {"description": "PROPOSAL: How about trying Redis caching next quarter?", "rationale_or_context": "Unvoted proposal", "decided_by": "Amit"},
            {"description": "Approved increasing staging environment memory to 32GB.", "rationale_or_context": "OOM prevention", "decided_by": "DevOps Lead"},
        ],
        "action_items": [],
        "risks": [],
        "unresolved_questions": [],
    }

    payload = adapter._sanitize_and_validate_payload(raw_data)

    # Suggestions and proposals must be filtered out
    descriptions = [d.description for d in payload.decisions]
    assert "We officially decide to adopt PostgreSQL 16 on Aurora." in descriptions
    assert "Approved increasing staging environment memory to 32GB." in descriptions
    assert not any("SUGGESTION" in d or "PROPOSAL" in d for d in descriptions)
    assert len(payload.decisions) == 2


def test_11_missing_responsible_persons():
    """11. Action items with missing assignees must be safely accepted with owner_name=None."""
    adapter = OllamaExtractor()

    raw_data = {
        "executive_summary": "Team sync summary.",
        "detailed_summary": None,
        "key_takeaways": [],
        "topics": [],
        "decisions": [],
        "action_items": [
            {"description": "Review API documentation", "owner_name": None, "deadline": None, "status": "PENDING"},
            {"description": "Update Dockerfile base image", "owner_name": "Unassigned", "deadline": "Tomorrow", "status": "PENDING"},
            {"description": "Send meeting minutes", "owner_name": "N/A", "deadline": None, "status": "IN_PROGRESS"},
            {"description": "Perform penetration test", "owner_name": "Priya", "deadline": "Friday", "status": "PENDING"},
        ],
        "risks": [],
        "unresolved_questions": [],
    }

    payload = adapter._sanitize_and_validate_payload(raw_data)
    assert len(payload.action_items) == 4
    # Unassigned, N/A, and None should be normalized to None without error
    assert payload.action_items[0].owner_name is None
    assert payload.action_items[1].owner_name is None
    assert payload.action_items[2].owner_name is None
    assert payload.action_items[3].owner_name == "Priya"


def test_12_preservation_of_original_evidence_and_timestamps(db_session: Session):
    """12. Strict preservation of evidence segment IDs and timestamps into persistent database records."""
    meeting, transcript, seg = create_test_meeting(db_session, "Evidence Grounding Meeting")

    mock_ollama_extractor = MagicMock(spec=OllamaExtractor)
    mock_ollama_extractor.model_name = "qwen3:8b"
    mock_ollama_extractor.extract.return_value = StructuredExtractionPayload(
        executive_summary="Evidence grounding verified.",
        detailed_summary=None,
        key_takeaways=[],
        topics=[],
        decisions=[DecisionCreate(
            description="Deploy on AWS",
            rationale_or_context="Cloud stability",
            decided_by="Aarav",
            evidence_segment_id=seg.id,
            timestamp_seconds=seg.start_time,
        )],
        action_items=[ActionItemCreate(
            description="Set up the cluster",
            owner_name="Priya",
            deadline=None,
            status="PENDING",
            evidence_segment_id=seg.id,
            timestamp_seconds=seg.start_time,
        )],
        risks=[],
        unresolved_questions=[],
    )

    result = extract_meeting_intelligence(
        meeting_id=meeting.id,
        db=db_session,
        provider_override="ollama",
        ollama_extractor=mock_ollama_extractor,
        persist=True,
    )

    assert result.has_intelligence is True
    # Verify in DB
    db_decision = db_session.query(Decision).filter(Decision.meeting_id == meeting.id).first()
    assert db_decision is not None
    assert db_decision.evidence_segment_id == seg.id
    assert db_decision.timestamp_seconds == seg.start_time

    db_action = db_session.query(ActionItem).filter(ActionItem.meeting_id == meeting.id).first()
    assert db_action is not None
    assert db_action.evidence_segment_id == seg.id
    assert db_action.timestamp_seconds == seg.start_time
