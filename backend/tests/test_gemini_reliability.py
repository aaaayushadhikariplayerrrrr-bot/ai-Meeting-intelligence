"""Dedicated reliability, timeout, retry, and fallback test suite for Gemini extraction.

Verifies:
1. A short transcript extraction.
2. A longer multi-chunk transcript extraction.
3. Successful Gemini extraction with evidence grounding.
4. Transient timeout with exponential backoff retry.
5. Rate limit (HTTP 429) handling and retry.
6. Gemini failure followed by deterministic fallback.
7. Malformed Gemini JSON response handling and repair.
"""

import json
from unittest.mock import MagicMock, patch
import httpx
import pytest

from app.core.config import settings
from app.models.identity import Organization, User, OrganizationMembership, OrgRole
from app.models.meeting import Meeting
from app.models.transcription import Transcript, TranscriptSegment, TranscriptType
from app.schemas.intelligence import (
    StructuredExtractionPayload,
    DecisionCreate,
    ActionItemCreate,
    RiskCreate,
    UnresolvedQuestionCreate,
)
from app.services.intelligence.base import ProviderExecutionError, ProviderType
from app.services.intelligence.gemini_adapter import GeminiExtractor
from app.services.intelligence.engine import extract_meeting_intelligence
from app.services.intelligence.deterministic import DeterministicExtractor


# ==============================================================================
# Helper Mock Factories
# ==============================================================================

def make_valid_gemini_json():
    return {
        "executive_summary": "The engineering team agreed to migrate the database to PostgreSQL on AWS Aurora.",
        "detailed_summary": "The team reviewed options between MongoDB and PostgreSQL. PostgreSQL 16 was selected.",
        "key_takeaways": [
            "PostgreSQL 16 selected for relational compliance",
            "Schema migration to be finalized by Friday",
        ],
        "topics": [
            {"title": "Database Architecture", "description": "Review of database options", "start_time": 0.0, "end_time": 30.0}
        ],
        "decisions": [
            {
                "description": "Migrate primary storage to PostgreSQL 16 on Aurora",
                "rationale_or_context": "Relational integrity and ACID compliance",
                "decided_by": "Alex Rivera",
                "evidence_segment_id": "seg-001",
            }
        ],
        "action_items": [
            {
                "description": "Provision AWS Aurora PostgreSQL cluster in staging",
                "owner_name": "Sarah Chen",
                "deadline": "2026-12-05T18:00:00Z",
                "status": "PENDING",
                "evidence_segment_id": "seg-002",
            }
        ],
        "risks": [
            {
                "description": "Potential cutover replication lag causing read downtime",
                "severity": "HIGH",
                "mitigation_suggestion": "Set up CloudWatch alert for replication lag",
                "evidence_segment_id": "seg-002",
            }
        ],
        "unresolved_questions": [
            {
                "question_text": "Does existing Terraform module support zero-downtime failover?",
                "raised_by": "Alex Rivera",
                "status": "OPEN",
                "evidence_segment_id": "seg-003",
            }
        ],
    }


def make_gemini_http_response(json_data, status_code=200):
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.json.return_value = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"text": json.dumps(json_data)}
                    ]
                }
            }
        ]
    }
    return resp


# ==============================================================================
# 1. Short Transcript Extraction
# ==============================================================================

def test_short_transcript_extraction():
    """Verify GeminiExtractor processes a short 2-segment transcript cleanly."""
    extractor = GeminiExtractor(api_key="fake-key", model_name="gemini-3.8-flash")
    mock_resp = make_gemini_http_response(make_valid_gemini_json())

    seg1 = TranscriptSegment(id="seg-001", transcript_id="t-1", sequence_order=1, start_time=0.0, end_time=15.0, text="We decided on PostgreSQL.", language_code="en")
    seg2 = TranscriptSegment(id="seg-002", transcript_id="t-1", sequence_order=2, start_time=15.0, end_time=30.0, text="Sarah will provision the cluster.", language_code="en")

    with patch.object(extractor.client, "post", return_value=mock_resp):
        payload = extractor.extract(
            segments=[seg1, seg2],
            meeting_title="Short Sync",
            output_language="en",
        )

    assert isinstance(payload, StructuredExtractionPayload)
    assert "PostgreSQL" in payload.executive_summary
    assert len(payload.decisions) == 1
    assert payload.decisions[0].evidence_segment_id == "seg-001"
    assert len(payload.action_items) == 1
    assert payload.action_items[0].owner_name == "Sarah Chen"


# ==============================================================================
# 2. Longer Transcript Multi-Chunk Extraction
# ==============================================================================

def test_longer_transcript_multi_chunk_extraction(db_session):
    """Verify engine processes a longer transcript across multiple chunks, merging safely."""
    org = Organization(name="Long Transcript Org", slug="long-org")
    user = User(email="long_user@example.com", hashed_password="hashed_password", full_name="Long User")
    db_session.add_all([org, user])
    db_session.flush()

    meeting = Meeting(organization_id=org.id, title="Quarterly Engineering Strategy")
    db_session.add(meeting)
    db_session.flush()

    transcript = Transcript(
        meeting_id=meeting.id,
        transcript_type=TranscriptType.CLEANED_REDACTED.value,
        version=1,
    )
    db_session.add(transcript)
    db_session.flush()

    # Create 30 segments to simulate longer discussion
    segments = []
    for i in range(30):
        seg = TranscriptSegment(
            transcript_id=transcript.id,
            sequence_order=i,
            start_time=float(i * 30),
            end_time=float((i + 1) * 30),
            text=f"Discussion item {i}: Topic coverage and technical alignment regarding system architecture and reliability goals.",
            language_code="en",
        )
        segments.append(seg)
    db_session.add_all(segments)
    db_session.commit()

    extractor = GeminiExtractor(api_key="fake-key", model_name="gemini-3.8-flash")
    mock_resp = make_gemini_http_response(make_valid_gemini_json())

    with patch.object(extractor.client, "post", return_value=mock_resp):
        # Force low max_chars to test chunking & merging
        with patch.object(settings, "INTELLIGENCE_MAX_TRANSCRIPT_CHARS", 500):
            result = extract_meeting_intelligence(
                meeting_id=meeting.id,
                db=db_session,
                provider_override="gemini",
                gemini_extractor=extractor,
            )

    assert result.has_intelligence is True
    assert result.provider_used == "gemini"
    assert result.is_heuristic is False
    assert result.summary is not None
    assert len(result.decisions) >= 1


# ==============================================================================
# 3. Successful Gemini Extraction with Evidence Grounding
# ==============================================================================

def test_gemini_successful_extraction_and_grounding(db_session):
    """Verify Gemini extraction maps decisions and action items to grounded DB segments."""
    org = Organization(name="Grounding Org", slug="ground-org")
    user = User(email="ground_user@example.com", hashed_password="hashed_password", full_name="Ground User")
    db_session.add_all([org, user])
    db_session.flush()

    meeting = Meeting(organization_id=org.id, title="Architecture Alignment")
    db_session.add(meeting)
    db_session.flush()

    transcript = Transcript(
        meeting_id=meeting.id,
        transcript_type=TranscriptType.CLEANED_REDACTED.value,
        version=1,
    )
    db_session.add(transcript)
    db_session.flush()

    seg1 = TranscriptSegment(
        id="seg-001",
        transcript_id=transcript.id,
        sequence_order=1,
        start_time=10.0,
        end_time=25.0,
        text="Alex: We decided to proceed with PostgreSQL on Aurora.",
        language_code="en",
    )
    seg2 = TranscriptSegment(
        id="seg-002",
        transcript_id=transcript.id,
        sequence_order=2,
        start_time=25.0,
        end_time=40.0,
        text="Sarah: I will set up the staging cluster by Friday.",
        language_code="en",
    )
    db_session.add_all([seg1, seg2])
    db_session.commit()

    extractor = GeminiExtractor(api_key="fake-key", model_name="gemini-3.8-flash")
    mock_resp = make_gemini_http_response(make_valid_gemini_json())

    with patch.object(extractor.client, "post", return_value=mock_resp):
        result = extract_meeting_intelligence(
            meeting_id=meeting.id,
            db=db_session,
            provider_override="gemini",
            gemini_extractor=extractor,
        )

    assert result.has_intelligence is True
    assert result.provider_used == "gemini"
    assert len(result.decisions) == 1
    # Evidence must match verified segment ID
    assert result.decisions[0].evidence_segment_id == "seg-001"
    assert result.decisions[0].timestamp_seconds == 10.0


# ==============================================================================
# 4. Transient Timeout and Exponential Backoff Retry
# ==============================================================================

def test_transient_timeout_and_successful_retry():
    """Verify GeminiExtractor retries on transient ReadTimeout and succeeds on 2nd attempt."""
    extractor = GeminiExtractor(
        api_key="fake-key",
        model_name="gemini-3.8-flash",
        timeout=120.0,
        max_retries=2,
    )

    mock_success = make_gemini_http_response(make_valid_gemini_json())
    timeout_err = httpx.ReadTimeout("Request timed out", request=MagicMock())

    # Fail on attempt 0 with timeout, succeed on attempt 1
    with patch.object(extractor.client, "post", side_effect=[timeout_err, mock_success]) as mock_post:
        with patch("time.sleep", return_value=None) as mock_sleep:
            payload = extractor.extract_payload("Alex: We agreed on PostgreSQL.", output_language="en")

    assert mock_post.call_count == 2
    assert mock_sleep.call_count == 1
    assert isinstance(payload, StructuredExtractionPayload)
    assert len(payload.decisions) == 1


# ==============================================================================
# 5. API Rate-Limit Handling (HTTP 429)
# ==============================================================================

def test_rate_limit_429_retry_and_recovery():
    """Verify HTTP 429 triggers exponential backoff retry and succeeds when rate limit clears."""
    extractor = GeminiExtractor(
        api_key="fake-key",
        model_name="gemini-3.8-flash",
        max_retries=2,
    )

    resp_429 = MagicMock(spec=httpx.Response)
    resp_429.status_code = 429
    resp_429.text = "Quota exceeded temporarily"
    resp_429.request = MagicMock()

    mock_success = make_gemini_http_response(make_valid_gemini_json())

    # Attempt 0: 429, Attempt 1: 200 OK
    with patch.object(extractor.client, "post", side_effect=[resp_429, mock_success]) as mock_post:
        with patch("time.sleep", return_value=None) as mock_sleep:
            payload = extractor.extract_payload("Priya: We finalized the release schedule.", output_language="en")

    assert mock_post.call_count == 2
    assert mock_sleep.call_count == 1
    assert isinstance(payload, StructuredExtractionPayload)


# ==============================================================================
# 6. Gemini Failure Followed by Deterministic Fallback
# ==============================================================================

def test_gemini_persistent_failure_triggers_deterministic_fallback(db_session):
    """Verify that when Gemini completely fails across all retries, deterministic fallback is executed seamlessly."""
    org = Organization(name="Fallback Org", slug="fallback-org")
    user = User(email="fb_user@example.com", hashed_password="hashed_password", full_name="FB User")
    db_session.add_all([org, user])
    db_session.flush()

    meeting = Meeting(organization_id=org.id, title="Critical Project Planning")
    db_session.add(meeting)
    db_session.flush()

    transcript = Transcript(
        meeting_id=meeting.id,
        transcript_type=TranscriptType.CLEANED_REDACTED.value,
        version=1,
    )
    db_session.add(transcript)
    db_session.flush()

    seg1 = TranscriptSegment(
        id="seg-10",
        transcript_id=transcript.id,
        sequence_order=1,
        start_time=0.0,
        end_time=15.0,
        text="Alex: We officially decide to deploy on Friday. Sarah will monitor the rollout.",
        language_code="en",
    )
    db_session.add(seg1)
    db_session.commit()

    extractor = GeminiExtractor(api_key="fake-key", model_name="gemini-3.8-flash", max_retries=0)
    timeout_err = httpx.ReadTimeout("Google Gemini API request timed out after 120.0s.", request=MagicMock())

    with patch.object(extractor.client, "post", side_effect=timeout_err), \
         patch("app.services.intelligence.engine.OllamaExtractor.extract", side_effect=ProviderExecutionError("Ollama daemon offline")):
        result = extract_meeting_intelligence(
            meeting_id=meeting.id,
            db=db_session,
            provider_override="gemini",
            gemini_extractor=extractor,
            allow_fallback=True,
        )

    # Must cleanly succeed using deterministic fallback
    assert result.has_intelligence is True
    assert result.provider_used == "deterministic"
    assert result.is_heuristic is True
    assert result.fallback_warning is not None
    assert "Gemini extraction failed" in result.fallback_warning
    assert "timed out" in result.fallback_warning


# ==============================================================================
# 7. Malformed Gemini JSON Response Handling & Repair
# ==============================================================================

def test_parse_json_safely_handles_markdown_and_trailing_commas():
    """Verify _parse_json_safely repairs markdown blocks, preambles, and trailing commas."""
    extractor = GeminiExtractor(api_key="fake-key")

    # Case A: Markdown code block with preamble and postscript
    raw_markdown = """Here is the structured intelligence you requested:
```json
{
  "executive_summary": "Preamble test summary",
  "decisions": []
}
```
Hope this is helpful!"""
    parsed_a = extractor._parse_json_safely(raw_markdown)
    assert parsed_a["executive_summary"] == "Preamble test summary"

    # Case B: Trailing comma in arrays and objects (common LLM flaw)
    raw_trailing = """{
  "executive_summary": "Trailing comma summary",
  "decisions": [
    {"description": "Decision 1",},
  ],
}"""
    parsed_b = extractor._parse_json_safely(raw_trailing)
    assert parsed_b["executive_summary"] == "Trailing comma summary"
    assert len(parsed_b["decisions"]) == 1

    # Case C: Completely malformed non-JSON
    raw_invalid = "This is plain unstructured English text with no JSON brackets at all."
    with pytest.raises(ProviderExecutionError, match="could not be parsed as valid JSON"):
        extractor._parse_json_safely(raw_invalid)
