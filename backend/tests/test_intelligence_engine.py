"""Comprehensive test suite for Phase 7.3 Core Structured Intelligence Extraction Engine.

Covers:
- Safe transcript input enforcement (rejects ORIGINAL, CLEANED; accepts REDACTED, CLEANED_REDACTED)
- Provider selection (disabled, deterministic, ollama)
- Deterministic extraction across English, Hindi, Gujarati, and Hinglish segments
- Evidence attribution grounding and rejection of fabricated segment IDs
- Timestamp derivation strictly from database segments
- Ollama adapter with mocked HTTP responses (success, connection error, timeout, HTTP 404)
- Invalid JSON and schema validation error handling
- Fallback transparency and warning preservation
- Segment chunking, partial-processing limits, and multi-chunk deduplication
- Database persistence and re-extraction duplicate prevention
"""

import json
from datetime import datetime, timezone
import httpx
import pytest
from sqlalchemy.orm import Session

from app.models.identity import Organization, User
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
    StructuredExtractionPayload,
    TopicCreate,
    DecisionCreate,
    ActionItemCreate,
    RiskCreate,
    UnresolvedQuestionCreate,
)
from app.services.intelligence.base import (
    ProviderType,
    IntelligenceError,
    SafeTranscriptRequiredError,
    IntelligenceDisabledError,
    ProviderExecutionError,
)
from app.services.intelligence.deterministic import DeterministicExtractor
from app.services.intelligence.ollama_adapter import OllamaExtractor
from app.services.intelligence.evidence import ground_and_verify_evidence
from app.services.intelligence.chunking import chunk_transcript_segments, merge_extracted_payloads
from app.services.intelligence.engine import (
    resolve_privacy_safe_transcript,
    extract_meeting_intelligence,
)


# ==============================================================================
# 1. Privacy-Safe Transcript Enforcement Tests
# ==============================================================================

def test_resolve_transcript_rejects_unredacted_original_and_cleaned(db_session):
    """Verify that intelligence extraction strictly rejects ORIGINAL and CLEANED transcripts."""
    org = Organization(name="Privacy Test Org", slug="privacy-test-org")
    meeting = Meeting(organization_id=org.id, title="Unsafe Transcript Test")
    db_session.add_all([org, meeting])
    db_session.flush()

    # 1. ORIGINAL transcript
    t_orig = Transcript(
        meeting_id=meeting.id,
        transcript_type=TranscriptType.ORIGINAL.value,
        version=1,
    )
    # 2. CLEANED transcript (stutter-removed but unredacted)
    t_clean = Transcript(
        meeting_id=meeting.id,
        transcript_type=TranscriptType.CLEANED.value,
        version=1,
    )
    db_session.add_all([t_orig, t_clean])
    db_session.commit()

    # Reject explicit ORIGINAL
    with pytest.raises(SafeTranscriptRequiredError, match="strictly requires privacy-safe transcripts"):
        resolve_privacy_safe_transcript(meeting.id, db_session, requested_transcript_id=t_orig.id)

    # Reject explicit CLEANED
    with pytest.raises(SafeTranscriptRequiredError, match="strictly requires privacy-safe transcripts"):
        resolve_privacy_safe_transcript(meeting.id, db_session, requested_transcript_id=t_clean.id)

    # When no redacted version exists, auto-resolution must fail with clear instruction
    with pytest.raises(SafeTranscriptRequiredError, match="No privacy-safe.*transcript available"):
        resolve_privacy_safe_transcript(meeting.id, db_session, requested_transcript_id=None)


def test_resolve_transcript_prefers_cleaned_redacted_over_redacted(db_session):
    """Verify priority: CLEANED_REDACTED > REDACTED, and picks latest version."""
    org = Organization(name="Org Priority", slug="org-priority")
    meeting = Meeting(organization_id=org.id, title="Priority Meeting")
    db_session.add_all([org, meeting])
    db_session.flush()

    t_red_v1 = Transcript(meeting_id=meeting.id, transcript_type=TranscriptType.REDACTED.value, version=1)
    t_red_v2 = Transcript(meeting_id=meeting.id, transcript_type=TranscriptType.REDACTED.value, version=2)
    t_clean_red_v1 = Transcript(meeting_id=meeting.id, transcript_type=TranscriptType.CLEANED_REDACTED.value, version=1)
    db_session.add_all([t_red_v1, t_red_v2, t_clean_red_v1])
    db_session.commit()

    # Should select CLEANED_REDACTED
    resolved = resolve_privacy_safe_transcript(meeting.id, db_session)
    assert resolved.id == t_clean_red_v1.id
    assert resolved.transcript_type == TranscriptType.CLEANED_REDACTED.value

    # If CLEANED_REDACTED does not exist, select latest REDACTED (v2)
    db_session.delete(t_clean_red_v1)
    db_session.commit()

    resolved_red = resolve_privacy_safe_transcript(meeting.id, db_session)
    assert resolved_red.id == t_red_v2.id
    assert resolved_red.version == 2


# ==============================================================================
# 2. Evidence Grounding & Fabricated Segment ID Rejection Tests
# ==============================================================================

def test_evidence_grounding_preserves_valid_and_strips_fabricated():
    """Verify genuine segment IDs derive timestamps from database segments, while fake IDs are stripped."""
    seg1 = TranscriptSegment(id="seg-1", start_time=12.5, end_time=18.0, text="We decided to launch next sprint.")
    seg2 = TranscriptSegment(id="seg-2", start_time=45.0, end_time=52.0, text="I will handle the deployment.")

    payload = StructuredExtractionPayload(
        executive_summary="Summary",
        topics=[],
        decisions=[
            # Genuine segment
            DecisionCreate(description="Launch next sprint", evidence_segment_id="seg-1", timestamp_seconds=999.9),
            # Fabricated segment ID
            DecisionCreate(description="Unverified decision", evidence_segment_id="fabricated-fake-id", timestamp_seconds=123.4),
        ],
        action_items=[
            # Genuine segment
            ActionItemCreate(description="Handle deployment", owner_name="Dev", evidence_segment_id="seg-2"),
            # Fabricated segment ID
            ActionItemCreate(description="Fake action", evidence_segment_id="bogus-uuid-1234"),
        ],
        risks=[
            RiskCreate(description="Risk genuine", evidence_segment_id="seg-1"),
            RiskCreate(description="Risk fake", evidence_segment_id="non-existent-seg"),
        ],
        unresolved_questions=[
            UnresolvedQuestionCreate(question_text="Valid question?", evidence_segment_id="seg-2"),
            UnresolvedQuestionCreate(question_text="Fake question?", evidence_segment_id="ghost-id"),
        ],
    )

    grounded = ground_and_verify_evidence(payload, [seg1, seg2])

    # 1. Genuine decision: verified and timestamp derived from seg1.start_time (12.5, not 999.9)
    assert grounded.decisions[0].evidence_segment_id == "seg-1"
    assert grounded.decisions[0].timestamp_seconds == 12.5

    # 2. Fabricated decision: evidence stripped
    assert grounded.decisions[1].evidence_segment_id is None
    assert grounded.decisions[1].timestamp_seconds is None

    # 3. Genuine action item: verified
    assert grounded.action_items[0].evidence_segment_id == "seg-2"
    assert grounded.action_items[0].timestamp_seconds == 45.0

    # 4. Fabricated action item: evidence stripped
    assert grounded.action_items[1].evidence_segment_id is None
    assert grounded.action_items[1].timestamp_seconds is None

    # 5. Risks
    assert grounded.risks[0].evidence_segment_id == "seg-1"
    assert grounded.risks[1].evidence_segment_id is None

    # 6. Questions
    assert grounded.unresolved_questions[0].evidence_segment_id == "seg-2"
    assert grounded.unresolved_questions[1].evidence_segment_id is None


# ==============================================================================
# 3. Deterministic Multilingual Extractor Tests
# ==============================================================================

def test_deterministic_extractor_multilingual_capabilities():
    """Verify deterministic extractor identifies commitments, decisions, risks, and questions in EN, HI, GU, and Hinglish."""
    extractor = DeterministicExtractor()

    segments = [
        TranscriptSegment(id="s1", sequence_order=0, start_time=0.0, end_time=10.0, text="Welcome everyone to the quarterly sync.", language_code="en"),
        # English decision
        TranscriptSegment(id="s2", sequence_order=1, start_time=10.5, end_time=20.0, text="We decided to deploy faster-whisper on RTX 5060.", language_code="en"),
        # Hinglish action item
        TranscriptSegment(id="s3", sequence_order=2, start_time=20.5, end_time=30.0, text="I will take care of quantization and testing.", language_code="hi-en"),
        # Hindi decision
        TranscriptSegment(id="s4", sequence_order=3, start_time=30.5, end_time=40.0, text="Yeh faisla liya ki Hindi ASR ko enable rakhenge.", language_code="hi"),
        # Gujarati risk
        TranscriptSegment(id="s5", sequence_order=4, start_time=40.5, end_time=50.0, text="Aa blocker chhe, 8GB VRAM might be tight under load.", language_code="gu"),
        # Multilingual question
        TranscriptSegment(id="s6", sequence_order=5, start_time=50.5, end_time=60.0, text="How will we handle concurrent requests?", language_code="en"),
    ]

    payload = extractor.extract(segments, meeting_title="Platform Sync")

    # Decisions detected
    assert len(payload.decisions) >= 2
    decision_texts = " ".join([d.description for d in payload.decisions])
    assert "RTX 5060" in decision_texts
    assert "faisla liya" in decision_texts or "Hindi ASR" in decision_texts

    # Action items detected
    assert len(payload.action_items) >= 1
    assert "quantization" in payload.action_items[0].description
    assert payload.action_items[0].evidence_segment_id == "s3"

    # Risks detected
    assert len(payload.risks) >= 1
    assert payload.risks[0].severity == RiskSeverity.HIGH

    # Questions detected
    assert len(payload.unresolved_questions) >= 1
    assert "concurrent requests" in payload.unresolved_questions[0].question_text

    # Executive summary synthesized
    assert "Platform Sync" in payload.executive_summary
    assert len(payload.topics) >= 1
    assert len(payload.key_takeaways) >= 1


# ==============================================================================
# 4. Ollama Adapter Tests (Mocked HTTP)
# ==============================================================================

def test_ollama_adapter_mocked_success():
    """Verify Ollama adapter parses valid structured JSON output."""
    mock_llm_json = {
        "executive_summary": "The board approved Q4 budget.",
        "detailed_summary": "Extensive discussions on roadmap.",
        "key_takeaways": ["Budget approved", "Hiring accelerated"],
        "topics": [{"title": "Budget Review", "description": "Financial sync", "start_time": 0.0, "end_time": 30.0}],
        "decisions": [{"description": "Approve $500k budget", "rationale_or_context": "Unanimous vote", "decided_by": "Board", "evidence_segment_id": "seg-10"}],
        "action_items": [{"description": "File financial report", "owner_name": "CFO", "deadline": None, "status": "PENDING", "evidence_segment_id": "seg-10"}],
        "risks": [{"description": "Tax regulation shifts", "severity": "MEDIUM", "mitigation_suggestion": "Consult legal", "evidence_segment_id": "seg-10"}],
        "unresolved_questions": [{"question_text": "Will Q1 require extra capital?", "raised_by": "Auditor", "status": "OPEN", "evidence_segment_id": "seg-10"}],
    }

    def mock_handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/generate"
        body = json.loads(request.content)
        assert body["model"] in {"llama3.2:3b", "qwen3:8b"}
        assert body["format"] == "json"
        return httpx.Response(200, json={"response": json.dumps(mock_llm_json)})

    transport = httpx.MockTransport(mock_handler)
    mock_client = httpx.Client(transport=transport)

    adapter = OllamaExtractor(client=mock_client)
    seg = TranscriptSegment(id="seg-10", start_time=5.0, end_time=15.0, text="We approve the budget.")
    payload = adapter.extract([seg], meeting_title="Budget Meeting")

    assert payload.executive_summary == "The board approved Q4 budget."
    assert len(payload.decisions) == 1
    assert payload.decisions[0].decided_by == "Board"
    assert payload.action_items[0].owner_name == "CFO"


def test_ollama_adapter_connection_error():
    """Verify Ollama adapter gracefully handles connection refusal."""
    def mock_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection refused by localhost:11434")

    transport = httpx.MockTransport(mock_handler)
    mock_client = httpx.Client(transport=transport)

    adapter = OllamaExtractor(client=mock_client)
    seg = TranscriptSegment(id="seg-1", start_time=0.0, end_time=5.0, text="Test utterance")

    with pytest.raises(ProviderExecutionError, match="Ollama server is not running or unreachable"):
        adapter.extract([seg])


def test_ollama_adapter_timeout_error():
    """Verify Ollama adapter gracefully handles request timeout."""
    def mock_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("Read timed out after 60s")

    transport = httpx.MockTransport(mock_handler)
    mock_client = httpx.Client(transport=transport)

    adapter = OllamaExtractor(client=mock_client)
    seg = TranscriptSegment(id="seg-1", start_time=0.0, end_time=5.0, text="Test utterance")

    with pytest.raises(ProviderExecutionError, match="timed out"):
        adapter.extract([seg])


def test_ollama_adapter_model_not_found_404():
    """Verify Ollama adapter reports missing model with pull command suggestion."""
    def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text='{"error":"model \'llama3.2:3b\' not found"}')

    transport = httpx.MockTransport(mock_handler)
    mock_client = httpx.Client(transport=transport)

    adapter = OllamaExtractor(client=mock_client)
    seg = TranscriptSegment(id="seg-1", start_time=0.0, end_time=5.0, text="Test utterance")

    with pytest.raises(ProviderExecutionError, match="Please pull it via 'ollama pull"):
        adapter.extract([seg])


def test_ollama_adapter_invalid_json():
    """Verify Ollama adapter rejects non-JSON responses."""
    def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"response": "This is plain text not JSON"})

    transport = httpx.MockTransport(mock_handler)
    mock_client = httpx.Client(transport=transport)

    adapter = OllamaExtractor(client=mock_client)
    seg = TranscriptSegment(id="seg-1", start_time=0.0, end_time=5.0, text="Test utterance")

    with pytest.raises(ProviderExecutionError, match="Ollama output was not valid JSON"):
        adapter.extract([seg])


# ==============================================================================
# 5. Chunking, Resource Management & Deduplication Tests
# ==============================================================================

def test_chunking_and_partial_processing_limits():
    """Verify segment chunking preserves boundaries and enforces max_chunks limit."""
    segments = [
        TranscriptSegment(id=f"seg-{i}", start_time=i * 10.0, end_time=(i + 1) * 10.0, text=f"Utterance number {i} with some content.")
        for i in range(10)
    ]

    # Max 2 chunks with small character threshold to force chunking
    chunks, is_partial, reason = chunk_transcript_segments(segments, max_chars=80, max_chunks=2)
    assert len(chunks) == 2
    assert is_partial is True
    assert "limit of 2 chunks" in reason


def test_merge_extracted_payloads_deduplicates():
    """Verify multi-chunk extraction payload merging eliminates duplicate items."""
    p1 = StructuredExtractionPayload(
        executive_summary="Chunk 1 overview.",
        key_takeaways=["Point A", "Point B"],
        topics=[TopicCreate(title="Roadmap")],
        decisions=[DecisionCreate(description="Deploy on RTX 5060")],
        action_items=[ActionItemCreate(description="Benchmark int8", owner_name="Dev")],
        risks=[RiskCreate(description="VRAM constraint")],
        unresolved_questions=[UnresolvedQuestionCreate(question_text="Timeline?")],
    )
    # Chunk 2 has duplicates of decision, action item, risk, and question
    p2 = StructuredExtractionPayload(
        executive_summary="Chunk 2 overview.",
        key_takeaways=["Point B", "Point C"],
        topics=[TopicCreate(title="Roadmap"), TopicCreate(title="Security")],
        decisions=[DecisionCreate(description="Deploy on RTX 5060"), DecisionCreate(description="Enable Argon2id")],
        action_items=[ActionItemCreate(description="Benchmark int8", owner_name="Dev"), ActionItemCreate(description="Audit logs", owner_name="QA")],
        risks=[RiskCreate(description="VRAM constraint"), RiskCreate(description="Thermal throttle")],
        unresolved_questions=[UnresolvedQuestionCreate(question_text="Timeline?"), UnresolvedQuestionCreate(question_text="Budget?")],
    )

    merged = merge_extracted_payloads([p1, p2])

    assert "Chunk 1 overview." in merged.executive_summary
    assert "Chunk 2 overview." in merged.executive_summary
    assert len(merged.key_takeaways) == 3  # Point A, B, C (Point B deduplicated)
    assert len(merged.topics) == 2        # Roadmap, Security (Roadmap deduplicated)
    assert len(merged.decisions) == 2     # RTX 5060, Argon2id (RTX 5060 deduplicated)
    assert len(merged.action_items) == 2  # Benchmark int8, Audit logs
    assert len(merged.risks) == 2         # VRAM constraint, Thermal throttle
    assert len(merged.unresolved_questions) == 2  # Timeline?, Budget?


# ==============================================================================
# 6. Engine Integration, Provider Fallback & Database Persistence Tests
# ==============================================================================

def test_extract_intelligence_disabled_provider_raises_error(db_session):
    """Verify disabled provider throws IntelligenceDisabledError."""
    org = Organization(name="Disabled Org", slug="disabled-org")
    meeting = Meeting(organization_id=org.id, title="Disabled Provider Test")
    db_session.add_all([org, meeting])
    db_session.flush()

    t_red = Transcript(meeting_id=meeting.id, transcript_type=TranscriptType.REDACTED.value, version=1)
    db_session.add(t_red)
    db_session.flush()

    seg = TranscriptSegment(transcript_id=t_red.id, sequence_order=0, start_time=0.0, end_time=5.0, text="Hello world", language_code="en")
    db_session.add(seg)
    db_session.commit()

    with pytest.raises(IntelligenceDisabledError, match="disabled in configuration"):
        extract_meeting_intelligence(
            meeting_id=meeting.id,
            organization_id=org.id,
            db=db_session,
            provider_override="disabled",
        )


def test_extract_intelligence_fallback_transparency(db_session):
    """Verify when Ollama fails, fallback to deterministic extractor is transparent and recorded."""
    org = Organization(name="Fallback Org", slug="fallback-org")
    meeting = Meeting(organization_id=org.id, title="Fallback Meeting")
    db_session.add_all([org, meeting])
    db_session.flush()

    t_red = Transcript(meeting_id=meeting.id, transcript_type=TranscriptType.CLEANED_REDACTED.value, version=1)
    db_session.add(t_red)
    db_session.flush()

    seg = TranscriptSegment(
        transcript_id=t_red.id,
        sequence_order=0,
        start_time=10.0,
        end_time=20.0,
        text="We decided to use deterministic fallback.",
        language_code="en",
    )
    db_session.add(seg)
    db_session.commit()

    # Mock failing Ollama client
    def mock_fail_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Ollama daemon offline")

    failing_transport = httpx.MockTransport(mock_fail_handler)
    failing_client = httpx.Client(transport=failing_transport)
    failing_ollama = OllamaExtractor(client=failing_client)

    result = extract_meeting_intelligence(
        meeting_id=meeting.id,
        organization_id=org.id,
        db=db_session,
        provider_override="ollama",
        allow_fallback=True,
        persist=False,
        ollama_extractor=failing_ollama,
    )

    # Provider used was deterministic
    assert result.provider_used == "deterministic"
    assert result.is_heuristic is True
    assert result.fallback_warning is not None
    assert "Ollama daemon offline" in result.fallback_warning
    assert len(result.decisions) >= 1
    assert result.decisions[0].evidence_segment_id == seg.id


def test_extract_intelligence_fallback_disabled_raises_provider_error(db_session):
    """Verify when fallback is disabled, Ollama failure raises ProviderExecutionError."""
    org = Organization(name="Strict Org", slug="strict-org")
    meeting = Meeting(organization_id=org.id, title="Strict Meeting")
    db_session.add_all([org, meeting])
    db_session.flush()

    t_red = Transcript(meeting_id=meeting.id, transcript_type=TranscriptType.REDACTED.value, version=1)
    db_session.add(t_red)
    db_session.flush()

    seg = TranscriptSegment(transcript_id=t_red.id, sequence_order=0, start_time=0.0, end_time=5.0, text="Important discussion", language_code="en")
    db_session.add(seg)
    db_session.commit()

    def mock_fail_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Ollama offline")

    failing_client = httpx.Client(transport=httpx.MockTransport(mock_fail_handler))
    failing_ollama = OllamaExtractor(client=failing_client)

    with pytest.raises(ProviderExecutionError, match="Ollama server is not running"):
        extract_meeting_intelligence(
            meeting_id=meeting.id,
            organization_id=org.id,
            db=db_session,
            provider_override="ollama",
            allow_fallback=False,
            persist=False,
            ollama_extractor=failing_ollama,
        )


def test_end_to_end_extraction_and_reprocessing_duplicate_prevention(db_session):
    """Verify full extraction persists entities to DB and reprocessing updates without duplicating rows."""
    org = Organization(name="Persistence Org", slug="persistence-org")
    meeting = Meeting(organization_id=org.id, title="Quarterly Strategy")
    db_session.add_all([org, meeting])
    db_session.flush()

    t_red = Transcript(meeting_id=meeting.id, transcript_type=TranscriptType.CLEANED_REDACTED.value, version=1)
    db_session.add(t_red)
    db_session.flush()

    segs = [
        TranscriptSegment(transcript_id=t_red.id, sequence_order=0, start_time=0.0, end_time=15.0, text="We decided to deploy Phase 7.3 on RTX 5060.", language_code="en"),
        TranscriptSegment(transcript_id=t_red.id, sequence_order=1, start_time=15.1, end_time=30.0, text="I will write unit tests by Friday.", language_code="en"),
        TranscriptSegment(transcript_id=t_red.id, sequence_order=2, start_time=30.1, end_time=45.0, text="Risk is potential memory pressure under concurrent load.", language_code="en"),
        TranscriptSegment(transcript_id=t_red.id, sequence_order=3, start_time=45.1, end_time=60.0, text="How will we measure inference latency?", language_code="en"),
    ]
    db_session.add_all(segs)
    db_session.commit()

    # 1. First Extraction
    res1 = extract_meeting_intelligence(
        meeting_id=meeting.id,
        organization_id=org.id,
        db=db_session,
        provider_override="deterministic",
        persist=True,
    )
    assert res1.summary is not None
    assert res1.summary.version == 1
    assert len(res1.decisions) == 1
    assert len(res1.action_items) == 1
    assert len(res1.risks) == 1
    assert len(res1.unresolved_questions) == 1

    # Verify counts in database
    assert db_session.query(Decision).filter(Decision.meeting_id == meeting.id).count() == 1
    assert db_session.query(ActionItem).filter(ActionItem.meeting_id == meeting.id).count() == 1
    assert db_session.query(Risk).filter(Risk.meeting_id == meeting.id).count() == 1
    assert db_session.query(UnresolvedQuestion).filter(UnresolvedQuestion.meeting_id == meeting.id).count() == 1

    # 2. Second Extraction (reprocessing)
    res2 = extract_meeting_intelligence(
        meeting_id=meeting.id,
        organization_id=org.id,
        db=db_session,
        provider_override="deterministic",
        persist=True,
    )
    assert res2.summary.version == 2

    # Verify counts in database did NOT duplicate!
    assert db_session.query(Decision).filter(Decision.meeting_id == meeting.id).count() == 1
    assert db_session.query(ActionItem).filter(ActionItem.meeting_id == meeting.id).count() == 1
    assert db_session.query(Risk).filter(Risk.meeting_id == meeting.id).count() == 1
    assert db_session.query(UnresolvedQuestion).filter(UnresolvedQuestion.meeting_id == meeting.id).count() == 1
    assert db_session.query(MeetingSummary).filter(MeetingSummary.meeting_id == meeting.id).count() == 1


def test_attribution_deterministic_provider_persisted(db_session: Session):
    """Verify deterministic provider attribution is accurately recorded and persisted."""
    org = Organization(name="Attribution Org 1", slug="attrib-org-1")
    meeting = Meeting(title="Attribution Meeting 1", organization=org)
    db_session.add_all([org, meeting])
    db_session.flush()

    t_red = Transcript(meeting_id=meeting.id, transcript_type=TranscriptType.CLEANED_REDACTED.value, version=1)
    db_session.add(t_red)
    db_session.flush()

    seg = TranscriptSegment(transcript_id=t_red.id, sequence_order=0, start_time=0.0, end_time=10.0, text="We decided on standard rules.", language_code="en")
    db_session.add(seg)
    db_session.commit()

    result = extract_meeting_intelligence(
        meeting_id=meeting.id,
        organization_id=org.id,
        db=db_session,
        provider_override="deterministic",
        persist=True,
    )

    assert result.provider_used == "deterministic"
    assert result.is_heuristic is True
    assert result.fallback_warning is None
    assert result.summary.provider_used == "deterministic"
    assert result.summary.is_heuristic is True
    assert result.summary.fallback_warning is None

    # Check database record
    summary_in_db = db_session.query(MeetingSummary).filter(MeetingSummary.meeting_id == meeting.id).first()
    assert summary_in_db is not None
    assert summary_in_db.provider_used == "deterministic"
    assert summary_in_db.is_heuristic is True
    assert summary_in_db.fallback_warning is None


def test_attribution_ollama_success_persisted(db_session: Session):
    """Verify successful Ollama execution records ollama provider and is_heuristic=False."""
    org = Organization(name="Attribution Org 2", slug="attrib-org-2")
    meeting = Meeting(title="Attribution Meeting 2", organization=org)
    db_session.add_all([org, meeting])
    db_session.flush()

    t_red = Transcript(meeting_id=meeting.id, transcript_type=TranscriptType.CLEANED_REDACTED.value, version=1)
    db_session.add(t_red)
    db_session.flush()

    seg = TranscriptSegment(transcript_id=t_red.id, sequence_order=0, start_time=0.0, end_time=10.0, text="The team decided to adopt Ollama.", language_code="en")
    db_session.add(seg)
    db_session.commit()

    class FakeOllamaSuccess:
        def extract(self, segments, meeting_title, output_language):
            return StructuredExtractionPayload(
                executive_summary="Ollama generated summary.",
                detailed_summary="Comprehensive discussion.",
                key_takeaways=["LLM takeaway 1"],
                topics=[],
                decisions=[
                    DecisionCreate(
                        description="Adopt Ollama",
                        decided_by="Architecture Team",
                        evidence_segment_id=segments[0].id,
                    )
                ],
                action_items=[],
                risks=[],
                unresolved_questions=[],
            )

    result = extract_meeting_intelligence(
        meeting_id=meeting.id,
        organization_id=org.id,
        db=db_session,
        provider_override="ollama",
        persist=True,
        ollama_extractor=FakeOllamaSuccess(),
    )

    assert result.provider_used == "ollama"
    assert result.is_heuristic is False
    assert result.fallback_warning is None
    assert result.summary.provider_used == "ollama"
    assert result.summary.is_heuristic is False
    assert result.summary.fallback_warning is None

    summary_in_db = db_session.query(MeetingSummary).filter(MeetingSummary.meeting_id == meeting.id).first()
    assert summary_in_db is not None
    assert summary_in_db.provider_used == "ollama"
    assert summary_in_db.is_heuristic is False
    assert summary_in_db.fallback_warning is None


def test_attribution_ollama_failure_followed_by_deterministic_fallback(db_session: Session):
    """Verify Ollama failure transparently falls back to deterministic with warning persisted."""
    org = Organization(name="Attribution Org 3", slug="attrib-org-3")
    meeting = Meeting(title="Attribution Meeting 3", organization=org)
    db_session.add_all([org, meeting])
    db_session.flush()

    t_red = Transcript(meeting_id=meeting.id, transcript_type=TranscriptType.CLEANED_REDACTED.value, version=1)
    db_session.add(t_red)
    db_session.flush()

    seg = TranscriptSegment(transcript_id=t_red.id, sequence_order=0, start_time=0.0, end_time=10.0, text="We decided on fallback strategy.", language_code="en")
    db_session.add(seg)
    db_session.commit()

    class FailingOllama:
        def extract(self, segments, meeting_title, output_language):
            raise ProviderExecutionError("Connection refused to Ollama daemon on 127.0.0.1:11434")

    result = extract_meeting_intelligence(
        meeting_id=meeting.id,
        organization_id=org.id,
        db=db_session,
        provider_override="ollama",
        allow_fallback=True,
        persist=True,
        ollama_extractor=FailingOllama(),
    )

    assert result.provider_used == "deterministic"
    assert result.is_heuristic is True
    assert result.fallback_warning is not None
    assert "Connection refused" in result.fallback_warning
    assert result.summary.provider_used == "deterministic"
    assert result.summary.is_heuristic is True
    assert result.summary.fallback_warning is not None

    summary_in_db = db_session.query(MeetingSummary).filter(MeetingSummary.meeting_id == meeting.id).first()
    assert summary_in_db is not None
    assert summary_in_db.provider_used == "deterministic"
    assert summary_in_db.is_heuristic is True
    assert "Connection refused" in summary_in_db.fallback_warning


def test_failed_re_extraction_preserves_previous_attribution(db_session: Session):
    """Verify failed re-extraction preserves previous intelligence and its attribution intact."""
    org = Organization(name="Attribution Org 4", slug="attrib-org-4")
    meeting = Meeting(title="Attribution Meeting 4", organization=org)
    db_session.add_all([org, meeting])
    db_session.flush()

    t_red = Transcript(meeting_id=meeting.id, transcript_type=TranscriptType.CLEANED_REDACTED.value, version=1)
    db_session.add(t_red)
    db_session.flush()

    seg = TranscriptSegment(transcript_id=t_red.id, sequence_order=0, start_time=0.0, end_time=10.0, text="Initial decision on architecture.", language_code="en")
    db_session.add(seg)
    db_session.commit()

    # 1. Successful initial extraction
    res1 = extract_meeting_intelligence(
        meeting_id=meeting.id,
        organization_id=org.id,
        db=db_session,
        provider_override="deterministic",
        persist=True,
    )
    assert res1.summary.version == 1
    assert res1.summary.provider_used == "deterministic"

    # 2. Second extraction fails (e.g. invalid provider without fallback)
    class ExplodingOllama:
        def extract(self, segments, meeting_title, output_language):
            raise ProviderExecutionError("Fatal unrecoverable LLM crash")

    with pytest.raises(ProviderExecutionError):
        extract_meeting_intelligence(
            meeting_id=meeting.id,
            organization_id=org.id,
            db=db_session,
            provider_override="ollama",
            allow_fallback=False,  # strictly fail
            persist=True,
            ollama_extractor=ExplodingOllama(),
        )

    # 3. Assert previous version and attribution are completely preserved
    preserved_summary = db_session.query(MeetingSummary).filter(MeetingSummary.meeting_id == meeting.id).first()
    assert preserved_summary is not None
    assert preserved_summary.version == 1
    assert preserved_summary.provider_used == "deterministic"
    assert preserved_summary.is_heuristic is True
    assert preserved_summary.fallback_warning is None
    assert "Attribution Meeting 4" in preserved_summary.executive_summary
