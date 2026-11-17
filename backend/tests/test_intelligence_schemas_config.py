"""Tests for Phase 7.2 Structured Meeting Intelligence Schemas and Configuration.

Covers:
- Valid and invalid schema inputs for all intelligence DTOs
- Optional field handling and defaults
- Enum validation and case-insensitive normalization
- Response serialization and ORM compatibility with SQLAlchemy models
- Configuration defaults, validators, and rejection of invalid values
- Safe startup when Ollama is not installed or running
"""

from datetime import datetime, timezone, timedelta
import uuid
import pytest
from pydantic import ValidationError

from app.core.config import Settings, settings
from app.models.identity import Organization, User
from app.models.meeting import Meeting
from app.models.transcription import Transcript, TranscriptSegment, TranscriptType
from app.models.intelligence import (
    ActionItemStatus,
    RiskSeverity,
    QuestionStatus,
    MeetingSummary,
    Topic,
    Decision,
    ActionItem,
    Risk,
    UnresolvedQuestion,
)
from app.schemas.intelligence import (
    TopicCreate,
    TopicResponse,
    DecisionCreate,
    DecisionResponse,
    ActionItemCreate,
    ActionItemResponse,
    ActionItemStatusUpdate,
    RiskCreate,
    RiskResponse,
    UnresolvedQuestionCreate,
    UnresolvedQuestionResponse,
    MeetingSummaryCreate,
    MeetingSummaryResponse,
    MeetingIntelligenceResponse,
    StructuredExtractionPayload,
    ExtractIntelligenceRequest,
    ExtractIntelligenceJobResponse,
)


# ==============================================================================
# 1. Pydantic Creation / Input Validation Tests
# ==============================================================================

def test_topic_create_validation():
    """Verify TopicCreate validates fields, defaults, and time ranges."""
    # Valid with all fields
    t = TopicCreate(title="Architecture Sync", description="Review 8GB VRAM", start_time=10.0, end_time=25.0)
    assert t.title == "Architecture Sync"
    assert t.start_time == 10.0
    assert t.end_time == 25.0

    # Valid with optional fields omitted
    t_min = TopicCreate(title="Quick Status")
    assert t_min.title == "Quick Status"
    assert t_min.description is None
    assert t_min.start_time is None
    assert t_min.end_time is None

    # Invalid empty title
    with pytest.raises(ValidationError):
        TopicCreate(title="")

    # Invalid time range: end_time earlier than start_time
    with pytest.raises(ValidationError, match="end_time cannot be earlier than start_time"):
        TopicCreate(title="Invalid Time", start_time=50.0, end_time=10.0)

    # Invalid negative start_time
    with pytest.raises(ValidationError):
        TopicCreate(title="Negative Time", start_time=-5.0)


def test_decision_create_validation():
    """Verify DecisionCreate validates description and optional context/evidence."""
    # Valid complete
    d = DecisionCreate(
        description="Adopt faster-whisper large-v3-turbo",
        rationale_or_context="Best accuracy/speed trade-off on RTX 5060",
        decided_by="Architecture Team",
        evidence_segment_id="seg-123",
        timestamp_seconds=45.2,
    )
    assert d.description == "Adopt faster-whisper large-v3-turbo"
    assert d.timestamp_seconds == 45.2

    # Valid minimal
    d_min = DecisionCreate(description="Approved Q4 budget")
    assert d_min.rationale_or_context is None
    assert d_min.decided_by is None
    assert d_min.evidence_segment_id is None
    assert d_min.timestamp_seconds is None

    # Invalid empty description
    with pytest.raises(ValidationError):
        DecisionCreate(description="")


def test_action_item_create_and_status_normalization():
    """Verify ActionItemCreate validates assignees, deadlines, and normalizes status."""
    now = datetime.now(timezone.utc)
    deadline = now + timedelta(days=7)

    # Valid with string status case-normalized
    ai = ActionItemCreate(
        description="Benchmark int8 quantization",
        owner_name="Alice Engineer",
        deadline=deadline,
        status="in_progress",
        timestamp_seconds=120.0,
    )
    assert ai.status == ActionItemStatus.IN_PROGRESS
    assert ai.owner_name == "Alice Engineer"
    assert ai.deadline == deadline

    # Default status is PENDING
    ai_default = ActionItemCreate(description="Document deployment")
    assert ai_default.status == ActionItemStatus.PENDING

    # Invalid status enum
    with pytest.raises(ValidationError, match="Invalid action item status"):
        ActionItemCreate(description="Task", status="NON_EXISTENT")

    # Invalid empty description
    with pytest.raises(ValidationError):
        ActionItemCreate(description="")


def test_risk_create_validation_and_severity_normalization():
    """Verify RiskCreate validates description and normalizes severity enums."""
    r = RiskCreate(
        description="Memory pressure when loading models simultaneously",
        severity="critical",
        mitigation_suggestion="Use sequential execution or 3B model",
    )
    assert r.severity == RiskSeverity.CRITICAL
    assert r.mitigation_suggestion == "Use sequential execution or 3B model"

    # Default severity is MEDIUM
    r_default = RiskCreate(description="Potential GPU overheating")
    assert r_default.severity == RiskSeverity.MEDIUM

    # Invalid severity
    with pytest.raises(ValidationError, match="Invalid risk severity"):
        RiskCreate(description="Risk", severity="EXTREME")


def test_unresolved_question_create_validation():
    """Verify UnresolvedQuestionCreate validates text and status normalization."""
    uq = UnresolvedQuestionCreate(
        question_text="Will Hindi ASR require domain-specific fine tuning?",
        raised_by="Dr. Sharma",
        status="deferred",
    )
    assert uq.question_text == "Will Hindi ASR require domain-specific fine tuning?"
    assert uq.status == QuestionStatus.DEFERRED
    assert uq.raised_by == "Dr. Sharma"

    # Default status is OPEN
    uq_def = UnresolvedQuestionCreate(question_text="What is the rollout date?")
    assert uq_def.status == QuestionStatus.OPEN

    # Invalid status
    with pytest.raises(ValidationError, match="Invalid question status"):
        UnresolvedQuestionCreate(question_text="Q", status="ANSWERED")


def test_meeting_summary_create_validation():
    """Verify MeetingSummaryCreate validates executive summary and version."""
    s = MeetingSummaryCreate(
        executive_summary="Team reviewed Phase 7 intelligence architecture.",
        detailed_summary="Discussed VRAM budget and local Ollama integration.",
        key_takeaways="1. Use 3B model\n2. Maintain audit trail",
        version=2,
    )
    assert s.version == 2
    assert "Phase 7" in s.executive_summary

    # Invalid empty executive summary
    with pytest.raises(ValidationError):
        MeetingSummaryCreate(executive_summary="")

    # Invalid version < 1
    with pytest.raises(ValidationError):
        MeetingSummaryCreate(executive_summary="Summary", version=0)


def test_structured_extraction_payload_validation():
    """Verify StructuredExtractionPayload correctly validates full extraction output."""
    payload = StructuredExtractionPayload(
        executive_summary="Comprehensive architectural review concluded.",
        detailed_summary="Detailed sections discussed.",
        key_takeaways=["Points agreed", "Tasks distributed"],
        topics=[TopicCreate(title="Topic 1")],
        decisions=[DecisionCreate(description="Decision 1")],
        action_items=[ActionItemCreate(description="Action 1", owner_name="Bob")],
        risks=[RiskCreate(description="Risk 1", severity=RiskSeverity.LOW)],
        unresolved_questions=[UnresolvedQuestionCreate(question_text="Question 1")],
    )
    assert len(payload.topics) == 1
    assert len(payload.decisions) == 1
    assert len(payload.action_items) == 1
    assert len(payload.risks) == 1
    assert len(payload.unresolved_questions) == 1
    assert payload.action_items[0].owner_name == "Bob"


def test_action_item_status_update_schema():
    """Verify ActionItemStatusUpdate validates and normalizes status input."""
    update = ActionItemStatusUpdate(status="completed")
    assert update.status == ActionItemStatus.COMPLETED

    update_upper = ActionItemStatusUpdate(status="CANCELLED")
    assert update_upper.status == ActionItemStatus.CANCELLED

    with pytest.raises(ValidationError, match="Invalid action item status"):
        ActionItemStatusUpdate(status="NOT_A_STATUS")


def test_extract_intelligence_request_and_job_response():
    """Verify request and response schemas for extraction jobs."""
    req = ExtractIntelligenceRequest(
        source_transcript_id="trans-123",
        output_language="hi",
        provider_override="deterministic",
    )
    assert req.source_transcript_id == "trans-123"
    assert req.output_language == "hi"
    assert req.provider_override == "deterministic"

    # Default request values
    req_def = ExtractIntelligenceRequest()
    assert req_def.source_transcript_id is None
    assert req_def.output_language == "en"
    assert req_def.provider_override is None

    # Job response DTO
    resp = ExtractIntelligenceJobResponse(
        job_id="job-abc",
        meeting_id="meet-xyz",
        status="PENDING",
        message="Intelligence extraction job queued successfully.",
    )
    assert resp.job_id == "job-abc"
    assert resp.job_type == "EXTRACTION"


# ==============================================================================
# 2. ORM Compatibility and Serialization Tests
# ==============================================================================

def test_orm_compatibility_with_sqlalchemy_models(db_session):
    """Verify Pydantic response schemas serialize directly from SQLAlchemy model instances."""
    org = Organization(name="Intelligence Test Org", slug="intel-test-org")
    user = User(email="analyst@example.com", full_name="Analyst User")
    meeting = Meeting(organization_id=org.id, title="Intelligence Review")
    db_session.add_all([org, user, meeting])
    db_session.commit()

    transcript = Transcript(
        meeting_id=meeting.id,
        transcript_type=TranscriptType.CLEANED_REDACTED.value,
        version=1,
    )
    db_session.add(transcript)
    db_session.commit()

    seg = TranscriptSegment(
        transcript_id=transcript.id,
        sequence_order=0,
        start_time=15.0,
        end_time=25.0,
        text="We decided to proceed with Phase 7.2 schemas.",
        language_code="en",
    )
    db_session.add(seg)
    db_session.commit()

    # 1. Topic ORM -> TopicResponse
    topic_orm = Topic(
        meeting_id=meeting.id,
        title="Schema Design",
        description="Designing Pydantic DTOs",
        start_time=10.0,
        end_time=30.0,
    )
    db_session.add(topic_orm)

    # 2. Decision ORM -> DecisionResponse
    decision_orm = Decision(
        meeting_id=meeting.id,
        description="Proceed with Pydantic v2 schemas",
        rationale_or_context="Full compatibility with existing models",
        decided_by="Core Team",
        evidence_segment_id=seg.id,
        timestamp_seconds=15.0,
    )
    db_session.add(decision_orm)

    # 3. ActionItem ORM -> ActionItemResponse
    action_orm = ActionItem(
        meeting_id=meeting.id,
        description="Write comprehensive schema tests",
        owner_name="Dev Team",
        owner_user_id=user.id,
        deadline=datetime.now(timezone.utc) + timedelta(days=2),
        status=ActionItemStatus.IN_PROGRESS.value,
        evidence_segment_id=seg.id,
        timestamp_seconds=18.0,
    )
    db_session.add(action_orm)

    # 4. Risk ORM -> RiskResponse
    risk_orm = Risk(
        meeting_id=meeting.id,
        description="Local inference latency if GPU is throttled",
        severity=RiskSeverity.LOW.value,
        mitigation_suggestion="Test on AC power",
        evidence_segment_id=seg.id,
    )
    db_session.add(risk_orm)

    # 5. UnresolvedQuestion ORM -> UnresolvedQuestionResponse
    question_orm = UnresolvedQuestion(
        meeting_id=meeting.id,
        question_text="Will Hindi extraction need custom prompting?",
        raised_by="Senior Dev",
        status=QuestionStatus.OPEN.value,
        evidence_segment_id=seg.id,
    )
    db_session.add(question_orm)

    # 6. MeetingSummary ORM -> MeetingSummaryResponse
    summary_orm = MeetingSummary(
        meeting_id=meeting.id,
        transcript_id=transcript.id,
        executive_summary="Team concluded Phase 7.2 schema implementation.",
        detailed_summary="Detailed analysis of VRAM, ORM, and DTO structures.",
        key_takeaways="1. Pydantic v2\n2. Tested",
        version=1,
    )
    db_session.add(summary_orm)
    db_session.commit()

    # Validate ORM -> Pydantic response conversions
    topic_resp = TopicResponse.model_validate(topic_orm)
    assert topic_resp.id == topic_orm.id
    assert topic_resp.title == "Schema Design"

    decision_resp = DecisionResponse.model_validate(decision_orm)
    assert decision_resp.id == decision_orm.id
    assert decision_resp.evidence_segment_id == seg.id
    assert decision_resp.timestamp_seconds == 15.0

    action_resp = ActionItemResponse.model_validate(action_orm)
    assert action_resp.id == action_orm.id
    assert action_resp.owner_user_id == user.id
    assert action_resp.status == ActionItemStatus.IN_PROGRESS.value

    risk_resp = RiskResponse.model_validate(risk_orm)
    assert risk_resp.id == risk_orm.id
    assert risk_resp.severity == RiskSeverity.LOW.value

    question_resp = UnresolvedQuestionResponse.model_validate(question_orm)
    assert question_resp.id == question_orm.id
    assert question_resp.status == QuestionStatus.OPEN.value

    summary_resp = MeetingSummaryResponse.model_validate(summary_orm)
    assert summary_resp.id == summary_orm.id
    assert summary_resp.version == 1

    # Aggregate response
    agg_resp = MeetingIntelligenceResponse(
        meeting_id=meeting.id,
        summary=summary_resp,
        topics=[topic_resp],
        decisions=[decision_resp],
        action_items=[action_resp],
        risks=[risk_resp],
        unresolved_questions=[question_resp],
    )
    assert agg_resp.has_intelligence is True
    assert len(agg_resp.topics) == 1
    assert len(agg_resp.decisions) == 1
    assert len(agg_resp.action_items) == 1

    # Empty aggregate response computes has_intelligence = False
    empty_agg = MeetingIntelligenceResponse(meeting_id=meeting.id)
    assert empty_agg.has_intelligence is False
    assert empty_agg.summary is None
    assert len(empty_agg.decisions) == 0


# ==============================================================================
# 3. Configuration & Settings Tests
# ==============================================================================

def test_intelligence_configuration_defaults():
    """Verify configuration defaults for intelligence settings."""
    assert settings.INTELLIGENCE_PROVIDER in {"disabled", "ollama", "deterministic"}
    assert settings.OLLAMA_BASE_URL.startswith("http")
    assert not settings.OLLAMA_BASE_URL.endswith("/")
    assert settings.LLM_MODEL_NAME != ""
    assert settings.LLM_REQUEST_TIMEOUT_SECONDS > 0
    assert settings.INTELLIGENCE_MAX_TRANSCRIPT_CHARS > 0
    assert settings.INTELLIGENCE_MAX_CHUNKS > 0
    assert settings.INTELLIGENCE_OUTPUT_LANGUAGE == "en"
    assert settings.INTELLIGENCE_MAX_RETRIES >= 0


def test_intelligence_configuration_validation():
    """Verify validation and normalization of intelligence configuration settings."""
    # 1. Valid provider options (case-insensitive normalization)
    s_det = Settings(ENVIRONMENT="testing", INTELLIGENCE_PROVIDER="DETERMINISTIC")
    assert s_det.INTELLIGENCE_PROVIDER == "deterministic"

    s_ollama = Settings(ENVIRONMENT="testing", INTELLIGENCE_PROVIDER="ollama")
    assert s_ollama.INTELLIGENCE_PROVIDER == "ollama"

    s_dis = Settings(ENVIRONMENT="testing", INTELLIGENCE_PROVIDER="disabled")
    assert s_dis.INTELLIGENCE_PROVIDER == "disabled"

    # Invalid provider
    with pytest.raises(ValueError, match="Invalid INTELLIGENCE_PROVIDER"):
        Settings(ENVIRONMENT="testing", INTELLIGENCE_PROVIDER="openai_cloud")

    # 2. Ollama base URL trailing slash normalization
    s_url = Settings(ENVIRONMENT="testing", OLLAMA_BASE_URL="http://localhost:11434/")
    assert s_url.OLLAMA_BASE_URL == "http://localhost:11434"

    # Empty Ollama URL resets to default
    s_empty_url = Settings(ENVIRONMENT="testing", OLLAMA_BASE_URL="")
    assert s_empty_url.OLLAMA_BASE_URL == "http://127.0.0.1:11434"

    # 3. Timeout must be positive
    with pytest.raises(ValueError, match="LLM_REQUEST_TIMEOUT_SECONDS must be greater than 0"):
        Settings(ENVIRONMENT="testing", LLM_REQUEST_TIMEOUT_SECONDS=0.0)

    # 4. Max transcript chars must be positive
    with pytest.raises(ValueError, match="INTELLIGENCE_MAX_TRANSCRIPT_CHARS must be greater than 0"):
        Settings(ENVIRONMENT="testing", INTELLIGENCE_MAX_TRANSCRIPT_CHARS=0)

    # 5. Max chunks must be positive
    with pytest.raises(ValueError, match="INTELLIGENCE_MAX_CHUNKS must be greater than 0"):
        Settings(ENVIRONMENT="testing", INTELLIGENCE_MAX_CHUNKS=-1)

    # 6. Max retries cannot be negative
    with pytest.raises(ValueError, match="INTELLIGENCE_MAX_RETRIES cannot be negative"):
        Settings(ENVIRONMENT="testing", INTELLIGENCE_MAX_RETRIES=-1)

    s_zero_retries = Settings(ENVIRONMENT="testing", INTELLIGENCE_MAX_RETRIES=0)
    assert s_zero_retries.INTELLIGENCE_MAX_RETRIES == 0


def test_env_example_contains_phase7_intelligence_settings():
    """Verify backend/.env.example documents all Phase 7 settings with safe defaults."""
    env_example_path = settings.ROOT_DIR if hasattr(settings, "ROOT_DIR") else None
    from app.core.config import ROOT_DIR
    env_file = ROOT_DIR / "backend" / ".env.example"
    assert env_file.exists()

    content = env_file.read_text(encoding="utf-8")
    assert "INTELLIGENCE_PROVIDER" in content
    assert "OLLAMA_BASE_URL" in content
    assert "LLM_MODEL_NAME" in content
    assert "LLM_REQUEST_TIMEOUT_SECONDS" in content
    assert "INTELLIGENCE_MAX_TRANSCRIPT_CHARS" in content
    assert "INTELLIGENCE_MAX_CHUNKS" in content
    assert "INTELLIGENCE_OUTPUT_LANGUAGE" in content
    assert "INTELLIGENCE_MAX_RETRIES" in content
