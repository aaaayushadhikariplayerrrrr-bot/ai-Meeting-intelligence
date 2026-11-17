import uuid
from datetime import datetime, timezone, timedelta
import pytest
from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError

from app.models.identity import User, Organization, OrganizationMembership, OrgRole
from app.models.meeting import (
    Meeting,
    MeetingParticipant,
    MediaAsset,
    ProcessingJob,
    InputSourceType,
    MeetingStatus,
    JobType,
    JobStatus,
)
from app.models.transcription import (
    Speaker,
    Transcript,
    TranscriptSegment,
    TranscriptType,
)
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
from app.models.memory import (
    KnowledgeRecord,
    SourceReference,
    AuditLog,
    KnowledgeSourceType,
    ReferenceType,
)


def test_database_initialization(db_session):
    """Verify that the test database initializes and executes queries."""
    result = db_session.execute(text("SELECT 1")).scalar()
    assert result == 1


def test_organization_creation(db_session):
    """Verify organization creation, defaults, and unique slug constraint."""
    org = Organization(
        name="Acme Corp",
        slug="acme-corp",
        description="Global technology leader"
    )
    db_session.add(org)
    db_session.commit()

    assert org.id is not None
    assert len(org.id) == 36
    assert org.is_active is True
    assert org.created_at is not None
    assert org.updated_at is not None

    # Test unique slug constraint
    duplicate_org = Organization(name="Acme Duplicate", slug="acme-corp")
    db_session.add(duplicate_org)
    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()


def test_user_and_membership_relationships(db_session):
    """Verify users, global super-admin vs tenant roles, and organization memberships."""
    # Global super admin user
    super_admin = User(
        email="sysadmin@example.com",
        full_name="System Admin",
        is_super_admin=True
    )
    # Standard tenant member
    regular_user = User(
        email="john@example.com",
        full_name="John Doe",
        is_super_admin=False
    )
    org = Organization(name="Alpha Org", slug="alpha-org")
    db_session.add_all([super_admin, regular_user, org])
    db_session.commit()

    # Create tenant-level membership with distinct roles
    membership_admin = OrganizationMembership(
        organization_id=org.id,
        user_id=super_admin.id,
        role=OrgRole.ORG_ADMIN.value
    )
    membership_member = OrganizationMembership(
        organization_id=org.id,
        user_id=regular_user.id,
        role=OrgRole.MEMBER.value
    )
    db_session.add_all([membership_admin, membership_member])
    db_session.commit()

    # Verify relationships
    assert len(org.memberships) == 2
    assert len(regular_user.memberships) == 1
    assert regular_user.memberships[0].role == OrgRole.MEMBER.value
    assert super_admin.memberships[0].role == OrgRole.ORG_ADMIN.value
    assert regular_user.is_super_admin is False
    assert super_admin.is_super_admin is True

    # Verify unique constraint: user cannot have duplicate membership in same org
    duplicate_membership = OrganizationMembership(
        organization_id=org.id,
        user_id=regular_user.id,
        role=OrgRole.MEMBER.value
    )
    db_session.add(duplicate_membership)
    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()


def test_meeting_creation(db_session):
    """Verify meeting creation supporting audio/video/YouTube, participants, media metadata, and jobs."""
    org = Organization(name="Beta Innovations", slug="beta-innovations")
    user = User(email="organizer@example.com", full_name="Meeting Host")
    db_session.add_all([org, user])
    db_session.commit()

    # Create meeting with YouTube input source
    meeting = Meeting(
        organization_id=org.id,
        created_by_user_id=user.id,
        title="Q4 Architecture & Strategy",
        description="Review of multilingual ASR pipeline and dual RAG memory.",
        input_source_type=InputSourceType.YOUTUBE.value,
        source_url_or_reference="https://youtube.com/watch?v=sample123",
        status=MeetingStatus.INGESTING.value,
        scheduled_start_time=datetime.now(timezone.utc)
    )
    db_session.add(meeting)
    db_session.commit()

    # Add participants
    participant_1 = MeetingParticipant(
        meeting_id=meeting.id,
        user_id=user.id,
        display_name="Meeting Host",
        email=user.email,
        role="HOST"
    )
    participant_2 = MeetingParticipant(
        meeting_id=meeting.id,
        display_name="External Consultant",
        email="consultant@partner.com",
        role="PRESENTER"
    )
    db_session.add_all([participant_1, participant_2])

    # Add media asset metadata (NOT binary media in SQLite!)
    media_asset = MediaAsset(
        meeting_id=meeting.id,
        file_name="q4_architecture_audio.wav",
        storage_path="/storage/media/meetings/q4_architecture_audio.wav",
        media_type="AUDIO",
        mime_type="audio/wav",
        file_size_bytes=45219800,
        duration_seconds=3600.5,
        checksum_sha256="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        is_raw_deleted=False
    )
    db_session.add(media_asset)

    # Add background processing job
    job = ProcessingJob(
        meeting_id=meeting.id,
        job_type=JobType.TRANSCRIPTION.value,
        status=JobStatus.IN_PROGRESS.value,
        progress=45.0
    )
    db_session.add(job)
    db_session.commit()

    # Verify associations
    assert len(meeting.participants) == 2
    assert len(meeting.media_assets) == 1
    assert meeting.media_assets[0].file_size_bytes == 45219800
    assert len(meeting.processing_jobs) == 1
    assert meeting.processing_jobs[0].progress == 45.0


def test_transcript_and_segment_relationships(db_session):
    """Verify speakers, transcripts, and multilingual segments (Hindi, Gujarati, English, Hinglish)."""
    org = Organization(name="Global Polyglot", slug="global-polyglot")
    meeting = Meeting(organization_id=org.id, title="Multilingual Sprint Retrospective")
    db_session.add_all([org, meeting])
    db_session.commit()

    # Speakers
    speaker_1 = Speaker(meeting_id=meeting.id, speaker_label="SPEAKER_00", display_name="Amit Patel")
    speaker_2 = Speaker(meeting_id=meeting.id, speaker_label="SPEAKER_01", display_name="Sarah Smith")
    db_session.add_all([speaker_1, speaker_2])
    db_session.commit()

    # Transcript
    transcript = Transcript(
        meeting_id=meeting.id,
        transcript_type=TranscriptType.ORIGINAL.value,
        version=1,
        language_detected="multilingual",
        is_primary=True
    )
    db_session.add(transcript)
    db_session.commit()

    # Multilingual segments showing Hindi, English, Gujarati, and Hinglish code-switching
    segments = [
        TranscriptSegment(
            transcript_id=transcript.id,
            speaker_id=speaker_1.id,
            sequence_order=1,
            start_time=0.0,
            end_time=4.5,
            text="Welcome everyone, let's start the review.",
            language_code="en",
            confidence=0.98
        ),
        TranscriptSegment(
            transcript_id=transcript.id,
            speaker_id=speaker_1.id,
            sequence_order=2,
            start_time=4.6,
            end_time=9.2,
            text="આજના એજન્ડામાં ડેટાબેઝ આર્કિટેક્ચર મહત્વનું છે.",
            language_code="gu",  # Gujarati
            confidence=0.94
        ),
        TranscriptSegment(
            transcript_id=transcript.id,
            speaker_id=speaker_2.id,
            sequence_order=3,
            start_time=9.3,
            end_time=14.0,
            text="हम इस स्प्रिंट में RAG पाइपलाइन को finalize कर देंगे।",
            language_code="hi-en",  # Hinglish / Code-switching
            confidence=0.96
        ),
        TranscriptSegment(
            transcript_id=transcript.id,
            speaker_id=speaker_1.id,
            sequence_order=4,
            start_time=14.1,
            end_time=18.5,
            text="बिल्कुल सही, privacy controls को प्राथमिकता दी जाएगी।",
            language_code="hi",  # Hindi
            confidence=0.97
        ),
    ]
    db_session.add_all(segments)
    db_session.commit()

    # Verify query and ordering
    loaded_transcript = db_session.get(Transcript, transcript.id)
    assert len(loaded_transcript.segments) == 4
    assert loaded_transcript.segments[0].language_code == "en"
    assert loaded_transcript.segments[1].language_code == "gu"
    assert loaded_transcript.segments[2].language_code == "hi-en"
    assert loaded_transcript.segments[3].language_code == "hi"
    assert loaded_transcript.segments[2].speaker.display_name == "Sarah Smith"


def test_original_and_cleaned_transcripts_coexistence(db_session):
    """Verify that original and cleaned/redacted transcripts coexist without overwriting."""
    org = Organization(name="Privacy Shield Inc", slug="privacy-shield")
    meeting = Meeting(organization_id=org.id, title="Confidential Board Meeting")
    db_session.add_all([org, meeting])
    db_session.commit()

    # 1. Original Transcript (raw ASR output)
    original_transcript = Transcript(
        meeting_id=meeting.id,
        transcript_type=TranscriptType.ORIGINAL.value,
        version=1,
        language_detected="en",
        is_primary=False
    )
    db_session.add(original_transcript)
    db_session.commit()

    raw_segment = TranscriptSegment(
        transcript_id=original_transcript.id,
        sequence_order=1,
        start_time=0.0,
        end_time=5.0,
        text="My secret API key is sk-123456789 and my email is john@corp.com.",
        language_code="en",
        is_redacted=False
    )
    db_session.add(raw_segment)

    # 2. Cleaned / Redacted Transcript (PII-masked output)
    cleaned_transcript = Transcript(
        meeting_id=meeting.id,
        transcript_type=TranscriptType.CLEANED_REDACTED.value,
        version=1,
        language_detected="en",
        is_primary=True
    )
    db_session.add(cleaned_transcript)
    db_session.commit()

    cleaned_segment = TranscriptSegment(
        transcript_id=cleaned_transcript.id,
        sequence_order=1,
        start_time=0.0,
        end_time=5.0,
        text="My secret API key is [REDACTED_API_KEY] and my email is [REDACTED_EMAIL].",
        language_code="en",
        is_redacted=True
    )
    db_session.add(cleaned_segment)
    db_session.commit()

    # Query meeting transcripts
    loaded_meeting = db_session.get(Meeting, meeting.id)
    assert len(loaded_meeting.transcripts) == 2

    # Verify both versions coexist
    transcripts_by_type = {t.transcript_type: t for t in loaded_meeting.transcripts}
    assert TranscriptType.ORIGINAL.value in transcripts_by_type
    assert TranscriptType.CLEANED_REDACTED.value in transcripts_by_type

    # Verify content differences
    assert "sk-123456789" in transcripts_by_type[TranscriptType.ORIGINAL.value].segments[0].text
    assert "[REDACTED_API_KEY]" in transcripts_by_type[TranscriptType.CLEANED_REDACTED.value].segments[0].text
    assert transcripts_by_type[TranscriptType.CLEANED_REDACTED.value].segments[0].is_redacted is True


def test_action_item_and_decision_relationships(db_session):
    """Verify meeting intelligence entities (decisions, action items, risks, questions) with evidence references."""
    org = Organization(name="Strategy Labs", slug="strategy-labs")
    user = User(email="owner@example.com", full_name="Project Owner")
    meeting = Meeting(organization_id=org.id, title="Product Sync")
    db_session.add_all([org, user, meeting])
    db_session.commit()

    transcript = Transcript(meeting_id=meeting.id, transcript_type=TranscriptType.ORIGINAL.value)
    db_session.add(transcript)
    db_session.commit()

    segment = TranscriptSegment(
        transcript_id=transcript.id,
        sequence_order=1,
        start_time=120.0,
        end_time=145.0,
        text="We decided to deploy the ASR pipeline on RTX 5060 GPUs with 8GB VRAM.",
        language_code="en"
    )
    db_session.add(segment)
    db_session.commit()

    # Meeting Summary
    summary = MeetingSummary(
        meeting_id=meeting.id,
        transcript_id=transcript.id,
        executive_summary="Team aligned on on-premise hardware deployment.",
        detailed_summary="Discussed VRAM budget and PyTorch CUDA acceleration.",
        version=1
    )
    # Decision grounded in transcript segment
    decision = Decision(
        meeting_id=meeting.id,
        description="Deploy Whisper Large-v3 Turbo locally on RTX 5060.",
        rationale_or_context="Guarantees data privacy and avoids cloud egress costs.",
        decided_by="Architecture Team",
        evidence_segment_id=segment.id,
        timestamp_seconds=120.0
    )
    # Action Item grounded in transcript segment with owner and deadline
    deadline = datetime.now(timezone.utc) + timedelta(days=7)
    action_item = ActionItem(
        meeting_id=meeting.id,
        description="Benchmark Whisper Large-v3 Turbo inference latency on RTX 5060.",
        owner_name="Project Owner",
        owner_user_id=user.id,
        deadline=deadline,
        status=ActionItemStatus.PENDING.value,
        evidence_segment_id=segment.id,
        timestamp_seconds=125.0
    )
    # Risk
    risk = Risk(
        meeting_id=meeting.id,
        description="8GB VRAM might be tight when loading large LLMs concurrently with ASR.",
        severity=RiskSeverity.HIGH.value,
        mitigation_suggestion="Offload LLM to Ollama with 4-bit quantization or sequential execution.",
        evidence_segment_id=segment.id
    )
    # Unresolved question
    question = UnresolvedQuestion(
        meeting_id=meeting.id,
        question_text="Will Hindi and Gujarati models require fine-tuning or zero-shot Whisper?",
        raised_by="AI Researcher",
        status=QuestionStatus.OPEN.value,
        evidence_segment_id=segment.id
    )
    db_session.add_all([summary, decision, action_item, risk, question])
    db_session.commit()

    # Verify associations and evidence linkage
    loaded_decision = db_session.get(Decision, decision.id)
    assert loaded_decision.evidence_segment.text == segment.text
    assert loaded_decision.meeting.title == "Product Sync"

    loaded_action = db_session.get(ActionItem, action_item.id)
    assert loaded_action.owner_user.email == "owner@example.com"
    assert loaded_action.status == ActionItemStatus.PENDING.value
    assert loaded_action.evidence_segment_id == segment.id


def test_foreign_key_enforcement(db_session):
    """Verify that foreign key constraints are enforced (invalid FK raises IntegrityError)."""
    fake_org_id = str(uuid.uuid4())

    # Attempt to insert a Meeting with a non-existent organization_id
    invalid_meeting = Meeting(
        organization_id=fake_org_id,
        title="Invalid Meeting"
    )
    db_session.add(invalid_meeting)

    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()


def test_cascade_delete(db_session):
    """Verify cascade delete from Organization -> Meeting -> Transcripts -> Segments."""
    org = Organization(name="Delete Test Org", slug="delete-test-org")
    db_session.add(org)
    db_session.commit()

    meeting = Meeting(organization_id=org.id, title="Meeting to Cascade")
    db_session.add(meeting)
    db_session.commit()

    transcript = Transcript(meeting_id=meeting.id)
    db_session.add(transcript)
    db_session.commit()

    segment = TranscriptSegment(
        transcript_id=transcript.id,
        sequence_order=1,
        start_time=0.0,
        end_time=1.0,
        text="Test segment"
    )
    db_session.add(segment)
    db_session.commit()

    segment_id = segment.id
    meeting_id = meeting.id

    # Deleting organization should cascade to meeting, transcript, and segments
    db_session.delete(org)
    db_session.commit()

    assert db_session.get(Meeting, meeting_id) is None
    assert db_session.get(TranscriptSegment, segment_id) is None


def test_organization_data_isolation(db_session):
    """Verify that queries enforce organization data isolation at the schema level."""
    org_a = Organization(name="Tenant Alpha", slug="tenant-alpha")
    org_b = Organization(name="Tenant Beta", slug="tenant-beta")
    db_session.add_all([org_a, org_b])
    db_session.commit()

    # Meetings for Org A
    meeting_a1 = Meeting(organization_id=org_a.id, title="Alpha Meeting 1")
    meeting_a2 = Meeting(organization_id=org_a.id, title="Alpha Meeting 2")
    # Meetings for Org B
    meeting_b1 = Meeting(organization_id=org_b.id, title="Beta Meeting 1")
    db_session.add_all([meeting_a1, meeting_a2, meeting_b1])

    # Knowledge Records for Org A and Org B
    record_a = KnowledgeRecord(
        organization_id=org_a.id,
        title="Alpha Proprietary Roadmap",
        content="Confidential strategy for Org A",
        knowledge_type=KnowledgeSourceType.INTERNAL_MEETING.value
    )
    record_b = KnowledgeRecord(
        organization_id=org_b.id,
        title="Beta Proprietary Roadmap",
        content="Confidential strategy for Org B",
        knowledge_type=KnowledgeSourceType.INTERNAL_MEETING.value
    )
    db_session.add_all([record_a, record_b])
    db_session.commit()

    # Query Org A meetings only
    org_a_meetings = db_session.execute(
        select(Meeting).where(Meeting.organization_id == org_a.id)
    ).scalars().all()
    assert len(org_a_meetings) == 2
    assert all(m.organization_id == org_a.id for m in org_a_meetings)

    # Query Org B knowledge records only
    org_b_records = db_session.execute(
        select(KnowledgeRecord).where(KnowledgeRecord.organization_id == org_b.id)
    ).scalars().all()
    assert len(org_b_records) == 1
    assert org_b_records[0].title == "Beta Proprietary Roadmap"


def test_knowledge_record_and_source_references(db_session):
    """Verify logical separation of internal vs external knowledge and evidence citation links."""
    org = Organization(name="Knowledge Org", slug="knowledge-org")
    meeting = Meeting(organization_id=org.id, title="Q1 Sync")
    db_session.add_all([org, meeting])
    db_session.commit()

    transcript = Transcript(meeting_id=meeting.id)
    db_session.add(transcript)
    db_session.commit()

    segment = TranscriptSegment(
        transcript_id=transcript.id,
        sequence_order=1,
        start_time=10.0,
        end_time=20.0,
        text="The compliance policy requires data retention for 90 days."
    )
    db_session.add(segment)
    db_session.commit()

    # 1. Internal meeting-derived knowledge record
    internal_record = KnowledgeRecord(
        organization_id=org.id,
        title="Data Retention Policy",
        content="Organizational data retention limit established at 90 days.",
        knowledge_type=KnowledgeSourceType.INTERNAL_MEETING.value,
        category="POLICY",
        meeting_id=meeting.id
    )
    db_session.add(internal_record)
    db_session.commit()

    # Evidence link to meeting transcript segment
    ref_meeting = SourceReference(
        knowledge_record_id=internal_record.id,
        meeting_id=meeting.id,
        transcript_segment_id=segment.id,
        reference_type=ReferenceType.MEETING_SEGMENT.value,
        timestamp_start=10.0,
        timestamp_end=20.0,
        text_snippet=segment.text
    )
    db_session.add(ref_meeting)

    # 2. External reference knowledge record
    external_record = KnowledgeRecord(
        organization_id=org.id,
        title="GDPR Article 17 - Right to Erasure",
        content="Legal requirement for data deletion policies.",
        knowledge_type=KnowledgeSourceType.EXTERNAL_REFERENCE.value,
        category="LEGAL_REFERENCE"
    )
    db_session.add(external_record)
    db_session.commit()

    ref_external = SourceReference(
        knowledge_record_id=external_record.id,
        reference_type=ReferenceType.WEB_URL.value,
        external_url="https://gdpr-info.eu/art-17-gdpr/",
        external_title="General Data Protection Regulation Article 17",
        text_snippet="The data subject shall have the right to obtain erasure..."
    )
    db_session.add(ref_external)
    db_session.commit()

    # Verify logical separation for Dual-Source RAG
    internal_results = db_session.execute(
        select(KnowledgeRecord).where(
            KnowledgeRecord.organization_id == org.id,
            KnowledgeRecord.knowledge_type == KnowledgeSourceType.INTERNAL_MEETING.value
        )
    ).scalars().all()
    assert len(internal_results) == 1
    assert internal_results[0].source_references[0].transcript_segment_id == segment.id

    external_results = db_session.execute(
        select(KnowledgeRecord).where(
            KnowledgeRecord.organization_id == org.id,
            KnowledgeRecord.knowledge_type == KnowledgeSourceType.EXTERNAL_REFERENCE.value
        )
    ).scalars().all()
    assert len(external_results) == 1
    assert external_results[0].source_references[0].external_url == "https://gdpr-info.eu/art-17-gdpr/"


def test_audit_log_creation(db_session):
    """Verify audit log records privacy-sensitive actions and deletions."""
    org = Organization(name="Audit Corp", slug="audit-corp")
    user = User(email="auditor@example.com", full_name="Compliance Auditor")
    db_session.add_all([org, user])
    db_session.commit()

    # Create audit log for raw audio deletion action
    audit = AuditLog(
        organization_id=org.id,
        user_id=user.id,
        action="DELETE_RAW_AUDIO",
        entity_type="MediaAsset",
        entity_id=str(uuid.uuid4()),
        details='{"policy": "PII_AUDIO_PURGE_AFTER_TRANSCRIPTION", "status": "success"}',
        ip_address="192.168.1.100"
    )
    db_session.add(audit)
    db_session.commit()

    assert audit.id is not None
    assert audit.created_at is not None
    assert audit.action == "DELETE_RAW_AUDIO"
    assert audit.organization.name == "Audit Corp"
