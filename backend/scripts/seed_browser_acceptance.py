"""Seed isolated database for Phase 9.4 Browser Acceptance Testing."""

import os
import sys
from datetime import datetime, timezone

# Ensure backend directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DB_PATH = (ROOT_DIR / "test_browser_acceptance.db").resolve().as_posix()
DB_URL = f"sqlite:///{DB_PATH}"

os.environ["DATABASE_URL"] = DB_URL

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.security import hash_password
from app.models.identity import User, Organization, OrganizationMembership, OrgRole
from app.models.meeting import Meeting, MeetingStatus
from app.models.transcription import Transcript, TranscriptSegment, TranscriptType
from app.models.intelligence import MeetingSummary, Decision, ActionItem, Risk, UnresolvedQuestion
from app.services.memory.embedding import MultilingualE5EmbeddingService
from app.services.memory.storage import ChunkStorageManager
from app.services.memory.types import SemanticChunk

def seed():
    engine = create_engine(DB_URL, connect_args={"check_same_thread": False, "timeout": 30.0})
    SessionFactory = sessionmaker(bind=engine)
    db = SessionFactory()

    try:
        # Check if already seeded
        if db.query(Organization).filter_by(id="org-acme-001").first():
            print("test_browser_acceptance.db already seeded!")
            return

        print("Seeding test_browser_acceptance.db...")

        # 1. Organizations
        org_acme = Organization(
            id="org-acme-001",
            name="Acme Global",
            slug="acme-global",
            is_active=True,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        org_beta = Organization(
            id="org-beta-002",
            name="Beta Corp",
            slug="beta-corp",
            is_active=True,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        db.add_all([org_acme, org_beta])
        db.commit()

        # 2. Users
        pwd_hash = hash_password("Password123!")
        alice = User(
            id="user-alice-001",
            email="alice@acme.com",
            full_name="Alice Admin",
            hashed_password=pwd_hash,
            is_super_admin=True,
            is_active=True,
            token_version=1,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        bob = User(
            id="user-bob-002",
            email="bob@acme.com",
            full_name="Bob Member",
            hashed_password=pwd_hash,
            is_super_admin=False,
            is_active=True,
            token_version=1,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        charlie = User(
            id="user-charlie-003",
            email="charlie@beta.com",
            full_name="Charlie Beta",
            hashed_password=pwd_hash,
            is_super_admin=False,
            is_active=True,
            token_version=1,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        db.add_all([alice, bob, charlie])
        db.commit()

        # 3. Memberships
        m_alice = OrganizationMembership(
            id="mem-alice-001",
            organization_id=org_acme.id,
            user_id=alice.id,
            role=OrgRole.ORG_ADMIN.value,
            is_active=True,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        m_bob = OrganizationMembership(
            id="mem-bob-002",
            organization_id=org_acme.id,
            user_id=bob.id,
            role=OrgRole.MEMBER.value,
            is_active=True,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        m_charlie = OrganizationMembership(
            id="mem-charlie-003",
            organization_id=org_beta.id,
            user_id=charlie.id,
            role=OrgRole.ORG_ADMIN.value,
            is_active=True,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        db.add_all([m_alice, m_bob, m_charlie])
        db.commit()

        # 4. Meeting 1: Acme Q3 Architecture Review
        m1_date = datetime(2026, 9, 15, 10, 0, 0, tzinfo=timezone.utc)
        meeting1 = Meeting(
            id="meeting-acme-001",
            organization_id=org_acme.id,
            title="Q3 Architecture Review",
            description="Quarterly review of infrastructure and database choices.",
            status=MeetingStatus.COMPLETED.value,
            created_at=m1_date,
            updated_at=m1_date,
        )
        db.add(meeting1)
        db.commit()

        # Transcripts for Meeting 1
        t1_orig = Transcript(
            id="trans-acme-001-orig",
            meeting_id=meeting1.id,
            transcript_type=TranscriptType.ORIGINAL.value,
            version=1,
            language_detected="en",
            is_primary=False,
            created_at=m1_date,
            updated_at=m1_date,
        )
        t1_clean = Transcript(
            id="trans-acme-001-clean",
            meeting_id=meeting1.id,
            transcript_type=TranscriptType.CLEANED.value,
            version=2,
            language_detected="en",
            is_primary=False,
            parent_transcript_id=t1_orig.id,
            created_at=m1_date,
            updated_at=m1_date,
        )
        t1_redacted = Transcript(
            id="trans-acme-001-red",
            meeting_id=meeting1.id,
            transcript_type=TranscriptType.CLEANED_REDACTED.value,
            version=3,
            language_detected="en",
            is_primary=True,
            parent_transcript_id=t1_clean.id,
            created_at=m1_date,
            updated_at=m1_date,
        )
        db.add_all([t1_orig, t1_clean, t1_redacted])
        db.commit()

        seg1_1 = TranscriptSegment(
            id="seg-m1-001",
            transcript_id=t1_redacted.id,
            sequence_order=1,
            start_time=0.0,
            end_time=5.0,
            text="Welcome everyone to the architecture review. Today we will decide our database strategy.",
            language_code="en",
            created_at=m1_date,
            updated_at=m1_date,
        )
        seg1_2 = TranscriptSegment(
            id="seg-m1-002",
            transcript_id=t1_redacted.id,
            sequence_order=2,
            start_time=5.0,
            end_time=15.0,
            text="We have decided to adopt PostgreSQL for all analytical reporting workloads.",
            language_code="en",
            created_at=m1_date,
            updated_at=m1_date,
        )
        seg1_3 = TranscriptSegment(
            id="seg-m1-003",
            transcript_id=t1_redacted.id,
            sequence_order=3,
            start_time=15.0,
            end_time=25.0,
            text="Alice will complete the benchmark of analytical queries by October first.",
            language_code="en",
            created_at=m1_date,
            updated_at=m1_date,
        )
        # Duplicate for original
        seg1_1_orig = TranscriptSegment(
            id="seg-m1-orig-001",
            transcript_id=t1_orig.id,
            sequence_order=1,
            start_time=0.0,
            end_time=5.0,
            text="Um, welcome everyone to the architecture review. Today we will decide our database strategy.",
            language_code="en",
            created_at=m1_date,
            updated_at=m1_date,
        )
        db.add_all([seg1_1, seg1_2, seg1_3, seg1_1_orig])
        db.commit()

        # Intelligence for Meeting 1
        sum1 = MeetingSummary(
            id="sum-acme-001",
            meeting_id=meeting1.id,
            transcript_id=t1_redacted.id,
            executive_summary="The architecture team reviewed database options and selected PostgreSQL for analytical reporting.",
            detailed_summary="The team evaluated options and agreed on PostgreSQL for initial analytical queries.",
            key_takeaways='["Database Architecture", "PostgreSQL", "Query Benchmarking"]',
            version=1,
            provider_used="deterministic",
            is_heuristic=False,
            created_at=m1_date,
            updated_at=m1_date,
        )
        dec1 = Decision(
            id="dec-acme-001",
            meeting_id=meeting1.id,
            description="Adopted PostgreSQL for analytical reporting workloads.",
            rationale_or_context="Evaluated for early query support.",
            status="SUPERSEDED",
            evidence_segment_id=seg1_2.id,
            timestamp_seconds=5.0,
            created_at=m1_date,
            updated_at=m1_date,
        )
        ai1 = ActionItem(
            id="ai-acme-001",
            meeting_id=meeting1.id,
            description="Complete benchmark of analytical queries",
            owner_name="Alice",
            deadline=datetime(2026, 10, 1, 0, 0, 0, tzinfo=timezone.utc),
            status="COMPLETED",
            evidence_segment_id=seg1_3.id,
            timestamp_seconds=15.0,
            created_at=m1_date,
            updated_at=m1_date,
        )
        db.add_all([sum1, dec1, ai1])
        db.commit()

        # 5. Meeting 2: Acme Q4 Database Migration Strategy
        m2_date = datetime(2026, 11, 1, 14, 0, 0, tzinfo=timezone.utc)
        meeting2 = Meeting(
            id="meeting-acme-002",
            organization_id=org_acme.id,
            title="Q4 Database Migration Strategy",
            description="Follow-up on analytical performance and ClickHouse migration.",
            status=MeetingStatus.COMPLETED.value,
            created_at=m2_date,
            updated_at=m2_date,
        )
        db.add(meeting2)
        db.commit()

        t2_redacted = Transcript(
            id="trans-acme-002-red",
            meeting_id=meeting2.id,
            transcript_type=TranscriptType.CLEANED_REDACTED.value,
            version=1,
            language_detected="en",
            is_primary=True,
            created_at=m2_date,
            updated_at=m2_date,
        )
        db.add(t2_redacted)
        db.commit()

        seg2_1 = TranscriptSegment(
            id="seg-m2-001",
            transcript_id=t2_redacted.id,
            sequence_order=1,
            start_time=0.0,
            end_time=6.0,
            text="Reviewing our Q3 decision, PostgreSQL has reached query latency limits on large datasets.",
            language_code="en",
            created_at=m2_date,
            updated_at=m2_date,
        )
        seg2_2 = TranscriptSegment(
            id="seg-m2-002",
            transcript_id=t2_redacted.id,
            sequence_order=2,
            start_time=6.0,
            end_time=16.0,
            text="We have officially decided to migrate analytical reporting workloads from PostgreSQL to ClickHouse.",
            language_code="en",
            created_at=m2_date,
            updated_at=m2_date,
        )
        seg2_3 = TranscriptSegment(
            id="seg-m2-003",
            transcript_id=t2_redacted.id,
            sequence_order=3,
            start_time=16.0,
            end_time=24.0,
            text="Bob will provision the staging ClickHouse cluster by November twentieth.",
            language_code="en",
            created_at=m2_date,
            updated_at=m2_date,
        )
        db.add_all([seg2_1, seg2_2, seg2_3])
        db.commit()

        sum2 = MeetingSummary(
            id="sum-acme-002",
            meeting_id=meeting2.id,
            transcript_id=t2_redacted.id,
            executive_summary="The team approved migrating analytical reporting workloads from PostgreSQL to ClickHouse due to scalability requirements.",
            detailed_summary="Latency limits were reached on PostgreSQL, leading to a migration decision.",
            key_takeaways='["ClickHouse Migration", "Database Scalability", "PostgreSQL Supersession"]',
            version=1,
            provider_used="deterministic",
            is_heuristic=False,
            created_at=m2_date,
            updated_at=m2_date,
        )
        dec3 = Decision(
            id="dec-acme-003",
            meeting_id=meeting2.id,
            description="Migrate analytical reporting workloads from PostgreSQL to ClickHouse.",
            rationale_or_context="PostgreSQL reached query latency limits on large datasets.",
            status="ACTIVE",
            evidence_segment_id=seg2_2.id,
            timestamp_seconds=6.0,
            supersedes_decision_id="dec-acme-001",
            created_at=m2_date,
            updated_at=m2_date,
        )
        ai2 = ActionItem(
            id="ai-acme-002",
            meeting_id=meeting2.id,
            description="Provision staging ClickHouse cluster",
            owner_name="Bob",
            deadline=datetime(2026, 11, 20, 0, 0, 0, tzinfo=timezone.utc),
            status="IN_PROGRESS",
            evidence_segment_id=seg2_3.id,
            timestamp_seconds=16.0,
            created_at=m2_date,
            updated_at=m2_date,
        )
        db.add_all([sum2, dec3, ai2])
        db.commit()

        # 6. Meeting 3: Beta Corp Secret Planning (Tenant Isolation verification)
        m3_date = datetime(2026, 10, 20, 11, 0, 0, tzinfo=timezone.utc)
        meeting3 = Meeting(
            id="meeting-beta-001",
            organization_id=org_beta.id,
            title="Beta Corp Secret Roadmap",
            description="Confidential strategy meeting strictly for Beta Corp.",
            status=MeetingStatus.COMPLETED.value,
            created_at=m3_date,
            updated_at=m3_date,
        )
        db.add(meeting3)
        db.commit()

        t3_redacted = Transcript(
            id="trans-beta-001-red",
            meeting_id=meeting3.id,
            transcript_type=TranscriptType.CLEANED_REDACTED.value,
            version=1,
            language_detected="en",
            is_primary=True,
            created_at=m3_date,
            updated_at=m3_date,
        )
        db.add(t3_redacted)
        db.commit()

        seg3_1 = TranscriptSegment(
            id="seg-m3-001",
            transcript_id=t3_redacted.id,
            sequence_order=1,
            start_time=0.0,
            end_time=10.0,
            text="Beta Corp confidential decision: We selected Oracle Database for Beta Project Phoenix.",
            language_code="en",
            created_at=m3_date,
            updated_at=m3_date,
        )
        db.add(seg3_1)
        db.commit()

        # 7. Index memory chunks into FTS5 and vector store for search and Ask AI
        print("Indexing semantic chunks into FTS5 and vector storage...")
        embedder = MultilingualE5EmbeddingService(lazy_load=False)
        storage_mgr = ChunkStorageManager()

        # Chunks for Meeting 1
        c1 = SemanticChunk(
            text="Welcome everyone to the architecture review. Today we will decide our database strategy. We have decided to adopt PostgreSQL for all analytical reporting workloads.",
            chunk_index=0,
            start_time=0.0,
            end_time=15.0,
            token_count=25,
            segment_ids=[seg1_1.id, seg1_2.id],
            language_codes=["en"],
            dominant_language="en",
            meeting_id=meeting1.id,
            transcript_id=t1_redacted.id,
            organization_id=org_acme.id,
        )
        c2 = SemanticChunk(
            text="Alice will complete the benchmark of analytical queries by October first.",
            chunk_index=1,
            start_time=15.0,
            end_time=25.0,
            token_count=12,
            segment_ids=[seg1_3.id],
            language_codes=["en"],
            dominant_language="en",
            meeting_id=meeting1.id,
            transcript_id=t1_redacted.id,
            organization_id=org_acme.id,
        )
        m1_emb_results = embedder.embed_chunks([c1, c2])
        storage_mgr.store_chunks(db, m1_emb_results, org_acme.id, t1_redacted)

        # Chunks for Meeting 2
        c3 = SemanticChunk(
            text="Reviewing our Q3 decision, PostgreSQL has reached query latency limits on large datasets. We have officially decided to migrate analytical reporting workloads from PostgreSQL to ClickHouse.",
            chunk_index=0,
            start_time=0.0,
            end_time=16.0,
            token_count=30,
            segment_ids=[seg2_1.id, seg2_2.id],
            language_codes=["en"],
            dominant_language="en",
            meeting_id=meeting2.id,
            transcript_id=t2_redacted.id,
            organization_id=org_acme.id,
        )
        c4 = SemanticChunk(
            text="Bob will provision the staging ClickHouse cluster by November twentieth.",
            chunk_index=1,
            start_time=16.0,
            end_time=24.0,
            token_count=12,
            segment_ids=[seg2_3.id],
            language_codes=["en"],
            dominant_language="en",
            meeting_id=meeting2.id,
            transcript_id=t2_redacted.id,
            organization_id=org_acme.id,
        )
        m2_emb_results = embedder.embed_chunks([c3, c4])
        storage_mgr.store_chunks(db, m2_emb_results, org_acme.id, t2_redacted)

        # Chunks for Beta Corp Meeting 3
        c5 = SemanticChunk(
            text="Beta Corp confidential decision: We selected Oracle Database for Beta Project Phoenix.",
            chunk_index=0,
            start_time=0.0,
            end_time=10.0,
            token_count=15,
            segment_ids=[seg3_1.id],
            language_codes=["en"],
            dominant_language="en",
            meeting_id=meeting3.id,
            transcript_id=t3_redacted.id,
            organization_id=org_beta.id,
        )
        m3_emb_results = embedder.embed_chunks([c5])
        storage_mgr.store_chunks(db, m3_emb_results, org_beta.id, t3_redacted)

        print("Seeding complete! Organizations, users, meetings, transcripts, and FTS5 memory chunks ready.")

    finally:
        db.close()

if __name__ == "__main__":
    seed()
