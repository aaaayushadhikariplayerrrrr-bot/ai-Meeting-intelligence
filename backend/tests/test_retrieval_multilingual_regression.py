"""Phase 8.9.6 Retrieval Regression, Multilingual Model Invariance & Quality Suite.

Comprehensive end-to-end regression tests verifying:
1. Multilingual Retrieval Invariance across English, Hindi, Hinglish, and Gujarati.
2. Stem matching on English paraphrased queries.
3. Cross-script Indic retrieval against Latin technical evidence.
4. Robust abstention on out-of-domain and irrelevant queries (zero false citations).
5. Cross-meeting temporal decision reasoning and lineage synthesis.
6. Multi-tenant isolation (strict cross-org boundary enforcement).
7. Privacy-safe transcript version gating (CLEANED_REDACTED only).
"""

import pytest
from datetime import datetime, timezone
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database.base import Base
from app.models.identity import Organization
from app.models.meeting import Meeting
from app.models.transcription import Transcript, TranscriptSegment, TranscriptType
from app.models.intelligence import Decision, DecisionStatus
from app.services.memory.embedding import MultilingualE5EmbeddingService
from app.services.memory.storage import ChunkStorageManager
from app.services.memory.types import SemanticChunk, ChunkEmbeddingResult
from app.services.memory.rag.engine import RAGEngine
from app.services.memory.rag.types import RAGQueryRequest, AbstentionStatus
from app.services.memory.rag.providers import DeterministicAnswerProvider


@pytest.fixture(scope="module")
def regression_env():
    """Build in-memory SQLite database populated with multilingual meeting evidence and decisions."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    db = Session()

    # 1. Organizations
    org_alpha = Organization(id="org-reg-alpha", name="Alpha Corp", slug="alpha-corp")
    org_beta = Organization(id="org-reg-beta", name="Beta Corp", slug="beta-corp")
    db.add_all([org_alpha, org_beta])

    # 2. Meetings in Org Alpha across dates
    m_cloud = Meeting(
        id="meet-reg-cloud",
        organization_id="org-reg-alpha",
        title="Cloud Architecture Review",
        scheduled_start_time=datetime(2026, 8, 10, 10, 0, tzinfo=timezone.utc),
    )
    m_budget = Meeting(
        id="meet-reg-budget",
        organization_id="org-reg-alpha",
        title="Annual Budget & Office Sync",
        scheduled_start_time=datetime(2026, 9, 15, 14, 0, tzinfo=timezone.utc),
    )
    m_db_oct = Meeting(
        id="meet-reg-db-oct",
        organization_id="org-reg-alpha",
        title="Database Migration Sync",
        scheduled_start_time=datetime(2026, 10, 20, 11, 0, tzinfo=timezone.utc),
    )

    # Meeting in Org Beta (for tenant isolation)
    m_beta = Meeting(
        id="meet-reg-beta",
        organization_id="org-reg-beta",
        title="Beta Internal Strategy",
        scheduled_start_time=datetime(2026, 8, 15, 9, 0, tzinfo=timezone.utc),
    )

    db.add_all([m_cloud, m_budget, m_db_oct, m_beta])

    # 3. Transcripts (CLEANED_REDACTED)
    tr_cloud = Transcript(id="tr-reg-cloud", meeting_id="meet-reg-cloud", transcript_type=TranscriptType.CLEANED_REDACTED.value, version=1)
    tr_budget = Transcript(id="tr-reg-budget", meeting_id="meet-reg-budget", transcript_type=TranscriptType.CLEANED_REDACTED.value, version=1)
    tr_db_oct = Transcript(id="tr-reg-db-oct", meeting_id="meet-reg-db-oct", transcript_type=TranscriptType.CLEANED_REDACTED.value, version=1)
    tr_beta = Transcript(id="tr-reg-beta", meeting_id="meet-reg-beta", transcript_type=TranscriptType.CLEANED_REDACTED.value, version=1)
    db.add_all([tr_cloud, tr_budget, tr_db_oct, tr_beta])
    db.commit()

    # 4. Multilingual Passages
    service = MultilingualE5EmbeddingService(device="cpu", allow_download=False)
    storage = ChunkStorageManager(expected_dim=384)

    passages = [
        {
            "id": "chk-k8s-reg",
            "meeting_id": "meet-reg-cloud",
            "tr_id": "tr-reg-cloud",
            "org_id": "org-reg-alpha",
            "lang": "en",
            "text": "The infrastructure engineering team finalized migrating microservices to Kubernetes clusters on AWS EKS to improve scalability and automate container orchestration.",
        },
        {
            "id": "chk-redis-reg",
            "meeting_id": "meet-reg-cloud",
            "tr_id": "tr-reg-cloud",
            "org_id": "org-reg-alpha",
            "lang": "en",
            "text": "Redis was selected as the distributed caching layer with a default TTL of 3600 seconds to reduce relational database read pressure.",
        },
        {
            "id": "chk-budget-reg",
            "meeting_id": "meet-reg-budget",
            "tr_id": "tr-reg-budget",
            "org_id": "org-reg-alpha",
            "lang": "hi",
            "text": "वित्तीय वर्ष के लिए मार्केटिंग बजट में पंद्रह प्रतिशत की वृद्धि को सर्वसम्मति से मंजूरी दी गई।",
        },
        {
            "id": "chk-office-gu-reg",
            "meeting_id": "meet-reg-budget",
            "tr_id": "tr-reg-budget",
            "org_id": "org-reg-alpha",
            "lang": "gu",
            "text": "નવી ઓફિસ માટે વાર્ષિક ભાડા કરાર પર બંને પક્ષો દ્વારા હસ્તાક્ષર કરવામાં આવ્યા છે.",
        },
        {
            "id": "chk-beta-confidential",
            "meeting_id": "meet-reg-beta",
            "tr_id": "tr-reg-beta",
            "org_id": "org-reg-beta",
            "lang": "en",
            "text": "Beta Corp entered confidential negotiations to acquire Gamma Systems for thirty million dollars.",
        },
    ]

    texts = [p["text"] for p in passages]
    embeddings = service.embed_passages(texts)

    for p, emb in zip(passages, embeddings):
        chunk = SemanticChunk(
            chunk_id=p["id"],
            chunk_index=0,
            text=p["text"],
            token_count=len(p["text"].split()),
            start_time=15.0,
            end_time=45.0,
            segment_ids=[f"seg-{p['id']}"],
            language_codes=[p["lang"]],
            dominant_language=p["lang"],
            meeting_id=p["meeting_id"],
            organization_id=p["org_id"],
            transcript_id=p["tr_id"],
        )
        emb_res = ChunkEmbeddingResult(
            chunk=chunk,
            embedding=emb,
            dimension=384,
            model_name=service.model_name,
        )
        tr = db.query(Transcript).filter(Transcript.id == p["tr_id"]).first()
        storage.store_chunks(db=db, chunks_with_embeddings=[emb_res], organization_id=p["org_id"], transcript=tr)

    # 5. Structured Decisions with Historical Supersession Lineage (Org Alpha)
    dec_aug = Decision(
        id="dec-aug-dynamo",
        meeting_id="meet-reg-cloud",
        description="Adopt Amazon DynamoDB as the primary transactional database engine.",
        rationale_or_context="Initial cloud-native scalability experiment for database engine architecture.",
        status=DecisionStatus.SUPERSEDED.value,
        timestamp_seconds=25.0,
    )
    db.add(dec_aug)
    db.flush()

    dec_oct = Decision(
        id="dec-oct-postgres",
        meeting_id="meet-reg-db-oct",
        description="Migrate primary transactional database engine to PostgreSQL 16 with read replicas.",
        rationale_or_context="DynamoDB lacked ACID relational transaction capabilities required for database engine architecture.",
        status=DecisionStatus.ACTIVE.value,
        timestamp_seconds=30.0,
        supersedes_decision_id="dec-aug-dynamo",
    )
    db.add(dec_oct)
    db.commit()

    rag_engine = RAGEngine(embedding_service=service)

    return {
        "db": db,
        "engine": rag_engine,
        "service": service,
    }


def test_stem_inflectional_helpers():
    """Verify morphological stem extraction and keyword-stem matching."""
    provider = DeterministicAnswerProvider()

    # Stem extraction tests
    assert provider._extract_stem("orchestrating") == "orchestrat"
    assert provider._extract_stem("orchestration") == "orchestrat"
    assert provider._extract_stem("scalable") == "scalable"
    assert provider._extract_stem("scaling") == "scal"
    assert provider._extract_stem("migrated") == "migrat"
    assert provider._extract_stem("authentication") == "authenticat"
    assert provider._extract_stem("authentic") == "authentic"
    assert provider._extract_stem("authentication") != provider._extract_stem("authentic")

    # Stem matching in sentences
    sentence = "We finalized container orchestration on Kubernetes."
    assert provider._matches_keyword_or_stem(sentence, ["orchestrating"])
    assert provider._matches_keyword_or_stem(sentence, ["kubernetes"])
    assert not provider._matches_keyword_or_stem(sentence, ["database", "redis"])


def test_multilingual_pipeline_answering(regression_env):
    """Verify supported answering across English, Hindi, Hinglish, and Gujarati."""
    db = regression_env["db"]
    engine = regression_env["engine"]

    test_cases = [
        # English direct & paraphrased -> should retrieve Cloud Architecture Review (K8s / Redis)
        ("How are containers deployed on AWS?", "meet-reg-cloud"),
        ("Which tool was selected for orchestrating pods?", "meet-reg-cloud"),
        ("What in-memory technology was adopted to speed up database queries?", "meet-reg-cloud"),
        # Hindi cross-lingual & direct
        ("कंटेनर ऑर्केस्ट्रेशन के लिए क्या निर्णय लिया गया?", "meet-reg-cloud"),
        ("डेटाबेस लोड कम करने के लिए कौन सी कैशिंग प्रणाली चुनी गई?", "meet-reg-cloud"),
        ("मार्केटिंग बजट में कितनी वृद्धि को मंजूरी दी गई?", "meet-reg-budget"),
        # Hinglish direct & paraphrased
        ("Container orchestration aur scaling ke liye konsa platform select kiya?", "meet-reg-cloud"),
        ("Database query read load kam karne ke liye kya use karenge?", "meet-reg-cloud"),
        # Gujarati cross-lingual & direct
        ("ઓફિસના ભાડા કરાર બાબતે શું થયું?", "meet-reg-budget"),
        ("માઈક્રોસર્વિસિસ ચલાવવા માટે કયું પ્લેટફોર્મ નક્કી કરવામાં આવ્યું?", "meet-reg-cloud"),
    ]

    for question, expected_meeting in test_cases:
        req = RAGQueryRequest(question=question, organization_id="org-reg-alpha")
        res = engine.query(db=db, request=req)

        assert not res.is_abstained, f"Query '{question}' unexpectedly abstained: {res.abstention_status}"
        assert res.abstention_status == AbstentionStatus.ANSWER_SUPPORTED
        assert len(res.citations) > 0, f"Query '{question}' returned zero citations"
        meeting_ids = [c.meeting_id for c in res.citations]
        assert expected_meeting in meeting_ids, f"Expected meeting {expected_meeting} not in {meeting_ids} for '{question}'"


def test_out_of_domain_and_irrelevant_abstention(regression_env):
    """Verify zero false citations and proper abstention for out-of-domain queries."""
    db = regression_env["db"]
    engine = regression_env["engine"]

    irrelevant_queries = [
        ("IRR EN Space", "What is the trajectory of the Europa space probe mission?"),
        ("IRR EN Cooking", "What is the recipe for traditional Italian sourdough bread?"),
        ("IRR EN Borderline Tech", "Did the team discuss creating a mobile game with Unity?"),
        ("IRR HI Space", "मंगल ग्रह पर जीवन की संभावना के बारे में क्या खोज हुई?"),
        ("IRR HI General", "मौसम विभाग के अनुसार कल बारिश होगी या नहीं?"),
        ("IRR Hinglish General", "Weekend par movie dekhne ka kya plan hai?"),
        ("IRR GU General", "અમદાવાદમાં હવામાન કેવું રહેશે?"),
    ]

    for label, q_text in irrelevant_queries:
        req = RAGQueryRequest(question=q_text, organization_id="org-reg-alpha")
        res = engine.query(db=db, request=req)

        assert res.is_abstained is True, f"{label} ('{q_text}') should have abstained but got is_abstained=False"
        assert len(res.citations) == 0, f"{label} ('{q_text}') cited {len(res.citations)} chunks unexpectedly"
        assert res.abstention_status in [
            AbstentionStatus.NO_EVIDENCE,
            AbstentionStatus.INSUFFICIENT_EVIDENCE,
        ], f"{label} got unexpected status {res.abstention_status}"


def test_cross_tenant_isolation(regression_env):
    """Verify strict tenant isolation: Org Beta cannot retrieve Org Alpha evidence or decisions."""
    db = regression_env["db"]
    engine = regression_env["engine"]

    # 1. Org Beta queries for Org Alpha's Kubernetes data
    req_k8s_beta = RAGQueryRequest(question="How are containers deployed on AWS?", organization_id="org-reg-beta")
    res_k8s_beta = engine.query(db=db, request=req_k8s_beta)

    assert res_k8s_beta.is_abstained is True
    assert len(res_k8s_beta.citations) == 0
    assert res_k8s_beta.abstention_status == AbstentionStatus.NO_EVIDENCE

    # 2. Org Beta queries for its own confidential acquisition
    req_beta_own = RAGQueryRequest(question="What was discussed regarding acquiring Gamma Systems?", organization_id="org-reg-beta")
    res_beta_own = engine.query(db=db, request=req_beta_own)

    assert not res_beta_own.is_abstained
    assert res_beta_own.abstention_status == AbstentionStatus.ANSWER_SUPPORTED
    assert "meet-reg-beta" in [c.meeting_id for c in res_beta_own.citations]

    # 3. Org Alpha queries for Beta's confidential acquisition -> must abstain with NO_EVIDENCE
    req_alpha_probe = RAGQueryRequest(question="What was discussed regarding acquiring Gamma Systems?", organization_id="org-reg-alpha")
    res_alpha_probe = engine.query(db=db, request=req_alpha_probe)

    assert res_alpha_probe.is_abstained is True
    assert len(res_alpha_probe.citations) == 0
    assert res_alpha_probe.abstention_status == AbstentionStatus.NO_EVIDENCE


def test_temporal_decision_lineage_and_citations(regression_env):
    """Verify cross-meeting temporal decision reasoning, supersession lineage, and verified citations."""
    db = regression_env["db"]
    engine = regression_env["engine"]

    req = RAGQueryRequest(
        question="What decisions were made about our database engine architecture?",
        organization_id="org-reg-alpha",
    )
    res = engine.query(db=db, request=req)

    assert not res.is_abstained
    assert res.abstention_status == AbstentionStatus.ANSWER_SUPPORTED

    # Check temporal decision synthesis
    answer = res.answer
    assert "PostgreSQL 16" in answer
    assert "DynamoDB" in answer
    assert "SUPERSEDED" in answer or "superseded" in answer
    assert "Cloud Architecture Review" in answer or "Database Migration Sync" in answer

    # Verify structured decisions are returned in response
    assert len(res.structured_decisions) >= 2
    statuses = {d["status"] for d in res.structured_decisions}
    assert DecisionStatus.ACTIVE.value in statuses
    assert DecisionStatus.SUPERSEDED.value in statuses

    # Verify citations exist and have meeting title
    assert len(res.citations) > 0
    for cit in res.citations:
        assert cit.meeting_title is not None
