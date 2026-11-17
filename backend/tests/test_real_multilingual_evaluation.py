"""Controlled Multilingual Retrieval Evaluation for intfloat/multilingual-e5-small.

Evaluates:
1. English query -> English evidence
2. Hindi query -> English evidence (cross-lingual)
3. English query -> Hindi evidence (cross-lingual)
4. Hinglish query -> Multilingual evidence
5. Gujarati query -> Gujarati evidence
6. Irrelevant queries and no-evidence handling
7. Meeting and date range filters
8. Duplicate passage handling
9. Cross-organization tenant isolation
10. Comparison: FTS5 keyword vs dense vector vs Reciprocal Rank Fusion (RRF)
"""

import math
import time
import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database.base import Base
from app.models.identity import Organization
from app.models.meeting import Meeting
from app.models.transcription import Transcript, TranscriptSegment, TranscriptType
from app.models.memory import TranscriptChunkRecord
from app.services.memory.embedding import MultilingualE5EmbeddingService
from app.services.memory.storage import ChunkStorageManager
from app.services.memory.types import SemanticChunk, ChunkEmbeddingResult
from app.services.memory.retrieval import search_keyword, search_semantic, search_hybrid_rrf


@pytest.fixture(scope="module")
def multilingual_eval_env():
    """Build in-memory SQLite database populated with multilingual meeting evidence."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    db = Session()

    # 1. Organizations
    org_a = Organization(id="org-alpha", name="Alpha Technologies", slug="alpha-tech")
    org_b = Organization(id="org-beta", name="Beta Industries", slug="beta-ind")
    db.add_all([org_a, org_b])

    # 2. Meetings in Org Alpha
    meet_en = Meeting(id="meet-en", organization_id="org-alpha", title="Q3 Cloud Infrastructure Architecture")
    meet_hi = Meeting(id="meet-hi", organization_id="org-alpha", title="वार्षिक वित्तीय एवं बजट समीक्षा")
    meet_hinglish = Meeting(id="meet-hinglish", organization_id="org-alpha", title="Engineering Sprint Planning & Release Velocity")
    meet_gu = Meeting(id="meet-gu", organization_id="org-alpha", title="ઓફિસ સંચાલન અને સુવિધા સમીક્ષા")
    
    # Meeting in Org Beta (Tenant Isolation)
    meet_beta = Meeting(id="meet-beta", organization_id="org-beta", title="Beta Confidential M&A Strategy")
    
    db.add_all([meet_en, meet_hi, meet_hinglish, meet_gu, meet_beta])

    # 3. Transcripts (Privacy-safe CLEANED_REDACTED)
    tr_en = Transcript(id="tr-en", meeting_id="meet-en", transcript_type=TranscriptType.CLEANED_REDACTED.value, version=1)
    tr_hi = Transcript(id="tr-hi", meeting_id="meet-hi", transcript_type=TranscriptType.CLEANED_REDACTED.value, version=1)
    tr_hinglish = Transcript(id="tr-hinglish", meeting_id="meet-hinglish", transcript_type=TranscriptType.CLEANED_REDACTED.value, version=1)
    tr_gu = Transcript(id="tr-gu", meeting_id="meet-gu", transcript_type=TranscriptType.CLEANED_REDACTED.value, version=1)
    tr_beta = Transcript(id="tr-beta", meeting_id="meet-beta", transcript_type=TranscriptType.CLEANED_REDACTED.value, version=1)
    
    db.add_all([tr_en, tr_hi, tr_hinglish, tr_gu, tr_beta])
    db.commit()

    # 4. Passages to Index
    passages = [
        # English passages
        {
            "id": "chk-en-k8s",
            "meeting_id": "meet-en",
            "org_id": "org-alpha",
            "tr_id": "tr-en",
            "lang": "en",
            "text": "The infrastructure engineering team finalized migrating microservices to Kubernetes clusters on AWS EKS to improve scalability and automate container orchestration.",
        },
        {
            "id": "chk-en-redis",
            "meeting_id": "meet-en",
            "org_id": "org-alpha",
            "tr_id": "tr-en",
            "lang": "en",
            "text": "Redis was selected as the distributed caching layer with a default TTL of 3600 seconds to reduce relational database read pressure.",
        },
        # Hindi passages
        {
            "id": "chk-hi-budget",
            "meeting_id": "meet-hi",
            "org_id": "org-alpha",
            "tr_id": "tr-hi",
            "lang": "hi",
            "text": "वित्तीय वर्ष के लिए मार्केटिंग बजट में पंद्रह प्रतिशत की वृद्धि को सर्वसम्मति से मंजूरी दी गई।",
        },
        {
            "id": "chk-hi-hiring",
            "meeting_id": "meet-hi",
            "org_id": "org-alpha",
            "tr_id": "tr-hi",
            "lang": "hi",
            "text": "नई इंजीनियरिंग टीम की भर्ती के लिए कुल पचास लाख रुपये का अतिरिक्त फंड निर्धारित किया गया है।",
        },
        # Hinglish passages
        {
            "id": "chk-hinglish-freeze",
            "meeting_id": "meet-hinglish",
            "org_id": "org-alpha",
            "tr_id": "tr-hinglish",
            "lang": "hi-Latn",
            "text": "Next sprint me ham production deployment freeze karenge taaki QA team performance testing complete kar sake.",
        },
        {
            "id": "chk-hinglish-backup",
            "meeting_id": "meet-hinglish",
            "org_id": "org-alpha",
            "tr_id": "tr-hinglish",
            "lang": "hi-Latn",
            "text": "Database backup validation har Sunday midnight ko run hoga automated cron job ke through.",
        },
        # Gujarati passages
        {
            "id": "chk-gu-lease",
            "meeting_id": "meet-gu",
            "org_id": "org-alpha",
            "tr_id": "tr-gu",
            "lang": "gu",
            "text": "નવી ઓફિસ માટે વાર્ષિક ભાડા કરાર પર બંને પક્ષો દ્વારા હસ્તાક્ષર કરવામાં આવ્યા છે.",
        },
        {
            "id": "chk-gu-fire",
            "meeting_id": "meet-gu",
            "org_id": "org-alpha",
            "tr_id": "tr-gu",
            "lang": "gu",
            "text": "કર્મચારીઓની સુરક્ષા માટે નવા ફાયર સેફ્ટી સાધનો આવતા સપ્તાહે સ્થાપિત કરવામાં આવશે.",
        },
        # Tenant isolation passage (Org Beta)
        {
            "id": "chk-beta-mna",
            "meeting_id": "meet-beta",
            "org_id": "org-beta",
            "tr_id": "tr-beta",
            "lang": "en",
            "text": "Competitor Gamma submitted a confidential thirty million dollar acquisition offer for the cloud subsidiary.",
        },
    ]

    service = MultilingualE5EmbeddingService(device="cpu", allow_download=False)
    storage = ChunkStorageManager(expected_dim=384)

    # Embed and store all passages
    texts = [p["text"] for p in passages]
    vectors = service.embed_passages(texts)

    items_by_org = {}
    for p, v in zip(passages, vectors):
        chunk = SemanticChunk(
            chunk_id=p["id"],
            chunk_index=0,
            text=p["text"],
            token_count=len(p["text"].split()),
            start_time=10.0,
            end_time=25.0,
            segment_ids=[f"seg-{p['id']}"],
            language_codes=[p["lang"]],
            dominant_language=p["lang"],
            meeting_id=p["meeting_id"],
            organization_id=p["org_id"],
            transcript_id=p["tr_id"],
            metadata={"chunk_id": p["id"]},
        )
        item = ChunkEmbeddingResult(
            chunk=chunk,
            embedding=v,
            dimension=384,
            model_name=service.model_name,
        )
        items_by_org.setdefault(p["org_id"], []).append((item, p["tr_id"]))

    for org_id, items in items_by_org.items():
        for item, tr_id in items:
            tr = db.query(Transcript).filter_by(id=tr_id).first()
            storage.store_chunks(
                db=db,
                chunks_with_embeddings=[item],
                organization_id=org_id,
                transcript=tr,
            )
    db.commit()

    yield {
        "db": db,
        "service": service,
        "storage": storage,
    }

    db.close()


def test_eval_1_english_query_to_english_evidence(multilingual_eval_env):
    """1. English Query -> English Evidence: 'How are containers deployed on AWS?'"""
    env = multilingual_eval_env
    db = env["db"]
    service = env["service"]

    query = "How are containers deployed on AWS?"
    q_vec = service.embed_query(query)

    # Hybrid Search
    results = search_hybrid_rrf(
        db=db,
        query=query,
        query_vector=q_vec,
        organization_id="org-alpha",
        limit=5,
    )

    assert len(results) > 0
    top = results[0]
    assert top.chunk_id == "chk-en-k8s", f"Expected chk-en-k8s, got {top.chunk_id}"
    assert "Kubernetes clusters on AWS EKS" in top.text
    print(f"\n[EVAL 1 PASS] EN->EN: Top rank '{top.chunk_id}' (RRF score: {top.rrf_score:.4f})")


def test_eval_2_hindi_query_to_english_evidence(multilingual_eval_env):
    """2. Hindi Query -> English Evidence (Cross-lingual): 'कंटेनर ऑर्केस्ट्रेशन के लिए क्या निर्णय लिया गया?'"""
    env = multilingual_eval_env
    db = env["db"]
    service = env["service"]

    query = "कंटेनर ऑर्केस्ट्रेशन के लिए क्या निर्णय लिया गया?"
    q_vec = service.embed_query(query)

    # Dense semantic search should match English Kubernetes passage despite Hindi query
    dense_results = search_semantic(
        db=db,
        query_vector=q_vec,
        organization_id="org-alpha",
        limit=5,
    )

    assert len(dense_results) > 0
    top = dense_results[0]
    assert top.chunk_id == "chk-en-k8s", f"Cross-lingual Hindi query should retrieve English K8s chunk, got {top.chunk_id}"
    assert top.cosine_similarity > 0.70, f"Cross-lingual similarity should exceed 0.70, got {top.cosine_similarity:.4f}"
    print(f"\n[EVAL 2 PASS] HI->EN Cross-Lingual: Top match '{top.chunk_id}' with cosine similarity: {top.cosine_similarity:.4f}")


def test_eval_3_english_query_to_hindi_evidence(multilingual_eval_env):
    """3. English Query -> Hindi Evidence (Cross-lingual): 'How much was the marketing budget increased?'"""
    env = multilingual_eval_env
    db = env["db"]
    service = env["service"]

    query = "How much was the marketing budget increased?"
    q_vec = service.embed_query(query)

    dense_results = search_semantic(
        db=db,
        query_vector=q_vec,
        organization_id="org-alpha",
        limit=5,
    )

    assert len(dense_results) > 0
    top = dense_results[0]
    assert top.chunk_id == "chk-hi-budget", f"Cross-lingual English query should retrieve Hindi budget chunk, got {top.chunk_id}"
    assert top.cosine_similarity > 0.70, f"Cross-lingual similarity should exceed 0.70, got {top.cosine_similarity:.4f}"
    assert "पंद्रह प्रतिशत की वृद्धि" in top.text
    print(f"\n[EVAL 3 PASS] EN->HI Cross-Lingual: Top match '{top.chunk_id}' with cosine similarity: {top.cosine_similarity:.4f}")


def test_eval_4_hinglish_query_to_hinglish_evidence(multilingual_eval_env):
    """4. Hinglish Query -> Hinglish/Multilingual Evidence: 'Production deployment kab freeze hoga?'"""
    env = multilingual_eval_env
    db = env["db"]
    service = env["service"]

    query = "Production deployment kab freeze hoga?"
    q_vec = service.embed_query(query)

    hybrid_results = search_hybrid_rrf(
        db=db,
        query=query,
        query_vector=q_vec,
        organization_id="org-alpha",
        limit=5,
    )

    assert len(hybrid_results) > 0
    top = hybrid_results[0]
    assert top.chunk_id == "chk-hinglish-freeze", f"Expected chk-hinglish-freeze, got {top.chunk_id}"
    print(f"\n[EVAL 4 PASS] Hinglish Retrieval: Top match '{top.chunk_id}' (RRF score: {top.rrf_score:.4f})")


def test_eval_5_gujarati_query_to_gujarati_evidence(multilingual_eval_env):
    """5. Gujarati Query -> Gujarati Evidence: 'ઓફિસના ભાડા કરાર બાબતે શું થયું?'"""
    env = multilingual_eval_env
    db = env["db"]
    service = env["service"]

    query = "ઓફિસના ભાડા કરાર બાબતે શું થયું?"
    q_vec = service.embed_query(query)

    hybrid_results = search_hybrid_rrf(
        db=db,
        query=query,
        query_vector=q_vec,
        organization_id="org-alpha",
        limit=5,
    )

    assert len(hybrid_results) > 0
    top = hybrid_results[0]
    assert top.chunk_id == "chk-gu-lease", f"Expected chk-gu-lease, got {top.chunk_id}"
    assert "વાર્ષિક ભાડા કરાર" in top.text
    print(f"\n[EVAL 5 PASS] Gujarati Retrieval: Top match '{top.chunk_id}' (RRF score: {top.rrf_score:.4f})")


def test_eval_6_irrelevant_query_no_evidence(multilingual_eval_env):
    """6. Irrelevant Queries: Verify zero false-positive keyword matches on completely off-topic query."""
    env = multilingual_eval_env
    db = env["db"]

    query = "orbital mechanics of Jupiter moon Europa space probe trajectory"
    
    # FTS5 should match 0 rows
    fts_results = search_keyword(
        db=db,
        query=query,
        organization_id="org-alpha",
        limit=5,
    )
    assert len(fts_results) == 0, f"Expected 0 FTS5 matches for unrelated query, got {len(fts_results)}"
    print(f"\n[EVAL 6 PASS] Irrelevant Query: FTS5 correctly returned 0 keyword candidates.")


def test_eval_7_meeting_filter_enforcement(multilingual_eval_env):
    """7. Meeting Scope Filter: Scoping search to 'meet-hi' strictly excludes 'meet-en'."""
    env = multilingual_eval_env
    db = env["db"]
    service = env["service"]

    query = "budget and finance allocation"
    q_vec = service.embed_query(query)

    # Search scoped to meet-hi
    hi_results = search_hybrid_rrf(
        db=db,
        query=query,
        query_vector=q_vec,
        organization_id="org-alpha",
        meeting_id="meet-hi",
        limit=5,
    )

    for r in hi_results:
        assert r.meeting_id == "meet-hi", f"Expected only meet-hi, got {r.meeting_id}"
    print(f"\n[EVAL 7 PASS] Meeting filter enforcement verified. All {len(hi_results)} results scoped to meet-hi.")


def test_eval_8_duplicate_passage_handling(multilingual_eval_env):
    """8. Duplicate Passage Handling: Re-indexing an existing transcript cleans prior chunks."""
    env = multilingual_eval_env
    db = env["db"]
    storage = env["storage"]

    # Create temporary transcript to test atomic replacement
    temp_meet = Meeting(id="meet-dup-test", organization_id="org-alpha", title="Duplicate Test Meeting")
    temp_tr = Transcript(id="tr-dup-test", meeting_id="meet-dup-test", transcript_type=TranscriptType.CLEANED_REDACTED.value, version=1)
    db.add_all([temp_meet, temp_tr])
    db.commit()

    chunk = SemanticChunk(
        chunk_id="chk-dup-1",
        chunk_index=0,
        text="Temporary chunk for deletion test.",
        token_count=5,
        start_time=0.0,
        end_time=5.0,
        segment_ids=["seg-dup-1"],
        language_codes=["en"],
        dominant_language="en",
        meeting_id="meet-dup-test",
        organization_id="org-alpha",
        transcript_id="tr-dup-test",
    )
    real_vec = env["service"].embed_passages([chunk.text])[0]
    dummy_item = ChunkEmbeddingResult(
        chunk=chunk,
        embedding=real_vec,
        dimension=384,
        model_name="intfloat/multilingual-e5-small",
    )
    storage.store_chunks(db, [dummy_item], "org-alpha", temp_tr)
    db.commit()

    init_count = db.query(TranscriptChunkRecord).filter_by(meeting_id="meet-dup-test").count()
    assert init_count == 1

    # Simulate atomic re-indexing: delete by transcript
    storage.delete_chunks_by_transcript(db, transcript_id="tr-dup-test", organization_id="org-alpha")
    db.commit()
    new_count = db.query(TranscriptChunkRecord).filter_by(meeting_id="meet-dup-test").count()
    assert new_count == 0, "delete_chunks_by_transcript should cleanly wipe old chunks before re-inserting"
    print(f"\n[EVAL 8 PASS] Stale/duplicate chunk cleanup verified.")


def test_eval_9_tenant_isolation(multilingual_eval_env):
    """9. Strict Cross-Tenant Isolation: Org Alpha cannot retrieve Org Beta chunks and vice versa."""
    env = multilingual_eval_env
    db = env["db"]
    service = env["service"]

    # Query for Beta's confidential acquisition offer
    query = "confidential thirty million dollar acquisition offer"
    q_vec = service.embed_query(query)

    # 1. Org Alpha search MUST return 0 results for Beta's content
    alpha_results = search_hybrid_rrf(
        db=db,
        query=query,
        query_vector=q_vec,
        organization_id="org-alpha",
        limit=5,
    )
    assert not any(r.organization_id == "org-beta" for r in alpha_results)
    assert not any(r.chunk_id == "chk-beta-mna" for r in alpha_results)

    # 2. Org Beta search MUST retrieve chk-beta-mna
    beta_results = search_hybrid_rrf(
        db=db,
        query=query,
        query_vector=q_vec,
        organization_id="org-beta",
        limit=5,
    )
    assert len(beta_results) > 0
    assert beta_results[0].chunk_id == "chk-beta-mna"
    assert beta_results[0].organization_id == "org-beta"
    print(f"\n[EVAL 9 PASS] Strict tenant boundary isolation verified.")


def test_eval_10_retrieval_mode_comparison(multilingual_eval_env):
    """10. Metric Comparison: FTS5 vs Dense Cosine vs Reciprocal Rank Fusion (RRF)."""
    env = multilingual_eval_env
    db = env["db"]
    service = env["service"]

    query = "Redis distributed caching database TTL"
    q_vec = service.embed_query(query)

    # 1. Keyword (FTS5)
    kw = search_keyword(db=db, query=query, organization_id="org-alpha", limit=3)
    kw_ranks = {r.chunk_id: r.bm25_rank for r in kw}

    # 2. Dense Semantic (Cosine)
    dense = search_semantic(db=db, query_vector=q_vec, organization_id="org-alpha", limit=3)
    dense_sims = {r.chunk_id: r.cosine_similarity for r in dense}

    # 3. Hybrid RRF
    hybrid = search_hybrid_rrf(db=db, query=query, query_vector=q_vec, organization_id="org-alpha", limit=3)
    hybrid_scores = {r.chunk_id: r.rrf_score for r in hybrid}

    assert "chk-en-redis" in kw_ranks
    assert "chk-en-redis" in dense_sims
    assert "chk-en-redis" in hybrid_scores
    assert dense_sims["chk-en-redis"] > 0.80

    print(f"\n[EVAL 10 PASS] Retrieval Mode Comparison for 'chk-en-redis':")
    print(f"  - FTS5 Keyword Rank: {kw_ranks['chk-en-redis']}")
    print(f"  - Dense Cosine Similarity: {dense_sims['chk-en-redis']:.4f}")
    print(f"  - Combined RRF Score: {hybrid_scores['chk-en-redis']:.6f}")
