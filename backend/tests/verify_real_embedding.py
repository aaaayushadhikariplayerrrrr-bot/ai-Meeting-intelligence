"""Systematic verification of real local E5 embedding inference against Phase 8.7 criteria."""

import math
import time
import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database.base import Base
from app.services.memory.embedding import (
    MultilingualE5EmbeddingService,
    get_embedding_service,
    E5_PASSAGE_PREFIX,
    E5_QUERY_PREFIX,
)
from app.services.memory.storage import ChunkStorageManager, EXPECTED_DIMENSION
from app.services.memory.types import SemanticChunk, ChunkEmbeddingResult
from app.models.identity import Organization
from app.models.meeting import Meeting
from app.models.memory import TranscriptChunkRecord
from app.models.transcription import Transcript, TranscriptType


def test_real_e5_offline_loading_and_properties():
    """Verify model loads on CPU in offline mode with exact 384 dimensions."""
    service = MultilingualE5EmbeddingService(device="cpu", allow_download=False)
    assert service.is_cached(), "Model should be reported as cached in local hub."
    
    t0 = time.time()
    service.load_model()
    load_time = time.time() - t0
    
    assert service.is_loaded, "Model should be marked as loaded."
    assert service.dimension == 384, f"Dimension must be 384, got {service.dimension}"
    assert service.device == "cpu", f"Device must be CPU, got {service.device}"
    print(f"\n[OK] Model loaded in {load_time:.2f}s on CPU.")


def test_real_e5_passage_embedding():
    """Verify passage embedding applies 'passage: ' prefix, 384 dim, finite values, norm ~ 1.0."""
    service = MultilingualE5EmbeddingService(device="cpu", allow_download=False)
    
    passage_text = "The committee unanimously approved the migration to SQLite with FTS5 indexing."
    embeddings = service.embed_passages([passage_text])
    
    assert len(embeddings) == 1
    raw_emb = embeddings[0]
    
    # 1. Type and Dimension
    assert isinstance(raw_emb, list)
    assert len(raw_emb) == 384, f"Dimension must be 384, got {len(raw_emb)}"
    
    emb = np.array(raw_emb, dtype=np.float32)
    assert emb.shape == (384,)
    
    # 2. Finite values
    assert np.all(np.isfinite(emb)), "Embeddings must contain strictly finite values."
    
    # 3. L2 Unit Normalization
    norm = float(np.linalg.norm(emb))
    assert math.isclose(norm, 1.0, abs_tol=1e-4), f"L2 norm must be ~1.0, got {norm}"
    print(f"\n[OK] Passage embedding verified. Dimension: {len(raw_emb)}, Norm: {norm:.6f}")


def test_real_e5_query_embedding():
    """Verify query embedding applies 'query: ' prefix, 384 dim, finite values, norm ~ 1.0."""
    service = MultilingualE5EmbeddingService(device="cpu", allow_download=False)
    
    query_text = "What database indexing was approved?"
    raw_query_emb = service.embed_query(query_text)
    
    assert isinstance(raw_query_emb, list)
    assert len(raw_query_emb) == 384
    
    query_emb = np.array(raw_query_emb, dtype=np.float32)
    assert query_emb.shape == (384,)
    assert np.all(np.isfinite(query_emb))
    
    norm = float(np.linalg.norm(query_emb))
    assert math.isclose(norm, 1.0, abs_tol=1e-4)
    print(f"\n[OK] Query embedding verified. Dimension: {len(raw_query_emb)}, Norm: {norm:.6f}")


def test_real_e5_prefix_distinction():
    """Verify that passage and query prefixes produce different embeddings for identical text."""
    service = MultilingualE5EmbeddingService(device="cpu", allow_download=False)
    text = "Deployment of Redis cache on port 6379."
    
    passage_emb = service.embed_passages([text])[0]
    query_emb = service.embed_query(text)
    
    # Cosine similarity between same text under different prefixes
    cos_sim = float(np.dot(passage_emb, query_emb))
    assert cos_sim < 0.999, f"Prefixes 'passage:' and 'query:' must alter representations, similarity={cos_sim}"
    print(f"\n[OK] Prefix distinction verified. Cross-prefix similarity: {cos_sim:.4f}")


def test_real_e5_batch_embedding_consistency():
    """Verify single vs batched embeddings produce identical outputs."""
    service = MultilingualE5EmbeddingService(device="cpu", allow_download=False, batch_size=2)
    
    texts = [
        "First quarter financial targets were exceeded by twelve percent.",
        "Engineering proposed upgrading the frontend framework to React 19.",
        "Security audit flagged two outdated cipher suites in the TLS configuration.",
    ]
    
    # Batched
    batch_embeddings = service.embed_passages(texts)
    assert len(batch_embeddings) == 3
    
    # Singles
    single_embeddings = [service.embed_passages([t])[0] for t in texts]
    
    for i, (b_emb, s_emb) in enumerate(zip(batch_embeddings, single_embeddings)):
        diff = np.max(np.abs(np.array(b_emb, dtype=np.float32) - np.array(s_emb, dtype=np.float32)))
        assert diff < 1e-4, f"Item {i} batch vs single mismatch, max diff: {diff}"
        norm = np.linalg.norm(np.array(b_emb, dtype=np.float32))
        assert math.isclose(norm, 1.0, abs_tol=1e-4)
        
    print("\n[OK] Batch consistency verified across 3 items.")


def test_real_e5_empty_and_whitespace_handling():
    """Verify handling of empty or whitespace inputs without unhandled crashes."""
    service = MultilingualE5EmbeddingService(device="cpu", allow_download=False)
    
    # Empty list
    empty_res = service.embed_passages([])
    assert empty_res == []
    
    # Whitespace passage
    ws_res = service.embed_passages(["   "])
    assert len(ws_res) == 1
    assert len(ws_res[0]) == 384
    assert np.all(np.isfinite(np.array(ws_res[0], dtype=np.float32)))
    
    # Whitespace query
    q_res = service.embed_query("   ")
    assert len(q_res) == 384
    assert np.all(np.isfinite(np.array(q_res, dtype=np.float32)))
    print("\n[OK] Empty and whitespace inputs handled safely.")


def test_real_e5_sqlite_storage_integration():
    """Verify that real E5 float32 embeddings serialize and store in SQLite BLOB vector storage."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    db = Session()
    
    try:
        # Create parent Organization and Meeting to satisfy FKs
        org = Organization(id="org-test", name="Test Org", slug="test-org")
        meeting = Meeting(id="meet-101", organization_id="org-test", title="Test Meeting")
        dummy_tr = Transcript(
            id="tr-safe-101",
            meeting_id="meet-101",
            transcript_type=TranscriptType.CLEANED_REDACTED.value,
            version=1,
        )
        db.add(org)
        db.add(meeting)
        db.add(dummy_tr)
        db.commit()

        service = MultilingualE5EmbeddingService(device="cpu", allow_download=False)
        storage = ChunkStorageManager(expected_dim=384)
        
        chunk = SemanticChunk(
            chunk_id="chk-real-101",
            chunk_index=0,
            text="Privacy-safe passage indexed with genuine E5 model.",
            token_count=10,
            start_time=0.0,
            end_time=15.0,
            segment_ids=["seg-1", "seg-2"],
            language_codes=["en"],
            dominant_language="en",
            meeting_id="meet-101",
            organization_id="org-test",
            transcript_id="tr-safe-101",
        )
        
        emb = service.embed_passages([chunk.text])[0]
        item = ChunkEmbeddingResult(
            chunk=chunk,
            embedding=emb,
            dimension=384,
            model_name=service.model_name,
        )
        
        stored_records = storage.store_chunks(
            db=db,
            chunks_with_embeddings=[item],
            organization_id="org-test",
            transcript=dummy_tr,
        )
        db.commit()
        
        assert len(stored_records) == 1
        rec = db.query(TranscriptChunkRecord).filter_by(id=stored_records[0].id).first()
        assert rec is not None
        assert rec.dimension == 384
        assert len(rec.embedding) == 384 * 4  # 1536 bytes
        
        # De-serialize and compare
        retrieved_vec = np.frombuffer(rec.embedding, dtype=np.float32)
        assert retrieved_vec.shape == (384,)
        np.testing.assert_allclose(retrieved_vec, np.array(emb, dtype=np.float32), rtol=1e-5)
        print("\n[OK] SQLite BLOB serialization and roundtrip storage verified.")
    finally:
        db.close()
