"""Benchmark Profiling for Real Multilingual E5 Model and RAG Operations.

Measures:
1. Model loading time (cold CPU load)
2. Single embedding latency (1 passage, 1 query)
3. Batched embedding latency (batches of 4, 8, 16)
4. Memory footprint (System RAM delta and GPU VRAM usage)
5. End-to-end Indexing throughput
6. Search latency (/api/v1/memory/search)
7. RAG query latency (/api/v1/memory/query)
"""

import gc
import json
import os
import sys
import time
import uuid
import psutil
import torch
import numpy as np

from fastapi.testclient import TestClient
from app.main import app
from app.database.session import SessionLocal
from app.core.security import create_access_token
from app.models.identity import Organization, User, OrganizationMembership, OrgRole
from app.models.meeting import Meeting, MeetingStatus
from app.models.memory import TranscriptChunkRecord
from app.models.transcription import Transcript, TranscriptSegment, TranscriptType
from app.services.memory.embedding import MultilingualE5EmbeddingService, get_embedding_service
from app.services.memory.storage import ChunkStorageManager
from app.services.memory.chunking import chunk_privacy_safe_transcript
from app.services.memory.types import ChunkEmbeddingResult


def get_process_memory_mb() -> float:
    """Return current process Resident Set Size (RSS) in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def get_gpu_memory_mb() -> float:
    """Return PyTorch allocated GPU VRAM in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 * 1024)
    return 0.0


def run_benchmark():
    results = {}
    print("=" * 70)
    print("PHASE 8.7 REAL E5 EMBEDDING & RAG PERFORMANCE BENCHMARK")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # 1. Baseline Memory
    # -------------------------------------------------------------------------
    gc.collect()
    ram_before = get_process_memory_mb()
    vram_before = get_gpu_memory_mb()
    print(f"[*] Baseline Process RAM: {ram_before:.2f} MB")
    print(f"[*] Baseline GPU VRAM:    {vram_before:.2f} MB")

    # -------------------------------------------------------------------------
    # 2. Cold Model Loading
    # -------------------------------------------------------------------------
    t_load_start = time.perf_counter()
    service = MultilingualE5EmbeddingService(device="cpu", allow_download=False)
    # Trigger actual model initialization
    service.load_model()
    t_load_end = time.perf_counter()
    load_time_s = t_load_end - t_load_start

    ram_after_load = get_process_memory_mb()
    vram_after_load = get_gpu_memory_mb()
    model_ram_delta = ram_after_load - ram_before
    model_vram_delta = vram_after_load - vram_before

    print(f"\n[1] Cold Model Load Time:  {load_time_s:.3f} s")
    print(f"    RAM Footprint (Delta): {model_ram_delta:.2f} MB")
    print(f"    GPU VRAM Delta:        {model_vram_delta:.2f} MB (Target: 0.0 MB on CPU)")
    results["model_load_time_s"] = round(load_time_s, 3)
    results["model_ram_delta_mb"] = round(model_ram_delta, 2)
    results["model_vram_delta_mb"] = round(model_vram_delta, 2)

    # -------------------------------------------------------------------------
    # 3. Single Embedding Latency (Passage & Query)
    # -------------------------------------------------------------------------
    sample_passage = (
        "The engineering team completed the deployment of Postgres 16 cluster with streaming replication."
    )
    sample_query = "How is database replication configured?"

    # Warmup
    _ = service.embed_passages([sample_passage])
    _ = service.embed_query(sample_query)

    # Passage Latency (10 iterations)
    passage_times = []
    for _ in range(10):
        t0 = time.perf_counter()
        _ = service.embed_passages([sample_passage])
        passage_times.append(time.perf_counter() - t0)
    avg_passage_ms = (sum(passage_times) / len(passage_times)) * 1000

    # Query Latency (10 iterations)
    query_times = []
    for _ in range(10):
        t0 = time.perf_counter()
        _ = service.embed_query(sample_query)
        query_times.append(time.perf_counter() - t0)
    avg_query_ms = (sum(query_times) / len(query_times)) * 1000

    print(f"\n[2] Single Passage Latency: {avg_passage_ms:.2f} ms (avg of 10 runs)")
    print(f"    Single Query Latency:   {avg_query_ms:.2f} ms (avg of 10 runs)")
    results["single_passage_latency_ms"] = round(avg_passage_ms, 2)
    results["single_query_latency_ms"] = round(avg_query_ms, 2)

    # -------------------------------------------------------------------------
    # 4. Batched Embedding Latency & Throughput
    # -------------------------------------------------------------------------
    batch_sentences = [
        f"Transcript chunk {i}: Project milestone discussion regarding microservices architecture and Kubernetes rollout."
        for i in range(16)
    ]

    print("\n[3] Batched Embedding Latency & Throughput:")
    batch_results = {}
    for batch_size in [4, 8, 16]:
        batch_slice = batch_sentences[:batch_size]
        service.batch_size = batch_size
        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            _ = service.embed_passages(batch_slice)
            times.append(time.perf_counter() - t0)
        avg_batch_s = sum(times) / len(times)
        throughput = batch_size / avg_batch_s
        latency_per_chunk_ms = (avg_batch_s / batch_size) * 1000
        print(
            f"    Batch Size {batch_size:2d}: {avg_batch_s*1000:6.2f} ms total "
            f"({latency_per_chunk_ms:5.2f} ms/chunk | {throughput:5.1f} chunks/sec)"
        )
        batch_results[f"batch_{batch_size}"] = {
            "total_ms": round(avg_batch_s * 1000, 2),
            "per_chunk_ms": round(latency_per_chunk_ms, 2),
            "throughput_chunks_per_sec": round(throughput, 2),
        }
    results["batch_benchmarks"] = batch_results

    # -------------------------------------------------------------------------
    # 5. Full Pipeline Indexing for Realistic Meeting
    # -------------------------------------------------------------------------
    db = SessionLocal()
    client = TestClient(app)
    try:
        # Create test org, user, meeting with 20 segments
        org = Organization(
            id=str(uuid.uuid4()),
            name="Benchmark Org",
            slug=f"bench-org-{uuid.uuid4().hex[:6]}"
        )
        user = User(
            id=str(uuid.uuid4()),
            email=f"bench_{uuid.uuid4().hex[:6]}@example.com",
            full_name="Benchmark User",
            is_active=True,
        )
        db.add_all([org, user])
        db.commit()

        membership = OrganizationMembership(
            id=str(uuid.uuid4()),
            user_id=user.id,
            organization_id=org.id,
            role=OrgRole.ORG_ADMIN.value,
        )
        db.add(membership)
        db.commit()

        token = create_access_token(subject=user.id, token_version=user.token_version)

        meeting = Meeting(
            id=str(uuid.uuid4()),
            organization_id=org.id,
            created_by_user_id=user.id,
            title="Benchmark 45-Minute Sprint Planning",
            status=MeetingStatus.COMPLETED.value,
        )
        db.add(meeting)
        db.commit()

        tr = Transcript(
            id=str(uuid.uuid4()),
            meeting_id=meeting.id,
            transcript_type=TranscriptType.CLEANED_REDACTED.value,
            version=1,
            language_detected="en",
        )
        db.add(tr)
        db.commit()

        # Add 20 meeting segments (simulating ~10-15 minute discussion)
        segments_data = [
            "Welcome everyone to sprint planning. Today we prioritize the payment gateway migration.",
            "Stripe integration is currently in staging with 99.8% webhook delivery reliability.",
            "We observed intermittent timeout issues when communicating with Razorpay fallback.",
            "The database team has migrated the user table schema to partition by billing region.",
            "Redis cache hit ratio improved from 72% to 91% following key expiration tuning.",
            "Frontend team has finalized the React 19 server components architecture.",
            "Mobile app release is scheduled for Thursday pending Google Play Store review.",
            "Security audit identified two moderate vulnerabilities in third-party npm packages.",
            "All CVE patches have been applied and tested in the ephemeral review environments.",
            "Infrastructure costs were reduced by 14% by terminating idle development instances.",
            "Next sprint goal is to implement automated disaster recovery failover across regions.",
            "DevOps verified backup restoration procedures take under 12 minutes for 500GB data.",
            "Customer support reported an increase in inquiries regarding SSO SAML configuration.",
            "Okta integration guide has been published to the developer documentation portal.",
            "API rate limiting has been enforced at 100 requests per minute per IP address.",
            "Telemetry dashboards on Grafana are showing normal P99 latency across all routes.",
            "Elasticsearch indexing pipeline was replaced by SQLite FTS5 for local organizational search.",
            "The hybrid retrieval benchmark demonstrates high recall across keyword and dense queries.",
            "Action items have been assigned to lead engineers for immediate follow-up.",
            "Meeting adjourned. Next sync scheduled for next Monday at 10:00 AM.",
        ]

        segments = []
        for i, text in enumerate(segments_data):
            seg = TranscriptSegment(
                id=str(uuid.uuid4()),
                transcript_id=tr.id,
                sequence_order=i + 1,
                start_time=float(i * 30),
                end_time=float((i + 1) * 30),
                text=text,
                language_code="en",
            )
            segments.append(seg)
        db.add_all(segments)
        db.commit()

        # Measure Chunking + Embedding + SQLite storage
        t_index_start = time.perf_counter()

        chunks = chunk_privacy_safe_transcript(
            transcript=tr,
            meeting=meeting,
            segments=segments,
            target_tokens=150,
            overlap_tokens=30,
        )

        texts = [c.text for c in chunks]
        service.batch_size = 8
        embeddings = service.embed_passages(texts)

        storage_mgr = ChunkStorageManager()
        chunk_results = [
            ChunkEmbeddingResult(
                chunk=chunk,
                embedding=emb,
                dimension=384,
                model_name=service.model_name,
            )
            for chunk, emb in zip(chunks, embeddings)
        ]
        stored = storage_mgr.store_chunks(
            db=db,
            chunks_with_embeddings=chunk_results,
            organization_id=org.id,
            transcript=tr,
        )
        db.commit()
        t_index_end = time.perf_counter()
        total_index_s = t_index_end - t_index_start

        print(f"\n[4] Meeting Indexing Latency (20 Segments -> {len(stored)} Chunks):")
        print(f"    Total Indexing Time: {total_index_s:.3f} s")
        print(f"    Rate:                {len(segments)/total_index_s:.1f} segments/sec")
        results["meeting_indexing_time_s"] = round(total_index_s, 3)
        results["chunks_created"] = len(stored)

        # -------------------------------------------------------------------------
        # 6. REST API Latency: /api/v1/memory/search
        # -------------------------------------------------------------------------
        headers = {
            "Authorization": f"Bearer {token}",
            "X-Organization-Id": org.id,
        }

        search_times = []
        for _ in range(5):
            t0 = time.perf_counter()
            resp = client.post(
                "/api/v1/memory/search",
                headers=headers,
                json={"query": "Stripe webhook delivery and payment", "limit": 5},
            )
            assert resp.status_code == 200
            search_times.append(time.perf_counter() - t0)
        avg_search_ms = (sum(search_times) / len(search_times)) * 1000

        print(f"\n[5] Authenticated /memory/search Latency: {avg_search_ms:.2f} ms (avg of 5 runs)")
        results["api_search_latency_ms"] = round(avg_search_ms, 2)

        # -------------------------------------------------------------------------
        # 7. REST API Latency: /api/v1/memory/query (Deterministic RAG)
        # -------------------------------------------------------------------------
        query_times = []
        for _ in range(5):
            t0 = time.perf_counter()
            resp = client.post(
                "/api/v1/memory/query",
                headers=headers,
                json={"question": "What is the backup restoration time?", "max_evidence_chunks": 5},
            )
            assert resp.status_code == 200
            query_times.append(time.perf_counter() - t0)
        avg_rag_ms = (sum(query_times) / len(query_times)) * 1000

        print(f"\n[6] Authenticated /memory/query Latency:  {avg_rag_ms:.2f} ms (avg of 5 runs)")
        results["api_query_latency_ms"] = round(avg_rag_ms, 2)

    finally:
        db.close()

    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY (JSON):")
    print(json.dumps(results, indent=2))
    print("=" * 70)

    # Save results to json for reporting
    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    run_benchmark()
