# MEMORA Architecture & Technical Design

This document details the internal architecture, asynchronous pipelines, data flow, storage model, and retrieval mechanisms of the Multilingual Meeting Intelligence System (MEMORA).

---

## 1. High-Level System Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                        React 19 Frontend (Vite)                       │
│  - Auth & Org Isolation     - Live Job Polling & Progress             │
│  - Safe Transcript View     - Cross-Meeting Decision Lineage          │
│  - Ask AI & Evidence Jump   - Commitment Tracking & Conflict Alerts   │
└────────────────────────────────────┬───────────────────────────────────┘
                                     │ HTTP / REST (JWT Bearer)
                                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        FastAPI Backend Server                         │
│  - Route Handlers           - TenantContext Dependency Injection      │
│  - Argon2id & JWT Auth      - Rate Limiting Middleware (Memory)       │
│  - Media Validation         - Role-Based Access Control               │
└──────────────────┬─────────────────────────────────┬───────────────────┘
                   │                                 │
                   ▼                                 ▼
┌─────────────────────────────────────┐  ┌───────────────────────────────┐
│       JobWorker & Async Tasks       │  │        SQLite Database        │
│  - Atomic Compare-and-Swap Claiming │  │  - PRAGMA journal_mode=WAL   │
│  - FFmpeg Normalization             │  │  - PRAGMA busy_timeout=30000 │
│  - Faster-Whisper GPU Engine        │  │  - FTS5 Full-Text Search     │
│  - PII Cleaner & Redactor           │  │  - Dense Vector Storage      │
│  - Chunker & E5 Embedding Engine    │  │  - Audit Logs & Lineage      │
│  - Status Reconciler                │  │  - Versioned Transcripts     │
└─────────────────────────────────────┘  └───────────────────────────────┘
```

---

## 2. Ingestion & Processing Pipeline

The media ingestion and processing pipeline progresses through deterministic, versioned states:

```
[Media Upload / YouTube]
          │
          ▼
1. Validation & Storage:
   - Check extension, validate audio stream via ffprobe
   - Verify duration <= 4.0 hours, size <= 500 MB
   - Stream to disk at `data/storage/<meeting_id>/<asset_id>.<ext>`
          │
          ▼
2. Normalization:
   - FFmpeg extracts & resamples audio to 16 kHz mono WAV
   - Register MediaAsset (media_type='NORMALIZED_AUDIO')
          │
          ▼
3. Speech Recognition (ASR):
   - Background JobWorker claims TRANSCRIPTION job atomically
   - Faster-Whisper `large-v3-turbo` on NVIDIA GPU (or CPU fallback)
   - Chunked processing with overlap deduplication for long recordings
   - Persists Transcript (version=1, type=ORIGINAL) and TranscriptSegments
          │
          ▼
4. Derivation & Privacy Redaction:
   - Cleaner removes stutter and transcription artifacts -> Transcript (CLEANED)
   - Redactor strips PII (Aadhaar, PAN, SSN, emails, phones, API keys, tokens)
   - Persists privacy-safe Transcript (CLEANED_REDACTED)
          │
          ▼
5. Post-Transcription Auto-Delete:
   - Verified commit of TranscriptSegments triggers raw media unlinking
   - Physically deletes raw upload and intermediate WAVs
   - Writes AuditLog (`action="DELETE_RAW_AUDIO"`)
   - Marks `is_raw_deleted = True`
          │
          ▼
6. Memory Indexing:
   - Semantic chunker segments redacted transcripts (200-300 tokens)
   - `intfloat/multilingual-e5-small` generates 384-dim embeddings (CPU)
   - Stores dense vectors in `transcript_chunks` and keywords in `transcript_chunks_fts`
          │
          ▼
7. Meeting Status Reconciliation:
   - Centralized `reconcile_meeting_status()` evaluates all job outcomes
   - Advances meeting status: `INGESTING` -> `PROCESSING` -> `COMPLETED`
```

---

## 3. Asynchronous Worker & Concurrency Model

### Single-Worker Asyncio Architecture
To prevent CPU/GPU oversubscription on local workstation hardware:
* `JobWorker` runs a continuous polling loop with a configurable interval (`JOB_POLL_INTERVAL_SECONDS = 2.0`).
* **Atomic Compare-and-Swap Claiming**:
  ```sql
  UPDATE processing_jobs
  SET status = 'IN_PROGRESS', attempt_count = attempt_count + 1, started_at = :now
  WHERE id = :job_id AND status = 'PENDING'
  ```
  This guarantees that multiple worker threads or background tasks cannot double-process the same job.
* **Non-Blocking Thread Offload**: Synchronous heavy computations (FFmpeg subprocesses, CTranslate2 ASR inference, PyTorch embeddings) are dispatched via `asyncio.to_thread()`, keeping the FastAPI event loop responsive.
* **Startup Crash Recovery**:
  On server boot, `reconcile_stranded_jobs_on_startup()` marks abandoned `IN_PROGRESS` jobs as `FAILED` with `error_details="ServerRestartInterruption"`, preventing locks from indefinitely blocking retry attempts.

---

## 4. SQLite Concurrency & WAL Hardening

MEMORA hardens SQLite for multi-threaded application performance:
1. **Write-Ahead Logging (WAL)**: `PRAGMA journal_mode = WAL;` enables concurrent readers to execute without blocking writers, and writers do not block readers.
2. **Busy Timeout**: `PRAGMA busy_timeout = 30000;` configures a 30-second driver-level spin-lock before raising a `database locked` error.
3. **Synchronous Mode**: `PRAGMA synchronous = NORMAL;` minimizes disk flush overhead during WAL checkpoints while maintaining ACID guarantees against power failures.
4. **Foreign Key Enforcement**: `PRAGMA foreign_keys = ON;` is enforced on every connection pool checkout.

---

## 5. Organizational Memory & Hybrid Retrieval Architecture

```
User Query: "What did we decide about Redis caching?"
                          │
         ┌────────────────┴────────────────┐
         ▼                                 ▼
   Dense Retrieval                 Keyword Retrieval
(`multilingual-e5-small`)       (SQLite FTS5 BM25 Search)
         │                                 │
         │ Cosine Similarity               │ Keyword Matching
         │ (weight: 0.6)                   │ (weight: 0.4)
         └────────────────┬────────────────┘
                          ▼
            Reciprocal Rank Fusion (RRF)
                          │
                          ▼
             Tenant Isolation Enforcement
          (Filter by `organization_id`)
                          │
                          ▼
             Top-K Context Segment Chunks
                          │
         ┌────────────────┴────────────────┐
         ▼                                 ▼
Generative LLM (Ollama)          Deterministic Fallback
(If configured & available)      (Default / Offline Provider)
         │                                 │
         └────────────────┬────────────────┘
                          ▼
Answer Synthesis + Grounded Evidence Citations (Timestamps, Meeting ID)
```

### Honest Abstention Model
When retrieved evidence has low cosine similarity (`score < 0.65`) or contains contradictory statements without resolution, MEMORA returns an explicit abstention state (`INSUFFICIENT_EVIDENCE` or `CONFLICTING_EVIDENCE`) rather than hallucinating an answer.

---

## 6. Cross-Meeting Decision Lineage & Conflict Tracking

* **Lineage Modeling**: Decisions have optional `supersedes_decision_id` links. When a decision is superseded:
  * The predecessor's status becomes `SUPERSEDED`.
  * The successor's status becomes `ACTIVE`.
  * The UI visualizes the chronological evolution and notes what changed.
* **Conflict Detection Engine**:
  * Scans decisions across meetings within the same organization.
  * Compares entity overlaps, keyword negation flips (`will use` vs `will not use`), and contradictory constraints.
  * Flags unlinked contradictory decisions in the **Potential Conflicts View** with direct links to transcript evidence.
* **Commitment Continuity**: Action items track ownership, target completion dates, and review histories across meetings, surfacing overdue deliverables.

---

## 7. Role-Based Access Control (RBAC) Architecture

```
┌─────────────────────────┬──────────────┬──────────────┐
│ Capability              │ ORG_ADMIN    │ MEMBER       │
├─────────────────────────┼──────────────┼──────────────┤
│ Upload Media            │ Yes          │ Yes          │
│ View Safe Transcripts   │ Yes          │ Yes          │
│ View Original Audio     │ Yes          │ No (Hidden)  │
│ View Original Text      │ Yes (Audited)│ No (403)     │
│ Delete Raw Media        │ Yes          │ No (403)     │
│ Cross-Meeting Memory    │ Yes          │ Yes          │
│ Supersede Decisions     │ Yes          │ Yes          │
│ View Hardware Status    │ Yes (Admin)  │ No (Hidden)  │
└─────────────────────────┴──────────────┴──────────────┘
```
