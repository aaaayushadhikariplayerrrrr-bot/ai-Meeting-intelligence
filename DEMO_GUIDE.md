# MEMORA — 5 to 7 Minute College Demonstration Guide

A practical, step-by-step walkthrough script for presenting MEMORA to professors, examiners, or project evaluators.

---

## 1. Quick Presentation Pitch (1 Minute)

> *"Good morning, professors. Today I am presenting MEMORA, a Multilingual Meeting Intelligence and Organizational Memory System.*
>
> *Existing tools like Otter or Zoom Notes are single-meeting transcription tools: they transcribe audio into flat text, write a summary, and forget everything when the meeting ends. They don't track how decisions evolve over time, they often store sensitive voice recordings on external servers, and they struggle with multilingual Hinglish conversations.*
>
> *MEMORA is built on three technical pillars:*
> 1. **Cross-Meeting Intelligence**: Tracking how decisions and commitments evolve across meetings, detecting when a new decision contradicts an older one.
> 2. **Privacy-First Local Processing**: Accelerated ASR on local NVIDIA GPUs, local CPU vector embeddings, and automatic raw audio deletion immediately after verified transcript persistence.
> 3. **Verifiable Evidence Grounding**: Every summary, decision, conflict alert, and Ask AI answer links back to exact transcript segments with second-level timestamps."*

---

## 2. Live Demonstration Sequence (5 Minutes)

### Step 1: Login & Strict Tenant Isolation (30 Seconds)
1. Open `http://localhost:5173`.
2. **Account Authentication**:
   * **Fresh Account**: Click **Create Account**, enter your name, email, and chosen password (min 8 characters). A fresh tenant organization is created automatically, and you are assigned `ORG_ADMIN`.
   * **Acceptance Fixture (Pre-seeded)**: If demonstrating against `test_browser_acceptance.db`, log in with `alice@acme.com` / `Password123!` (`ORG_ADMIN` in Acme Global) or `bob@acme.com` / `Password123!` (standard `MEMBER`).
3. Highlight:
   * **Tenant Isolation**: Users select their organization (e.g. `Acme Global`). All queries, meetings, and memory chunks are isolated to this tenant.
   * **Security Badge**: Argon2id password hashing and database-tracked JWT token version revocation on logout.

### Step 2: Upload Synthetic Meeting & GPU ASR Pipeline (1 Minute)
1. Click **Upload Meeting**.
2. Select the synthetic test fixture at `data/test_fixtures/synthetic_team_sync.wav`.
3. Title: *"Sprint 42 Architecture Sync"*.
4. Click **Upload & Ingest**.
5. Highlight:
   * **Real-time Pipeline**: Explain that the backend worker validates codecs, extracts a standardized 16 kHz mono WAV via FFmpeg, and dispatches to **Faster-Whisper** (`large-v3-turbo`) on the local NVIDIA RTX 5060 GPU.
   * **Live Status**: Point to the status transition from `INGESTING` to `PROCESSING` to `COMPLETED` managed by the centralized meeting lifecycle reconciler.

### Step 3: Dual Transcripts & Privacy Redaction (1 Minute)
1. Click on the uploaded meeting to open **Meeting Intelligence View**.
2. Switch between **Original Transcript** and **Privacy-Safe (Redacted)** tabs.
3. Highlight:
   * Show that filler words and stutter are cleaned.
   * Show that PII (emails, phone numbers, Aadhaar/ID numbers) is redacted while contractual qualification clauses (`shall`, `must not`, `unless`) are strictly preserved.
   * **Privacy Deletion**: Explain that the raw uploaded audio file on disk has already been unlinked and deleted via standard OS unlinking immediately upon verified transcript persistence, preventing biometric voice retention.

### Step 4: Meeting Intelligence & Decision Extraction (1 Minute)
1. Scroll down to inspect the structured tabs:
   * **Executive Summary & Topics**
   * **Decisions**: Show decisions with confidence scores.
   * **Action Items**: Show assignees, due dates, and status checkboxes.
2. Click **View in Transcript** on any decision:
   * Notice that the UI smoothly jumps to and highlights the exact spoken segment.
3. **Honest Diarization Note**:
   * Point out that speaker labels show as null/unassigned because an external speaker diarization model is unconfigured—MEMORA honestly displays unassigned speakers rather than fabricating fake speaker labels.

### Step 5: Cross-Meeting Intelligence & Decision Evolution (1 Minute)
1. Click the top navigation tab: **Cross-Meeting Intelligence**.
2. Show **Decision History**:
   * Show how an older decision (e.g. *"Use Memcached"*) is linked to a newer decision (*"Migrate to Redis for Distributed Caching"*).
   * Show the `SUPERSEDED` vs `ACTIVE` status badges.
3. Show **Commitment Tracking**:
   * Filter action items across multiple meetings, highlighting overdue deliverables.
4. Show **Potential Conflicts**:
   * Highlight the alert showing two meetings with contradictory decisions, complete with evidence links.

### Step 6: Ask AI & Evidence Grounding (1 Minute)
1. Switch to **Organizational Memory** -> **Ask AI**.
2. Ask: *"What caching solution did the team agree to deploy?"*
3. Highlight:
   * **Hybrid Search**: Query runs dense vector search (`multilingual-e5-small` on CPU) fused with SQLite FTS5 BM25 keyword search.
   * **Deterministic Provider Attribution**: Clarify that when a local generative LLM is not running, MEMORA uses an honest deterministic synthesis engine that extracts verified answers directly from citations without generative hallucination.
   * **Grounded Citations**: The answer displays the meeting title, date, and exact quote.
   * **Click to Verify**: Click the citation pill — the UI navigates directly to the cited timestamp in the privacy-safe transcript.
4. Test an unanswerable query: *"What is the company budget for 2030?"*
   * Point out the honest **Abstention Banner** (`INSUFFICIENT_EVIDENCE`): the system explicitly refuses to speculate rather than hallucinating an answer.

---

## 3. Anticipated Professor Questions & Honest Answers

### Q1: *"Did you train your own speech recognition model?"*
> **Answer**: *"No. Training a state-of-the-art ASR model from scratch requires thousands of hours of speech data. We use Faster-Whisper (`large-v3-turbo`) with CTranslate2 runtime optimized for NVIDIA Tensor Cores. Our engineering contribution is the production orchestration: long-audio chunking with overlap deduplication, dynamic Windows CUDA DLL loading, GPU VRAM bounds, and fallback to CPU."*

### Q2: *"Is the Ask AI using GPT-4 or OpenAI?"*
> **Answer**: *"No external cloud APIs are used for search or inference. For embeddings, we run HuggingFace's `multilingual-e5-small` locally on CPU. For answer synthesis, because high-parameter generative LLMs require significant dedicated VRAM alongside Whisper, we built an honest deterministic synthesis engine that extracts answers directly from verified citation chunks and abstains when evidence is missing. An Ollama adapter is also implemented if a local LLM is available."*

### Q3: *"How does SQLite handle multiple concurrent users?"*
> **Answer**: *"We configured SQLite in Write-Ahead Logging (WAL) mode with `PRAGMA synchronous=NORMAL` and a 30-second busy timeout. In WAL mode, read transactions never block write transactions, and writes never block reads. For local team and demo usage, SQLite WAL performs reliably. For large-scale enterprise deployments with dozens of concurrent background workers, the SQLAlchemy repository layer can be directed to PostgreSQL via the `DATABASE_URL` setting."*

### Q4: *"What prevents unauthorized users from seeing raw recordings?"*
> **Answer**: *"First, raw media files and intermediate WAVs are automatically deleted from disk immediately after transcription completes. Second, for the transcript text itself, MEMORA enforces role-based access control: regular members can only retrieve PII-redacted transcripts (HTTP 403 on original unredacted transcripts), and all admin retrievals of unredacted text generate immutable audit logs."*
