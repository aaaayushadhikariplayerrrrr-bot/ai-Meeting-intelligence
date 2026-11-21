# MEMORA: Multilingual Meeting Intelligence & Organizational Memory System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.13-3776AB.svg?style=flat&logo=python&logoColor=white" alt="Python 3.13" />
  <img src="https://img.shields.io/badge/FastAPI-0.163.0-009688.svg?style=flat&logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/React-19.3.1-61DAFB.svg?style=flat&logo=react&logoColor=black" alt="React 19" />
  <img src="https://img.shields.io/badge/TypeScript-7.0.2-3178C6.svg?style=flat&logo=typescript&logoColor=white" alt="TypeScript" />
  <img src="https://img.shields.io/badge/Vite-8.3.6-646CFF.svg?style=flat&logo=vite&logoColor=white" alt="Vite" />
  <img src="https://img.shields.io/badge/SQLite-WAL%20%2B%20FTS5-003B57.svg?style=flat&logo=sqlite&logoColor=white" alt="SQLite" />
  <img src="https://img.shields.io/badge/Faster--Whisper-1.2.1-FF6F00.svg?style=flat" alt="Faster-Whisper" />
  <img src="https://img.shields.io/badge/CTranslate2-4.9.0-4B32C3.svg?style=flat" alt="CTranslate2" />
  <img src="https://img.shields.io/badge/License-Not%20Specified-lightgrey.svg?style=flat" alt="License" />
</p>

A privacy-first, locally-deployable meeting intelligence and cross-meeting organizational memory platform designed for English, Hindi, and Hinglish (code-switched) business conversations.

---

## Table of Contents

- [1. Problem Statement & Technical Novelty](#1-problem-statement--technical-novelty)
- [2. Key Features](#2-key-features)
  - [2.1 Implemented Capabilities](#21-implemented-capabilities)
  - [2.2 Planned Capabilities](#22-planned-capabilities)
- [3. Demo and Screenshots](#3-demo-and-screenshots)
- [4. System Architecture](#4-system-architecture)
  - [4.1 Architecture Diagram](#41-architecture-diagram)
  - [4.2 End-to-End Data Flow](#42-end-to-end-data-flow)
- [5. Technology Stack](#5-technology-stack)
- [6. AI and Machine Learning Implementation](#6-ai-and-machine-learning-implementation)
  - [6.1 Speech Recognition (ASR)](#61-speech-recognition-asr)
  - [6.2 Multilingual & Code-Switching Handling](#62-multilingual--code-switching-handling)
  - [6.3 Transcript Cleaning & Readability](#63-transcript-cleaning--readability)
  - [6.4 PII & Privacy Redaction Engine](#64-pii--privacy-redaction-engine)
  - [6.5 Structured Meeting Intelligence](#65-structured-meeting-intelligence)
  - [6.6 Embeddings & Hybrid Organizational Memory (RAG)](#66-embeddings--hybrid-organizational-memory-rag)
  - [6.7 Hardware Management & CPU/GPU Fallbacks](#67-hardware-management--cpugpu-fallbacks)
- [7. Project Folder Structure](#7-project-folder-structure)
- [8. Hardware Requirements](#8-hardware-requirements)
  - [8.1 CPU-Only Deployment (Default)](#81-cpu-only-deployment-default)
  - [8.2 Optional NVIDIA GPU Acceleration](#82-optional-nvidia-gpu-acceleration)
- [9. Installation and Setup](#9-installation-and-setup)
  - [9.1 Prerequisites](#91-prerequisites)
  - [9.2 Windows (PowerShell) Setup](#92-windows-powershell-setup)
  - [9.3 Linux (Bash) Setup](#93-linux-bash-setup)
  - [9.4 Environment Configuration](#94-environment-configuration)
  - [9.5 Database Setup](#95-database-setup)
- [10. Running the Application](#10-running-the-application)
  - [10.1 Starting the Backend](#101-starting-the-backend)
  - [10.2 Starting the Frontend](#102-starting-the-frontend)
  - [10.3 Step-by-Step Usage Guide](#103-step-by-step-usage-guide)
- [11. API Documentation](#11-api-documentation)
- [12. Testing](#12-testing)
  - [12.1 Backend Tests](#121-backend-tests)
  - [12.2 Frontend Tests & Build](#122-frontend-tests--build)
  - [12.3 Database Schema Drift Check](#123-database-schema-drift-check)
- [13. Privacy, Security & Data Lifecycle](#13-privacy-security--data-lifecycle)
- [14. Current Limitations vs. Future Improvements](#14-current-limitations-vs-future-improvements)
- [15. Project Roadmap](#15-project-roadmap)
- [16. Contributing](#16-contributing)
- [17. License](#17-license)
- [18. Acknowledgements](#18-acknowledgements)

---

## 1. Problem Statement & Technical Novelty

Traditional meeting transcription solutions function as isolated, single-meeting note-takers: they convert audio into a flat wall of text, output a generic summary, and discard all context once the call concludes. When organizations hold dozens of recurring standups, sprint reviews, and planning sessions, critical business context is fragmented across separate documents.

**MEMORA** addresses four systemic challenges in modern meeting intelligence:

1. **Cross-Meeting Decision Lineage**: Decisions and commitments do not exist in a vacuum. MEMORA establishes chronological lineage trees (`ACTIVE`, `SUPERSEDED`, `REVOKED`) across past meetings, alerting teams to contradictory mandates and tracking commitment reviews over time.
2. **Privacy-by-Design & Automated Media Deletion**: Raw audio and video files contain sensitive biometric voiceprints and facial data. Under strict retention policies (`AUTO_DELETE_RAW_POST_TRANSCRIPTION=True`), MEMORA automatically unlinks physical media files from disk immediately after transcription and segment persistence are committed to the database, leaving only timestamped text segments, PII-redacted derived versions, and vector embeddings.
3. **Multilingual Code-Switching (Hinglish)**: Real-world business discussions across India and multinational teams continuously mix English and Hindi within individual sentences. MEMORA natively accommodates code-switched Hindi-English speech without forcing translation, preserving both spoken languages in transcript segments.
4. **Verifiable Evidence Navigation**: Instead of ungrounded generative summaries prone to hallucination, every decision, action item, conflict, and Ask AI query in MEMORA is linked to exact transcript segments with millisecond timestamps. Users can click any citation to navigate directly to the verified spoken evidence.

---

## 2. Key Features

### 2.1 Implemented Capabilities

| Feature Category | Implemented & Verified Capabilities |
| :--- | :--- |
| **Media Ingestion** | • Audio upload: `.wav`, `.mp3`, `.m4a`, `.flac`<br>• Video upload: `.mp4`, `.mkv`, `.mov`, `.webm`<br>• Authorized YouTube video ingestion via `yt-dlp`<br>• Background codec inspection via `ffprobe`<br>• Automatic normalization to 16 kHz mono WAV via FFmpeg<br>• Enforced upload limits: 500 MB max size, 4.0 hours max duration |
| **Speech Recognition** | • Faster-Whisper (`1.2.1`) powered by CTranslate2 (`4.9.0`)<br>• Default model: `large-v3-turbo`<br>• Automatic device selection (NVIDIA CUDA FP16 with automatic fallback to CPU INT8)<br>• Overlapping chunking (300s window, 2s overlap) for long audio processing<br>• Asynchronous job worker with atomic compare-and-swap database task claiming |
| **Multilingual Support** | • ISO language recognition (`en`, `hi`, `gu`, `auto`)<br>• Code-switched Hindi-English (`hinglish`) mode with multilingual prompt conditioning<br>• Preservation of original spoken words without forced translation |
| **Transcript Cleaning & PII Redaction** | • Stutter and repetition reduction while preserving meaningful cultural/rhetorical reduplications (e.g. Hindi *धीरे धीरे*, *बार बार*, English *very very*)<br>• PII masking: Indian Aadhaar (Verhoeff checksum validated), PAN cards, credit cards (Luhn validated), emails, phone numbers, Indian PIN codes, JWTs, Bearer tokens, and API keys<br>• Preservation of statutory/legal conditioning words (`shall`, `must not`, `unless`, `subject to`)<br>• Four derived transcript states: `ORIGINAL`, `CLEANED`, `REDACTED`, `CLEANED_REDACTED` |
| **Access Control (RBAC)** | • Multi-tenant organization isolation on every database query<br>• `ORG_ADMIN`: Full media management, access to original transcripts (audited), diagnostic monitoring<br>• `MEMBER`: Access strictly constrained to privacy-safe (`REDACTED`, `CLEANED_REDACTED`) transcripts and organizational memory |
| **Cross-Meeting Intelligence** | • **Decision History**: Explicit superseding and revocation with lineage tracking<br>• **Commitment Tracking**: Cross-meeting action items with overdue calculations and owner filtering<br>• **Conflict Detection**: Heuristic contradiction identification across active decisions<br>• **Unresolved Questions**: Tracking unanswered items across meetings until marked resolved |
| **Organizational Memory (RAG)** | • Semantic chunking (200–300 tokens, 30-token overlap)<br>• 384-dimensional dense embeddings via `intfloat/multilingual-e5-small`<br>• SQLite FTS5 BM25 keyword search combined with vector cosine similarity via Reciprocal Rank Fusion (RRF, `k=60`)<br>• Nine-stage RAG query engine with database citation verification<br>• Honest abstention handling (`NO_EVIDENCE`, `INSUFFICIENT_EVIDENCE`, `MODEL_UNAVAILABLE`) |
| **Security & Privacy** | • Argon2id password hashing (100 MB memory cost, 2 iterations, 8 threads)<br>• HS256 JWT authentication with per-user `token_version` instant logout revocation<br>• Process-local sliding-window rate limiters on auth and memory endpoints<br>• Automatic post-transcription file unlinking with immutable audit log recording |

### 2.2 Planned Capabilities

* **Speaker Diarization**: Multi-speaker clustering via PyAnnote Audio (currently defaults honestly to `NullDiarizationProvider` with null speaker labels).
* **Distributed Rate Limiting**: Centralized Redis storage for horizontal multi-instance deployments.
* **PostgreSQL Storage Backend**: Horizontal multi-writer database adapter for high-concurrency enterprise clusters.
* **Real-Time Streaming Transcription**: WebSocket-based live microphone audio ingestion.

---

## 3. Demo and Screenshots

Below are placeholders for the primary user interfaces in MEMORA. Replace the files in `docs/screenshots/` with actual captures from your deployment:

| View | Screenshot Placeholder |
| :--- | :--- |
| **Meeting Dashboard & Upload** | ![Meeting Dashboard](docs/screenshots/dashboard.png)<br>*Meeting ingestion queue, upload progress, and processing status.* |
| **Privacy-Safe Transcript Viewer** | ![Transcript View](docs/screenshots/transcript_view.png)<br>*Timestamped segments with active PII redaction and dual-layer privacy controls.* |
| **Structured Intelligence** | ![Meeting Intelligence](docs/screenshots/meeting_intelligence.png)<br>*Synthesized executive summary, key topics, decisions, action items, and risks.* |
| **Cross-Meeting Decision Lineage** | ![Decision Lineage](docs/screenshots/cross_meeting_decisions.png)<br>*Timeline of active, superseded, and revoked decisions with potential conflict warnings.* |
| **Ask AI & Grounded Citations** | ![Ask AI RAG](docs/screenshots/ask_ai_rag.png)<br>*Hybrid question answering with verified, clickable transcript citations.* |

---

## 4. System Architecture

### 4.1 Architecture Diagram

```mermaid
flowchart TD
    subgraph Frontend["Frontend (React 19 + TypeScript + Vite)"]
        UI["SPA Interface (Port 5173)"]
        AuthContext["Auth Context (JWT + X-Organization-Id)"]
        Views["Dashboard | Transcripts | Cross-Meeting | Ask AI"]
    end

    subgraph Backend["Backend Server (FastAPI + Uvicorn)"]
        Router["API Router (/api/v1)"]
        AuthModule["Authentication & RBAC (Argon2id + JWT)"]
        RateLimiter["Sliding-Window Rate Limiter"]
        TenantDeps["TenantContext Dependency Injection"]
    end

    subgraph AsyncPipeline["Asynchronous Background Workers"]
        JobWorker["JobWorker (Atomic SQLite CAS Claiming)"]
        FFmpegProc["FFmpeg 16 kHz Mono Normalizer"]
        WhisperASR["Faster-Whisper Engine (large-v3-turbo)"]
        Redactor["Privacy Redactor & Stutter Cleaner"]
        AutoDelete["Media Auto-Delete & Audit Logger"]
        Scheduler["RetentionScheduler (Daily Media Purge)"]
    end

    subgraph MemoryEngine["Organizational Memory & RAG"]
        Chunker["Semantic Chunker (200-300 Tokens)"]
        EmbeddingSvc["Multilingual-E5-Small (384-dim CPU)"]
        RAGEngine["9-Stage RAG Engine"]
        HybridSearch["RRF Hybrid Retrieval (FTS5 BM25 + Vector Cosine)"]
        CitationVerifier["Database Citation Verifier"]
        Providers["Deterministic | Ollama | Gemini"]
    end

    subgraph Storage["Persistent Data Layer (SQLite 3 WAL)"]
        RelationalDB[("Relational Tables (Users, Meetings, Transcripts, Decisions)")]
        FTS5Index[("FTS5 Virtual Index (transcript_chunks_fts)")]
        VectorStore[("Raw Float32 Vector BLOBs (transcript_chunks)")]
        DiskStorage[("Local File Storage (data/storage/)")]
    end

    UI -->|HTTP / REST (Bearer Token)| Router
    Router --> AuthModule
    Router --> RateLimiter
    Router --> TenantDeps
    TenantDeps --> RelationalDB

    Router -->|Queue Ingestion Job| RelationalDB
    JobWorker -->|Poll & Claim PENDING Jobs| RelationalDB
    JobWorker --> FFmpegProc
    FFmpegProc --> DiskStorage
    JobWorker --> WhisperASR
    WhisperASR --> Redactor
    Redactor --> RelationalDB
    Redactor --> AutoDelete
    AutoDelete -->|Unlink Physical Files| DiskStorage
    AutoDelete -->|Log Event| RelationalDB

    Redactor --> Chunker
    Chunker --> EmbeddingSvc
    EmbeddingSvc --> VectorStore
    Chunker --> FTS5Index

    Router -->|Ask AI Query| RAGEngine
    RAGEngine --> HybridSearch
    HybridSearch --> FTS5Index
    HybridSearch --> VectorStore
    RAGEngine --> Providers
    Providers --> CitationVerifier
    CitationVerifier --> RelationalDB
```

### 4.2 End-to-End Data Flow

1. **Ingestion**: The user uploads an audio/video file or inputs an authorized YouTube URL via the React UI. The file is validated (duration $\le$ 4 hours, size $\le$ 500 MB) and streamed to `data/storage/<meeting_id>/`.
2. **Audio Extraction**: In the background, FFmpeg extracts and normalizes the audio stream to a 16 kHz, single-channel, 16-bit PCM WAV file.
3. **ASR Transcription**: The `JobWorker` claims the job atomically using compare-and-swap SQL. Faster-Whisper transcribes the audio, applying VAD filtering and overlapping chunk deduplication. The unredacted output is stored as `Transcript (ORIGINAL)`.
4. **Cleaning & PII Redaction**: The transcript is cleaned of stutters/fillers, and private entities (Aadhaar, PAN, emails, phones, API keys) are masked. A derived, privacy-safe `Transcript (CLEANED_REDACTED)` is committed.
5. **Post-Transcription Deletion**: Immediately upon verified database persistence, the physical raw upload and intermediate WAVs are unlinked from disk, writing an unalterable `AuditLog` entry.
6. **Memory Indexing**: The privacy-safe transcript is broken into semantic chunks (200–300 tokens) with 30-token overlap. `multilingual-e5-small` generates 384-dimensional dense vectors stored as raw binary float32 BLOBs in SQLite, alongside an FTS5 full-text index record.
7. **Cross-Meeting Retrieval & RAG**: When an organization member queries "Ask AI", hybrid retrieval combines FTS5 BM25 keyword matching with dense vector cosine similarity via Reciprocal Rank Fusion. An answer is synthesized and grounded strictly in verified database citations.

---

## 5. Technology Stack

### Backend Technologies

| Technology | Verified Version | Purpose |
| :--- | :--- | :--- |
| **Python** | `3.13.x` (or `3.11+`) | Core backend runtime environment |
| **FastAPI** | `0.163.0` | High-performance asynchronous REST API framework |
| **Uvicorn** | `0.54.0` | ASGI production web server |
| **Pydantic** | `2.14.3` | Request/response data validation and settings management |
| **SQLAlchemy** | `2.1.8` | Declarative ORM with SQLite foreign-key support |
| **Alembic** | `1.20.1` | Database schema migrations and drift checking |
| **Argon2-cffi** | `25.1.0` | Secure password hashing (Argon2id algorithm) |
| **PyJWT** | `2.15.1` | JSON Web Token encoding, signing, and verification |
| **HTTPX** | `0.28.1` | Asynchronous HTTP client for external integrations |
| **yt-dlp** | `2026.10.26` | Ingestion of authorized public YouTube meeting recordings |
| **PyAV** | `19.0.2` | Python bindings for FFmpeg media decoding |

### AI / Machine Learning Frameworks

| Library / Model | Verified Version | Purpose |
| :--- | :--- | :--- |
| **Faster-Whisper** | `1.2.1` | High-performance speech recognition wrapper |
| **CTranslate2** | `4.9.0` | Optimized inference engine for Transformer models |
| **Whisper Model** | `large-v3-turbo` | Multilingual ASR model checkpoint |
| **Multilingual-E5** | `intfloat/multilingual-e5-small` | 384-dimensional dense multilingual sentence embeddings |
| **PyTorch** | User-selected via selector | Tensor computations for Multilingual-E5 embedding inference |
| **Transformers** | Transitive dependency | HuggingFace tokenizer and model loading for E5 |
| **ONNX Runtime** | `1.32.0` | High-performance runtime for auxiliary model evaluation |

### Database & Storage

* **SQLite 3**: Embedded database with Write-Ahead Logging (`PRAGMA journal_mode=WAL`), foreign key enforcement (`PRAGMA foreign_keys=ON`), and 30-second busy timeout (`PRAGMA busy_timeout=30000`).
* **SQLite FTS5**: Native full-text search virtual tables for BM25 keyword retrieval.
* **Vector Storage**: Exact NumPy float32 binary BLOB serialization (1,536 bytes per 384-dimensional vector).

### Frontend Technologies

| Technology | Verified Version | Purpose |
| :--- | :--- | :--- |
| **React** | `19.3.1` | Modern component-based user interface |
| **TypeScript** | `~7.0.2` | Static type safety across all frontend components |
| **Vite** | `^8.3.6` | Next-generation frontend build tool and dev server |
| **Lucide React** | `^1.73.0` | Lightweight, consistent UI icons |
| **Oxlint** | `^1.90.0` | High-performance JavaScript/TypeScript linter |

---

## 6. AI and Machine Learning Implementation

### 6.1 Speech Recognition (ASR)

Speech recognition is implemented in `backend/app/services/transcription/faster_whisper_engine.py` via `FasterWhisperEngine`:

* **Inference Engine**: Uses CTranslate2's optimized integer and float quantization routines.
* **Model Checkpoint**: Configured to `large-v3-turbo` by default (`settings.ASR_MODEL_NAME`).
* **Audio Chunking**: Recordings longer than 300 seconds are segmented into overlapping windows (300s window, 2s overlap). Segments across boundaries are merged and deduplicated using timestamp alignment (`merge_and_deduplicate_segments`).
* **VAD Filtering**: Built-in Voice Activity Detection suppresses background noise and silent intervals.

### 6.2 Multilingual & Code-Switching Handling

* **Language Support**: Supports English (`en`), Hindi (`hi`), Gujarati (`gu`), and automatic detection (`auto`).
* **Hinglish Mode**: Code-switched Hindi-English speech is treated as a specialized transcription mode rather than a forced ISO language code (passing `hi-en` directly causes runtime errors in Whisper). Setting `whisper_language=None` with an initial prompt (`"Hindi and English code-switching conversation, Hinglish."`) allows Whisper's multilingual decoder to transcribe both languages without translation.

### 6.3 Transcript Cleaning & Readability

Implemented in `backend/app/services/transcript_cleaning/cleaner.py`:

* Eliminates stuttered syllables and accidental repetitions (e.g. *"th- the"*, *"I I went"*).
* **Cultural Lexical Preservation**: Explicitly protects meaningful reduplications from being collapsed. Words such as Hindi *धीरे धीरे* (slowly), *बार बार* (repeatedly), *साथ साथ* (together), and English *very very* are preserved.
* Removes filler vocalizations (*"um"*, *"uh"*, *"you know"*, *"matlab"*, *"yani"*).

### 6.4 PII & Privacy Redaction Engine

Implemented in `backend/app/services/privacy_redaction/redactor.py`:

* **Indian Identity Documents**:
  * **Aadhaar Numbers**: Validates 12-digit numbers using the **Verhoeff checksum algorithm**, preventing ordinary 12-digit technical metrics or product codes from false redaction. Supports Devanagari (०-९) and Gujarati (૦-૯) numerals.
  * **PAN Cards**: Matches standard Income Tax alphanumeric formats (`[A-Z]{5}[0-9]{4}[A-Z]`).
* **Financial & Credentials**:
  * Credit/Debit cards validated via the **Luhn algorithm**.
  * API keys, JWT access tokens, Bearer authorization headers, and hexadecimal password hashes.
* **Statutory Conditioning Preservation**: Explicitly preserves legal qualification terms (`shall`, `must`, `must not`, `unless`, `subject to`, `provided that`) to maintain contractual integrity in recorded minutes.

### 6.5 Structured Meeting Intelligence

Structured intelligence is coordinated through three interchangeable providers (`backend/app/services/intelligence/`):

1. **Deterministic Extractor** (`deterministic.py`): Zero-dependency, offline heuristic rule-based engine. Uses multi-lingual regex patterns to identify decisions, action items, risks, and questions in English, Hindi, and Hinglish without consuming GPU memory.
2. **Ollama Local LLM** (`ollama_adapter.py`): Connects to a local Ollama daemon (`http://127.0.0.1:11434`) running `llama3.2:3b` or custom models using strict JSON schema validation.
3. **Google Gemini REST API** (`gemini_adapter.py`): Optional cloud provider connecting to Gemini Flash (`gemini-3.8-flash` or fallbacks) when `GEMINI_ENABLED=True` and a valid `GEMINI_API_KEY` is provided.

### 6.6 Embeddings & Hybrid Organizational Memory (RAG)

* **Model**: `intfloat/multilingual-e5-small` loaded via HuggingFace Transformers.
* **Vector Specifications**: 384 dimensions, normalized to unit L2 length ($\|v\|_2 = 1.0$).
* **Task Conditioning**: Enforces strict E5 prefixes: `"passage: "` for indexable transcript segments, and `"query: "` for search queries.
* **Hybrid Retrieval**: Combines SQLite FTS5 BM25 keyword ranking with vector cosine similarity using Reciprocal Rank Fusion:
  $$\text{RRF Score}(d) = \frac{1}{60 + \text{Rank}_{\text{FTS5}}(d)} + \frac{1}{60 + \text{Rank}_{\text{Vector}}(d)}$$
* **Citation Verification**: The 9-stage `RAGEngine` validates that cited segment IDs belong to the tenant organization before formulating the response. If insufficient evidence exists, the engine abstains with `NO_EVIDENCE` or `INSUFFICIENT_EVIDENCE`.

### 6.7 Hardware Management & CPU/GPU Fallbacks

Implemented in `backend/app/services/transcription/gpu_manager.py`:

* **Thread Bounding**: Restricts OpenMP, OpenBLAS, MKL, and PyTorch threads (`CPU_MAX_THREADS=4`, `TORCH_NUM_THREADS=4`, `TORCH_NUM_INTEROP_THREADS=2`) strictly once at application startup, preventing CPU oversubscription during concurrent requests.
* **Windows DLL Discovery**: Automatically detects CUDA DLLs (`cublas64_12.dll`, `cudnn64_9.dll`) inside the active PyTorch directory (`torch/lib`) and appends them to the Windows DLL search path via `os.add_dll_directory`.
* **VRAM Headroom Guard**: `can_use_gpu_for_embedding` checks if Faster-Whisper is actively using the GPU. On 8 GB VRAM GPUs, embedding inference remains on CPU by default to prevent CUDA out-of-memory errors during simultaneous transcription.

---

## 7. Project Folder Structure

```
Multilingual Meeting Intelligence System/
│
├── backend/                              # FastAPI Backend Application
│   ├── alembic/                          # Alembic database schema migrations
│   │   ├── versions/                     # Versioned migration scripts
│   │   └── env.py                        # Alembic migration environment
│   ├── app/
│   │   ├── api/                          # REST API routers & dependency injection
│   │   │   ├── endpoints/                # auth, media, transcription, intelligence, memory, cross_meeting
│   │   │   ├── deps.py                   # TenantContext & security dependencies
│   │   │   └── router.py                 # Master API router (/api/v1)
│   │   ├── core/                         # Configuration & security settings
│   │   │   ├── config.py                 # Pydantic BaseSettings (database, ASR, security)
│   │   │   ├── security.py               # Argon2id & JWT token generation
│   │   │   └── rate_limit.py             # Process-local sliding window rate limiters
│   │   ├── database/                     # Database connection & session factory
│   │   │   └── session.py                # SQLite WAL initialization & engine binding
│   │   ├── models/                       # SQLAlchemy declarative ORM models
│   │   │   ├── identity.py               # User, Organization, OrganizationMembership
│   │   │   ├── meeting.py                # Meeting, MediaAsset, ProcessingJob
│   │   │   ├── transcription.py          # Transcript, TranscriptSegment
│   │   │   ├── intelligence.py           # Summary, Topic, Decision, ActionItem, Risk
│   │   │   └── memory.py                 # TranscriptChunkRecord, AuditLog
│   │   ├── schemas/                      # Pydantic request & response schemas
│   │   ├── services/                     # Core business logic & AI pipelines
│   │   │   ├── transcription/            # FasterWhisperEngine & GPU manager
│   │   │   ├── transcript_cleaning/      # Stutter and filler word reduction
│   │   │   ├── privacy_redaction/        # PII, Aadhaar, PAN, and credentials masking
│   │   │   ├── transcript_processing/    # Derived transcript version orchestration
│   │   │   ├── intelligence/             # Deterministic, Ollama, & Gemini extractors
│   │   │   ├── memory/                   # E5 embeddings, SQLite vector storage, FTS5 RAG
│   │   │   ├── job_worker.py             # Asynchronous task processing loop
│   │   │   ├── lifecycle.py              # Media unlinking & retention enforcement
│   │   │   └── media_validation.py       # Codec inspection & file duration validation
│   │   └── main.py                       # FastAPI entry point, lifespan, & CORS setup
│   ├── tests/                            # Comprehensive backend test suites
│   ├── .env.example                      # Backend environment variable template
│   ├── alembic.ini                       # Alembic migration configuration
│   └── requirements.txt                  # Core pinned Python dependencies
│
├── frontend/                             # React 19 Frontend Application
│   ├── public/                           # Static assets
│   ├── src/
│   │   ├── components/                   # UI views & layout components
│   │   │   ├── cross_meeting/            # DecisionHistory, Commitments, Conflicts, Questions
│   │   │   ├── AuthScreen.tsx            # Login & registration screens
│   │   │   ├── LandingPage.tsx           # Public overview & landing page
│   │   │   ├── MeetingIntelligenceView.tsx # Transcripts & single-meeting intelligence
│   │   │   ├── OrganizationalMemoryView.tsx # Ask AI & Hybrid Memory Search
│   │   │   └── OrgSelectionScreen.tsx    # Multi-tenant organization switcher
│   │   ├── context/                      # AuthContext & tenant state management
│   │   ├── services/                     # API client services
│   │   ├── tests/                        # Frontend component & security test suites
│   │   ├── types/                        # TypeScript domain interfaces
│   │   ├── App.tsx                       # Master view router & navigation bar
│   │   └── main.tsx                      # React root entry point
│   ├── .env.example                      # Frontend environment variable template
│   ├── package.json                      # Frontend dependencies & npm scripts
│   ├── tsconfig.json                     # TypeScript compiler configuration
│   └── vite.config.ts                    # Vite dev server & backend API proxy setup
│
├── data/                                 # Runtime application storage (excluded from Git)
│   └── storage/                          # Tenant media uploads & temporary WAVs
├── docs/                                 # Architectural documentation & assets
│   ├── screenshots/                      # UI preview images
│   └── architecture.md                   # Detailed technical design specifications
├── ARCHITECTURE.md                       # High-level architecture summary
├── SECURITY.md                           # Privacy, tenant isolation, & retention policies
└── README.md                             # Project documentation
```

---

## 8. Hardware Requirements

### 8.1 CPU-Only Deployment (Default)

MEMORA is fully functional on systems without dedicated GPUs:

* **Transcription**: Runs Faster-Whisper on CPU using 8-bit integer quantization (`compute_type="int8"`). Processing time will be roughly 1.5× to 3× audio duration depending on CPU core count.
* **Embeddings**: `multilingual-e5-small` runs natively on CPU using PyTorch CPU wheels with bounded threads (`CPU_MAX_THREADS=4`).
* **Intelligence**: The default `deterministic` rule-based extractor executes instantly with zero GPU VRAM consumption.
* **Minimum Specifications**:
  * x86_64 CPU (4 physical cores, 8 threads minimum).
  * 16 GB system RAM.
  * 10 GB free disk space.

### 8.2 Optional NVIDIA GPU Acceleration

For accelerated performance during transcription:

* **Supported GPUs**: NVIDIA RTX series or modern data center GPUs (compute capability $\ge$ 8.0 recommended).
* **VRAM Requirements**: 6 GB VRAM minimum; 8 GB recommended for the `large-v3-turbo` model.
* **Software Requirements**:
  * NVIDIA Driver supporting CUDA 12.x.
  * *Note on `nvidia-smi`*: The CUDA version displayed in `nvidia-smi` represents the maximum CUDA version supported by your installed GPU driver. You do not need to install a CUDA Toolkit that exactly matches this number.
  * PyTorch wheels bundle their own CUDA runtime.

---

## 9. Installation and Setup

### 9.1 Prerequisites

1. **Python**: Verified on Python `3.13.x` (64-bit). Compatible with Python `3.11`–`3.13`.
2. **Node.js**: Node.js `v20+` and `npm 10+`.
3. **FFmpeg & FFprobe**: Must be installed and accessible in your system's `PATH`. Verify by running:
   ```bash
   ffmpeg -version
   ffprobe -version
   ```

### 9.2 Windows (PowerShell) Setup

Open PowerShell as an Administrator or regular user:

```powershell
# 1. Clone the repository
git clone https://github.com/<your-username>/memora.git
cd "memora"

# 2. Setup Python virtual environment in backend
cd backend
python -m venv venv
.\venv\Scripts\Activate.ps1

# 3. Install core dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Install PyTorch
# Visit https://pytorch.org/get-started/locally/ to select your exact wheel.
# Example for CPU-only:
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 5. Verify CTranslate2 installation
# backend/requirements.txt pins ctranslate2==4.9.0.
# If using NVIDIA GPU acceleration, ensure your NVIDIA drivers are up to date.
```

### 9.3 Linux (Bash) Setup

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/memora.git
cd memora

# 2. Setup virtual environment
cd backend
python3 -m venv venv
source venv/bin/activate

# 3. Install core dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Install PyTorch via official selector
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 9.4 Environment Configuration

#### Backend Configuration

Copy `backend/.env.example` to `backend/.env`:

```powershell
# On Windows PowerShell
cp .env.example .env

# On Linux
cp .env.example .env
```

Generate a cryptographically secure 256-bit secret key for JWT signing:

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

Open `backend/.env` and update the following settings:

```dotenv
# Application Settings
PROJECT_NAME="Multilingual Meeting Intelligence System"
ENVIRONMENT="development"
DEBUG=True
API_V1_STR="/api/v1"

# Database Path
DATABASE_URL="sqlite:///./meeting_intelligence.db"

# Security (Paste the 32+ character hex string generated above)
SECRET_KEY="replace-with-your-generated-32-char-secret-key"
JWT_ALGORITHM="HS256"
ACCESS_TOKEN_EXPIRE_MINUTES=60

# CORS Allowed Origins
ALLOWED_ORIGINS="http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000"

# ASR & Speech Recognition
ASR_ENGINE="faster-whisper"
ASR_MODEL_NAME="large-v3-turbo"
ASR_DEVICE="cuda"           # Change to "cpu" if no NVIDIA GPU is available
ASR_COMPUTE_TYPE="float16"  # Change to "int8" if running on CPU

# Structured Intelligence Provider ("deterministic", "ollama", or "disabled")
INTELLIGENCE_PROVIDER="deterministic"

# Privacy & Media Retention
AUTO_DELETE_RAW_POST_TRANSCRIPTION=True
AUTO_DELETE_NORMALIZED_POST_TRANSCRIPTION=True
ENABLE_RETENTION_SCHEDULER=True
RETENTION_DAYS=30
```

#### Frontend Configuration

Navigate to `frontend/` and configure its environment:

```powershell
cd ../frontend
cp .env.example .env
npm install
```

`frontend/.env` should contain:

```dotenv
VITE_API_URL=http://127.0.0.1:8000
```

### 9.5 Database Setup

Apply database schema migrations using Alembic from the `backend/` directory:

```powershell
cd backend
.\venv\Scripts\Activate.ps1
alembic upgrade head
```

---

## 10. Running the Application

### 10.1 Starting the Backend

From the `backend/` directory with the virtual environment activated:

```powershell
# Windows PowerShell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

The backend server initializes SQLite WAL mode, bounds CPU threads, reconciles any stranded jobs from prior sessions, and starts the background `JobWorker` and `RetentionScheduler`.

* **Backend Health Check**: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)
* **Interactive OpenAPI Swagger Docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
* **Alternative ReDoc Documentation**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

### 10.2 Starting the Frontend

In a separate terminal, navigate to the `frontend/` directory:

```powershell
cd frontend
npm run dev
```

The Vite dev server starts on port `5173`. Open your browser to:

* **Local Web Application**: [http://localhost:5173](http://localhost:5173)

### 10.3 Step-by-Step Usage Guide

1. **User Registration & Tenant Creation**:
   * Navigate to [http://localhost:5173](http://localhost:5173).
   * Click **Register**. Enter your name, email, password, and organization name (e.g. *Acme Engineering*).
   * The system creates your tenant organization and grants your account `ORG_ADMIN` privileges.
2. **Ingest a Meeting**:
   * Click **Upload Meeting** on the navigation bar.
   * Drag and drop an audio or video file (or provide an authorized YouTube link).
   * The background pipeline validates duration, extracts a 16 kHz mono WAV, and queues the transcription job.
3. **Monitor Ingestion & Transcription**:
   * The meeting dashboard displays live job status and progress.
   * Faster-Whisper transcribes speech into timestamped segments.
   * Stutter cleaning and PII redaction derive a privacy-safe `CLEANED_REDACTED` transcript.
   * Raw audio files are automatically deleted from disk if `AUTO_DELETE_RAW_POST_TRANSCRIPTION=True`.
4. **Inspect Single-Meeting Intelligence**:
   * View the transcript viewer to inspect timestamped utterances with redaction badges.
   * Access the **Intelligence** panel to review synthesized summaries, topics, decisions, action items, and identified risks.
5. **Manage Cross-Meeting Lineage**:
   * Switch to **Cross-Meeting Intelligence** in the top navigation.
   * Review **Decision History** to inspect decisions across multiple calls, explicitly superseding older mandates or revoking obsolete ones.
   * Track open commitments, view overdue warnings, and filter action items by owner.
   * Inspect **Potential Conflicts** to resolve contradictory decisions across departments.
6. **Query Organizational Memory (Ask AI)**:
   * Navigate to **Organizational Memory** $\rightarrow$ **Ask AI**.
   * Ask natural-language questions (e.g. *"What did we decide regarding database indexing strategies?"*).
   * Review grounded answers with clickable citations that jump directly to the verified transcript segment.

---

## 11. API Documentation

All protected endpoints require an `Authorization: Bearer <token>` header. Multi-tenant endpoints automatically scope queries to the user's active tenant (`X-Organization-Id` header).

### Authentication Endpoints (`/api/v1/auth`)

| Method | Endpoint | Description | Access |
| :--- | :--- | :--- | :--- |
| `POST` | `/api/v1/auth/register` | Register new user, create tenant organization, and assign `ORG_ADMIN`. | Public (Rate Limited) |
| `POST` | `/api/v1/auth/login` | Authenticate user credentials and issue minimal JWT bearer token. | Public (Rate Limited) |
| `GET` | `/api/v1/auth/me` | Retrieve profile and active tenant memberships for current user. | Authenticated |
| `GET` | `/api/v1/auth/organizations` | List all organizations available to the authenticated user. | Authenticated |
| `POST` | `/api/v1/auth/logout` | Revoke all issued access tokens by incrementing user's `token_version`. | Authenticated |

### Media Ingestion Endpoints (`/api/v1/media`)

| Method | Endpoint | Description | Access |
| :--- | :--- | :--- | :--- |
| `POST` | `/api/v1/media/upload` | Upload audio/video file; queues extraction and validation jobs. | Authenticated |
| `POST` | `/api/v1/media/youtube` | Ingest meeting from authorized YouTube URL. | Authenticated |
| `GET` | `/api/v1/media/meetings` | List all meetings belonging to the authenticated organization. | Authenticated |
| `GET` | `/api/v1/media/meetings/{id}` | Retrieve metadata, media assets, and jobs for a specific meeting. | Authenticated |
| `GET` | `/api/v1/media/jobs/{id}` | Poll progress and status of an ingestion or processing job. | Authenticated |
| `DELETE` | `/api/v1/media/assets/{id}` | Manually delete raw media asset from disk and record audit log. | `ORG_ADMIN` |

### Transcription Endpoints (`/api/v1/transcription`)

| Method | Endpoint | Description | Access |
| :--- | :--- | :--- | :--- |
| `POST` | `/api/v1/transcription/start/{meeting_id}` | Trigger asynchronous speech recognition job for a meeting. | Authenticated |
| `GET` | `/api/v1/transcription/{meeting_id}/default` | Retrieve default privacy-safe transcript (`CLEANED_REDACTED` or `REDACTED`). | Authenticated |
| `GET` | `/api/v1/transcription/{meeting_id}` | List saved transcripts and segments (unredacted requires `ORG_ADMIN`). | Authenticated |
| `POST` | `/api/v1/transcription/{meeting_id}/clean` | Generate derived `CLEANED` transcript version (stutter removal). | `ORG_ADMIN` |
| `POST` | `/api/v1/transcription/{meeting_id}/redact` | Generate derived `REDACTED` transcript version (PII masking). | Authenticated |
| `POST` | `/api/v1/transcription/{meeting_id}/process` | Generate combined `CLEANED_REDACTED` transcript version. | Authenticated |
| `GET` | `/api/v1/transcription/jobs/{id}` | Check status and progress of an active transcription job. | Authenticated |
| `GET` | `/api/v1/transcription/hardware/diagnostics` | Inspect CUDA, CTranslate2, and GPU VRAM diagnostic readiness. | Super Admin Only |

### Single-Meeting Intelligence (`/api/v1`)

| Method | Endpoint | Description | Access |
| :--- | :--- | :--- | :--- |
| `POST` | `/api/v1/meetings/{id}/intelligence/extract` | Trigger asynchronous structured meeting intelligence extraction. | Authenticated |
| `GET` | `/api/v1/meetings/{id}/intelligence` | Retrieve executive summaries, topics, decisions, actions, and risks. | Authenticated |
| `PATCH` | `/api/v1/action-items/{id}/status` | Update lifecycle status of an action item (`PENDING`, `IN_PROGRESS`, etc.). | Authenticated |

### Cross-Meeting Intelligence (`/api/v1/intelligence` or `/api/v1`)

| Method | Endpoint | Description | Access |
| :--- | :--- | :--- | :--- |
| `GET` | `/api/v1/decisions` | List decisions across organization with status and date filters. | Authenticated |
| `GET` | `/api/v1/decisions/{id}` | Get decision details and complete chronological lineage tree. | Authenticated |
| `POST` | `/api/v1/decisions/{id}/supersede` | Explicitly supersede a prior decision with a new decision. | Authenticated |
| `POST` | `/api/v1/decisions/{id}/revoke` | Revoke an existing decision with documented rationale. | Authenticated |
| `GET` | `/api/v1/commitments` | List commitments/action items across all meetings. | Authenticated |
| `POST` | `/api/v1/commitments/{id}/status` | Review and update commitment status across meetings. | Authenticated |
| `GET` | `/api/v1/commitments/overdue` | List open commitments that have exceeded their target deadline. | Authenticated |
| `GET` | `/api/v1/commitments/by-person/{name}` | List open commitments assigned to a specific team member. | Authenticated |
| `GET` | `/api/v1/conflicts` | Detect potential contradictory stances across active decisions. | Authenticated |
| `POST` | `/api/v1/questions/{id}/reopen` | Reopen a previously resolved question back to `OPEN` status. | Authenticated |

### Organizational Memory & RAG (`/api/v1/memory`)

| Method | Endpoint | Description | Access |
| :--- | :--- | :--- | :--- |
| `POST` | `/api/v1/memory/query` | Ask natural-language questions grounded in verified meeting memory. | Authenticated (Rate Limited) |
| `POST` | `/api/v1/memory/search` | Execute hybrid keyword (FTS5) and semantic vector search across chunks. | Authenticated (Rate Limited) |
| `GET` | `/api/v1/memory/conflicts` | Inspect potential decision conflicts indexed in organizational memory. | Authenticated (Rate Limited) |

---

## 12. Testing

### 12.1 Backend Tests

The backend test suite is executed using `pytest`. Run tests from the `backend/` directory:

```powershell
cd backend
.\venv\Scripts\Activate.ps1

# Run all backend unit and integration tests
python -m pytest tests/ -v

# Run targeted test suites
python -m pytest tests/test_security.py -v
python -m pytest tests/test_transcription.py -v
python -m pytest tests/test_memory_rag_engine.py -v
python -m pytest tests/test_cross_meeting_apis.py -v
```

### 12.2 Frontend Tests & Build

Run frontend unit tests and type checks from the `frontend/` directory:

```powershell
cd frontend

# Run frontend tests
npm run test

# Run Oxlint static analysis
npm run lint

# Validate TypeScript compilation and production build
npm run build
```

### 12.3 Database Schema Drift Check

Verify that SQLAlchemy ORM models match Alembic migration revisions:

```powershell
cd backend
.\venv\Scripts\Activate.ps1
python -m alembic check
```

---

## 13. Privacy, Security & Data Lifecycle

* **Multi-Tenant Scoping**: All database queries strictly enforce `organization_id == tenant.organization.id`. Cross-tenant queries return HTTP 404 or HTTP 403 without leaking entity existence.
* **Session Invalidation**: Passwords are hashed using Argon2id. When a user logs out, their database record increments `token_version`, immediately invalidating all previously issued JWT tokens without requiring a stateful token blacklist.
* **Automated Audio Deletion**: When `AUTO_DELETE_RAW_POST_TRANSCRIPTION=True`, raw user uploads and intermediate 16 kHz WAVs are unlinked from disk immediately upon verified transcription persistence. Deletion events are recorded in an immutable `audit_logs` table.
* **Local Processing Invariant**: Speech recognition, PII redaction, embedding generation, vector search, and deterministic intelligence extraction run entirely on local workstation hardware. No transcript text or audio recordings are transmitted to third-party cloud services unless the Google Gemini provider is explicitly configured.

---

## 14. Current Limitations vs. Future Improvements

| Current Operational Limitation | Planned Architecture Improvement |
| :--- | :--- |
| **Speaker Diarization**: Does not currently cluster individual speaker voices; defaults to `null` speaker labels. | Integration of PyAnnote Audio for multi-speaker diarization and speaker enrollment. |
| **Process-Local Rate Limiting**: In-memory sliding-window limiters are bounded to individual OS processes. | Redis-backed distributed token bucket rate limiting for multi-instance deployments. |
| **Database Concurrency**: SQLite WAL mode supports concurrent readers but serializes write transactions (30s busy timeout). | PostgreSQL migration pathway for enterprise deployments requiring heavy concurrent writes. |
| **Rule-Based Redaction**: Catches structured identifiers (Aadhaar, PAN, emails, phones, tokens) via regex and checksum algorithms, but cannot identify arbitrary informal names. | Named Entity Recognition (NER) token classification model for contextual name masking. |
| **File Deletion Level**: Executes standard operating system unlinking (`Path.unlink()`), not multi-pass cryptographic disk shredding. | Optional DoD 5220.22-M compliant multi-pass secure file shredding module. |

---

## 15. Project Roadmap

```
[Completed]
├── Phase 1: Multi-format media ingestion (audio, video, YouTube) & FFmpeg normalization
├── Phase 2: Faster-Whisper ASR pipeline with CUDA acceleration & CPU fallback
├── Phase 3: Hinglish code-switching support without forced translation
├── Phase 4: Stutter cleaning & PII redaction (Aadhaar Verhoeff, PAN, cards, credentials)
├── Phase 5: Post-transcription raw media auto-deletion & audit logging
├── Phase 6: Multi-tenant RBAC (ORG_ADMIN vs. MEMBER) & Argon2id authentication
├── Phase 7: Structured single-meeting intelligence (Summaries, Decisions, Action Items)
├── Phase 8: Cross-meeting decision lineage, commitment continuity, & conflict alerts
├── Phase 9: Hybrid organizational memory (Multilingual-E5 + SQLite FTS5 RRF RAG)
└── Phase 10: React 19 + TypeScript + Vite interactive web interface

[In Progress / Planned]
├── PostgreSQL adapter for horizontal multi-writer database scaling
├── Redis integration for distributed cross-process rate limiting
├── PyAnnote Audio integration for automated speaker voice diarization
└── Real-time WebSocket streaming audio transcription
```

---

## 16. Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork the Repository**: Create your feature branch (`git checkout -b feature/my-feature`).
2. **Preserve Invariants**: Ensure all database queries enforce tenant isolation (`organization_id`), and ensure raw media handling complies with privacy auto-deletion rules.
3. **Run Tests**: Verify that all backend and frontend test suites pass:
   ```bash
   cd backend && python -m pytest tests/
   cd ../frontend && npm run test && npm run build
   ```
4. **Submit a Pull Request**: Provide a detailed description of your changes, referencing any relevant issues.

---

## 17. License

No license has been specified for this project. All rights are reserved by the original authors. You may not distribute, commercialize, or modify this software without explicit written permission from the copyright owners.

---

## 18. Acknowledgements

* [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper) – Fast Whisper speech recognition library powered by CTranslate2.
* [CTranslate2](https://github.com/OpenNMT/CTranslate2) – Fast inference engine for Transformer models.
* [HuggingFace Transformers](https://github.com/huggingface/transformers) – Provider of the `intfloat/multilingual-e5-small` multilingual sentence embedding model.
* [FastAPI](https://fastapi.tiangolo.com/) – Modern, high-performance web framework for building APIs with Python.
* [SQLite](https://www.sqlite.org/) – In-process relational database engine with WAL mode and FTS5 full-text indexing.
* [React](https://react.dev/) & [Vite](https://vite.dev/) – Modern frontend framework and build tooling.
* [Lucide Icons](https://lucide.dev/) – Open-source UI iconography.
