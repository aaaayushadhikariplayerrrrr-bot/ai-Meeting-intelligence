# MEMORA Architecture & Technical Design

See comprehensive architectural documentation in [docs/architecture.md](docs/architecture.md).

Summary of Architectural Pillars:
1. **Asynchronous Micro-Pipelines**: Ingestion, Normalization, Faster-Whisper ASR, PII Redaction, Memory Indexing, and Lifecycle Status Reconciliation.
2. **SQLite WAL & Concurrency**: WAL journal mode with 30s busy timeout, FTS5 virtual tables, and single-writer coordination.
3. **Hybrid RAG & Memory**: 384-dimensional dense vectors (`multilingual-e5-small`) fused with SQLite FTS5 BM25 keyword retrieval.
4. **Strict Tenant & Role Boundaries**: Multi-tenant isolation at every database query, role-based redaction, and audited unredacted transcript access.
5. **Privacy Auto-Delete Lifecycle**: Automatic file unlinking and unalterable audit trail upon verified transcript persistence.
