"""Persistent tenant-isolated hybrid retrieval engine combining SQLite FTS5 and NumPy vector search."""

import json
import re
from typing import List, Optional, Dict, Any, Union
import numpy as np
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.models.meeting import Meeting
from app.models.memory import TranscriptChunkRecord
from app.services.memory.exceptions import (
    RetrievalError,
    VectorValidationError,
    TenantSecurityError,
)
from app.services.memory.storage import (
    deserialize_embedding,
    ensure_fts5_ready,
    EXPECTED_DIMENSION,
)
from app.services.memory.types import (
    KeywordSearchResult,
    SemanticSearchResult,
    HybridSearchResult,
)


# Common conversational, grammatical, and question-filler stopwords
# across English, romanized Hinglish, Devanagari (Hindi), and Gujarati.
CONVERSATIONAL_STOPWORDS = {
    # English grammatical & conversational filler
    "what", "when", "where", "which", "who", "whom", "whose", "why", "how",
    "is", "am", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "have", "has", "had", "having",
    "the", "a", "an", "and", "or", "in", "on", "at", "to", "for", "with", "by", "from", "of", "as", "into", "onto",
    "about", "regarding", "concerning", "related",
    "tell", "show", "give", "me", "us", "can", "could", "would", "should", "will", "shall",
    "use", "uses", "used", "using",
    "now", "current", "currently", "latest", "recent", "recently", "today", "yesterday",
    "make", "makes", "made", "making", "take", "takes", "took", "taken", "taking",
    "our", "my", "your", "their", "his", "her", "its",
    "we", "you", "they", "them", "he", "she", "it",
    "this", "that", "these", "those", "there", "here",
    "any", "some", "all", "each", "every", "both", "neither", "either",
    "across", "between", "among", "along", "after", "before", "during", "while",
    "please", "thanks", "thank", "okay", "hello", "hi", "hey",

    # Meeting & discussion conversational filler
    "meeting", "meetings", "discuss", "discussed", "discussing", "discussion", "discussions",
    "decide", "decided", "deciding", "decision", "decisions",
    "team", "teams", "member", "members", "everyone", "people", "folks", "anyone", "someone",
    "talk", "talks", "talked", "talking", "update", "updates", "updated", "updating",
    "sync", "syncs", "synced", "syncing", "call", "calls", "called", "calling",

    # Transliterated Hindi / Hinglish conversational
    "kya", "kaise", "kab", "kaha", "kahan", "kyu", "kyun", "kisne", "kiska", "kisko", "kaun",
    "hai", "hain", "tha", "thi", "the", "hoga", "hogi", "hoge",
    "ke", "ki", "ka", "ko", "se", "me", "mein", "par", "pe", "liye",
    "baare", "batao", "bataiye", "hume", "hum", "yeh", "woh", "kuch", "koi",
    "kare", "karna", "karne", "karni", "karta", "karte", "karti", "kar", "karein", "kiya", "gaya", "gayi", "gaye", "hua", "hui", "hue",
    "karenge", "karega", "karegi", "aur", "konsa", "kaunsa", "kaunse", "kaunsi", "kam",
    "faisla", "nirnay", "tay", "liya", "lekar", "baat",

    # Devanagari conversational
    "क्या", "कैसे", "कब", "कहाँ", "क्यों", "कौन", "किसने", "किसका", "किसको",
    "है", "हैं", "था", "थी", "थे", "होगा", "होगी", "होंगे",
    "के", "की", "का", "को", "से", "में", "पर", "लिए", "लिये",
    "बारे", "बताओ", "बताइए", "हमें", "हम", "यह", "वह", "कुछ", "कोई",
    "करें", "करना", "करने", "किया", "गया", "गई", "गए", "हुआ", "हुई", "हुए",
    "होने", "रहा", "रही", "रहे", "सकता", "सकती", "सकते", "चाहिए", "दिए", "दी", "दिया", "देना",
    "निर्णय", "फैसला", "तय", "लेना", "ली", "बात", "बारेमें",

    # Gujarati conversational
    "શું", "કેમ", "ક્યારે", "ક્યાં", "કોણ", "છે", "હતા", "હતી", "હતું", "હશે",
    "માટે", "વિશે", "અંગે", "કરો", "કર્યું", "થયું", "આ", "તે", "આવતી", "આવતા",
    "કરવા", "કરવામાં", "આવ્યા", "આવ્યું", "આવશે", "આવી", "આપ્યા", "આપ્યું",
    "થવા", "થવાની", "થશે", "જોઈએ", "કરેલ", "કરેલા", "થયેલ", "થયેલા",
    "કયું", "કઈ", "કયો", "કોઈ", "બધા", "દ્વારા", "પર", "અને", "કે",
    "નિર્ણય", "નક્કી", "લીધો", "લીધી", "વાત", "ચર્ચા", "કેવું", "કેવી", "કેવો", "રહેશે",
    "su", "kem", "kyare", "kya", "kon", "che", "hata", "hati", "matu", "ne", "ma", "mate", "vishe",
}


def sanitize_fts5_query(raw_query: str) -> str:
    """Sanitize and format a user query string safely for SQLite FTS5 MATCH expressions.

    Supports:
    1. Exact phrases enclosed in quotes: "machine learning"
    2. Conversational stopword stripping across English, Hinglish, Hindi, and Gujarati.
    3. Multilingual terms in Latin, Devanagari (Hindi), and Gujarati scripts.
    4. Safe FTS5 OR expression generation so conversational questions do not require
       every single token to appear in the candidate passage.
    5. Neutralizes FTS5 operator injection (AND, OR, NOT, NEAR, column prefixes, unclosed syntax).
    """
    if not raw_query:
        return ""

    raw = raw_query.strip()
    if not raw:
        return ""

    # 1. Exact phrase enclosed in quotes: e.g. "machine learning"
    if raw.startswith('"') and raw.endswith('"') and raw.count('"') == 2:
        inner = raw[1:-1].strip()
        inner = re.sub(r'["\\]', ' ', inner).strip()
        if inner:
            return f'"{inner}"'
        return ""

    # 2. Extract any embedded quoted phrases
    quoted_phrases = [
        re.sub(r'["\\]', ' ', p).strip()
        for p in re.findall(r'"([^"]+)"', raw)
        if re.sub(r'["\\]', ' ', p).strip()
    ]
    unquoted_text = re.sub(r'"[^"]+"', ' ', raw)

    # 3. Extract individual word tokens (preserving hyphenated compound terms like un-updated, zero-downtime)
    raw_tokens = re.findall(r'[\w\u0900-\u097F\u0A80-\u0AFF]+(?:-[\w\u0900-\u097F\u0A80-\u0AFF]+)*', unquoted_text)

    # 4. Filter conversational stopwords
    substantive_tokens = [
        t for t in raw_tokens
        if t.lower() not in CONVERSATIONAL_STOPWORDS
    ]

    # Combine terms
    terms = []
    for phrase in quoted_phrases:
        terms.append(f'"{phrase}"')

    for token in substantive_tokens:
        terms.append(f'"{token}"')

    if not terms:
        return ""

    # If single term, return exact quoted term
    if len(terms) == 1:
        return terms[0]

    # Multiple substantive terms: join with explicit OR
    return " OR ".join(terms)


def search_keyword(
    db: Session,
    query: str,
    organization_id: str,
    meeting_id: Optional[str] = None,
    project_id: Optional[str] = None,
    limit: int = 20,
) -> List[KeywordSearchResult]:
    """Execute tenant-isolated SQLite FTS5 keyword full-text search with BM25 ranking.

    Security guarantee:
    Organization boundary is strictly enforced in the database JOIN and WHERE clause.
    Candidates from other tenants are never retrieved.
    """
    if not organization_id or not organization_id.strip():
        raise TenantSecurityError("A valid organization_id is required for keyword search.")

    if limit <= 0:
        return []

    ensure_fts5_ready(db)

    sanitized = sanitize_fts5_query(query)
    if not sanitized:
        return []

    sql_parts = [
        "SELECT",
        "  tc.id AS chunk_id,",
        "  tc.organization_id,",
        "  tc.meeting_id,",
        "  tc.transcript_id,",
        "  tc.chunk_index,",
        "  tc.text,",
        "  tc.start_time,",
        "  tc.end_time,",
        "  tc.segment_ids,",
        "  tc.language_codes,",
        "  tc.dominant_language,",
        "  tc.token_count,",
        "  tc.extra_metadata,",
        "  bm25(transcript_chunks_fts) AS bm25_rank",
        "FROM transcript_chunks_fts fts",
        "JOIN transcript_chunks tc ON fts.chunk_id = tc.id",
        "WHERE fts.transcript_chunks_fts MATCH :query",
        "  AND tc.organization_id = :organization_id",
    ]
    params: Dict[str, Any] = {
        "query": sanitized,
        "organization_id": organization_id,
        "limit": limit,
    }

    if meeting_id is not None:
        sql_parts.append("  AND tc.meeting_id = :meeting_id")
        params["meeting_id"] = meeting_id

    if project_id is not None:
        sql_parts.append("  AND tc.meeting_id IN (SELECT id FROM meetings WHERE project_id = :project_id AND organization_id = :organization_id)")
        params["project_id"] = project_id

    # In SQLite FTS5, bm25() returns negative values where lower is more relevant (e.g. -1.5 < -1.0)
    sql_parts.append("ORDER BY bm25_rank ASC")
    sql_parts.append("LIMIT :limit")

    query_str = "\n".join(sql_parts)
    try:
        rows = db.execute(text(query_str), params).fetchall()
    except Exception as exc:
        raise RetrievalError(f"FTS5 keyword search failed: {exc}") from exc

    results: List[KeywordSearchResult] = []
    for rank_idx, row in enumerate(rows, start=1):
        segment_ids = json.loads(row.segment_ids) if row.segment_ids else []
        language_codes = json.loads(row.language_codes) if row.language_codes else []
        metadata = json.loads(row.extra_metadata) if row.extra_metadata else {}

        results.append(
            KeywordSearchResult(
                chunk_id=str(row.chunk_id),
                bm25_score=float(row.bm25_rank),
                bm25_rank=rank_idx,
                text=str(row.text),
                meeting_id=str(row.meeting_id),
                organization_id=str(row.organization_id),
                transcript_id=str(row.transcript_id),
                chunk_index=int(row.chunk_index),
                start_time=float(row.start_time),
                end_time=float(row.end_time),
                segment_ids=segment_ids,
                language_codes=language_codes,
                dominant_language=str(row.dominant_language),
                token_count=int(row.token_count),
                metadata=metadata,
            )
        )

    return results


def search_semantic(
    db: Session,
    query_vector: Union[List[float], np.ndarray],
    organization_id: str,
    meeting_id: Optional[str] = None,
    project_id: Optional[str] = None,
    limit: int = 20,
    expected_dim: int = EXPECTED_DIMENSION,
) -> List[SemanticSearchResult]:
    """Execute tenant-isolated dense vector cosine similarity search using NumPy.

    Critical security guarantee:
    Organization filtering MUST happen in the database query before candidate vectors
    are loaded into memory or compared. Cross-tenant vectors are never loaded.
    """
    if not organization_id or not organization_id.strip():
        raise TenantSecurityError("A valid organization_id is required for semantic search.")

    if limit <= 0:
        return []

    # Validate query vector
    if query_vector is None:
        raise VectorValidationError("Query vector cannot be None.")

    try:
        q_arr = np.asarray(query_vector, dtype=np.float32)
    except Exception as exc:
        raise VectorValidationError(f"Invalid query vector: {exc}") from exc

    if q_arr.ndim != 1 or q_arr.shape[0] != expected_dim:
        raise VectorValidationError(
            f"Query vector dimension mismatch: expected {expected_dim}, got {q_arr.shape}."
        )

    if not np.all(np.isfinite(q_arr)):
        raise VectorValidationError("Query vector contains NaN or infinite values.")

    q_norm_val = float(np.linalg.norm(q_arr))
    if q_norm_val == 0.0:
        raise VectorValidationError("Query vector cannot be all zeros.")

    # Unit normalize the query vector for accurate cosine similarity
    q_normalized = q_arr / q_norm_val

    # Query candidate chunks strictly within the tenant boundary in the database
    db_query = (
        db.query(
            TranscriptChunkRecord.id,
            TranscriptChunkRecord.organization_id,
            TranscriptChunkRecord.meeting_id,
            TranscriptChunkRecord.transcript_id,
            TranscriptChunkRecord.chunk_index,
            TranscriptChunkRecord.text,
            TranscriptChunkRecord.start_time,
            TranscriptChunkRecord.end_time,
            TranscriptChunkRecord.segment_ids,
            TranscriptChunkRecord.language_codes,
            TranscriptChunkRecord.dominant_language,
            TranscriptChunkRecord.token_count,
            TranscriptChunkRecord.extra_metadata,
            TranscriptChunkRecord.embedding,
        )
        .filter(TranscriptChunkRecord.organization_id == organization_id)
    )

    if meeting_id is not None:
        db_query = db_query.filter(TranscriptChunkRecord.meeting_id == meeting_id)

    if project_id is not None:
        db_query = db_query.filter(
            TranscriptChunkRecord.meeting_id.in_(
                db.query(Meeting.id).filter(
                    Meeting.project_id == project_id,
                    Meeting.organization_id == organization_id
                )
            )
        )

    candidates = db_query.all()
    if not candidates:
        return []

    # Unpack candidate vectors into an (N, 384) matrix
    num_candidates = len(candidates)
    candidate_matrix = np.empty((num_candidates, expected_dim), dtype=np.float32)

    for i, candidate in enumerate(candidates):
        candidate_matrix[i] = deserialize_embedding(candidate.embedding, expected_dim=expected_dim)

    # Cosine similarity for unit-normalized vectors is dot product
    scores = np.dot(candidate_matrix, q_normalized)

    # Rank descending
    top_indices = np.argsort(scores)[::-1][:limit]

    results: List[SemanticSearchResult] = []
    for rank_idx, idx in enumerate(top_indices, start=1):
        row = candidates[idx]
        segment_ids = json.loads(row.segment_ids) if row.segment_ids else []
        language_codes = json.loads(row.language_codes) if row.language_codes else []
        metadata = json.loads(row.extra_metadata) if row.extra_metadata else {}

        results.append(
            SemanticSearchResult(
                chunk_id=str(row.id),
                cosine_similarity=float(scores[idx]),
                semantic_rank=rank_idx,
                text=str(row.text),
                meeting_id=str(row.meeting_id),
                organization_id=str(row.organization_id),
                transcript_id=str(row.transcript_id),
                chunk_index=int(row.chunk_index),
                start_time=float(row.start_time),
                end_time=float(row.end_time),
                segment_ids=segment_ids,
                language_codes=language_codes,
                dominant_language=str(row.dominant_language),
                token_count=int(row.token_count),
                metadata=metadata,
            )
        )

    return results


def search_hybrid_rrf(
    db: Session,
    query: str,
    query_vector: Union[List[float], np.ndarray],
    organization_id: str,
    meeting_id: Optional[str] = None,
    project_id: Optional[str] = None,
    limit: int = 20,
    k: int = 60,
    expected_dim: int = EXPECTED_DIMENSION,
) -> List[HybridSearchResult]:
    """Combine FTS5 keyword ranking and dense vector semantic ranking using Reciprocal Rank Fusion (RRF).

    Formula:
        RRF_score(chunk) = sum(1 / (k + rank))
        where rank is 1-based.

    Preserves individual keyword and semantic ranks/scores for transparency.
    Note: The RRF score is a relative reciprocal rank fusion score, NOT a probability or confidence score.
    """
    if k < 1:
        raise RetrievalError(f"RRF smoothing parameter k must be >= 1, got {k}.")

    if limit <= 0:
        return []

    # Execute tenant-isolated keyword search
    keyword_results = search_keyword(
        db=db,
        query=query,
        organization_id=organization_id,
        meeting_id=meeting_id,
        project_id=project_id,
        limit=limit * 2,  # Oversample to allow rich fusion
    )

    # Execute tenant-isolated semantic search
    semantic_results = search_semantic(
        db=db,
        query_vector=query_vector,
        organization_id=organization_id,
        meeting_id=meeting_id,
        project_id=project_id,
        limit=limit * 2,  # Oversample to allow rich fusion
        expected_dim=expected_dim,
    )

    # Merge by chunk_id
    fused_entries: Dict[str, Dict[str, Any]] = {}

    for kw in keyword_results:
        cid = kw.chunk_id
        if cid not in fused_entries:
            fused_entries[cid] = {
                "chunk_id": cid,
                "text": kw.text,
                "meeting_id": kw.meeting_id,
                "organization_id": kw.organization_id,
                "transcript_id": kw.transcript_id,
                "chunk_index": kw.chunk_index,
                "start_time": kw.start_time,
                "end_time": kw.end_time,
                "segment_ids": kw.segment_ids,
                "language_codes": kw.language_codes,
                "dominant_language": kw.dominant_language,
                "token_count": kw.token_count,
                "metadata": kw.metadata,
                "keyword_rank": kw.bm25_rank,
                "keyword_score": kw.bm25_score,
                "semantic_rank": None,
                "semantic_score": None,
                "rrf_score": 1.0 / (k + kw.bm25_rank),
            }

    for sem in semantic_results:
        cid = sem.chunk_id
        sem_contrib = 1.0 / (k + sem.semantic_rank)
        if cid in fused_entries:
            fused_entries[cid]["semantic_rank"] = sem.semantic_rank
            fused_entries[cid]["semantic_score"] = sem.cosine_similarity
            fused_entries[cid]["rrf_score"] += sem_contrib
        else:
            fused_entries[cid] = {
                "chunk_id": cid,
                "text": sem.text,
                "meeting_id": sem.meeting_id,
                "organization_id": sem.organization_id,
                "transcript_id": sem.transcript_id,
                "chunk_index": sem.chunk_index,
                "start_time": sem.start_time,
                "end_time": sem.end_time,
                "segment_ids": sem.segment_ids,
                "language_codes": sem.language_codes,
                "dominant_language": sem.dominant_language,
                "token_count": sem.token_count,
                "metadata": sem.metadata,
                "keyword_rank": None,
                "keyword_score": None,
                "semantic_rank": sem.semantic_rank,
                "semantic_score": sem.cosine_similarity,
                "rrf_score": sem_contrib,
            }

    # Sort descending by RRF score
    sorted_candidates = sorted(
        fused_entries.values(),
        key=lambda x: x["rrf_score"],
        reverse=True,
    )[:limit]

    return [HybridSearchResult(**item) for item in sorted_candidates]
