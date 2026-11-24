"""Structured memory retrieval engine for Organizational Memory.

Retrieves and scores structured meeting intelligence directly from database records:
1. Decisions (with lifecycle status, rationale, origin meeting, and lineage).
2. Commitments (with owner, deadline, overdue state, and complete review history).
3. Unresolved Questions (with status, raised_by, and resolution details).
4. Potential Conflicts (between active decisions detected via ConflictDetector).

Integrates with RAG without re-embedding structured entities into the transcript vector store.
"""

import logging
import re
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any, Set
from pydantic import BaseModel, Field
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.models.identity import Organization
from app.models.meeting import Meeting
from app.models.intelligence import Decision
from app.schemas.intelligence import (
    CrossMeetingDecisionResponse,
    CrossMeetingCommitmentResponse,
    CrossMeetingQuestionResponse,
    DecisionResponse,
)
from app.schemas.memory import DecisionConflictItem
from app.services.intelligence.cross_meeting import (
    DecisionLifecycleService,
    CommitmentTrackingService,
    UnresolvedQuestionService,
)
from app.services.intelligence.lineage import get_decision_lineage
from app.services.memory.conflict_detector import ConflictDetector
from app.services.memory.retrieval import CONVERSATIONAL_STOPWORDS

logger = logging.getLogger(__name__)


# Intent trigger keywords
DECISION_KEYWORDS = {
    "decision", "decisions", "decide", "decided", "deciding", "supersede", "superseded", "revoke", "revoked",
    "policy", "architecture", "choice", "agreed", "chuna", "tay", "final", "made", "taken",
    "faisla", "faisle", "निर्णय", "फैसला"
}

TEMPORAL_KEYWORDS = {
    # Earlier / Previous / Initial / First
    "previous", "previously", "earlier", "initial", "initially", "original", "originally",
    "first", "prior", "past", "old", "older", "before", "preceding", "shuru", "pehle", "pehli", "pehla",
    "purani", "purana", "purane",
    # Latest / Recent / Current / Last
    "latest", "recent", "recently", "current", "currently", "last", "new", "newest", "newer",
    "now", "present", "subsequent", "later", "afterward", "afterwards", "then", "hal", "naya", "nayi", "naye", "aakhri",
    # Change / Supersession / Revision
    "change", "changed", "changes", "changing", "revise", "revised", "revision", "replace",
    "replaced", "supersede", "superseded", "supersedes", "superseding", "revoke", "revoked",
    "update", "updated", "modify", "modified", "badla", "badlav", "badle", "badal",
    # Comparison / Timeline / History
    "compare", "compared", "comparing", "comparison", "difference", "differences", "differ",
    "differing", "versus", "vs", "timeline", "history", "historical", "evolution", "across",
    "between", "progression", "tulna", "farak", "antarr", "itihas",
    # Time units & months
    "january", "february", "march", "april", "may", "june", "july", "august", "september",
    "october", "november", "december", "month", "months", "year", "years", "week", "weeks",
    # Indic script (Devanagari)
    "पहले", "पिछला", "पिछली", "पिछले", "शुरुआती", "शुरू", "आरंभिक", "मूल",
    "नवीनतम", "नया", "नयी", "नए", "हाल", "वर्तमान", "आखिरी", "अंतिम",
    "बदलाव", "बदला", "बदली", "परिवर्तन", "संशोधन",
    "तुलना", "अंतर", "फर्क", "इतिहास", "क्रम", "महीने", "साल",
}

COMMITMENT_KEYWORDS = {
    "action", "actions", "item", "items", "task", "tasks", "commitment", "commitments",
    "todo", "assign", "assigned", "owner", "deadline", "overdue", "due", "kam", "zimmedari",
    "review", "reviews", "reviewed", "status", "progress"
}

QUESTION_KEYWORDS = {
    "question", "questions", "unresolved", "open", "deferred", "resolve", "resolved",
    "resolution", "pending", "pending question", "swal", "doubt", "unanswered", "remain", "remains", "remaining"
}

CONFLICT_KEYWORDS = {
    "conflict", "conflicts", "conflicting", "contradict", "contradiction", "contradictory",
    "disagree", "dispute", "opposing", "differ", "clash"
}


class StructuredMemoryContext(BaseModel):
    """Container holding structured organizational intelligence retrieved for a query."""
    decisions: List[CrossMeetingDecisionResponse] = Field(default_factory=list)
    commitments: List[CrossMeetingCommitmentResponse] = Field(default_factory=list)
    questions: List[CrossMeetingQuestionResponse] = Field(default_factory=list)
    conflicts: List[DecisionConflictItem] = Field(default_factory=list)
    has_structured_evidence: bool = False


class StructuredMemoryRetriever:
    """Retrieves structured meeting intelligence matching user queries within tenant boundaries."""

    @staticmethod
    def _extract_query_words(query: str) -> List[str]:
        return [w.lower() for w in re.findall(r"[\w\u0900-\u097F\u0A80-\u0AFF]+", query) if len(w) >= 2]

    @classmethod
    def search_structured_memory(
        cls,
        db: Session,
        organization_id: str,
        query: str,
        meeting_id: Optional[str] = None,
        project_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        max_per_type: int = 5,
    ) -> StructuredMemoryContext:
        """Execute tenant-scoped SQL-based structured search across decisions, commitments, questions, and conflicts.

        Args:
            db: Database session.
            organization_id: Authenticated organization UUID.
            query: User's natural-language question or search query.
            meeting_id: Optional meeting UUID filter.
            start_date: Optional start datetime filter.
            end_date: Optional end datetime filter.
            max_per_type: Maximum results to retrieve per intelligence category.

        Returns:
            StructuredMemoryContext with matching entities and detected conflicts.
        """
        if not organization_id or not organization_id.strip():
            return StructuredMemoryContext()

        clean_query = query.strip()
        words_list = cls._extract_query_words(clean_query)
        words_set = set(words_list)

        # Detect query intents
        has_decision_intent = bool(words_set.intersection(DECISION_KEYWORDS))
        has_commitment_intent = bool(words_set.intersection(COMMITMENT_KEYWORDS))
        has_question_intent = bool(words_set.intersection(QUESTION_KEYWORDS))
        has_conflict_intent = bool(words_set.intersection(CONFLICT_KEYWORDS))
        has_temporal_intent = bool(words_set.intersection(TEMPORAL_KEYWORDS))

        # Check for specific filter indicators in query
        is_overdue_query = "overdue" in words_set
        is_superseded_query = "superseded" in words_set
        is_unresolved_query = "unresolved" in words_set or "open" in words_set

        # Substantive search terms excluding intent and temporal stopwords
        all_intent_stops = (
            DECISION_KEYWORDS | COMMITMENT_KEYWORDS | QUESTION_KEYWORDS | CONFLICT_KEYWORDS | TEMPORAL_KEYWORDS | CONVERSATIONAL_STOPWORDS | {
                "what", "which", "where", "when", "how", "who", "whom", "why", "any", "are", "there", "were", "was",
                "is", "made", "about", "regarding", "concerning", "related", "show", "tell", "list", "find",
                "the", "for", "with", "have", "has", "had", "been", "our", "all", "remain", "remains", "remaining",
                "pending", "taken", "did", "do", "does", "we", "they", "our", "us", "meeting", "meetings",
                "discuss", "discussed", "take", "give", "given", "across", "between",
                # Transliterated / Romanized Indic particles
                "kya", "kaise", "kab", "kaha", "kahan", "kyu", "kyun", "kisne", "kiska", "kisko",
                "su", "kem", "kyare", "kon", "che", "hata", "hati", "mate", "vishe", "naki", "karyu",
                "mein", "me", "ko", "se", "ne", "par", "ke", "ki", "ka", "tha", "thi", "the", "hai", "hain",
                "huan", "hua", "hui", "hue", "lekar", "baare", "bare", "kare", "karna", "gaya", "gayi", "gaye",
                "liya", "diya", "hona", "chahiye", "sakte", "sakta",
                # Devanagari particles
                "में", "से", "को", "पर", "के", "की", "का", "था", "थी", "थे", "है", "हैं", "हुई", "हुए", "हुआ",
                "बारे", "लिया", "गया", "गई", "गए", "करना", "चाहिए", "सकते", "सकता"
            }
        )
        substantive_words = [w for w in words_list if w not in all_intent_stops]
        search_term = " ".join(substantive_words) if substantive_words else None

        retrieved_decisions: List[CrossMeetingDecisionResponse] = []
        retrieved_commitments: List[CrossMeetingCommitmentResponse] = []
        retrieved_questions: List[CrossMeetingQuestionResponse] = []
        detected_conflicts: List[DecisionConflictItem] = []

        # 1. Retrieve Decisions if intent matches, temporal intent is present, or general query
        if (
            has_decision_intent
            or has_temporal_intent
            or has_conflict_intent
            or not (has_commitment_intent or has_question_intent)
        ):
            # For temporal queries, never restrict status to SUPERSEDED so we get the full timeline
            dec_status = "SUPERSEDED" if (is_superseded_query and not has_temporal_intent) else None
            dec_limit = max_per_type * 2 if has_temporal_intent else max_per_type

            dec_list = DecisionLifecycleService.list_decisions(
                db=db,
                organization_id=organization_id,
                status=dec_status,
                meeting_id=meeting_id,
                start_date=start_date,
                end_date=end_date,
                search=search_term,
                limit=dec_limit,
            )
            # If search term was too specific, try individual substantive words if decision intent was explicit
            if not dec_list and (has_decision_intent or has_temporal_intent):
                if substantive_words and len(substantive_words) > 1:
                    for w in substantive_words:
                        dec_list = DecisionLifecycleService.list_decisions(
                            db=db,
                            organization_id=organization_id,
                            status=dec_status,
                            meeting_id=meeting_id,
                            start_date=start_date,
                            end_date=end_date,
                            search=w,
                            limit=dec_limit,
                        )
                        if dec_list:
                            break
                if not dec_list and not meeting_id:
                    dec_list = DecisionLifecycleService.list_decisions(
                        db=db,
                        organization_id=organization_id,
                        status=dec_status,
                        start_date=start_date,
                        end_date=end_date,
                        limit=dec_limit,
                    )

            # Reconstitute and expand full lineage for all retrieved decisions
            if dec_list:
                decisions_by_id: Dict[str, CrossMeetingDecisionResponse] = {d.id: d for d in dec_list}

                # 1a. Traverse ancestors
                for base_d in list(decisions_by_id.values()):
                    if base_d.supersedes_decision_id:
                        ancestors = get_decision_lineage(db, base_d.id)
                        for anc in ancestors:
                            if meeting_id and anc.meeting_id != meeting_id:
                                continue
                            if anc.id not in decisions_by_id:
                                anc_m = (
                                    db.query(Meeting)
                                    .filter(
                                        Meeting.id == anc.meeting_id,
                                        Meeting.organization_id == organization_id,
                                    )
                                    .first()
                                )
                                if anc_m:
                                    anc_mdate = (
                                        anc_m.actual_start_time
                                        or anc_m.scheduled_start_time
                                        or anc_m.created_at
                                    )
                                    if start_date:
                                        cmp_anc = anc_mdate.replace(tzinfo=timezone.utc) if (anc_mdate and anc_mdate.tzinfo is None) else anc_mdate
                                        cmp_start = start_date.replace(tzinfo=timezone.utc) if (start_date and start_date.tzinfo is None) else start_date
                                        if cmp_anc and cmp_start and cmp_anc < cmp_start:
                                            continue

                                    decisions_by_id[anc.id] = CrossMeetingDecisionResponse(
                                        id=anc.id,
                                        meeting_id=anc.meeting_id,
                                        description=anc.description,
                                        rationale_or_context=anc.rationale_or_context,
                                        decided_by=anc.decided_by,
                                        evidence_segment_id=anc.evidence_segment_id,
                                        timestamp_seconds=anc.timestamp_seconds,
                                        status=anc.status,
                                        supersedes_decision_id=anc.supersedes_decision_id,
                                        created_at=anc.created_at,
                                        updated_at=anc.updated_at,
                                        meeting_title=anc_m.title,
                                        meeting_date=anc_mdate,
                                    )

                # 1b. Traverse descendants (superseding decisions)
                for base_d in list(decisions_by_id.values()):
                    curr_id = base_d.id
                    while curr_id:
                        effective_date_expr = func.coalesce(
                            Meeting.actual_start_time,
                            Meeting.scheduled_start_time,
                            Meeting.created_at,
                        ).label("meeting_date")
                        superseding_query = (
                            db.query(Decision, Meeting.title, effective_date_expr)
                            .join(Meeting, Decision.meeting_id == Meeting.id)
                            .filter(
                                Decision.supersedes_decision_id == curr_id,
                                Meeting.organization_id == organization_id,
                            )
                        )
                        if meeting_id:
                            superseding_query = superseding_query.filter(Decision.meeting_id == meeting_id)
                        if end_date:
                            superseding_query = superseding_query.filter(effective_date_expr <= end_date)

                        superseding_row = superseding_query.first()
                        if superseding_row:
                            s_dec, s_mtitle, s_mdate = superseding_row
                            if s_dec.id not in decisions_by_id:
                                decisions_by_id[s_dec.id] = CrossMeetingDecisionResponse(
                                    id=s_dec.id,
                                    meeting_id=s_dec.meeting_id,
                                    description=s_dec.description,
                                    rationale_or_context=s_dec.rationale_or_context,
                                    decided_by=s_dec.decided_by,
                                    evidence_segment_id=s_dec.evidence_segment_id,
                                    timestamp_seconds=s_dec.timestamp_seconds,
                                    status=s_dec.status,
                                    supersedes_decision_id=s_dec.supersedes_decision_id,
                                    created_at=s_dec.created_at,
                                    updated_at=s_dec.updated_at,
                                    meeting_title=s_mtitle,
                                    meeting_date=s_mdate,
                                )
                            curr_id = s_dec.id
                        else:
                            break

                # 1c. Populate lineage chains and links on each decision
                for d in decisions_by_id.values():
                    d_ancestors = get_decision_lineage(db, d.id)
                    d.lineage = [DecisionResponse.model_validate(a) for a in d_ancestors]
                    d.lineage_depth = len(d.lineage)
                    superseding_id = next(
                        (other.id for other in decisions_by_id.values() if other.supersedes_decision_id == d.id),
                        None,
                    )
                    if superseding_id:
                        d.superseded_by_decision_id = superseding_id

                # 1d. Chronologically sort by actual database meeting date and created_at
                def _decision_sort_key(item: CrossMeetingDecisionResponse):
                    m_dt = item.meeting_date
                    if m_dt is None:
                        m_dt = datetime.min.replace(tzinfo=timezone.utc)
                    elif m_dt.tzinfo is None:
                        m_dt = m_dt.replace(tzinfo=timezone.utc)

                    c_dt = item.created_at
                    if c_dt is None:
                        c_dt = datetime.min.replace(tzinfo=timezone.utc)
                    elif c_dt.tzinfo is None:
                        c_dt = c_dt.replace(tzinfo=timezone.utc)

                    return (m_dt, c_dt)

                retrieved_decisions = sorted(decisions_by_id.values(), key=_decision_sort_key)
            else:
                retrieved_decisions = []

        # 2. Retrieve Commitments / Action Items if intent matches or if general query
        if has_commitment_intent or not (has_decision_intent or has_question_intent or has_conflict_intent):
            overdue_filter = True if is_overdue_query else None
            ai_list = CommitmentTrackingService.list_commitments(
                db=db,
                organization_id=organization_id,
                is_overdue=overdue_filter,
                meeting_id=meeting_id,
                search=search_term,
                limit=max_per_type,
            )
            if not ai_list and has_commitment_intent:
                if substantive_words and len(substantive_words) > 1:
                    for w in substantive_words:
                        ai_list = CommitmentTrackingService.list_commitments(
                            db=db,
                            organization_id=organization_id,
                            is_overdue=overdue_filter,
                            meeting_id=meeting_id,
                            search=w,
                            limit=max_per_type,
                        )
                        if ai_list:
                            break
                if not ai_list and not meeting_id and not substantive_words:
                    ai_list = CommitmentTrackingService.list_commitments(
                        db=db,
                        organization_id=organization_id,
                        is_overdue=overdue_filter,
                        limit=max_per_type,
                    )
            retrieved_commitments = ai_list

        # 3. Retrieve Unresolved Questions if intent matches or if general query
        if has_question_intent or not (has_decision_intent or has_commitment_intent or has_conflict_intent):
            q_status = "OPEN" if is_unresolved_query else None
            q_list = UnresolvedQuestionService.list_questions(
                db=db,
                organization_id=organization_id,
                status=q_status,
                meeting_id=meeting_id,
                start_date=start_date,
                end_date=end_date,
                search=search_term,
                limit=max_per_type,
            )
            if not q_list and has_question_intent:
                if substantive_words and len(substantive_words) > 1:
                    for w in substantive_words:
                        q_list = UnresolvedQuestionService.list_questions(
                            db=db,
                            organization_id=organization_id,
                            status=q_status,
                            meeting_id=meeting_id,
                            start_date=start_date,
                            end_date=end_date,
                            search=w,
                            limit=max_per_type,
                        )
                        if q_list:
                            break
                if not q_list and not meeting_id and not substantive_words:
                    q_list = UnresolvedQuestionService.list_questions(
                        db=db,
                        organization_id=organization_id,
                        status=q_status,
                        start_date=start_date,
                        end_date=end_date,
                        limit=max_per_type,
                    )
            retrieved_questions = q_list

        # 4. Check for conflicts if conflict intent detected or if decisions are retrieved
        if has_conflict_intent or retrieved_decisions:
            detected_conflicts = ConflictDetector.detect_conflicts(
                db=db,
                organization_id=organization_id,
                subject_filter=search_term,
            )

        # Filter all retrieved items by project_id if specified
        if project_id:
            proj_meeting_ids = set(
                r[0] for r in db.query(Meeting.id).filter(
                    Meeting.project_id == project_id,
                    Meeting.organization_id == organization_id
                ).all()
            )
            retrieved_decisions = [d for d in retrieved_decisions if d.meeting_id in proj_meeting_ids]
            retrieved_commitments = [c for c in retrieved_commitments if c.meeting_id in proj_meeting_ids]
            retrieved_questions = [q for q in retrieved_questions if q.meeting_id in proj_meeting_ids]
            proj_dec_ids = {d.id for d in retrieved_decisions}
            detected_conflicts = [
                c for c in detected_conflicts
                if (c.decision_a_id in proj_dec_ids or c.decision_b_id in proj_dec_ids)
            ]

        has_evidence = bool(
            retrieved_decisions or retrieved_commitments or retrieved_questions or detected_conflicts
        )

        return StructuredMemoryContext(
            decisions=retrieved_decisions,
            commitments=retrieved_commitments,
            questions=retrieved_questions,
            conflicts=detected_conflicts,
            has_structured_evidence=has_evidence,
        )
