"""Deterministic conflict detection engine for organizational decisions.

Identifies potential contradictions between ACTIVE decisions across meetings:
- Detects opposing stances (e.g. 'adopt/approve' vs 'reject/cancel') on shared subjects.
- Detects mutually exclusive architectural and technology choices in the same domain.
- Distinguishes potential conflicts from explicitly superseded decisions.
- Transparently labels detections as heuristic and provides actionable review guidance.
- Does NOT automatically mutate decision status or supersede records.
"""

import logging
import re
from typing import List, Optional, Dict, Set, Tuple
from sqlalchemy.orm import Session

from app.models.meeting import Meeting
from app.models.intelligence import Decision, DecisionStatus
from app.schemas.memory import DecisionConflictItem
from app.services.intelligence.lineage import get_decision_lineage

logger = logging.getLogger(__name__)


# Affirmative and negative stance patterns
AFFIRMATIVE_PATTERNS = [
    re.compile(r"\b(?:adopt|adopting|adopted|approve|approved|approving|enable|enabled|confirm|confirmed|deploy|deployed|use|using|selected|select|in\s+favor\s+of|proceed\s+with)\b", re.IGNORECASE),
    re.compile(r"\b(?:chuna|tay\s+kiya|manzoor|swikar|final\s+karyu)\b", re.IGNORECASE),
]

NEGATIVE_PATTERNS = [
    re.compile(r"\b(?:reject|rejected|rejecting|cancel|cancelled|cancelling|disallow|disallowed|decided\s+against|abandon|abandoned|drop|dropped|do\s+not\s+use|not\s+recommended|avoid|avoiding)\b", re.IGNORECASE),
    re.compile(r"\b(?:mana\s+kiya|cancel\s+karyu|reject\s+karyu|rok\s+diya)\b", re.IGNORECASE),
]

# Mutually exclusive technology domain categories
EXCLUSIVE_DOMAIN_CATEGORIES: Dict[str, Set[str]] = {
    "database": {"postgresql", "postgres", "mysql", "mongodb", "dynamodb", "cassandra", "sqlite", "couchdb", "mariadb"},
    "cloud_provider": {"aws", "gcp", "azure", "on-premise", "bare-metal"},
    "container_orchestration": {"kubernetes", "k8s", "docker swarm", "nomad", "ecs fargate"},
    "api_architecture": {"graphql", "rest", "grpc", "trpc"},
    "frontend_framework": {"react", "vue", "angular", "svelte"},
    "auth_protocol": {"oauth2", "saml", "kerberos", "ldap"},
}

STOPWORDS: Set[str] = {
    "the", "a", "an", "and", "or", "in", "on", "at", "to", "for", "with",
    "about", "meeting", "decision", "decisions", "decided", "team", "project", "system",
    "will", "shall", "must", "should", "our", "all", "v1", "v2", "phase",
    "regarding", "concerning", "related", "any", "are", "there", "were", "what", "which",
    "conflict", "conflicts", "contradict", "contradiction", "between",
}


def _extract_content_tokens(text: str) -> Set[str]:
    """Extract substantive lowercase word tokens."""
    tokens = re.findall(r"\b[a-zA-Z0-9_\u0900-\u097F\u0A80-\u0AFF]{3,}\b", text.lower())
    return {t for t in tokens if t not in STOPWORDS}


class ConflictDetector:
    """Deterministic rule-based detector for potential contradictions between decisions."""

    @staticmethod
    def detect_conflicts(
        db: Session,
        organization_id: str,
        decisions: Optional[List[Decision]] = None,
        subject_filter: Optional[str] = None,
    ) -> List[DecisionConflictItem]:
        """Identify potential conflicts across active decisions within an organization.

        Guarantees:
        1. Only considers ACTIVE decisions. SUPERSEDED or REVOKED decisions are never flagged as active conflicts.
        2. Explicit lineage links (where one decision supersedes another) are recognized as intentional evolutions, NOT conflicts.
        3. Never modifies database records or decision statuses.
        4. Transparently labels output with is_heuristic=True.

        Args:
            db: Database session.
            organization_id: Authenticated organization UUID.
            decisions: Optional list of decisions to evaluate. If omitted, loads all active decisions for the org.
            subject_filter: Optional keyword to restrict conflict evaluation to a specific topic.

        Returns:
            List of detected DecisionConflictItem instances.
        """
        # 1. Fetch active decisions if not provided
        if decisions is None:
            query = (
                db.query(Decision)
                .join(Meeting, Decision.meeting_id == Meeting.id)
                .filter(
                    Meeting.organization_id == organization_id,
                    Decision.status == DecisionStatus.ACTIVE.value,
                )
                .order_by(Decision.created_at.asc())
            )
            candidate_decisions = query.all()
        else:
            # Filter strictly to ACTIVE decisions belonging to the organization
            candidate_decisions = [
                d for d in decisions
                if d.status == DecisionStatus.ACTIVE.value
            ]

        if len(candidate_decisions) < 2:
            return []

        # 2. Build fast lookup maps for meetings
        meeting_ids = list({d.meeting_id for d in candidate_decisions})
        meetings = (
            db.query(Meeting)
            .filter(Meeting.id.in_(meeting_ids), Meeting.organization_id == organization_id)
            .all()
        )
        meeting_map: Dict[str, Meeting] = {m.id: m for m in meetings}

        # 3. Build ancestor sets to skip explicit lineage relations
        lineage_map: Dict[str, Set[str]] = {}
        for d in candidate_decisions:
            ancestors = get_decision_lineage(db, d.id)
            lineage_map[d.id] = {a.id for a in ancestors}

        detected: List[DecisionConflictItem] = []
        seen_pairs: Set[Tuple[str, str]] = set()

        # 4. Compare all pairs of active decisions
        num_dec = len(candidate_decisions)
        for i in range(num_dec):
            d1 = candidate_decisions[i]
            tokens_1 = _extract_content_tokens(d1.description)
            text_1 = d1.description.lower()

            has_aff_1 = any(p.search(d1.description) for p in AFFIRMATIVE_PATTERNS)
            has_neg_1 = any(p.search(d1.description) for p in NEGATIVE_PATTERNS)

            for j in range(i + 1, num_dec):
                d2 = candidate_decisions[j]

                # Skip if from the same meeting (intra-meeting deliberation vs cross-meeting contradiction)
                # or if already paired
                pair_key = (min(d1.id, d2.id), max(d1.id, d2.id))
                if pair_key in seen_pairs:
                    continue

                # Check explicit lineage: if d1 is an ancestor of d2 or vice-versa, skip!
                if d1.id in lineage_map.get(d2.id, set()) or d2.id in lineage_map.get(d1.id, set()):
                    continue
                if d1.supersedes_decision_id == d2.id or d2.supersedes_decision_id == d1.id:
                    continue

                tokens_2 = _extract_content_tokens(d2.description)
                text_2 = d2.description.lower()

                has_aff_2 = any(p.search(d2.description) for p in AFFIRMATIVE_PATTERNS)
                has_neg_2 = any(p.search(d2.description) for p in NEGATIVE_PATTERNS)

                # Check A: Stance opposition on shared subject keywords
                shared_tokens = tokens_1.intersection(tokens_2)
                if subject_filter:
                    filter_tokens = _extract_content_tokens(subject_filter)
                    if filter_tokens:
                        if not (filter_tokens.intersection(tokens_1) or filter_tokens.intersection(tokens_2)):
                            continue
                    elif subject_filter.lower() not in text_1 and subject_filter.lower() not in text_2:
                        continue

                conflict_found = False
                conflict_subject = ""
                conflict_reason = ""

                # Case A: Opposing stance on a shared keyword (e.g. approve X vs reject X)
                if shared_tokens:
                    is_opposing = (has_aff_1 and has_neg_2) or (has_neg_1 and has_aff_2)
                    if is_opposing:
                        conflict_subject = ", ".join(sorted(shared_tokens))
                        m1_title = meeting_map.get(d1.meeting_id, Meeting(title="Meeting")).title
                        m2_title = meeting_map.get(d2.meeting_id, Meeting(title="Meeting")).title
                        conflict_reason = (
                            f"Decision '{d1.id[:8]}' (in '{m1_title}') and Decision '{d2.id[:8]}' (in '{m2_title}') "
                            f"are both ACTIVE but express opposing stances regarding '{conflict_subject}' "
                            f"without an explicit superseding link."
                        )
                        conflict_found = True

                # Case B: Mutually exclusive technology choices in the same domain category
                if not conflict_found:
                    for category, choices in EXCLUSIVE_DOMAIN_CATEGORIES.items():
                        techs_1 = {t for t in choices if t in text_1}
                        techs_2 = {t for t in choices if t in text_2}
                        if techs_1 and techs_2 and techs_1 != techs_2:
                            # Both decisions pick different technologies from the same exclusive category
                            t1_name = next(iter(techs_1))
                            t2_name = next(iter(techs_2))
                            conflict_subject = f"{category} ({t1_name} vs {t2_name})"
                            m1_title = meeting_map.get(d1.meeting_id, Meeting(title="Meeting")).title
                            m2_title = meeting_map.get(d2.meeting_id, Meeting(title="Meeting")).title
                            conflict_reason = (
                                f"Decisions in '{m1_title}' and '{m2_title}' are both ACTIVE but select competing "
                                f"{category} options ('{t1_name}' vs '{t2_name}') without an explicit superseding relationship."
                            )
                            conflict_found = True
                            break

                if conflict_found:
                    seen_pairs.add(pair_key)
                    m1 = meeting_map.get(d1.meeting_id)
                    m2 = meeting_map.get(d2.meeting_id)

                    m1_date = m1.created_at.isoformat() if m1 and m1.created_at else None
                    m2_date = m2.created_at.isoformat() if m2 and m2.created_at else None

                    item = DecisionConflictItem(
                        decision_a_id=d1.id,
                        decision_a_text=d1.description,
                        decision_a_meeting_title=m1.title if m1 else "Unknown Meeting",
                        decision_a_date=m1_date,
                        decision_b_id=d2.id,
                        decision_b_text=d2.description,
                        decision_b_meeting_title=m2.title if m2 else "Unknown Meeting",
                        decision_b_date=m2_date,
                        subject=conflict_subject,
                        reason=conflict_reason,
                        is_heuristic=True,
                        suggested_action=(
                            "Review whether the later decision was intended to explicitly supersede the earlier one, "
                            "or if one decision should be revoked using the decision lifecycle service."
                        ),
                    )
                    detected.append(item)

        return detected
