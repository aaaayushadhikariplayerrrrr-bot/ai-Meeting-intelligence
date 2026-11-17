"""Pluggable Answer Providers for Cross-Meeting RAG Question Answering."""

import json
import logging
import re
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Tuple, Set
import httpx

from app.core.config import settings
from app.services.intelligence.base import ProviderExecutionError
from app.services.memory.rag.context import ContextBuilder
from app.services.memory.retrieval import CONVERSATIONAL_STOPWORDS
from app.services.memory.structured_retrieval import TEMPORAL_KEYWORDS
from app.services.memory.rag.types import (
    EvidenceItem,
    RawAnswerResult,
    AbstentionStatus,
    DecisionConflictItem,
    TranscriptConflictItem,
)

logger = logging.getLogger(__name__)


class AnswerProvider(ABC):
    """Abstract base class for RAG answer generation providers."""

    @abstractmethod
    def generate_answer(
        self,
        question: str,
        evidence_items: List[EvidenceItem],
        output_language: str = "en",
        structured_context: Optional[Any] = None,
    ) -> RawAnswerResult:
        """Generate a grounded factual answer from assembled evidence items."""
        pass


class DeterministicAnswerProvider(AnswerProvider):
    """Conservative, evidence-grounded rule-based answer generator.

    Provides reliable factual responses without pretending to perform LLM reasoning.
    Transparently cites sources, detects cross-meeting contradictions, and abstains
    when evidence is insufficient or missing.
    """

    AFFIRMATIVE_MARKERS = [
        re.compile(r"\b(?:decided\s+to|agreed\s+to|approved|confirmed|finalized|selected|adopted|in\s+favor\s+of)\b", re.IGNORECASE),
        re.compile(r"\b(?:tay\s+hua|final\s+karyu|manzoor|pakka\s+kiya)\b", re.IGNORECASE),
    ]
    NEGATIVE_MARKERS = [
        re.compile(r"\b(?:decided\s+against|rejected|cancelled|abandoned|dropped|disagreed|not\s+recommended)\b", re.IGNORECASE),
        re.compile(r"\b(?:mana\s+kiya|cancel\s+karyu|reject\s+karyu)\b", re.IGNORECASE),
    ]

    # Substantive category, architecture, concept, and operational domain words that do NOT denote specific branded products/tools
    GENERIC_CONTEXT_WORDS = {
        "infrastructure", "database", "databases", "backend", "frontend", "architecture", "architectures",
        "platform", "platforms", "system", "systems", "service", "services", "microservice", "microservices",
        "server", "servers", "cloud", "storage", "caching", "cache", "network", "networking", "cluster", "clusters",
        "container", "containers", "orchestration", "orchestrating", "framework", "frameworks", "tool", "tools",
        "technology", "technologies", "stack", "stacks", "pipeline", "pipelines", "application", "applications", "app", "apps",
        "api", "apis", "protocol", "protocols", "security", "policy", "policies", "deployment", "deployments",
        "production", "prod", "dev", "staging", "environment", "environments", "code", "codebase",
        "component", "components", "layer", "layers", "solution", "solutions", "vendor", "vendors", "provider", "providers",
        "product", "products", "strategy", "setup", "configuration", "config", "model", "models",
        "data", "process", "processes", "method", "methods", "item", "items", "feature", "features",
        "option", "options", "alternative", "alternatives", "choice", "choices", "topic", "topics",
        "issue", "issues", "bug", "bugs", "ticket", "tickets", "sprint", "sprints", "project", "projects",
        "team", "teams", "member", "members", "org", "organization", "workspace", "workspaces",
        "meeting", "meetings", "discussion", "discussions", "sync", "review", "reviews",
        "pod", "pods", "node", "nodes", "instance", "instances", "vm", "vms",
        "engine", "engines", "memory", "in-memory", "speed", "fast", "faster",
        "disk", "disks", "io", "cpu", "cpus", "ram", "gpu", "gpus", "hardware",
        "tier", "tiers", "driver", "drivers", "connector", "connectors", "monolith",
        "budget", "cost", "costs", "expense", "expenses", "pricing", "office", "rent", "lease", "contract", "contracts",
        "agreement", "agreements", "compliance", "audit", "audits", "license", "licenses", "licensing",
        "client", "clients", "customer", "customers", "user", "users", "account", "accounts", "auth", "authentication",
        "authorization", "login", "access", "credential", "credentials", "password", "passwords", "secret", "secrets",
        "key", "keys", "token", "tokens", "session", "sessions", "role", "roles", "permission", "permissions",
        "tenant", "tenants", "tenancy", "isolation", "encryption", "redaction", "privacy",
        "monitoring", "logging", "logs", "metrics", "alerting", "alerts", "test", "tests", "testing",
        "ci", "cd", "repo", "repository", "repositories", "branch", "branches", "commit", "commits",
        "pr", "prs", "release", "releases", "version", "versions", "migration", "migrations",
        "scale", "scaling", "scalability", "performance", "latency", "throughput", "load", "pressure",
        "read", "write", "queries", "query", "table", "tables", "schema", "schemas", "record", "records",
        "index", "indices", "indexing", "search", "hybrid", "vector", "dense", "sparse",
        # Transliterated Hindi / Gujarati
        "kya", "kaise", "kab", "kaha", "kahan", "kyu", "kyun", "kisne", "kiska", "kisko",
        "shuru", "pehle", "pehli", "pehla", "purani", "purana", "purane", "aakhri", "naya", "nayi", "naye",
        "faisla", "faisle", "kam", "zimmedari", "swal", "charcha", "vishe", "baat", "baare",
        # Devanagari and Gujarati generic concepts
        "डेटाबेस", "इन्फ्रास्ट्रक्चर", "सिस्टम", "प्लेटफॉर्म", "सर्विस", "सर्विसेज", "क्लाउड",
        "सुरक्षा", "प्रणाली", "कैशिंग", "बजट", "ऑफिस", "मीटिंग", "परियोजना", "चर्चा", "फैसला", "निर्णय",
        "ઓફિસ", "બજેટ", "પ્લેટફોર્મ", "સિસ્ટમ", "કરાર", "ભાડા", "ચર્ચા", "નિર્ણય"
    }

    ACTION_DECISION_WORDS = {
        "adopt", "adopts", "adopted", "adopting", "adoption",
        "select", "selects", "selected", "selecting", "selection",
        "approve", "approves", "approved", "approving", "approval",
        "reject", "rejects", "rejected", "rejecting", "rejection",
        "confirm", "confirms", "confirmed", "confirming", "confirmation",
        "agree", "agrees", "agreed", "agreeing", "agreement",
        "decide", "decides", "decided", "deciding", "decision", "decisions",
        "choose", "chooses", "chose", "chosen", "choosing",
        "pick", "picks", "picked", "picking",
        "use", "uses", "used", "using", "usage",
        "implement", "implements", "implemented", "implementing", "implementation",
        "deploy", "deploys", "deployed", "deploying", "deployment",
        "migrate", "migrates", "migrated", "migrating", "migration",
        "replace", "replaces", "replaced", "replacing", "replacement",
        "switch", "switches", "switched", "switching",
        "change", "changes", "changed", "changing",
        "finalize", "finalizes", "finalized", "finalizing",
        "drop", "drops", "dropped", "dropping",
        "cancel", "cancels", "cancelled", "canceling", "cancellation",
        "abandon", "abandons", "abandoned", "abandoning",
        "consider", "considers", "considered", "considering", "consideration",
        "evaluate", "evaluates", "evaluated", "evaluating", "evaluation",
        "recommend", "recommends", "recommended", "recommending", "recommendation",
        "prefer", "prefers", "preferred", "preferring", "preference",
        "enable", "enables", "enabled", "enabling",
        "disable", "disables", "disabled", "disabling",
        "disallow", "disallows", "disallowed", "disallowing",
        "plan", "plans", "planned", "planning",
        "start", "starts", "started", "starting",
        "stop", "stops", "stopped", "stopping",
        "chuna", "chuni", "chune", "chunav", "rakha", "rakhi", "rakhe",
        "lagaya", "lagayi", "lagaye", "hataya", "hatayi", "hataye",
        "badla", "badli", "badle", "badlav", "tay", "manzoor", "swikar", "aswikar",
        "મંજૂર", "પસંદ"
    }

    @staticmethod
    def _extract_stem(word: str) -> str:
        """Return a normalized stem for common English inflectional suffixes."""
        w = word.lower()
        for suffix in ("ing", "ed", "es", "s", "ion"):
            if w.endswith(suffix) and len(w) - len(suffix) >= 4:
                return w[:-len(suffix)]
        return w

    def _matches_keyword_or_stem(self, text: str, keywords: List[str]) -> bool:
        """Check if text contains any keyword directly or via inflectional stem."""
        if not keywords:
            return True
        text_words = set(re.findall(r"[\w\u0900-\u097F\u0A80-\u0AFF]+", text.lower()))
        kw_set = set(keywords)
        if text_words.intersection(kw_set):
            return True
        text_stems = {self._extract_stem(w) for w in text_words}
        kw_stems = {self._extract_stem(k) for k in keywords}
        return bool(text_stems.intersection(kw_stems))

    def _extract_query_keywords(self, question: str) -> List[str]:
        """Extract substantive words from question across Latin, Devanagari, and Gujarati scripts."""
        words = re.findall(r"[\w\u0900-\u097F\u0A80-\u0AFF]+", question.lower())
        return [w for w in words if len(w) > 2 and w not in CONVERSATIONAL_STOPWORDS]

    def _extract_query_entities(self, question: str) -> List[str]:
        """Identify specific named entities, tools, products, or technologies in the question.

        Distinguishes specific named technologies (e.g. 'terraform', 'spanner', 'kubernetes', 'redis')
        from generic domain words (e.g. 'infrastructure', 'database') and action verbs (e.g. 'selected').
        """
        words = re.findall(r"[\w\u0900-\u097F\u0A80-\u0AFF]+", question.lower())
        substantive = [w for w in words if len(w) > 2 and w not in CONVERSATIONAL_STOPWORDS]
        entities = []
        for w in substantive:
            stem = self._extract_stem(w)
            if (
                w not in self.GENERIC_CONTEXT_WORDS
                and stem not in self.GENERIC_CONTEXT_WORDS
                and w not in self.ACTION_DECISION_WORDS
                and stem not in self.ACTION_DECISION_WORDS
                and w not in TEMPORAL_KEYWORDS
                and stem not in TEMPORAL_KEYWORDS
            ):
                entities.append(w)
        return entities

    def _detect_conflicts(self, evidence_items: List[EvidenceItem], keywords: List[str]) -> Optional[Tuple[EvidenceItem, EvidenceItem, str]]:
        """Identify potential conflicts across separate meetings on the same topic."""
        if len(evidence_items) < 2:
            return None

        meeting_groups: Dict[str, List[EvidenceItem]] = {}
        for item in evidence_items:
            meeting_groups.setdefault(item.meeting_id, []).append(item)

        if len(meeting_groups) < 2:
            return None

        # Check for opposing stance across meetings on substantive topic keywords
        # Exclude pure action/decision verbs (e.g. 'adopted', 'approved') from being used as topic subjects.
        topic_keywords = [
            kw for kw in keywords
            if kw not in self.ACTION_DECISION_WORDS
            and self._extract_stem(kw) not in self.ACTION_DECISION_WORDS
        ]
        if not topic_keywords:
            topic_keywords = keywords

        affirmative_items: List[Tuple[EvidenceItem, str]] = []
        negative_items: List[Tuple[EvidenceItem, str]] = []

        for item in evidence_items:
            matching_kws = [kw for kw in topic_keywords if self._matches_keyword_or_stem(item.text, [kw])]
            if not matching_kws:
                continue

            has_aff = any(p.search(item.text) for p in self.AFFIRMATIVE_MARKERS)
            has_neg = any(p.search(item.text) for p in self.NEGATIVE_MARKERS)

            if has_neg:
                negative_items.append((item, matching_kws[0]))
            elif has_aff:
                affirmative_items.append((item, matching_kws[0]))

        for aff_item, kw in affirmative_items:
            for neg_item, nkw in negative_items:
                if aff_item.meeting_id != neg_item.meeting_id and (kw == nkw or self._extract_stem(kw) == self._extract_stem(nkw)):
                    return (aff_item, neg_item, kw)

        return None

    def _clean_and_condense_item_text(self, text: str, keywords: List[str]) -> str:
        """Collapse redundant adjacent redaction tokens and extract relevant sentences if multi-sentence."""
        # 1. Collapse duplicate adjacent redaction placeholders (e.g. [PHONE_REDACTED] [PHONE_REDACTED])
        collapsed = re.sub(
            r'(\[(?:[A-Z0-9_]+_REDACTED|REDACTED)\](?:\s+)){2,}',
            r'\1',
            text
        )
        collapsed = collapsed.strip()

        # 2. Split into sentences respecting English and Indic sentence terminators
        raw_sentences = re.split(r'(?<=[.?!।॥\n])\s+', collapsed)
        sentences = [s.strip() for s in raw_sentences if s.strip()]
        if len(sentences) <= 1:
            return collapsed

        # 3. If keywords provided, extract only sentences matching at least one keyword or stem
        if keywords:
            matching = [
                s for s in sentences
                if self._matches_keyword_or_stem(s, keywords)
            ]
            if matching:
                return " ".join(matching)

        return collapsed

    def _synthesize_decisions(
        self,
        question: str,
        decisions: List[Any],
        conflicts: List[Any],
    ) -> List[str]:
        """Synthesize chronological temporal reasoning and decision lineage across meetings."""
        if not decisions:
            return []

        q_lower = question.lower()

        # Detect specific temporal question intents
        is_latest = any(w in q_lower for w in [
            "latest", "recent", "recently", "current", "currently", "last", "newest", "newer",
            "hal", "naya", "nayi", "naye", "aakhri", "नवीनतम", "हाल", "वर्तमान"
        ])
        is_supersession = any(w in q_lower for w in [
            "supersede", "superseded", "supersedes", "change", "changed", "changes", "revise",
            "revised", "replace", "replaced", "initially", "earlier", "initial", "previous",
            "pehle", "pehli", "badla", "badlav", "बदलाव", "बदला", "संशोधन", "what changed"
        ])
        is_comparison = any(w in q_lower for w in [
            "compare", "comparing", "comparison", "difference", "differences", "differ",
            "versus", "vs", "tulna", "farak", "antarr", "तुलना", "अंतर", "फर्क"
        ])
        is_history = (
            any(w in q_lower for w in ["history", "timeline", "evolution", "progression", "itihas", "इतिहास"])
            or ("across" in q_lower and ("meeting" in q_lower or "meetings" in q_lower))
            or ("previous" in q_lower and ("meeting" in q_lower or "meetings" in q_lower))
            or ("earlier" in q_lower and ("meeting" in q_lower or "meetings" in q_lower))
            or ("pehle" in q_lower and ("meeting" in q_lower or "meetings" in q_lower))
            or ("pehli" in q_lower and ("meeting" in q_lower or "meetings" in q_lower))
            or ("पहले" in q_lower and ("meeting" in q_lower or "meetings" in q_lower))
        )

        # Check for confirmed supersession relationships among decisions
        supersession_pairs: List[Tuple[Any, Any]] = []
        for d in decisions:
            if getattr(d, "supersedes_decision_id", None):
                earlier = next((ed for ed in decisions if ed.id == d.supersedes_decision_id), None)
                if earlier:
                    supersession_pairs.append((earlier, d))

        lines: List[str] = []

        # Case 1: Single decision retrieved
        if len(decisions) == 1:
            d = decisions[0]
            m_date = getattr(d, "meeting_date", None)
            date_str = f" on {str(m_date)[:10]}" if m_date else ""
            ev_ref = f"Evidence Reference: Segment [{d.evidence_segment_id}]" if getattr(d, "evidence_segment_id", None) else "Evidence Reference: Unavailable"
            superseded_note = f" (Supersedes decision {d.supersedes_decision_id[:8]})" if getattr(d, "supersedes_decision_id", None) else ""
            if d.status == "SUPERSEDED":
                superseded_note += " [SUPERSEDED by a later decision]"
            elif d.status == "REVOKED":
                superseded_note += " [REVOKED]"

            lines.append("Recorded Decisions in Organizational Memory:")
            line = f"- [{d.status}] \"{d.description}\" (Decided in '{d.meeting_title}'{date_str}){superseded_note}"
            if getattr(d, "rationale_or_context", None):
                line += f"\n  Rationale: {d.rationale_or_context}"
            line += f"\n  {ev_ref}"
            lines.append(line)

            if is_comparison or is_history or is_supersession:
                lines.append(
                    "\nHistorical Context Note:\n"
                    f"Available meeting records contain only a single recorded decision on this topic ('{d.meeting_title}'{date_str}); "
                    "there is insufficient historical data across multiple meetings to establish an evolution, comparison, or change over time."
                )
            elif is_latest:
                lines.append("\nLatest Decision Analysis:")
                if d.status == "ACTIVE":
                    lines.append(
                        f"The latest (and only) recorded decision regarding this topic was introduced in '{d.meeting_title}'{date_str}.\n"
                        f"Status: Confirmed CURRENT and ACTIVE in organizational records."
                    )
                elif d.status == "REVOKED":
                    lines.append(
                        f"The latest recorded decision was introduced in '{d.meeting_title}'{date_str}, but was REVOKED.\n"
                        "Status: REVOKED. There is currently no active decision on this topic."
                    )
                elif d.status == "SUPERSEDED":
                    lines.append(
                        f"The latest recorded decision in '{d.meeting_title}'{date_str} is marked SUPERSEDED in organizational records."
                    )
            return lines

        # Case 2: Multiple decisions across meetings / dates
        # Present full chronological history
        lines.append("Chronological Decision History:")
        for idx, d in enumerate(decisions, 1):
            m_date = getattr(d, "meeting_date", None)
            date_str = f" ({str(m_date)[:10]})" if m_date else ""
            ev_ref = f"Evidence Reference: Segment [{d.evidence_segment_id}]" if getattr(d, "evidence_segment_id", None) else "Evidence Reference: Unavailable"

            status_note = ""
            if d.status == "SUPERSEDED":
                status_note = "\n     Status Note: SUPERSEDED by a later decision"
            elif d.status == "REVOKED":
                status_note = "\n     Status Note: REVOKED"
            elif d.status == "ACTIVE":
                if getattr(d, "supersedes_decision_id", None):
                    status_note = "\n     Status Note: ACTIVE (Supersedes earlier decision)"
                else:
                    status_note = "\n     Status Note: ACTIVE"

            rat_str = f"\n     Rationale: {d.rationale_or_context}" if getattr(d, "rationale_or_context", None) else ""
            lines.append(
                f"  {idx}. [{d.status}] \"{d.description}\"\n"
                f"     Meeting: '{d.meeting_title}'{date_str}{rat_str}\n"
                f"     {ev_ref}{status_note}"
            )

        # Synthesize according to relationship and query intent
        if supersession_pairs:
            # Confirmed explicit lineage exists
            if is_latest:
                latest = decisions[-1]
                latest_mdate = getattr(latest, "meeting_date", None)
                latest_date = f" on {str(latest_mdate)[:10]}" if latest_mdate else ""
                lines.append("\nLatest Decision Analysis:")
                lines.append(f"The latest recorded decision regarding this topic was introduced in '{latest.meeting_title}'{latest_date}:")
                lines.append(f"- [{latest.status}] \"{latest.description}\"")
                if getattr(latest, "rationale_or_context", None):
                    lines.append(f"  Rationale: {latest.rationale_or_context}")
                if getattr(latest, "supersedes_decision_id", None):
                    earlier = next((ed for ed in decisions if ed.id == latest.supersedes_decision_id), None)
                    if earlier:
                        e_mdate = getattr(earlier, "meeting_date", None)
                        e_date = f" ({str(e_mdate)[:10]})" if e_mdate else ""
                        lines.append(f"  Lineage: Explicitly superseded the earlier decision from '{earlier.meeting_title}'{e_date}.")
                if latest.status == "ACTIVE":
                    lines.append("  Status: Confirmed CURRENT and ACTIVE in organizational records.")
                elif latest.status == "REVOKED":
                    lines.append("  Status: REVOKED. This decision was revoked, leaving no active decision on this topic.")
                elif latest.status == "SUPERSEDED":
                    lines.append("  Status: SUPERSEDED by a later decision.")

            elif is_comparison:
                lines.append("\nCross-Meeting Comparison & Changes:")
                for earlier, later in supersession_pairs:
                    e_mdate = getattr(earlier, "meeting_date", None)
                    l_mdate = getattr(later, "meeting_date", None)
                    e_date = f" ({str(e_mdate)[:10]})" if e_mdate else ""
                    l_date = f" ({str(l_mdate)[:10]})" if l_mdate else ""
                    lines.append(
                        f"- In '{earlier.meeting_title}'{e_date}, the decision was: [{earlier.status}] \"{earlier.description}\".\n"
                        f"- In '{later.meeting_title}'{l_date}, the decision was: [{later.status}] \"{later.description}\".\n"
                        f"- Relationship: The decision in '{later.meeting_title}' explicitly superseded the earlier decision in '{earlier.meeting_title}'."
                    )
                    if getattr(later, "rationale_or_context", None):
                        lines.append(f"  Recorded Rationale: {later.rationale_or_context}")

            else:
                # History / Supersession / General Timeline
                lines.append("\nDecision Lineage & Supersession Summary:")
                for earlier, later in supersession_pairs:
                    e_mdate = getattr(earlier, "meeting_date", None)
                    l_mdate = getattr(later, "meeting_date", None)
                    e_date = f" ({str(e_mdate)[:10]})" if e_mdate else ""
                    l_date = f" ({str(l_mdate)[:10]})" if l_mdate else ""
                    lines.append(
                        f"- Earlier Decision: [{earlier.status}] \"{earlier.description}\" decided in '{earlier.meeting_title}'{e_date}.\n"
                        f"- Superseding Decision: [{later.status}] \"{later.description}\" decided in '{later.meeting_title}'{l_date}."
                    )
                    if getattr(later, "rationale_or_context", None):
                        lines.append(f"  Recorded Change / Rationale: {later.rationale_or_context}")

                latest = decisions[-1]
                latest_mdate = getattr(latest, "meeting_date", None)
                l_date = f" on {str(latest_mdate)[:10]}" if latest_mdate else ""
                if latest.status == "ACTIVE":
                    lines.append(f"- Current Status: [{latest.status}] \"{latest.description}\" from '{latest.meeting_title}'{l_date} is currently the active recorded decision.")
                elif latest.status == "REVOKED":
                    lines.append(f"- Current Status: The latest decision was REVOKED; organizational records do not establish an active effective decision on this topic.")
        else:
            # Multiple decisions without explicit supersession link
            lines.append("\nLineage & Conflict Analysis:")
            lines.append(
                "Note: No explicit superseding relationship is recorded between these decisions. "
                "Both decisions currently remain recorded in organizational records without a confirmed lineage link."
            )
            active_decs = [d for d in decisions if d.status == "ACTIVE"]
            if len(active_decs) >= 2:
                lines.append(
                    "Because multiple decisions remain marked ACTIVE on this topic across separate meetings without an explicit superseding link, "
                    "the available evidence is insufficient to confirm which decision is current, representing a potential decision conflict rather than a confirmed supersession."
                )
            elif is_latest:
                latest = decisions[-1]
                latest_mdate = getattr(latest, "meeting_date", None)
                l_date = f" on {str(latest_mdate)[:10]}" if latest_mdate else ""
                lines.append(
                    f"The latest recorded decision chronologically was made in '{latest.meeting_title}'{l_date}: \"{latest.description}\" [{latest.status}]. "
                    "However, because no explicit lineage links the decisions, the records do not confirm that the earlier policy was formally superseded."
                )

        return lines

    def generate_answer(
        self,
        question: str,
        evidence_items: List[EvidenceItem],
        output_language: str = "en",
        structured_context: Optional[Any] = None,
    ) -> RawAnswerResult:
        """Deterministically synthesize a grounded factual response from retrieved evidence and structured memory."""
        # 1. Check for explicit decision conflicts detected in structured records (P2)
        conflicts = getattr(structured_context, "conflicts", None) if structured_context else None

        if conflicts:
            # P2: Independent of query wording (with or without 'conflict'), if unlinked active decision
            # conflicts exist, return CONFLICTING_EVIDENCE and abstain from arbitrarily selecting a winner.
            cf_blocks = []
            for cf in conflicts:
                cf_blocks.append(
                    f"• Potential contradiction regarding '{cf.subject}':\n"
                    f"  - Decision in '{cf.decision_a_meeting_title}': \"{cf.decision_a_text}\"\n"
                    f"  - Decision in '{cf.decision_b_meeting_title}': \"{cf.decision_b_text}\"\n"
                    f"  - Reason: {cf.reason}\n"
                    f"  - Recommended Action: {cf.suggested_action}"
                )
            conflict_msg = (
                "[HEURISTIC CONFLICT WARNING] Potentially contradictory active decisions were identified across meetings:\n\n"
                + "\n\n".join(cf_blocks)
                + "\n\nNote: Both decisions currently remain marked ACTIVE in organizational records without an explicit superseding link or revocation. "
                "The system refrains from arbitrarily selecting the newest decision as the current decision."
            )
            answer_parts = [conflict_msg]
            if structured_context and getattr(structured_context, "decisions", None):
                answer_parts.append("\nRecorded Conflicting Decisions in Timeline:")
                for d in structured_context.decisions:
                    m_date = getattr(d, "meeting_date", None)
                    d_str = f" on {str(m_date)[:10]}" if m_date else ""
                    answer_parts.append(f"- [{d.status}] \"{d.description}\" in '{d.meeting_title}'{d_str}")

            conflict_cited_ids = []
            if evidence_items:
                keywords = self._extract_query_keywords(question)
                matched_items = [
                    item for item in evidence_items
                    if not keywords or self._matches_keyword_or_stem(item.text, keywords)
                ]
                if matched_items:
                    answer_parts.append("\nSupporting Discussion Passages from Transcripts:")
                    for item in matched_items[:4]:
                        condensed = self._clean_and_condense_item_text(item.text, keywords)
                        conflict_cited_ids.append(item.chunk_id)
                        answer_parts.append(f'- In \'{item.meeting_title}\': "{condensed}" [{item.chunk_id}]')
                else:
                    conflict_cited_ids = [item.chunk_id for item in evidence_items[:2]]

            return RawAnswerResult(
                answer_text="\n".join(answer_parts),
                cited_chunk_ids=conflict_cited_ids,
                abstention_status=AbstentionStatus.CONFLICTING_EVIDENCE,
                abstention_reason="Potential contradiction detected between active decisions without explicit superseding link.",
                provider_used="deterministic",
                is_heuristic=True,
                detected_conflicts=conflicts,
                structured_decisions=[d.model_dump() for d in structured_context.decisions] if structured_context and getattr(structured_context, "decisions", None) else [],
                structured_commitments=[c.model_dump() for c in structured_context.commitments] if structured_context and getattr(structured_context, "commitments", None) else [],
                structured_questions=[q.model_dump() for q in structured_context.questions] if structured_context and getattr(structured_context, "questions", None) else [],
            )

        # 2. Check for relevant structured records (Decisions, Commitments, Questions)
        entities = self._extract_query_entities(question)
        is_indic_query = bool(re.search(r'[\u0900-\u097F\u0A80-\u0AFF]', question))

        has_matching_decisions = bool(
            structured_context
            and getattr(structured_context, "decisions", None)
            and (
                not entities
                or any(
                    any(
                        self._matches_keyword_or_stem(d.description, [e])
                        or (getattr(d, "decided_by", None) and self._matches_keyword_or_stem(d.decided_by, [e]))
                        or (getattr(d, "rationale_or_context", None) and self._matches_keyword_or_stem(d.rationale_or_context, [e]))
                        for e in entities
                    )
                    for d in structured_context.decisions
                )
            )
        )
        has_matching_commitments = bool(
            structured_context
            and getattr(structured_context, "commitments", None)
            and (
                not entities
                or any(
                    any(
                        self._matches_keyword_or_stem(c.description, [e])
                        or (getattr(c, "owner_name", None) and self._matches_keyword_or_stem(c.owner_name, [e]))
                        for e in entities
                    )
                    for c in structured_context.commitments
                )
            )
        )
        has_matching_questions = bool(
            structured_context
            and getattr(structured_context, "questions", None)
            and (
                not entities
                or any(
                    any(
                        self._matches_keyword_or_stem(q.question_text, [e])
                        or (getattr(q, "raised_by", None) and self._matches_keyword_or_stem(q.raised_by, [e]))
                        for e in entities
                    )
                    for q in structured_context.questions
                )
            )
        )
        has_structured = has_matching_decisions or has_matching_commitments or has_matching_questions

        if has_structured:
            answer_sections: List[str] = []
            structured_cited_ids: List[str] = []

            # Format decisions if present
            if structured_context.decisions and has_matching_decisions:
                matching_decs = [
                    d for d in structured_context.decisions
                    if not entities or any(
                        self._matches_keyword_or_stem(d.description, [e])
                        or (getattr(d, "decided_by", None) and self._matches_keyword_or_stem(d.decided_by, [e]))
                        or (getattr(d, "rationale_or_context", None) and self._matches_keyword_or_stem(d.rationale_or_context, [e]))
                        for e in entities
                    )
                ]
                dec_lines = self._synthesize_decisions(
                    question=question,
                    decisions=matching_decs,
                    conflicts=conflicts or [],
                )
                answer_sections.extend(dec_lines)

            # Format commitments if present
            if structured_context.commitments and has_matching_commitments:
                matching_comms = [
                    c for c in structured_context.commitments
                    if not entities or any(
                        self._matches_keyword_or_stem(c.description, [e])
                        or (getattr(c, "owner_name", None) and self._matches_keyword_or_stem(c.owner_name, [e]))
                        for e in entities
                    )
                ]
                if answer_sections:
                    answer_sections.append("")
                answer_sections.append("Committed Action Items:")
                for c in matching_comms:
                    overdue_label = " [OVERDUE]" if c.is_overdue else ""
                    deadline_str = f"{str(c.deadline)[:10]}" if c.deadline else "No deadline set"
                    owner_str = c.owner_name or "Unassigned"
                    rev_str = f" | Last reviewed in '{c.last_reviewed_meeting_title}'" if c.last_reviewed_meeting_title else ""
                    ev_str = f" | Evidence: Segment [{c.evidence_segment_id}]" if c.evidence_segment_id else " | Evidence: Unavailable"
                    line = (
                        f"- [{c.status}{overdue_label}] \"{c.description}\"\n"
                        f"  Owner: {owner_str} | Deadline: {deadline_str} | Origin: '{c.origin_meeting_title}'{rev_str}{ev_str}"
                    )
                    answer_sections.append(line)

            # Format questions if present
            if structured_context.questions and has_matching_questions:
                matching_qs = [
                    q for q in structured_context.questions
                    if not entities or any(
                        self._matches_keyword_or_stem(q.question_text, [e])
                        or (getattr(q, "raised_by", None) and self._matches_keyword_or_stem(q.raised_by, [e]))
                        for e in entities
                    )
                ]
                if answer_sections:
                    answer_sections.append("")
                answer_sections.append("Follow-up & Unresolved Questions:")
                for q in matching_qs:
                    raised_str = f" (Raised by {q.raised_by})" if q.raised_by else ""
                    res_str = ""
                    if q.status == "RESOLVED":
                        res_m = f" in '{q.resolving_meeting_title}'" if q.resolving_meeting_title else ""
                        res_str = f"\n  Resolution: Resolved{res_m}. Notes: {q.resolution_notes or 'None'}"
                    ev_str = f" | Evidence: Segment [{q.evidence_segment_id}]" if q.evidence_segment_id else " | Evidence: Unavailable"
                    line = f"- [{q.status}] \"{q.question_text}\"{raised_str} from '{q.origin_meeting_title}'{res_str}{ev_str}"
                    answer_sections.append(line)

            # Check if there are also supporting transcript passages (P3: stem matching)
            keywords = self._extract_query_keywords(question)
            relevant_transcript_items = [
                item for item in evidence_items
                if (not keywords or self._matches_keyword_or_stem(item.text, keywords))
                and (not entities or any(self._matches_keyword_or_stem(item.text, [e]) for e in entities))
            ]
            if relevant_transcript_items:
                answer_sections.append("\nSupporting Discussion Passages from Transcripts:")
                for item in relevant_transcript_items[:3]:
                    condensed = self._clean_and_condense_item_text(item.text, keywords)
                    structured_cited_ids.append(item.chunk_id)
                    answer_sections.append(f'- In \'{item.meeting_title}\': "{condensed}" [{item.chunk_id}]')

            full_answer = "\n".join(answer_sections)
            return RawAnswerResult(
                answer_text=full_answer,
                cited_chunk_ids=structured_cited_ids,
                abstention_status=AbstentionStatus.ANSWER_SUPPORTED,
                abstention_reason=None,
                provider_used="deterministic",
                is_heuristic=True,
                structured_decisions=[d.model_dump() for d in structured_context.decisions] if structured_context and getattr(structured_context, "decisions", None) else [],
                structured_commitments=[c.model_dump() for c in structured_context.commitments] if structured_context and getattr(structured_context, "commitments", None) else [],
                structured_questions=[q.model_dump() for q in structured_context.questions] if structured_context and getattr(structured_context, "questions", None) else [],
                detected_conflicts=conflicts or [],
            )

        # 3. No evidence check (neither structured nor transcript)
        if not evidence_items:
            return RawAnswerResult(
                answer_text="No relevant meeting discussions or decisions were found in your organization's recorded meetings for this question.",
                cited_chunk_ids=[],
                abstention_status=AbstentionStatus.NO_EVIDENCE,
                abstention_reason="No matching meeting evidence retrieved for query.",
                provider_used="deterministic",
                is_heuristic=True,
            )

        keywords = self._extract_query_keywords(question)

        # 4. Check for sufficient keyword / topic relevance in transcript evidence (P1)
        keyword_matched_items = [
            item for item in evidence_items
            if not keywords or self._matches_keyword_or_stem(item.text, keywords)
        ]

        if entities and not is_indic_query:
            # P1: Specific entity requirement.
            # Chunks that only match generic context words (e.g. 'infrastructure', 'database')
            # but do NOT mention the queried entity (e.g. Terraform, Spanner) cannot support claims about it.
            entity_matched_items = [
                item for item in keyword_matched_items
                if any(self._matches_keyword_or_stem(item.text, [e]) for e in entities)
            ]
            if not entity_matched_items:
                return RawAnswerResult(
                    answer_text=(
                        f"Retrieved meeting discussions mention related organizational background, "
                        f"but contain insufficient factual evidence or decisions regarding '{', '.join(entities)}'."
                    ),
                    cited_chunk_ids=[],
                    abstention_status=AbstentionStatus.INSUFFICIENT_EVIDENCE,
                    abstention_reason=f"Retrieved evidence does not mention queried entity '{', '.join(entities)}'.",
                    provider_used="deterministic",
                    is_heuristic=True,
                )

            # Check for multi-meeting disjoint evidence (P1 Requirement 4):
            # If the user asks whether an entity was decided / selected / approved / adopted:
            # At least one meeting must substantively connect the entity with an action or decision.
            # Chunks from other meetings that don't mention the entity cannot be combined to fabricate support.
            has_action_intent = any(
                self._matches_keyword_or_stem(question, [w])
                for w in self.ACTION_DECISION_WORDS
            ) or any(p.search(question) for p in self.AFFIRMATIVE_MARKERS + self.NEGATIVE_MARKERS)

            if has_action_intent:
                has_substantive_support = any(
                    any(p.search(item.text) for p in self.AFFIRMATIVE_MARKERS + self.NEGATIVE_MARKERS)
                    or self._matches_keyword_or_stem(item.text, list(self.ACTION_DECISION_WORDS))
                    for item in entity_matched_items
                )
                if not has_substantive_support:
                    return RawAnswerResult(
                        answer_text=(
                            f"Retrieved meeting discussions mention '{', '.join(entities)}', "
                            f"but do not establish a confirmed decision, approval, selection, or adoption."
                        ),
                        cited_chunk_ids=[],
                        abstention_status=AbstentionStatus.INSUFFICIENT_EVIDENCE,
                        abstention_reason="Evidence mentions the entity but does not support the queried decision or action.",
                        provider_used="deterministic",
                        is_heuristic=True,
                    )

            relevant_items = entity_matched_items
        elif keyword_matched_items:
            relevant_items = keyword_matched_items
        elif is_indic_query:
            # Cross-script Indic retrieval: query is in Indic script against Latin (English) evidence
            relevant_items = [
                item for item in evidence_items
                if not bool(re.search(r'[\u0900-\u097F\u0A80-\u0AFF]', item.text)) and item.semantic_rank == 1
            ]
        else:
            relevant_items = []

        if not relevant_items:
            return RawAnswerResult(
                answer_text=(
                    "Retrieved meeting discussions mention related organizational background, "
                    "but contain insufficient factual evidence to answer this specific question."
                ),
                cited_chunk_ids=[],
                abstention_status=AbstentionStatus.INSUFFICIENT_EVIDENCE,
                abstention_reason="Retrieved evidence does not contain relevant query keywords or details.",
                provider_used="deterministic",
                is_heuristic=True,
            )

        # 5. Check for cross-meeting conflicts in transcript passages (P4)
        conflict = self._detect_conflicts(relevant_items, keywords)
        if conflict:
            aff_item, neg_item, topic = conflict
            date_aff = f" on {aff_item.meeting_date[:10]}" if aff_item.meeting_date else ""
            date_neg = f" on {neg_item.meeting_date[:10]}" if neg_item.meeting_date else ""

            conflict_text = (
                f"Potentially conflicting statements were identified across meetings regarding '{topic}':\n\n"
                f"- In '{aff_item.meeting_title}'{date_aff}: \"{aff_item.text}\" [{aff_item.chunk_id}]\n"
                f"- In '{neg_item.meeting_title}'{date_neg}: \"{neg_item.text}\" [{neg_item.chunk_id}]\n\n"
                "The meetings do not indicate a single unanimous resolution."
            )
            transcript_conflict = TranscriptConflictItem(
                chunk_a_id=aff_item.chunk_id,
                chunk_a_text=aff_item.text,
                chunk_a_meeting_title=aff_item.meeting_title,
                chunk_a_date=aff_item.meeting_date,
                chunk_b_id=neg_item.chunk_id,
                chunk_b_text=neg_item.text,
                chunk_b_meeting_title=neg_item.meeting_title,
                chunk_b_date=neg_item.meeting_date,
                subject=topic,
                reason=(
                    f"Passage in '{aff_item.meeting_title}' expresses an affirmative stance regarding '{topic}', "
                    f"while passage in '{neg_item.meeting_title}' expresses a negative stance without an explicit resolution."
                ),
                is_heuristic=True,
                suggested_action="Review the cited meeting segments to determine whether the policy was updated or remains disputed.",
            )
            return RawAnswerResult(
                answer_text=conflict_text,
                cited_chunk_ids=[aff_item.chunk_id, neg_item.chunk_id],
                abstention_status=AbstentionStatus.CONFLICTING_EVIDENCE,
                abstention_reason=f"Conflicting decisions/statements found between '{aff_item.meeting_title}' and '{neg_item.meeting_title}'.",
                provider_used="deterministic",
                is_heuristic=True,
                transcript_conflicts=[transcript_conflict],
            )

        # 6. Standard Factual Evidence Synthesis grouped by meeting
        meetings_order: List[str] = []
        meeting_to_items: Dict[str, List[EvidenceItem]] = {}
        for item in relevant_items:
            if item.meeting_id not in meeting_to_items:
                meetings_order.append(item.meeting_id)
                meeting_to_items[item.meeting_id] = []
            meeting_to_items[item.meeting_id].append(item)

        answer_lines: List[str] = []
        cited_chunk_ids: List[str] = []

        is_multi_meeting = len(meetings_order) > 1
        if is_multi_meeting:
            answer_lines.append(f"Based on discussions across {len(meetings_order)} meetings:")
        else:
            first_m = meeting_to_items[meetings_order[0]][0]
            answer_lines.append(f"Based on discussion in '{first_m.meeting_title}':")

        for mid in meetings_order:
            m_items = meeting_to_items[mid]
            m_title = m_items[0].meeting_title
            m_date = f" ({m_items[0].meeting_date[:10]})" if m_items[0].meeting_date else ""

            if is_multi_meeting:
                answer_lines.append(f"\nIn '{m_title}'{m_date}:")

            seen_texts: set = set()
            for item in m_items:
                condensed = self._clean_and_condense_item_text(item.text, keywords)
                if condensed in seen_texts:
                    continue
                seen_texts.add(condensed)
                cited_chunk_ids.append(item.chunk_id)
                prefix = "- " if is_multi_meeting else ""
                answer_lines.append(f'{prefix}"{condensed}" [{item.chunk_id}]')

        full_answer = "\n".join(answer_lines)
        return RawAnswerResult(
            answer_text=full_answer,
            cited_chunk_ids=cited_chunk_ids,
            abstention_status=AbstentionStatus.ANSWER_SUPPORTED,
            abstention_reason=None,
            provider_used="deterministic",
            is_heuristic=True,
        )


class OllamaAnswerProvider(AnswerProvider):
    """Optional Ollama LLM provider for generative question answering.

    Disabled by default. If enabled but unreachable, raises ProviderExecutionError
    to trigger transparent fallback to DeterministicAnswerProvider.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout: Optional[float] = None,
        client: Optional[httpx.Client] = None,
    ):
        self.base_url = (base_url or getattr(settings, "OLLAMA_BASE_URL", "http://127.0.0.1:11434")).rstrip("/")
        self.model_name = model_name or getattr(settings, "LLM_MODEL_NAME", "llama3.2:3b")
        self.timeout = timeout or getattr(settings, "LLM_REQUEST_TIMEOUT_SECONDS", 60.0)
        self._custom_client = client
        self.context_builder = ContextBuilder()

    def generate_answer(
        self,
        question: str,
        evidence_items: List[EvidenceItem],
        output_language: str = "en",
        structured_context: Optional[Any] = None,
    ) -> RawAnswerResult:
        """Call Ollama /api/generate with assembled context and extract cited chunk IDs."""
        has_any_evidence = bool(
            evidence_items
            or (structured_context and (
                getattr(structured_context, "decisions", None)
                or getattr(structured_context, "commitments", None)
                or getattr(structured_context, "questions", None)
                or getattr(structured_context, "conflicts", None)
            ))
        )
        if not has_any_evidence:
            return RawAnswerResult(
                answer_text="No relevant meeting discussions or decisions were found in your organization's recorded meetings for this question.",
                cited_chunk_ids=[],
                abstention_status=AbstentionStatus.NO_EVIDENCE,
                abstention_reason="No matching meeting evidence retrieved for query.",
                provider_used="ollama",
                is_heuristic=False,
            )

        prompts = self.context_builder.build_llm_prompt(
            question=question,
            evidence_items=evidence_items,
            output_language=output_language,
            structured_context=structured_context,
        )

        request_body = {
            "model": self.model_name,
            "prompt": prompts["prompt"],
            "system": prompts["system"],
            "stream": False,
            "options": {
                "temperature": 0.1,
            },
        }

        endpoint = f"{self.base_url}/api/generate"
        client = self._custom_client or httpx.Client(timeout=self.timeout)

        try:
            response = client.post(endpoint, json=request_body)
            if response.status_code != 200:
                raise ProviderExecutionError(
                    f"Ollama generation returned non-200 status {response.status_code}: {response.text[:200]}"
                )

            data = response.json()
            raw_response = data.get("response", "").strip()
            if not raw_response:
                raise ProviderExecutionError("Ollama returned an empty response.")

            # Extract cited chunk IDs matching retrieved evidence
            retrieved_chunk_ids = {e.chunk_id for e in evidence_items}
            found_citations: List[str] = []
            for cid in retrieved_chunk_ids:
                if cid in raw_response:
                    found_citations.append(cid)

            # Check if LLM abstained
            is_abstained = any(
                phrase in raw_response.lower()
                for phrase in ["insufficient evidence", "no evidence", "not mentioned", "not discussed"]
            )
            abstention_status = AbstentionStatus.INSUFFICIENT_EVIDENCE if is_abstained else AbstentionStatus.ANSWER_SUPPORTED

            return RawAnswerResult(
                answer_text=raw_response,
                cited_chunk_ids=found_citations,
                abstention_status=abstention_status,
                abstention_reason="LLM indicated evidence was insufficient." if is_abstained else None,
                provider_used="ollama",
                is_heuristic=False,
            )

        except (httpx.ConnectError, httpx.TimeoutException) as conn_err:
            raise ProviderExecutionError(f"Ollama daemon unreachable: {conn_err}") from conn_err
        except Exception as exc:
            if isinstance(exc, ProviderExecutionError):
                raise
            raise ProviderExecutionError(f"Ollama generation failed: {exc}") from exc
        finally:
            if not self._custom_client:
                client.close()
