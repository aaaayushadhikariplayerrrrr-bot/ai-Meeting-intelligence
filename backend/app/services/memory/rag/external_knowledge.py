"""External Knowledge Integration Service for MEMORA Organizational Memory.

Provides background encyclopedic, legal, and technical knowledge for subjects
mentioned in meetings when the meeting discussion does not provide a full explanation.

Privacy Safeguard:
- Only sanitized public topic/subject strings are queried externally.
- No private transcripts, attendee identities, internal IDs, or organization data are ever transmitted.
"""

import json
import logging
import re
import urllib.parse
from typing import List, Optional, Dict, Any
import httpx

from app.core.config import settings
from app.schemas.memory import (
    ExternalKnowledgeCitation,
    ExternalKnowledgeResult,
    EvidenceItem,
)

logger = logging.getLogger(__name__)

# User-Agent conforming to Wikimedia API policy
WIKIMEDIA_USER_AGENT = "MEMORA-Intelligence/1.0 (organizational-memory@memora.internal)"

# Question regex patterns that explicitly seek external definitions, explanations, or background
EXTERNAL_QUESTION_PATTERNS = [
    re.compile(r"^(?:what\s+(?:is|was|are|were)|what\s+does|can\s+you\s+explain|please\s+explain|explain|tell\s+me\s+about)\s+(?:the\s+|an?\s+)?(.+?)(?:\s+about|\s+mean|\s+stand\s+for|\s*\?)?$", re.IGNORECASE),
    re.compile(r"^(?:background|details|information)\s+on\s+(?:the\s+|an?\s+)?(.+?)(?:\s*\?)?$", re.IGNORECASE),
    re.compile(r"^what\s+(?:is|was)\s+(?:the\s+)?(.+?)\s+about\??$", re.IGNORECASE),
    re.compile(r"^(?:who\s+(?:is|was))\s+(?:the\s+)?(.+?)(?:\s*\?)?$", re.IGNORECASE),
]

# Recognizable statutory, constitutional, regulatory, technical, or standardized entities
KNOWN_ENTITY_PATTERNS = [
    re.compile(r"\b(?:\d+(?:st|nd|rd|th)?|one\s+hundred\s+and\s+(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth))\s+(?:constitutional\s+amendment|amendment(?:\s+act)?)\b", re.IGNORECASE),
    re.compile(r"\b(?:section\s+\d+[a-z]?|article\s+\d+[a-z]?)\b", re.IGNORECASE),
    re.compile(r"\b(?:gdpr|hipaa|soc\s*2|pci-dss|iso\s*27001|oauth(?:\s*2(?:\.0)?)?|saml|jwt|grpc|graphql|opentelemetry|kafka|kubernetes|terraform|docker|redis|elasticsearch|postgresql|mysql)\b", re.IGNORECASE),
]


class ExternalKnowledgeService:
    """Retrieves verified external background knowledge to contextualize meeting discussions."""

    def __init__(self, client: Optional[httpx.Client] = None, timeout: float = 6.0):
        self._client = client
        self.timeout = timeout

    def _get_client(self) -> httpx.Client:
        if self._client is not None:
            return self._client
        return httpx.Client(
            timeout=self.timeout,
            headers={"User-Agent": WIKIMEDIA_USER_AGENT},
            follow_redirects=True,
        )

    def extract_external_subject(
        self,
        question: str,
        evidence_items: Optional[List[EvidenceItem]] = None,
    ) -> Optional[str]:
        """Identify if a question or meeting evidence refers to an external subject needing background context.

        Returns the sanitized subject string or None.
        Privacy Guarantee: Extracts only the sanitized subject entity, never meeting text or attendee names.
        """
        clean_q = question.strip()
        if not clean_q:
            return None

        # 1. Check direct query patterns (e.g. "What is the 103rd Constitutional Amendment about?")
        for pattern in EXTERNAL_QUESTION_PATTERNS:
            match = pattern.match(clean_q)
            if match:
                candidate = match.group(1).strip()
                # Clean trailing question marks, prepositions, or punctuation
                candidate = re.sub(r"[?.,;!]+$", "", candidate).strip()
                candidate = re.sub(r"\s+about$", "", candidate, flags=re.IGNORECASE).strip()
                candidate = re.sub(r"\s+mean$", "", candidate, flags=re.IGNORECASE).strip()
                candidate = re.sub(r"\s+stand\s+for$", "", candidate, flags=re.IGNORECASE).strip()
                if len(candidate) >= 3 and not self._is_purely_conversational(candidate):
                    return self._sanitize_subject(candidate)

        # 2. Check for known entities or statutory patterns in the question
        for pattern in KNOWN_ENTITY_PATTERNS:
            match = pattern.search(clean_q)
            if match:
                return self._sanitize_subject(match.group(0))

        # 3. Check if meeting evidence mentions a recognized statutory/technical entity that the question asks about
        if evidence_items:
            for item in evidence_items:
                for pattern in KNOWN_ENTITY_PATTERNS:
                    match = pattern.search(item.text)
                    if match:
                        entity = match.group(0)
                        # If the user's question mentions this entity or a key word from it
                        entity_words = [w.lower() for w in entity.split() if len(w) > 2]
                        if any(w in clean_q.lower() for w in entity_words):
                            return self._sanitize_subject(entity)

        return None

    def _is_purely_conversational(self, text: str) -> bool:
        """Check if candidate text is purely conversational meeting jargon rather than an entity."""
        t_lower = text.lower().strip()
        stop_phrases = {
            "the next step", "our plan", "the decision", "the budget", "the meeting",
            "the agenda", "the status", "action items", "next steps", "we discussed",
            "he said", "she said", "everyone agreed", "who attended", "summary",
        }
        if t_lower in stop_phrases:
            return True

        # Check for first/second person or internal organizational workflow markers
        internal_workflow_tokens = {
            "our", "my", "we", "your", "sprint", "roadmap", "backlog", "standup", "retro",
            "retrospective", "milestone", "ticket", "jira", "assigned", "deadline", "quarter",
            "q1", "q2", "q3", "q4", "okr", "okrs", "kpi", "kpis", "meeting", "meetings"
        }
        words = set(re.findall(r"\w+", t_lower))
        if words.intersection(internal_workflow_tokens):
            return True

        return False

    def _sanitize_subject(self, subject: str) -> str:
        """Sanitize subject string to ensure no PII or sensitive data is included."""
        # Remove emails, URLs, IP addresses
        cleaned = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "", subject)
        cleaned = re.sub(r"https?://\S+", "", cleaned)
        cleaned = re.sub(r"[^\w\s\-\.\']", " ", cleaned)
        cleaned = " ".join(cleaned.split())
        return cleaned[:100].strip()

    def _rank_best_wikipedia_title(self, subject: str, search_results: List[Dict[str, Any]]) -> Optional[str]:
        """Rank and select the most specific relevant article title from search results."""
        if not search_results:
            return None

        # Build candidate words
        subj_lower = subject.lower()
        num_match = re.search(r"\b(\d+)(?:st|nd|rd|th)?\b", subj_lower)
        num_str = num_match.group(1) if num_match else ""

        # Map common ordinals to word form (e.g. 103 -> 'one hundred and third', 42 -> 'forty-second')
        num_words = {
            "103": "one hundred and third",
            "104": "one hundred and fourth",
            "105": "one hundred and fifth",
            "106": "one hundred and sixth",
            "42": "forty-second",
            "44": "forty-fourth",
        }
        word_form = num_words.get(num_str, "")

        def score_candidate(item: Dict[str, Any]) -> int:
            title = item.get("title", "")
            t_lower = title.lower()
            score = 0

            # Penalize generic lists and overview catalogues
            if t_lower.startswith("list of"):
                score -= 10
            if "index of" in t_lower or "outline of" in t_lower:
                score -= 8

            # Reward number or spelled-out ordinal match
            if num_str and num_str in t_lower:
                score += 15
            if word_form and word_form in t_lower:
                score += 20

            # Reward subject keyword overlap
            subj_tokens = [w for w in re.split(r"\W+", subj_lower) if len(w) > 2]
            for tok in subj_tokens:
                if tok in t_lower:
                    score += 3

            # Exact or close title match
            if t_lower == subj_lower:
                score += 25

            return score

        best_item = max(search_results, key=score_candidate)
        return best_item.get("title")

    def fetch_wikipedia_knowledge(self, subject: str) -> Optional[ExternalKnowledgeResult]:
        """Fetch verified factual background information from Wikipedia REST API.

        Free, reliable, open encyclopedic reference without introducing paid dependencies.
        """
        client = self._get_client()
        try:
            # Step 1: Search Wikipedia for relevant article title
            search_url = "https://en.wikipedia.org/w/api.php"
            search_params = {
                "action": "query",
                "list": "search",
                "srsearch": subject,
                "utf8": "1",
                "format": "json",
                "srlimit": "5",
            }
            resp = client.get(search_url, params=search_params)
            if resp.status_code != 200:
                logger.warning(f"Wikipedia search returned status {resp.status_code} for '{subject}'")
                return None

            search_data = resp.json()
            search_results = search_data.get("query", {}).get("search", [])
            if not search_results:
                logger.info(f"Wikipedia returned zero search results for '{subject}'")
                return None

            best_title = self._rank_best_wikipedia_title(subject, search_results)
            if not best_title:
                best_title = search_results[0].get("title")
            if not best_title:
                return None

            # Step 2: Fetch article summary from Wikipedia REST API
            title_encoded = urllib.parse.quote(best_title.replace(" ", "_"), safe="")
            summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title_encoded}"
            summary_resp = client.get(summary_url)
            if summary_resp.status_code != 200:
                logger.warning(f"Wikipedia summary returned status {summary_resp.status_code} for '{best_title}'")
                return None

            summary_data = summary_resp.json()
            extract = summary_data.get("extract", "").strip()
            if not extract:
                return None

            canonical_url = (
                summary_data.get("content_urls", {}).get("desktop", {}).get("page")
                or f"https://en.wikipedia.org/wiki/{title_encoded}"
            )
            page_title = summary_data.get("title", best_title)

            # Derive key facts from sentences in extract
            raw_sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", extract) if len(s.strip()) > 15]
            key_facts = raw_sentences[:4] if raw_sentences else [extract]

            citation = ExternalKnowledgeCitation(
                source_title=f"Wikipedia: {page_title}",
                source_url=canonical_url,
                snippet=extract[:250] + ("..." if len(extract) > 250 else ""),
                published_date=summary_data.get("timestamp"),
            )

            return ExternalKnowledgeResult(
                subject=subject,
                summary=extract,
                key_facts=key_facts,
                citations=[citation],
                provider_used="wikipedia",
                is_available=True,
                unavailability_reason=None,
                disclaimer="External background knowledge retrieved for context. This information was not discussed during your organization's meetings.",
            )

        except Exception as exc:
            logger.warning(f"Wikipedia retrieval exception for '{subject}': {exc}")
            return None
        finally:
            if self._client is None:
                client.close()

    def fetch_gemini_knowledge(self, subject: str) -> Optional[ExternalKnowledgeResult]:
        """Fetch external encyclopedic knowledge using existing Gemini configuration if available."""
        api_key = getattr(settings, "GEMINI_API_KEY", "")
        if not api_key or not getattr(settings, "GEMINI_ENABLED", False):
            return None

        model_name = getattr(settings, "GEMINI_MODEL", "gemini-3.8-flash").replace("models/", "")
        endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"

        prompt = (
            f"Provide an authoritative, neutral background explanation of the following public subject: '{subject}'.\n"
            "Respond ONLY with a valid JSON object matching this schema:\n"
            "{\n"
            '  "summary": "2-3 sentences explaining what this subject is in clear, simple language.",\n'
            '  "key_facts": ["Key fact 1", "Key fact 2", "Key fact 3"],\n'
            '  "source_title": "Official name of the standard, law, or authority",\n'
            '  "source_url": "Authoritative public reference URL"\n'
            "}\n"
            "Do not include any proprietary, conversational, or meeting details."
        )

        request_body = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.1,
                "response_mime_type": "application/json",
            },
        }

        client = self._get_client()
        try:
            resp = client.post(endpoint, json=request_body, headers={"x-goog-api-key": api_key})
            if resp.status_code != 200:
                logger.warning(f"Gemini knowledge retrieval returned {resp.status_code} for '{subject}'")
                return None

            data = resp.json()
            candidates = data.get("candidates", [])
            if not candidates:
                return None

            text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
            if not text:
                return None

            # Parse JSON
            parsed = json.loads(text)
            summary = parsed.get("summary", "").strip()
            key_facts = parsed.get("key_facts", [])
            source_title = parsed.get("source_title", f"Authoritative Reference for {subject}")
            source_url = parsed.get("source_url", "https://en.wikipedia.org/wiki/" + urllib.parse.quote(subject))

            if not summary:
                return None

            citation = ExternalKnowledgeCitation(
                source_title=source_title,
                source_url=source_url,
                snippet=summary[:250],
            )

            return ExternalKnowledgeResult(
                subject=subject,
                summary=summary,
                key_facts=key_facts,
                citations=[citation],
                provider_used="gemini",
                is_available=True,
                unavailability_reason=None,
                disclaimer="External background knowledge retrieved for context via Gemini. This information was not discussed during your organization's meetings.",
            )

        except Exception as exc:
            logger.warning(f"Gemini knowledge retrieval exception for '{subject}': {exc}")
            return None
        finally:
            if self._client is None:
                client.close()

    def fetch_duckduckgo_knowledge(self, subject: str) -> Optional[ExternalKnowledgeResult]:
        """Fetch instant answer from DuckDuckGo API as a secondary fallback."""
        client = self._get_client()
        try:
            url = "https://api.duckduckgo.com/"
            params = {
                "q": subject,
                "format": "json",
                "no_html": "1",
                "skip_disambig": "1",
            }
            resp = client.get(url, params=params)
            if resp.status_code != 200:
                return None

            data = resp.json()
            abstract = data.get("AbstractText", "").strip()
            source_url = data.get("AbstractURL", "").strip()
            source_title = data.get("AbstractSource", "DuckDuckGo Instant Answer")

            if not abstract:
                # Try definition if abstract is empty
                abstract = data.get("Definition", "").strip()
                source_url = data.get("DefinitionURL", "").strip()

            if not abstract:
                return None

            citation = ExternalKnowledgeCitation(
                source_title=source_title,
                source_url=source_url or f"https://duckduckgo.com/?q={urllib.parse.quote(subject)}",
                snippet=abstract[:250],
            )

            raw_sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", abstract) if len(s.strip()) > 15]

            return ExternalKnowledgeResult(
                subject=subject,
                summary=abstract,
                key_facts=raw_sentences[:3] if raw_sentences else [abstract],
                citations=[citation],
                provider_used="duckduckgo",
                is_available=True,
                unavailability_reason=None,
                disclaimer="External background knowledge retrieved for context. This information was not discussed during your organization's meetings.",
            )

        except Exception as exc:
            logger.warning(f"DuckDuckGo knowledge retrieval exception for '{subject}': {exc}")
            return None
        finally:
            if self._client is None:
                client.close()

    def retrieve_external_knowledge(
        self,
        subject: str,
        prefer_gemini: bool = False,
    ) -> ExternalKnowledgeResult:
        """Orchestrate multi-tiered external knowledge retrieval with guaranteed honest fallback."""
        # 1. If Gemini configured and preferred, attempt Gemini
        if prefer_gemini:
            res = self.fetch_gemini_knowledge(subject)
            if res:
                return res

        # 2. Wikipedia (Primary reliable, open, free encyclopedic source)
        res = self.fetch_wikipedia_knowledge(subject)
        if res:
            return res

        # 3. DuckDuckGo Instant Answer Fallback
        res = self.fetch_duckduckgo_knowledge(subject)
        if res:
            return res

        # 4. Try Gemini if not preferred initially
        if not prefer_gemini:
            res = self.fetch_gemini_knowledge(subject)
            if res:
                return res

        # 5. Honest unavailability result (never hallucinate)
        return ExternalKnowledgeResult(
            subject=subject,
            summary=f"External background knowledge for '{subject}' could not be retrieved at this time.",
            key_facts=[],
            citations=[],
            provider_used="unavailable",
            is_available=False,
            unavailability_reason="External knowledge sources are currently unreachable or offline.",
            disclaimer="External background knowledge was requested for context, but external retrieval was unavailable.",
        )


_default_service: Optional[ExternalKnowledgeService] = None


def get_external_knowledge_service() -> ExternalKnowledgeService:
    """Singleton getter for ExternalKnowledgeService."""
    global _default_service
    if _default_service is None:
        _default_service = ExternalKnowledgeService()
    return _default_service
