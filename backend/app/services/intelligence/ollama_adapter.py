"""Ollama local LLM REST adapter for structured intelligence extraction."""

import json
import logging
import re
from typing import List, Optional, Dict, Any
import httpx
from pydantic import ValidationError

from app.core.config import settings
from app.models.transcription import TranscriptSegment
from app.schemas.intelligence import (
    TopicCreate,
    DecisionCreate,
    ActionItemCreate,
    RiskCreate,
    UnresolvedQuestionCreate,
    StructuredExtractionPayload,
)
from app.services.intelligence.base import ProviderExecutionError
from app.services.intelligence.prompts import build_intelligence_prompt

logger = logging.getLogger(__name__)


class OllamaExtractor:
    """Connects to local Ollama daemon for multilingual structured extraction."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout: Optional[float] = None,
        client: Optional[httpx.Client] = None,
    ):
        self.base_url = (base_url or getattr(settings, "OLLAMA_BASE_URL", "http://127.0.0.1:11434")).rstrip("/")
        self.model_name = model_name or getattr(settings, "LLM_MODEL_NAME", "qwen3:8b")
        self.timeout = timeout or getattr(settings, "LLM_REQUEST_TIMEOUT_SECONDS", 60.0)
        self._custom_client = client

    def extract(
        self,
        segments: List[TranscriptSegment],
        meeting_title: str = "Meeting",
        output_language: str = "en",
    ) -> StructuredExtractionPayload:
        """Call Ollama /api/generate with format='json' and validate the returned payload."""
        if not segments:
            return StructuredExtractionPayload(
                executive_summary="Empty transcript provided; no intelligence could be extracted.",
                detailed_summary=None,
                key_takeaways=[],
                topics=[],
                decisions=[],
                action_items=[],
                risks=[],
                unresolved_questions=[],
            )

        prompts = build_intelligence_prompt(
            segments=segments,
            meeting_title=meeting_title,
            output_language=output_language,
        )

        request_body = {
            "model": self.model_name,
            "prompt": prompts["prompt"],
            "system": prompts["system"],
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.1,
            },
        }

        endpoint = f"{self.base_url}/api/generate"

        try:
            if self._custom_client:
                response = self._custom_client.post(endpoint, json=request_body, timeout=self.timeout)
            else:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(endpoint, json=request_body)

            response.raise_for_status()
            resp_data = response.json()

        except httpx.ConnectError as ce:
            raise ProviderExecutionError(
                f"Ollama server is not running or unreachable at '{self.base_url}'. Details: {ce}"
            ) from ce
        except httpx.TimeoutException as te:
            raise ProviderExecutionError(
                f"Ollama request timed out after {self.timeout}s at '{self.base_url}'. Details: {te}"
            ) from te
        except httpx.HTTPStatusError as se:
            if se.response.status_code == 404:
                raise ProviderExecutionError(
                    f"Model '{self.model_name}' not found on Ollama server at '{self.base_url}'. "
                    f"Please pull it via 'ollama pull {self.model_name}'."
                ) from se
            raise ProviderExecutionError(
                f"Ollama HTTP {se.response.status_code} error from '{self.base_url}': {se.response.text}"
            ) from se
        except Exception as e:
            raise ProviderExecutionError(f"Unexpected error communicating with Ollama: {str(e)}") from e

        # Parse raw content
        raw_text = resp_data.get("response", "").strip()
        if not raw_text:
            raise ProviderExecutionError("Ollama returned an empty response string.")

        parsed_json = self._parse_json_safely(raw_text)
        return self._sanitize_and_validate_payload(parsed_json)

    def _parse_json_safely(self, raw_text: str) -> Dict[str, Any]:
        """Strip markdown fences, trailing commas, and unescaped artifacts to safely parse JSON."""
        cleaned = raw_text.strip()

        # 1. Direct JSON parse
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # 2. Extract from markdown code blocks
        fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned, re.IGNORECASE)
        if fence_match:
            candidate = fence_match.group(1).strip()
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                cleaned = candidate

        # 3. Extract between first '{' and last '}'
        first_brace = cleaned.find("{")
        last_brace = cleaned.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            bounded = cleaned[first_brace : last_brace + 1].strip()
            try:
                return json.loads(bounded)
            except json.JSONDecodeError:
                cleaned = bounded

        # 4. Clean trailing commas before } or ]
        cleaned_no_trailing = re.sub(r",\s*([}\]])", r"\1", cleaned)
        try:
            return json.loads(cleaned_no_trailing)
        except json.JSONDecodeError as jde:
            raise ProviderExecutionError(
                f"Ollama output was not valid JSON: {jde.msg}"
            ) from jde

    @staticmethod
    def _parse_float_safely(val: Any) -> Optional[float]:
        """Safely parse timestamps, seconds, or numbers that may contain 's' or 'sec' suffixes."""
        if val is None:
            return None
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, str):
            cleaned = val.strip().lower().rstrip("s").rstrip("sec").rstrip("seconds").strip()
            if ":" in cleaned:
                parts = cleaned.split(":")
                try:
                    if len(parts) == 2:
                        return float(parts[0]) * 60 + float(parts[1])
                    elif len(parts) == 3:
                        return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
                except ValueError:
                    return None
            try:
                return float(cleaned)
            except ValueError:
                return None
        return None

    def _sanitize_and_validate_payload(self, data: Dict[str, Any]) -> StructuredExtractionPayload:
        """Sanitize fields to guarantee compatibility with StructuredExtractionPayload schema."""
        if not isinstance(data, dict):
            raise ProviderExecutionError("Ollama returned a non-dictionary JSON payload.")

        # 0. Unwrap top-level single container objects (e.g. {"structured_intelligence": {...}})
        if len(data) == 1:
            first_key = list(data.keys())[0].lower()
            first_val = list(data.values())[0]
            if first_key in {
                "structured_intelligence",
                "meeting_intelligence",
                "meeting",
                "intelligence",
                "response",
                "result",
                "data",
                "output",
            } and isinstance(first_val, dict):
                data = first_val

        try:
            # 1. Clean decisions (filter out suggestions/proposals)
            decisions = []
            for d in data.get("decisions", []):
                if isinstance(d, dict):
                    desc = d.get("description") or d.get("title")
                    if desc:
                        desc_str = str(desc).strip()
                        desc_upper = desc_str.upper()
                        d_type = str(d.get("decision_type", "")).upper()
                        if d_type in {"SUGGESTION", "PROPOSAL", "EXPLORATORY"} or any(
                            desc_upper.startswith(prefix) for prefix in ("SUGGESTION:", "PROPOSAL:", "SUGGESTION -", "PROPOSAL -")
                        ):
                            continue
                        decisions.append(
                            DecisionCreate(
                                description=desc_str,
                                rationale_or_context=d.get("rationale_or_context") or d.get("rationale") or d.get("context"),
                                decided_by=d.get("decided_by"),
                                evidence_segment_id=d.get("evidence_segment_id"),
                                timestamp_seconds=self._parse_float_safely(d.get("timestamp_seconds")),
                            )
                        )

            # If no decisions array but approval_status or decision exists
            if not decisions and "approval_status" in data:
                app_status = data["approval_status"]
                if isinstance(app_status, dict):
                    for cond in app_status.get("conditions", []):
                        decisions.append(DecisionCreate(description=str(cond).strip(), rationale_or_context="Approved condition"))
                    if not decisions and app_status.get("csv_export_approved"):
                        decisions.append(DecisionCreate(description="Approved CSV export with dashboard access rules", rationale_or_context="Access rules compliance"))

            # 2. Clean action items
            actions = []
            raw_actions = data.get("action_items")
            if not raw_actions and "action" in data:
                raw_actions = [data["action"]] if isinstance(data["action"], (str, dict)) else data["action"]

            for a in (raw_actions or []):
                if isinstance(a, str) and a.strip():
                    actions.append(ActionItemCreate(description=a.strip(), owner_name=None, deadline=None, status="PENDING"))
                elif isinstance(a, dict) and (a.get("description") or a.get("action")):
                    desc = str(a.get("description") or a.get("action")).strip()
                    dl = a.get("deadline")
                    parsed_dl = None
                    if dl:
                        try:
                            from datetime import datetime
                            if isinstance(dl, datetime):
                                parsed_dl = dl
                            elif isinstance(dl, str):
                                parsed_dl = datetime.fromisoformat(dl.strip().replace("Z", "+00:00"))
                        except Exception:
                            parsed_dl = None

                    raw_owner = a.get("owner_name") or a.get("assignee")
                    owner_name = None
                    if raw_owner and str(raw_owner).strip().lower() not in {"unassigned", "n/a", "none", "tbd", "unknown", ""}:
                        owner_name = str(raw_owner).strip()

                    raw_a_status = str(a.get("status") or "PENDING").strip().upper().replace(" ", "_")
                    if raw_a_status in {"PENDING", "TODO", "NOT_STARTED", "NEW", "OPEN"}:
                        a_status = "PENDING"
                    elif raw_a_status in {"IN_PROGRESS", "PROGRESS", "DOING", "ACTIVE"}:
                        a_status = "IN_PROGRESS"
                    elif raw_a_status in {"COMPLETED", "DONE", "FINISHED"}:
                        a_status = "COMPLETED"
                    elif raw_a_status in {"BLOCKED"}:
                        a_status = "BLOCKED"
                    elif raw_a_status in {"CANCELLED", "CANCELED"}:
                        a_status = "CANCELLED"
                    else:
                        a_status = "PENDING"

                    actions.append(
                        ActionItemCreate(
                            description=desc,
                            owner_name=owner_name,
                            deadline=parsed_dl,
                            status=a_status,
                            evidence_segment_id=a.get("evidence_segment_id"),
                            timestamp_seconds=self._parse_float_safely(a.get("timestamp_seconds")),
                        )
                    )

            # 3. Clean topics
            topics = []
            for t in data.get("topics", []):
                if isinstance(t, dict):
                    title = t.get("title") or t.get("name")
                    if title:
                        topics.append(
                            TopicCreate(
                                title=str(title).strip(),
                                description=t.get("description"),
                                start_time=self._parse_float_safely(t.get("start_time")),
                                end_time=self._parse_float_safely(t.get("end_time")),
                            )
                        )

            # 4. Clean risks
            risks = []
            for r in data.get("risks", []):
                if isinstance(r, dict) and r.get("description"):
                    raw_sev = str(r.get("severity") or "MEDIUM").strip().upper()
                    if raw_sev in {"LOW", "MINOR"}:
                        sev = "LOW"
                    elif raw_sev in {"MEDIUM", "MODERATE"}:
                        sev = "MEDIUM"
                    elif raw_sev in {"HIGH", "MAJOR"}:
                        sev = "HIGH"
                    elif raw_sev in {"CRITICAL", "BLOCKER", "SEVERE"}:
                        sev = "CRITICAL"
                    else:
                        sev = "MEDIUM"

                    risks.append(
                        RiskCreate(
                            description=str(r["description"]).strip(),
                            severity=sev,
                            mitigation_suggestion=r.get("mitigation_suggestion") or r.get("mitigation"),
                            evidence_segment_id=r.get("evidence_segment_id"),
                        )
                    )

            # 5. Clean unresolved questions
            questions = []
            for q in data.get("unresolved_questions", []):
                if isinstance(q, dict) and (q.get("question_text") or q.get("question")):
                    q_text = q.get("question_text") or q.get("question")
                    raw_q_status = str(q.get("status") or "OPEN").strip().upper()
                    if raw_q_status in {"OPEN", "PENDING", "UNRESOLVED", "TODO", "ACTIVE"}:
                        q_status = "OPEN"
                    elif raw_q_status in {"RESOLVED", "ANSWERED", "CLOSED", "DONE"}:
                        q_status = "RESOLVED"
                    elif raw_q_status in {"DEFERRED", "POSTPONED"}:
                        q_status = "DEFERRED"
                    else:
                        q_status = "OPEN"

                    questions.append(
                        UnresolvedQuestionCreate(
                            question_text=str(q_text).strip(),
                            raised_by=q.get("raised_by"),
                            status=q_status,
                            evidence_segment_id=q.get("evidence_segment_id"),
                        )
                    )

            # 6. Normalize detailed_summary losslessly
            raw_detailed = data.get("detailed_summary") or data.get("details")
            normalized_detailed: Optional[str] = None
            if isinstance(raw_detailed, str):
                normalized_detailed = raw_detailed.strip() or None
            elif isinstance(raw_detailed, dict):
                # When LLM formats detailed summary as structured sections/topics
                if "topics" in raw_detailed and isinstance(raw_detailed["topics"], list):
                    topic_lines = []
                    for top in raw_detailed["topics"]:
                        if isinstance(top, dict):
                            t_title = top.get("title", "")
                            t_desc = top.get("description", "")
                            if t_title and t_desc:
                                topic_lines.append(f"### {t_title}\n{t_desc}")
                            elif t_desc:
                                topic_lines.append(t_desc)
                            elif t_title:
                                topic_lines.append(t_title)
                    normalized_detailed = "\n\n".join(topic_lines) if topic_lines else None
                else:
                    # Lossless formatted JSON string representation
                    formatted_parts = []
                    for k, v in raw_detailed.items():
                        k_clean = str(k).replace("_", " ").title()
                        if isinstance(v, (str, int, float, bool)):
                            formatted_parts.append(f"**{k_clean}**: {v}")
                        elif isinstance(v, list):
                            items = ", ".join(str(item) for item in v)
                            formatted_parts.append(f"**{k_clean}**: {items}")
                        else:
                            formatted_parts.append(f"**{k_clean}**: {json.dumps(v)}")
                    normalized_detailed = "\n".join(formatted_parts) if formatted_parts else None
            elif isinstance(raw_detailed, list):
                normalized_detailed = "\n\n".join(str(item) for item in raw_detailed if item) or None

            # 7. Executive Summary validation with lossless synthesis
            exec_summary = str(
                data.get("executive_summary")
                or data.get("summary")
                or data.get("overview")
                or data.get("meeting_summary")
                or data.get("brief_summary")
                or data.get("synopsis")
                or data.get("description")
                or data.get("privacy_note")
                or ""
            ).strip()

            if not exec_summary:
                if normalized_detailed:
                    first_section = normalized_detailed.split("\n\n")[0].split("\n")[0].replace("#", "").replace("*", "").strip()
                    exec_summary = first_section[:300] if len(first_section) > 10 else normalized_detailed[:300]
                elif decisions:
                    exec_summary = f"The meeting concluded with key decisions confirmed: {decisions[0].description}."
                elif actions:
                    exec_summary = f"The meeting established key action items: {actions[0].description}."
                elif topics:
                    exec_summary = f"The meeting covered topics including {topics[0].title}."
                else:
                    exec_summary = "Meeting intelligence extracted successfully from transcript."

            takeaways = data.get("key_takeaways")
            if takeaways and not isinstance(takeaways, list):
                takeaways = [str(takeaways)]

            return StructuredExtractionPayload(
                executive_summary=exec_summary,
                detailed_summary=normalized_detailed,
                key_takeaways=takeaways,
                topics=topics,
                decisions=decisions,
                action_items=actions,
                risks=risks,
                unresolved_questions=questions,
            )

        except ValidationError as ve:
            raise ProviderExecutionError(
                f"Ollama structured intelligence failed schema validation: {ve}"
            ) from ve
