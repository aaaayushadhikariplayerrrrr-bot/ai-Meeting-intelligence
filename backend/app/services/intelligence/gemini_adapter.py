"""Google Gemini REST API adapter for structured meeting intelligence extraction."""

import json
import logging
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
from app.services.intelligence.base import ProviderExecutionError, ProviderType
from app.services.intelligence.prompts import build_intelligence_prompt, SYSTEM_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)

# Fallback generation models if configured model is unavailable (e.g. 503 overload, timeout, or 404)
FALLBACK_GEMINI_MODELS = [
    "gemini-3.1-flash-lite",
    "gemini-flash-latest",
    "gemini-flash-lite-latest",
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
]


class GeminiExtractor:
    """Connects to Google Gemini REST API for multilingual structured intelligence extraction."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout: Optional[float] = None,
        client: Optional[httpx.Client] = None,
    ):
        self.api_key = api_key or getattr(settings, "GEMINI_API_KEY", "")
        self.model_name = (model_name or getattr(settings, "GEMINI_MODEL", "gemini-3.8-flash")).replace("models/", "")
        self.timeout = timeout or getattr(settings, "GEMINI_REQUEST_TIMEOUT_SECONDS", 30.0)
        self.provider_type = ProviderType.GEMINI

        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key,
        }
        if client is not None:
            self.client = client
            if self.api_key and "x-goog-api-key" not in self.client.headers:
                self.client.headers["x-goog-api-key"] = self.api_key
        else:
            self.client = httpx.Client(timeout=self.timeout, headers=headers)
        self._custom_client = self.client

    def __repr__(self) -> str:
        return f"<GeminiExtractor(model={self.model_name}, provider={self.provider_type.value})>"

    def __str__(self) -> str:
        return f"GeminiExtractor(model={self.model_name})"

    def extract(
        self,
        segments: List[TranscriptSegment],
        meeting_title: str = "Meeting",
        output_language: str = "en",
    ) -> StructuredExtractionPayload:
        """Call Gemini generateContent with application/json schema and validate returned intelligence."""
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

        if not self.api_key or not self.api_key.strip():
            raise ProviderExecutionError(
                "Gemini API key is not configured. Please configure GEMINI_API_KEY in the backend environment."
            )

        prompts = build_intelligence_prompt(
            segments=segments,
            meeting_title=meeting_title,
            output_language=output_language,
        )

        request_body = {
            "system_instruction": {
                "parts": [{"text": prompts["system"]}]
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompts["prompt"]}]
                }
            ],
            "generationConfig": {
                "temperature": 0.1,
                "response_mime_type": "application/json",
            },
        }

        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key,
        }

        models_to_try = [self.model_name]
        for fb in FALLBACK_GEMINI_MODELS:
            if fb != self.model_name:
                models_to_try.append(fb)

        last_error: Optional[Exception] = None

        for idx, model_to_use in enumerate(models_to_try):
            endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model_to_use}:generateContent"
            try:
                response = self.client.post(
                    endpoint, headers=headers, json=request_body, timeout=self.timeout
                )

                if response.status_code in (404, 503) and idx < len(models_to_try) - 1:
                    logger.warning(
                        f"Gemini model '{model_to_use}' returned HTTP {response.status_code}. "
                        f"Attempting fallback model '{models_to_try[idx + 1]}'."
                    )
                    continue

                response.raise_for_status()
                resp_data = response.json()

                candidates = resp_data.get("candidates", [])
                if not candidates:
                    raise ProviderExecutionError("Gemini returned zero response candidates.")

                parts = candidates[0].get("content", {}).get("parts", [])
                if not parts or "text" not in parts[0]:
                    raise ProviderExecutionError("Gemini returned empty content parts.")

                raw_text = parts[0]["text"].strip()
                parsed_json = self._parse_json_safely(raw_text)

                return self._sanitize_and_validate_payload(parsed_json)

            except httpx.ConnectError as ce:
                last_error = ce
                if idx < len(models_to_try) - 1:
                    continue
                raise ProviderExecutionError(
                    f"Network error communicating with Google Gemini API: {type(ce).__name__}"
                ) from ce
            except httpx.TimeoutException as te:
                if idx < len(models_to_try) - 1:
                    logger.warning(f"Gemini request for model '{model_to_use}' timed out. Trying fallback.")
                    continue
                raise ProviderExecutionError(
                    f"Google Gemini API request timed out after {self.timeout}s."
                ) from te
            except httpx.HTTPStatusError as se:
                last_error = se
                status_code = se.response.status_code
                if status_code in (401, 403):
                    raise ProviderExecutionError(
                        "Gemini API authentication failed. Invalid, unauthorized, or suspended API key."
                    ) from se
                elif status_code == 429:
                    raise ProviderExecutionError(
                        "Google Gemini API quota or rate limit exceeded. Please check quota limits."
                    ) from se
                elif status_code in (404, 503) and idx < len(models_to_try) - 1:
                    continue
                else:
                    raise ProviderExecutionError(
                        f"Google Gemini API HTTP {status_code} error: {se.response.text[:200]}"
                    ) from se
            except ProviderExecutionError:
                raise
            except Exception as e:
                last_error = e
                logger.warning(f"Error during Gemini extraction with model '{model_to_use}': {e}")
                if idx < len(models_to_try) - 1:
                    continue
                raise ProviderExecutionError(
                    f"Unexpected error communicating with Gemini API: {str(e)}"
                ) from e

        if last_error:
            raise ProviderExecutionError(f"Gemini extraction failed across all models: {last_error}")

        raise ProviderExecutionError("Gemini extraction failed to produce a valid response.")

    def extract_payload(
        self,
        segments_text: str,
        output_language: str = "en",
        meeting_title: str = "Meeting",
    ) -> StructuredExtractionPayload:
        """Call Gemini API with formatted transcript text and validate returned structured intelligence."""
        if not segments_text or not segments_text.strip():
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

        if not self.api_key or not self.api_key.strip():
            raise ProviderExecutionError(
                "Gemini API key is not configured. Please configure GEMINI_API_KEY in the backend environment."
            )

        system_instruction = SYSTEM_PROMPT_TEMPLATE.format(output_language=output_language)
        user_prompt = (
            f"Meeting Title: {meeting_title}\n\n"
            f"--- PRIVACY-SAFE TRANSCRIPT BEGIN ---\n"
            f"{segments_text}\n"
            f"--- PRIVACY-SAFE TRANSCRIPT END ---\n\n"
            "Analyze the transcript and produce the required JSON intelligence extraction."
        )

        request_body = {
            "system_instruction": {
                "parts": [{"text": system_instruction}]
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": user_prompt}]
                }
            ],
            "generationConfig": {
                "temperature": 0.1,
                "response_mime_type": "application/json",
            },
        }

        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key,
        }

        models_to_try = [self.model_name]
        for fb in FALLBACK_GEMINI_MODELS:
            if fb != self.model_name:
                models_to_try.append(fb)

        last_error: Optional[Exception] = None

        for idx, model_to_use in enumerate(models_to_try):
            endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model_to_use}:generateContent"
            try:
                response = self.client.post(
                    endpoint, headers=headers, json=request_body, timeout=self.timeout
                )

                if response.status_code in (404, 503) and idx < len(models_to_try) - 1:
                    logger.warning(
                        f"Gemini model '{model_to_use}' returned HTTP {response.status_code}. "
                        f"Attempting fallback model '{models_to_try[idx + 1]}'."
                    )
                    continue

                response.raise_for_status()
                resp_data = response.json()

                candidates = resp_data.get("candidates", [])
                if not candidates:
                    raise ProviderExecutionError("Gemini returned zero response candidates.")

                parts = candidates[0].get("content", {}).get("parts", [])
                if not parts or "text" not in parts[0]:
                    raise ProviderExecutionError("Gemini returned empty content parts.")

                raw_text = parts[0]["text"].strip()
                parsed_json = self._parse_json_safely(raw_text)

                return self._sanitize_and_validate_payload(parsed_json)

            except httpx.ConnectError as ce:
                last_error = ce
                if idx < len(models_to_try) - 1:
                    continue
                raise ProviderExecutionError(
                    f"Network error communicating with Google Gemini API: {type(ce).__name__}"
                ) from ce
            except httpx.TimeoutException as te:
                if idx < len(models_to_try) - 1:
                    logger.warning(f"Gemini request for model '{model_to_use}' timed out. Trying fallback.")
                    continue
                raise ProviderExecutionError(
                    f"Google Gemini API request timed out after {self.timeout}s."
                ) from te
            except httpx.HTTPStatusError as se:
                last_error = se
                status_code = se.response.status_code
                if status_code in (401, 403):
                    raise ProviderExecutionError(
                        "Gemini API authentication failed. Invalid, unauthorized, or suspended API key."
                    ) from se
                elif status_code == 429:
                    raise ProviderExecutionError(
                        "Google Gemini API quota or rate limit exceeded. Please check quota limits."
                    ) from se
                elif status_code in (404, 503) and idx < len(models_to_try) - 1:
                    continue
                else:
                    raise ProviderExecutionError(
                        f"Google Gemini API HTTP {status_code} error: {se.response.text[:200]}"
                    ) from se
            except ProviderExecutionError:
                raise
            except Exception as e:
                last_error = e
                logger.warning(f"Error during Gemini extraction with model '{model_to_use}': {e}")
                if idx < len(models_to_try) - 1:
                    continue
                raise ProviderExecutionError(
                    f"Unexpected error communicating with Gemini API: {str(e)}"
                ) from e

        if last_error:
            raise ProviderExecutionError(f"Gemini extraction failed across all models: {last_error}")

        raise ProviderExecutionError("Gemini extraction failed to produce a valid response.")

    def _parse_json_safely(self, raw_text: str) -> Dict[str, Any]:
        """Strip markdown fences if present and parse json."""
        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as jde:
            raise ProviderExecutionError(
                f"Gemini output could not be parsed as valid JSON: {jde.msg}"
            ) from jde

    def _sanitize_and_validate_payload(self, data: Dict[str, Any]) -> StructuredExtractionPayload:
        """Sanitize fields to guarantee compatibility with Pydantic schemas."""
        try:
            # 1. Clean decisions (filter out suggestions/proposals)
            decisions = []
            for d in data.get("decisions", []):
                if isinstance(d, dict):
                    desc = d.get("description") or d.get("title")
                    if desc:
                        d_type = str(d.get("decision_type", "")).upper()
                        if d_type in {"SUGGESTION", "PROPOSAL", "EXPLORATORY"}:
                            continue
                        decisions.append(
                            DecisionCreate(
                                description=str(desc).strip(),
                                rationale_or_context=d.get("rationale_or_context") or d.get("rationale") or d.get("context"),
                                decided_by=d.get("decided_by"),
                                evidence_segment_id=d.get("evidence_segment_id"),
                                timestamp_seconds=d.get("timestamp_seconds"),
                            )
                        )

            # 2. Clean action items
            actions = []
            for a in data.get("action_items", []):
                if isinstance(a, dict) and a.get("description"):
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
                    actions.append(
                        ActionItemCreate(
                            description=str(a["description"]).strip(),
                            owner_name=a.get("owner_name") or a.get("assignee"),
                            deadline=parsed_dl,
                            status=a.get("status", "PENDING"),
                            evidence_segment_id=a.get("evidence_segment_id"),
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
                                start_time=t.get("start_time"),
                                end_time=t.get("end_time"),
                            )
                        )

            # 4. Clean risks
            risks = []
            for r in data.get("risks", []):
                if isinstance(r, dict) and r.get("description"):
                    risks.append(
                        RiskCreate(
                            description=str(r["description"]).strip(),
                            severity=r.get("severity", "MEDIUM"),
                            mitigation_suggestion=r.get("mitigation_suggestion") or r.get("mitigation"),
                            evidence_segment_id=r.get("evidence_segment_id"),
                        )
                    )

            # 5. Clean unresolved questions
            questions = []
            for q in data.get("unresolved_questions", []):
                if isinstance(q, dict) and (q.get("question_text") or q.get("question")):
                    q_text = q.get("question_text") or q.get("question")
                    questions.append(
                        UnresolvedQuestionCreate(
                            question_text=str(q_text).strip(),
                            raised_by=q.get("raised_by"),
                            status=q.get("status", "OPEN"),
                            evidence_segment_id=q.get("evidence_segment_id"),
                        )
                    )

            # 6. Assemble Payload
            exec_summary = str(data.get("executive_summary") or data.get("summary") or "").strip()
            if not exec_summary:
                exec_summary = "Meeting intelligence successfully extracted."

            takeaways = data.get("key_takeaways")
            if takeaways and not isinstance(takeaways, list):
                takeaways = [str(takeaways)]

            return StructuredExtractionPayload(
                executive_summary=exec_summary,
                detailed_summary=data.get("detailed_summary"),
                key_takeaways=takeaways,
                topics=topics,
                decisions=decisions,
                action_items=actions,
                risks=risks,
                unresolved_questions=questions,
            )

        except ValidationError as ve:
            raise ProviderExecutionError(
                f"Gemini structured intelligence failed schema validation: {ve}"
            ) from ve
