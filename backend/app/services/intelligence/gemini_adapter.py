"""Google Gemini REST API adapter for structured meeting intelligence extraction."""

import json
import logging
import random
import re
import time
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

# Verified Flash fallback generation models prioritized for low latency and high reliability
FALLBACK_GEMINI_MODELS = [
    "gemini-flash-lite-latest",
    "gemini-3.5-flash-lite",
    "gemini-3.6-flash",
    "gemini-3.1-flash-lite",
    "gemini-flash-latest",
]


class GeminiExtractor:
    """Connects to Google Gemini REST API for multilingual structured intelligence extraction."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
        client: Optional[httpx.Client] = None,
    ):
        self.api_key = api_key or getattr(settings, "GEMINI_API_KEY", "")
        self.model_name = (model_name or getattr(settings, "GEMINI_MODEL", "gemini-3.8-flash")).replace("models/", "")
        self.timeout = float(timeout or getattr(settings, "GEMINI_REQUEST_TIMEOUT_SECONDS", 120.0))
        self.max_retries = int(max_retries if max_retries is not None else getattr(settings, "GEMINI_MAX_RETRIES", 2))
        self.provider_type = ProviderType.GEMINI

        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key,
        }

        # Granular timeout: allow generous read timeout while failing fast on unreachable connect
        connect_timeout = min(15.0, self.timeout)
        self._httpx_timeout = httpx.Timeout(
            timeout=self.timeout,
            connect=connect_timeout,
            read=self.timeout,
            write=30.0,
        )

        if client is not None:
            self.client = client
            if self.api_key and "x-goog-api-key" not in self.client.headers:
                self.client.headers["x-goog-api-key"] = self.api_key
        else:
            self.client = httpx.Client(timeout=self._httpx_timeout, headers=headers)
        self._custom_client = self.client

    def __repr__(self) -> str:
        return f"<GeminiExtractor(model={self.model_name}, provider={self.provider_type.value}, timeout={self.timeout}s)>"

    def __str__(self) -> str:
        return f"GeminiExtractor(model={self.model_name})"

    def _send_generate_content_request(
        self,
        request_body: Dict[str, Any],
        context_description: str = "meeting transcript",
        char_count: int = 0,
        segment_count: int = 0,
    ) -> StructuredExtractionPayload:
        """Execute request across primary and fallback models with exponential backoff on transient errors."""
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key,
        }

        models_to_try = [self.model_name]
        for fb in FALLBACK_GEMINI_MODELS:
            if fb != self.model_name and fb not in models_to_try:
                models_to_try.append(fb)

        last_error: Optional[Exception] = None
        total_retries = 0

        for idx, model_to_use in enumerate(models_to_try):
            endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model_to_use}:generateContent"
            has_next_model = (idx < len(models_to_try) - 1)
            next_model_name = models_to_try[idx + 1] if has_next_model else None

            # Retry loop per model
            for attempt in range(self.max_retries + 1):
                t0 = time.time()
                is_last_attempt = (attempt == self.max_retries)

                try:
                    logger.debug(
                        f"Dispatching Gemini request: model={model_to_use}, attempt={attempt + 1}/{self.max_retries + 1}, "
                        f"timeout={self.timeout}s, chars={char_count}"
                    )

                    response = self.client.post(
                        endpoint,
                        headers=headers,
                        json=request_body,
                        timeout=self._httpx_timeout,
                    )
                    elapsed = time.time() - t0
                    status_code = response.status_code

                    # 1. Successful extraction
                    if status_code == 200:
                        resp_data = response.json()
                        candidates = resp_data.get("candidates", [])
                        if not candidates:
                            raise ProviderExecutionError("Gemini returned zero response candidates.")

                        parts = candidates[0].get("content", {}).get("parts", [])
                        if not parts or "text" not in parts[0]:
                            raise ProviderExecutionError("Gemini returned empty content parts.")

                        raw_text = parts[0]["text"].strip()
                        parsed_json = self._parse_json_safely(raw_text)
                        payload = self._sanitize_and_validate_payload(parsed_json)

                        logger.info(
                            f"Gemini extraction succeeded with model '{model_to_use}' in {elapsed:.2f}s "
                            f"(input_chars={char_count}, segments={segment_count}, total_retries={total_retries})."
                        )
                        return payload

                    # 2. Model not found or deprecated: advance immediately to next model without retrying
                    if status_code == 404:
                        logger.warning(
                            f"Gemini model '{model_to_use}' returned HTTP 404 (model not found or unsupported)."
                            + (f" Advancing to fallback model '{next_model_name}'." if has_next_model else "")
                        )
                        last_error = httpx.HTTPStatusError(
                            f"HTTP 404 model not found for '{model_to_use}'",
                            request=response.request,
                            response=response,
                        )
                        break  # Break attempt loop, try next model in models_to_try

                    # 3. Permanent auth / credential failure: do NOT retry
                    if status_code in (401, 403):
                        raise ProviderExecutionError(
                            "Gemini API authentication failed. Invalid, unauthorized, or suspended API key."
                        )

                    # 4. Bad request parameter: do NOT retry
                    if status_code == 400:
                        raise ProviderExecutionError(
                            f"Google Gemini API HTTP 400 bad request: {response.text[:200]}"
                        )

                    # 5. Rate limit (429): transient quota/rate limit error
                    if status_code == 429:
                        last_error = httpx.HTTPStatusError(
                            f"HTTP 429 rate limit exceeded for '{model_to_use}'",
                            request=response.request,
                            response=response,
                        )
                        if not is_last_attempt:
                            backoff = min(2.0 * (2 ** attempt) + random.uniform(0.1, 0.5), 10.0)
                            logger.warning(
                                f"Gemini API rate limit (HTTP 429) on model '{model_to_use}' "
                                f"(attempt {attempt + 1}/{self.max_retries + 1}). Retrying in {backoff:.2f}s..."
                            )
                            time.sleep(backoff)
                            total_retries += 1
                            continue
                        elif has_next_model:
                            logger.warning(
                                f"Gemini model '{model_to_use}' exhausted {self.max_retries + 1} attempts on rate limit (429). "
                                f"Attempting fallback model '{next_model_name}'."
                            )
                            break
                        else:
                            raise ProviderExecutionError(
                                "Google Gemini API quota or rate limit exceeded. Please check quota limits."
                            )

                    # 6. Transient server error (500, 502, 503, 504)
                    if status_code in (500, 502, 503, 504):
                        last_error = httpx.HTTPStatusError(
                            f"HTTP {status_code} server error for '{model_to_use}'",
                            request=response.request,
                            response=response,
                        )
                        if not is_last_attempt:
                            backoff = min(1.5 * (2 ** attempt) + random.uniform(0.1, 0.4), 8.0)
                            logger.warning(
                                f"Transient HTTP {status_code} on Gemini model '{model_to_use}' "
                                f"(attempt {attempt + 1}/{self.max_retries + 1}, duration: {elapsed:.2f}s). "
                                f"Retrying in {backoff:.2f}s..."
                            )
                            time.sleep(backoff)
                            total_retries += 1
                            continue
                        elif has_next_model:
                            logger.warning(
                                f"Gemini model '{model_to_use}' returned HTTP {status_code} after {self.max_retries + 1} attempts. "
                                f"Attempting fallback model '{next_model_name}'."
                            )
                            break
                        else:
                            raise ProviderExecutionError(
                                f"Google Gemini API HTTP {status_code} server error: {response.text[:200]}"
                            )

                    # Other unexpected status codes
                    response.raise_for_status()

                except httpx.ConnectError as ce:
                    last_error = ce
                    elapsed = time.time() - t0
                    if not is_last_attempt:
                        backoff = min(1.0 * (2 ** attempt) + random.uniform(0.1, 0.3), 5.0)
                        logger.warning(
                            f"Network connection error to Gemini API on model '{model_to_use}' "
                            f"(attempt {attempt + 1}/{self.max_retries + 1}, duration: {elapsed:.2f}s). "
                            f"Retrying in {backoff:.2f}s..."
                        )
                        time.sleep(backoff)
                        total_retries += 1
                        continue
                    elif has_next_model:
                        logger.warning(
                            f"Gemini model '{model_to_use}' failed connect error after {self.max_retries + 1} attempts. "
                            f"Attempting fallback model '{next_model_name}'."
                        )
                        break
                    else:
                        raise ProviderExecutionError(
                            f"Network error communicating with Google Gemini API: {type(ce).__name__}"
                        ) from ce

                except httpx.TimeoutException as te:
                    last_error = te
                    elapsed = time.time() - t0
                    if not is_last_attempt:
                        backoff = min(1.0 * (2 ** attempt) + random.uniform(0.1, 0.3), 5.0)
                        logger.warning(
                            f"Gemini request on model '{model_to_use}' timed out after {elapsed:.2f}s "
                            f"(attempt {attempt + 1}/{self.max_retries + 1}, limit: {self.timeout}s). "
                            f"Retrying in {backoff:.2f}s..."
                        )
                        time.sleep(backoff)
                        total_retries += 1
                        continue
                    elif has_next_model:
                        logger.warning(
                            f"Gemini model '{model_to_use}' timed out after {self.max_retries + 1} attempts. "
                            f"Attempting fallback model '{next_model_name}'."
                        )
                        break
                    else:
                        raise ProviderExecutionError(
                            f"Google Gemini API request timed out after {self.timeout}s."
                        ) from te

                except ProviderExecutionError:
                    raise
                except Exception as ex:
                    last_error = ex
                    logger.warning(
                        f"Unexpected error during Gemini extraction with model '{model_to_use}' "
                        f"(attempt {attempt + 1}/{self.max_retries + 1}): {ex}"
                    )
                    if not is_last_attempt:
                        time.sleep(1.0)
                        total_retries += 1
                        continue
                    elif has_next_model:
                        break
                    else:
                        raise ProviderExecutionError(
                            f"Unexpected error communicating with Gemini API: {str(ex)}"
                        ) from ex

        if last_error:
            if isinstance(last_error, httpx.TimeoutException):
                raise ProviderExecutionError(
                    f"Google Gemini API request timed out after {self.timeout}s."
                ) from last_error
            raise ProviderExecutionError(f"Gemini extraction failed across all models: {last_error}")

        raise ProviderExecutionError("Gemini extraction failed to produce a valid response.")

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

        total_chars = sum(len(s.text or "") for s in segments)
        return self._send_generate_content_request(
            request_body=request_body,
            context_description=f"meeting '{meeting_title}'",
            char_count=total_chars,
            segment_count=len(segments),
        )

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

        return self._send_generate_content_request(
            request_body=request_body,
            context_description=f"transcript text for '{meeting_title}'",
            char_count=len(segments_text),
            segment_count=segments_text.count("\n") + 1,
        )

    def _parse_json_safely(self, raw_text: str) -> Dict[str, Any]:
        """Strip markdown fences, trailing commas, and unescaped artifacts to safely parse JSON."""
        if not raw_text or not raw_text.strip():
            raise ProviderExecutionError("Gemini returned empty text payload.")

        cleaned = raw_text.strip()

        # 1. Attempt direct JSON parsing first
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # 2. Extract from markdown code blocks ```json ... ``` or ``` ... ```
        fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned, re.IGNORECASE)
        if fence_match:
            candidate = fence_match.group(1).strip()
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                cleaned = candidate

        # 3. Extract substring between first '{' and last '}'
        first_brace = cleaned.find("{")
        last_brace = cleaned.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            bounded_candidate = cleaned[first_brace : last_brace + 1].strip()
            try:
                return json.loads(bounded_candidate)
            except json.JSONDecodeError:
                cleaned = bounded_candidate

        # 4. Clean common LLM formatting flaws: trailing commas before } or ]
        cleaned_no_trailing = re.sub(r",\s*([}\]])", r"\1", cleaned)
        try:
            return json.loads(cleaned_no_trailing)
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
