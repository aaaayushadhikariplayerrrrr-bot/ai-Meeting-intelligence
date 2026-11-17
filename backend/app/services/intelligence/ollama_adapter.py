"""Ollama local LLM REST adapter for structured intelligence extraction."""

import json
import logging
from typing import List, Optional, Dict, Any
import httpx
from pydantic import ValidationError

from app.core.config import settings
from app.models.transcription import TranscriptSegment
from app.schemas.intelligence import StructuredExtractionPayload
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
        self.model_name = model_name or getattr(settings, "LLM_MODEL_NAME", "llama3.2:3b")
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

        try:
            parsed_json = json.loads(raw_text)
        except json.JSONDecodeError as je:
            raise ProviderExecutionError(f"Ollama output was not valid JSON: {je}") from je

        try:
            payload = StructuredExtractionPayload.model_validate(parsed_json)
        except ValidationError as ve:
            raise ProviderExecutionError(f"Ollama JSON output failed schema validation: {ve}") from ve

        return payload
