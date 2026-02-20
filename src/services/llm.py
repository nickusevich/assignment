"""
LLM client via OpenRouter (OpenAI-compatible API).
"""

import json
from typing import Type, TypeVar

from openai import OpenAI
from pydantic import BaseModel, ValidationError

from src.config.settings import settings
from src.logger import get_logger

logger = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMClient:
    def __init__(self):
        self.client = OpenAI(
            base_url=settings.openrouter_base_url,
            api_key=settings.openrouter_api_key,
        )

    def call(self, prompt: str, system: str | None = None) -> str:
        """Raw text response."""
        messages = self._build_messages(prompt, system)
        response = self.client.chat.completions.create(
            model=settings.llm_model,
            messages=messages,
        )
        return response.choices[0].message.content

    def call_structured(
        self,
        prompt: str,
        response_model: Type[T],
        system: str | None = None,
    ) -> T:
        """
        Call LLM with JSON mode and validate against a Pydantic model.

        response_format=json_object guarantees valid JSON from the API.
        Pydantic validates the schema. Malformed output never reaches the application
        logic.
        """
        messages = self._build_messages(prompt, system)

        response = self.client.chat.completions.create(
            model=settings.llm_model,
            messages=messages,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as e:
            logger.error("json_parse_failed", raw=raw[:300], error=str(e))
            raise ValueError(f"LLM returned invalid JSON: {e}")

        try:
            return response_model.model_validate(parsed)
        except ValidationError:
            logger.error(
                "response_validation_failed",
                model=response_model.__name__,
                parsed=parsed,
            )
            raise

    @staticmethod
    def _build_messages(prompt: str, system: str | None) -> list[dict]:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return messages
