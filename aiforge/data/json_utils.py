"""Helpers for parsing JSON payloads returned by LLMs."""

import json
from typing import Any


def strip_code_fences(text: str) -> str:
    """Remove a surrounding Markdown code fence (``` or ```json) if present."""
    text = (text or "").strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def parse_llm_json(text: str) -> Any:
    """Parse an LLM response as JSON, tolerating Markdown code fences.

    Raises ``json.JSONDecodeError`` if the payload is not valid JSON.
    """
    return json.loads(strip_code_fences(text))
