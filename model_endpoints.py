"""User-configured OpenAI-compatible API access for all benchmark suites."""

from __future__ import annotations

import os


def resolve(model: str) -> tuple[str, str]:
    """Return the public API configuration; never embed credentials in code."""
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("Set OPENAI_API_KEY (use a non-empty placeholder for a local server).")
    return base_url, api_key


def supports_temperature(model: str) -> bool:
    excluded = {"claude-opus-5", "claude-sonnet-5"}
    excluded.update(filter(None, os.getenv("NO_TEMPERATURE_MODELS", "").split(",")))
    return model not in excluded


def temperature_kwargs(model: str, temperature: float | None) -> dict:
    if temperature is None or not supports_temperature(model):
        return {}
    return {"temperature": temperature}


def make_client(model: str, max_retries: int = 3, **kwargs):
    from openai import OpenAI

    base_url, api_key = resolve(model)
    return OpenAI(base_url=base_url, api_key=api_key, max_retries=max_retries, **kwargs)
