from __future__ import annotations

import os
import sys

from openai import OpenAI

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from model_endpoints import resolve, temperature_kwargs  # noqa: E402,F401


def create_openai_client(model: str, planner_name: str, base_url: str | None = None) -> tuple[OpenAI, str]:
    routed_base_url, api_key = resolve(model)
    base_url = base_url or routed_base_url
    if not api_key:
        raise ValueError("未找到 API key，请设置 OPENAI_API_KEY")

    print(f"初始化 {planner_name}:")
    print(f"  - Model: {model}")
    print(f"  - Base URL: {base_url}")
    return OpenAI(base_url=base_url, api_key=api_key, max_retries=3), model
