"""Minimal Groq (OpenAI-compatible) chat client. Free-tier friendly; no SDK needed."""
from __future__ import annotations

import os

import requests

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


class LLMError(RuntimeError):
    pass


def chat(messages: list[dict], model: str | None = None, temperature: float = 0.2, timeout: int = 90) -> str:
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise LLMError("GROQ_API_KEY is not set. Copy .env.example to .env and add a free key from console.groq.com.")
    resp = requests.post(
        GROQ_URL,
        headers={"Authorization": f"Bearer {key}"},
        json={
            "model": model or os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
            "messages": messages,
            "temperature": temperature,
        },
        timeout=timeout,
    )
    if resp.status_code != 200:
        raise LLMError(f"Groq returned {resp.status_code}: {resp.text[:300]}")
    return resp.json()["choices"][0]["message"]["content"].strip()
