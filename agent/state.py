"""Shared graph state."""
from __future__ import annotations

from typing import TypedDict


class Source(TypedDict):
    id: int
    title: str
    url: str
    summary: str


class ResearchState(TypedDict, total=False):
    question: str
    queries: list[str]
    sources: list[Source]
    draft: str
    critique: dict          # {"score": int, "issues": [str], "passed": bool, "audit": {...}}
    revisions: int          # automatic rewrites since the last human decision
    human_rounds: int
    human_feedback: str
    approved: bool
    final: str
