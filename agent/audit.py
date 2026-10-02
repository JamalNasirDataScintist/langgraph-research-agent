"""Deterministic citation audit. Runs alongside the LLM critic so grounding is never left to the model alone."""
from __future__ import annotations

import re

_CITE = re.compile(r"\[(\d+)\]")


def citation_audit(markdown: str, n_sources: int) -> dict:
    cited = {int(m) for m in _CITE.findall(markdown)}
    invalid = sorted(n for n in cited if n < 1 or n > n_sources)
    uncited = []
    for block in re.split(r"\n\s*\n", markdown):
        text = block.strip()
        if not text or text.startswith("#") or len(text) < 80:
            continue
        if not _CITE.search(text):
            uncited.append(text[:70] + ("..." if len(text) > 70 else ""))
    unused = sorted(set(range(1, n_sources + 1)) - cited)
    issues = []
    if invalid:
        issues.append(f"cites sources that do not exist: {invalid}")
    if uncited:
        issues.append(f"{len(uncited)} paragraph(s) have no citation, e.g. '{uncited[0]}'")
    return {
        "ok": not invalid and not uncited,
        "cited": sorted(cited),
        "invalid": invalid,
        "uncited_paragraphs": uncited,
        "unused_sources": unused,
        "issues": issues,
    }
