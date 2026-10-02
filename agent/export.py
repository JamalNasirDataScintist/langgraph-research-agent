"""Render the final markdown report to a simple, clean PDF."""
from __future__ import annotations

import re

from fpdf import FPDF

_REPLACEMENTS = {"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-", "•": "-", "…": "..."}


def _clean(text: str) -> str:
    for src, dst in _REPLACEMENTS.items():
        text = text.replace(src, dst)
    text = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r"\1 (\2)", text)  # [title](url) -> title (url)
    text = re.sub(r"[*_`]+", "", text)
    return text.encode("latin-1", "replace").decode("latin-1")


def markdown_to_pdf(markdown: str) -> bytes:
    pdf = FPDF(unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    width = pdf.w - pdf.l_margin - pdf.r_margin
    for raw in markdown.splitlines():
        line = _clean(raw.rstrip())
        if not line.strip():
            pdf.ln(3)
            continue
        pdf.set_x(pdf.l_margin)
        if line.startswith("# "):
            pdf.set_font("Helvetica", "B", 18)
            pdf.multi_cell(width, 9, line[2:])
            pdf.ln(2)
        elif line.startswith("## "):
            pdf.ln(2)
            pdf.set_font("Helvetica", "B", 14)
            pdf.multi_cell(width, 8, line[3:])
        elif line.startswith("### "):
            pdf.set_font("Helvetica", "B", 12)
            pdf.multi_cell(width, 7, line[4:])
        elif line.lstrip().startswith(("- ", "> ")):
            pdf.set_font("Helvetica", "", 11)
            pdf.multi_cell(width, 6, "  - " + line.lstrip()[2:])
        else:
            pdf.set_font("Helvetica", "", 11)
            pdf.multi_cell(width, 6, line)
    return bytes(pdf.output())
