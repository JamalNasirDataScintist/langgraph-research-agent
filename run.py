"""Command-line runner with terminal human-in-the-loop approval.

  python run.py "What are the main trade-offs of solid-state batteries?"
"""
from __future__ import annotations

import argparse
import re
import uuid
from pathlib import Path

from dotenv import load_dotenv
from langgraph.types import Command

from agent.export import markdown_to_pdf
from agent.graph import build_graph
from agent.llm import LLMError, chat
from agent.tools import fetch_page, web_search

ROOT = Path(__file__).parent


def main() -> None:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    args = parser.parse_args()

    graph = build_graph(chat, web_search, fetch_page)
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    payload: dict | Command = {"question": args.question}
    try:
        while True:
            result = graph.invoke(payload, config)
            if "__interrupt__" not in result:
                break
            review = result["__interrupt__"][0].value
            critique = review["critique"]
            print("\n" + "=" * 70)
            print(f"REVIEW ROUND {review['round']}  |  critic score {critique.get('score')}/10  |  "
                  f"{'passed' if critique.get('passed') else 'NEEDS ATTENTION'}")
            for issue in critique.get("issues", []):
                print(f"  - {issue}")
            print("=" * 70 + "\n" + review["draft"] + "\n")
            answer = input("Approve? [y = approve / type feedback to request changes]: ").strip()
            approve = answer.lower() in ("y", "yes")
            payload = Command(resume={"approve": approve, "feedback": "" if approve else answer})
    except LLMError as exc:
        raise SystemExit(f"LLM error: {exc}")

    final = result["final"]
    out_dir = ROOT / "reports"
    out_dir.mkdir(exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", args.question.lower()).strip("-")[:50]
    (out_dir / f"{slug}.md").write_text(final, encoding="utf-8")
    (out_dir / f"{slug}.pdf").write_bytes(markdown_to_pdf(final))
    print(f"\nSaved reports/{slug}.md and reports/{slug}.pdf")


if __name__ == "__main__":
    main()
