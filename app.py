"""Streamlit UI with an approval step:  streamlit run app.py"""
from __future__ import annotations

import uuid
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langgraph.types import Command

from agent.export import markdown_to_pdf
from agent.graph import build_graph
from agent.llm import LLMError, chat
from agent.tools import fetch_page, web_search

load_dotenv(Path(__file__).parent / ".env")
st.set_page_config(page_title="Research Agent", page_icon="🧭", layout="wide")


@st.cache_resource
def get_graph():
    return build_graph(chat, web_search, fetch_page)  # one in-memory checkpointer shared across reruns


graph = get_graph()
st.title("🧭 Research agent with human approval")
st.caption("plan → search → write → critique (auto-rewrite) → **you approve** → final report with sources")

if "thread" not in st.session_state:
    st.session_state.thread, st.session_state.result = str(uuid.uuid4()), None
config = {"configurable": {"thread_id": st.session_state.thread}}


def advance(payload):
    try:
        with st.spinner("Working..."):
            st.session_state.result = graph.invoke(payload, config)
    except LLMError as exc:
        st.error(str(exc))


question = st.text_input("Research question", "What are the main trade-offs of solid-state batteries?")
c1, c2 = st.columns([1, 6])
if c1.button("Start", type="primary") and question:
    st.session_state.thread = str(uuid.uuid4())
    config = {"configurable": {"thread_id": st.session_state.thread}}
    advance({"question": question})
if c2.button("Reset"):
    st.session_state.result = None

result = st.session_state.result
if result and "__interrupt__" in result:
    review = result["__interrupt__"][0].value
    crit = review["critique"]
    st.subheader(f"Review round {review['round']}")
    m1, m2, m3 = st.columns(3)
    m1.metric("Critic score", f"{crit.get('score')}/10")
    m2.metric("Checks passed", "yes" if crit.get("passed") else "no")
    m3.metric("Sources used", len(review["sources"]))
    for issue in crit.get("issues", []):
        st.warning(issue)
    st.markdown(review["draft"])
    with st.expander("Sources"):
        for s in review["sources"]:
            st.markdown(f"**[{s['id']}] [{s['title']}]({s['url']})**  \n{s['summary']}")
    feedback = st.text_area("Instructions for a rewrite (leave empty to approve)")
    a, b = st.columns(2)
    if a.button("✅ Approve"):
        advance(Command(resume={"approve": True}))
        st.rerun()
    if b.button("🔁 Request changes", disabled=not feedback.strip()):
        advance(Command(resume={"approve": False, "feedback": feedback}))
        st.rerun()
elif result and result.get("final"):
    st.success("Approved. Final report:")
    st.markdown(result["final"])
    d1, d2 = st.columns(2)
    d1.download_button("Download Markdown", result["final"], "report.md")
    d2.download_button("Download PDF", markdown_to_pdf(result["final"]), "report.pdf", "application/pdf")
