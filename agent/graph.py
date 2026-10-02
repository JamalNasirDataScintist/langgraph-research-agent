"""Graph wiring:  plan -> research -> write -> critique -(pass)-> human_review -(approve)-> finalize
                                                    ^------(fail, max 2 rewrites)------'      |
                                                    '-----------(reject + feedback)-----------'
"""
from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from .nodes import make_nodes, route_after_critique, route_after_human
from .state import ResearchState


def build_graph(llm, search, fetch, checkpointer=None):
    nodes = make_nodes(llm, search, fetch)
    graph = StateGraph(ResearchState)
    for name, fn in nodes.items():
        graph.add_node(name, fn)
    graph.add_edge(START, "plan")
    graph.add_edge("plan", "research")
    graph.add_edge("research", "write")
    graph.add_edge("write", "critique")
    graph.add_conditional_edges("critique", route_after_critique, {"write": "write", "human_review": "human_review"})
    graph.add_conditional_edges("human_review", route_after_human, {"write": "write", "finalize": "finalize"})
    graph.add_edge("finalize", END)
    return graph.compile(checkpointer=checkpointer or MemorySaver())
