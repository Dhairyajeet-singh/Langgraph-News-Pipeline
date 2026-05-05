"""
Graph topology. Reads top-to-bottom like a flowchart:

    START → planner → researcher → writer → critic
                                              │
            ┌─────────────────────────────────┤
            │                                 │
       (retry_research)                  (retry_writing)
            │                                 │
            └────→ planner                    └────→ writer
                                              │
                                            (pass)
                                              │
                                              ▼
                                       human_review
                                       (interrupt here in HITL mode)
                                              │
                                              ▼
                                         publisher
                                              │
                                              ▼
                                            END

The single public function `build_graph(mode)` returns a compiled graph.
- mode="auto" : runs end-to-end without stopping
- mode="hitl" : interrupts before human_review, must be resumed externally
"""
from __future__ import annotations

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from agent.state import AgentState
from agent.nodes import (
    planner_node,
    researcher_node,
    writer_node,
    critic_node,
    human_review_node,
    publisher_node,
)


def _route_after_critic(state: AgentState) -> str:
    """Conditional edge after the critic. Drives the retry loop."""
    verdict = state.get("critic_verdict", "pass")
    if verdict == "retry_research":
        return "planner"
    if verdict == "retry_writing":
        return "writer"
    return "human_review"


def build_graph(mode: str = "auto"):
    """
    mode: 'auto' or 'hitl'
    Returns a compiled LangGraph graph ready to invoke / stream.
    """
    g = StateGraph(AgentState)

    g.add_node("planner", planner_node)
    g.add_node("researcher", researcher_node)
    g.add_node("writer", writer_node)
    g.add_node("critic", critic_node)
    g.add_node("human_review", human_review_node)
    g.add_node("publisher", publisher_node)

    g.add_edge(START, "planner")
    g.add_edge("planner", "researcher")
    g.add_edge("researcher", "writer")
    g.add_edge("writer", "critic")
    g.add_conditional_edges(
        "critic",
        _route_after_critic,
        {"planner": "planner", "writer": "writer", "human_review": "human_review"},
    )
    g.add_edge("human_review", "publisher")
    g.add_edge("publisher", END)

    checkpointer = MemorySaver()  # required for interrupt + resume
    interrupt_before = ["human_review"] if mode == "hitl" else []

    return g.compile(checkpointer=checkpointer, interrupt_before=interrupt_before)


def run_newsletter_agent(
    goal: str,
    recipient_email: str = "",
    top_n: int = 7,
    max_parallel: int = 4,
    mode: str = "auto",
    thread_id: str = "default",
) -> dict:
    """
    One function call that runs the full pipeline.

    For HITL mode, this returns after the interrupt; the caller
    (Flask route) is responsible for resuming with .invoke(None, ...)
    or with human_edits via update_state().
    """
    graph = build_graph(mode=mode)
    config = {"configurable": {"thread_id": thread_id}}

    initial: AgentState = {
        "goal": goal,
        "recipient_email": recipient_email,
        "top_n": top_n,
        "max_parallel": max_parallel,
        "mode": mode,
        "iteration": 0,
    }
    final_state = graph.invoke(initial, config=config)
    return final_state