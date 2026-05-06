"""
Shared state object that flows through every node in the graph.
LangGraph merges partial dicts returned by each node into this state.
"""
from __future__ import annotations
from typing import TypedDict, Optional


class AgentState(TypedDict, total=False):
    goal: str                  # prompt from the user
    top_n: int                 # how many articles to fetch (depth slider)
    max_parallel: int          # scraper concurrency
    recipient_email: str       # where to send the final newsletter
    mode: str                  # "auto" or "human in the loop"

    # ── Research phase ──
    search_query: str          # query derived from the goal by planner
    articles: list[dict]       # [{url, content}, ...] from scraper

    # ── Writing phase ──
    draft_markdown: str        # current newsletter draft

    # ── Critic loop ──
    critic_verdict: str        # "pass" | "retry_research" | "retry_writing"
    critic_notes: str          # what to fix
    iteration: int             # how many times we've looped through critic

    # ── Human-in-the-loop ──
    human_approved: Optional[bool]   # True/False after user reviews
    human_edits: Optional[str]       # optional edited markdown from user

    # ── Publishing ──
    output_path: str           # path to saved .md file
    email_sent: bool

    # ── Diagnostics ──
    error: Optional[str]