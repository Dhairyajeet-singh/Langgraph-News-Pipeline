"""
Graph nodes. Each is a function that takes AgentState and returns a partial
update dict. LangGraph merges that into the running state.

The orchestration topology lives in graph.py — keep this file focused on
'what each step does', not 'when each step runs'.
"""
from __future__ import annotations

import json
import re
from datetime import datetime

from agent import tools
from agent.prompts import PLANNER_PROMPT, WRITER_PROMPT, CRITIC_PROMPT
from agent.state import AgentState

MAX_CRITIC_ITERATIONS = 2  # absolute ceiling on retry loops


# 1. PLANNER — goal → search query
def planner_node(state: AgentState) -> dict:
    tools._log("info", "── Planner ──")
    goal = state["goal"]
    query = tools.call_gemini(
        system_prompt=PLANNER_PROMPT,
        user_message=f"Goal: {goal}\nQuery:",
        temperature=0.2,
    )
    # If critic asked for a research retry, append a hint
    if state.get("critic_verdict") == "retry_research" and state.get("critic_notes"):
        query = f"{query} {state['critic_notes'][:60]}"
        tools._log("info", f"Adjusted query based on critic notes")
    tools._log("ok", f"Search query: {query}")
    return {"search_query": query}


# 2. RESEARCHER — runs the scraper with the query, returns list of articles
def researcher_node(state: AgentState) -> dict:
    tools._log("info", "── Researcher ──")
    articles = tools.research(
        query=state["search_query"],
        top_n=state.get("top_n", 7),
        max_parallel=state.get("max_parallel", 4),
    )
    if not articles:
        return {"articles": [], "error": "No articles scraped"}
    return {"articles": articles}


# 3. WRITER — articles → markdown newsletter
def _format_articles_for_writer(articles: list[dict]) -> str:
    """Pack article list into a single user message for the writer LLM."""
    blocks = []
    for i, art in enumerate(articles, 1):
        # Truncate per-article to keep prompt size reasonable
        content = art["content"][:6000]
        blocks.append(f"--- ARTICLE {i} ---\nURL: {art['url']}\n\n{content}")
    return "\n\n".join(blocks)


def writer_node(state: AgentState) -> dict:
    tools._log("info", "── Writer ──")
    articles = state.get("articles", [])
    if not articles:
        return {"draft_markdown": "", "error": "No articles to write from"}

    user_msg = (
        f"GOAL: {state['goal']}\n\n"
        f"DATE: {datetime.now().strftime('%B %d, %Y')}\n\n"
    )
    if state.get("critic_verdict") == "retry_writing" and state.get("critic_notes"):
        user_msg += f"PREVIOUS DRAFT FEEDBACK: {state['critic_notes']}\nPlease address this in the new draft.\n\n"
    user_msg += f"ARTICLES TO COVER:\n{_format_articles_for_writer(articles)}"

    draft = tools.call_gemini(
        system_prompt=WRITER_PROMPT,
        user_message=user_msg,
        temperature=0.5,
    )
    # Strip any accidental code fences
    draft = re.sub(r"^```(?:markdown|md)?\s*\n", "", draft)
    draft = re.sub(r"\n```\s*$", "", draft)
    tools._log("ok", f"Draft generated ({len(draft):,} chars)")
    return {"draft_markdown": draft.strip()}


# 4. CRITIC — score the draft, decide whether to loop
def critic_node(state: AgentState) -> dict:
    tools._log("info", "── Critic ──")
    iteration = state.get("iteration", 0) + 1

    user_msg = (
        f"GOAL: {state['goal']}\n\n"
        f"DRAFT:\n---\n{state['draft_markdown']}\n---\n\n"
        f"Return ONLY the JSON verdict."
    )
    raw = tools.call_gemini(
        system_prompt=CRITIC_PROMPT,
        user_message=user_msg,
        temperature=0.1,
    )
    # Strip fences if Gemini added any
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    raw = re.sub(r"\s*```$", "", raw)

    try:
        verdict_obj = json.loads(raw)
        verdict = verdict_obj.get("verdict", "pass")
        notes = verdict_obj.get("notes", "")
        score = verdict_obj.get("score", 0)
    except json.JSONDecodeError:
        tools._log("warn", f"Critic returned non-JSON, defaulting to pass: {raw[:100]}")
        verdict, notes, score = "pass", "(critic parse failed)", 5

    # Hard ceiling — don't loop forever even if critic keeps failing the draft
    if iteration >= MAX_CRITIC_ITERATIONS and verdict != "pass":
        tools._log("warn", f"Hit max iterations ({MAX_CRITIC_ITERATIONS}); forcing pass")
        verdict = "pass"
        notes = f"(forced pass after {iteration} iterations) " + notes

    tools._log("ok", f"Critic: {verdict} (score={score}, iter={iteration}) — {notes}")
    return {
        "critic_verdict": verdict,
        "critic_notes": notes,
        "iteration": iteration,
    }


# 5. HUMAN REVIEW — placeholder node; graph interrupts BEFORE this in HITL
def human_review_node(state: AgentState) -> dict:
    """
    In HITL mode, the graph is compiled with interrupt_before=["human_review"],
    so execution pauses BEFORE this runs. The Flask route then resumes the
    graph after the user clicks Approve/Edit, optionally injecting human_edits.

    In auto mode, this just passes through.
    """
    tools._log("info", "── Human review ──")
    if state.get("human_edits"):
        tools._log("ok", "Applied human edits to draft")
        return {"draft_markdown": state["human_edits"], "human_approved": True}
    return {"human_approved": True}


# 6. PUBLISHER — save .md, send email
def publisher_node(state: AgentState) -> dict:
    tools._log("info", "── Publisher ──")
    md = state["draft_markdown"]

    # Extract first H1 as subject; fallback to generic
    subject_match = re.search(r"^#\s+(.+)$", md, re.MULTILINE)
    subject = subject_match.group(1).strip() if subject_match else "Your Newsletter"

    path = tools.save_markdown(md)

    sent = False
    recipient = state.get("recipient_email", "").strip()
    if recipient:
        sent = tools.send_email(recipient, subject, md, attach_path=path)
    else:
        tools._log("warn", "No recipient email; skipping send (file saved)")

    return {"output_path": path, "email_sent": sent}