---
title: Langgraph News Pipeline
emoji: 📰
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---
# Newsletter Agent

An autonomous AI agent that researches, writes, critiques, and emails a newsletter on any topic — given only a plain-English goal.

```
run_newsletter_agent("Create a weekly newsletter on the latest AI agent news.")
```

## Architecture

LangGraph state machine with six nodes and a self-correcting critic loop:

```
START → planner → researcher → writer → critic
                                          │
                ┌─────────────────────────┤
                │                         │
       (retry_research)            (retry_writing)
                │                         │
                └→ planner                └→ writer
                                          │
                                       (pass)
                                          │
                                          ▼
                                   human_review
                                   (interrupt here in HITL mode)
                                          │
                                          ▼
                                     publisher → END
```

**Node responsibilities:**

| Node         | Tool used                             | Output                     |
|--------------|---------------------------------------|----------------------------|
| `planner`    | Gpt-4o-mini                           | search query               |
| `researcher` | Playwright + DuckDuckGo + trafilatura | list of cleaned articles   |
| `writer`     | Gpt-4o-mini                           | markdown newsletter draft  |
| `critic`     | Gpt-4o-mini                           | `pass` / `retry_*` + notes |
| `human_review` | (interrupt point)                   | optionally edited markdown |
| `publisher`  | local FS + Gmail SMTP                 | saved `.md` + sent email   |

The critic loop is capped at 2 iterations to prevent runaway retries.

## Tools used (≥3, per assignment)

1. **Web search + scraper** — Playwright (Firefox), DuckDuckGo HTML, trafilatura, BeautifulSoup.
2. **LLM** — Google Gemini for planning, writing, and critiquing.
4. **File saver** — writes timestamped `.md` to `outputs/`.
5. **Email sender** — Gmail SMTP, sends rendered HTML body + `.md` attachment.

## File structure

```
newsletter-agent/
├── app.py                    # Flask frontend with SSE for live logs
├── agent/
│   ├── state.py              # AgentState TypedDict
│   ├── graph.py              # LangGraph topology (~50 lines)
│   ├── nodes.py              # node implementations
│   ├── tools.py              # Gemini, file save, email send
│   └── prompts.py            # all LLM prompts
├── scraper/
│   └── pipeline.py           # generalized scraper (sync wrapper around async Playwright)
├── templates/
│   └── index.html            # editorial-themed UI
├── outputs/                  # generated newsletters land here
├── requirements.txt
└── .env.example
```

## Setup

```bash
# 1. Install Python deps
pip install -r requirements.txt

# 2. Install Playwright browser
playwright install firefox

#3 Set up .env file with your openai api key and gmail address, app password (for sending mails to others)

# 4. Configure secrets
cp .env.example .env
# edit .env with your Gemini key + Gmail app password

# 5. Run
python app.py
# open http://localhost:5000
```

## Modes

**Fully autonomous** — agent runs end to end and emails the result without intervention.

**Human-in-the-loop** — pipeline runs through research → writing → critic loop, then pauses at the `human_review` checkpoint. The user sees the polished draft, can edit it inline, then approves. Implemented via LangGraph's `interrupt_before` + `MemorySaver` checkpointing.

## How `run_newsletter_agent` satisfies the assignment

| Requirement | Where it lives |
|---|---|
| Goal input | `app.py` POST /run, or `agent.graph.run_newsletter_agent(goal)` |
| Multi-step reasoning | 6-node LangGraph in `agent/graph.py` |
| Tool use (≥3) | scraper + Gemini + Ollama + file save + SMTP |
| Self-reflection | `critic_node` with `retry_research` / `retry_writing` verdicts |
| Single-call autonomy | `run_newsletter_agent(goal)` in `agent/graph.py` |
| Auto / HITL toggle | `mode` param + `interrupt_before` in `build_graph()` |
