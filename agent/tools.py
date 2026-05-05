"""
Tool layer. Three categories:
  1. LLM tool      → call_gemini()    wraps OpenAI (name kept for back-compat)
  2. Research tool → research()       wraps the async scraper
  3. IO tools      → save_markdown(), send_email()
"""
from __future__ import annotations

import os
import smtplib
import ssl
from datetime import datetime
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email import encoders
from typing import Optional

from openai import OpenAI
import markdown as md_lib

from scraper.pipeline import scrape_topic

# ── Logging hook (same pattern as scraper, lets Flask stream logs) ──────────
_log_callback = None

def set_log_callback(fn):
    global _log_callback
    _log_callback = fn

def _log(level: str, msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] [{level.upper():5s}] {msg}")
    if _log_callback:
        try:
            _log_callback(level, msg)
        except Exception:
            pass


# ── 1. LLM tool (OpenAI) ────────────────────────────────────────────────────
# Function is still named call_gemini() so nodes.py doesn't need to change.
# The default model is read from OPENAI_MODEL env var, falling back to gpt-4o-mini.
_openai_client = None

def _ensure_openai() -> OpenAI:
    global _openai_client
    if _openai_client is not None:
        return _openai_client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in environment (.env)")
    _openai_client = OpenAI(api_key=api_key)
    return _openai_client


def call_gemini(system_prompt: str, user_message: str,
                model: Optional[str] = None,
                temperature: float = 0.4) -> str:
    """
    Single-turn LLM call. Now backed by OpenAI under the hood.
    Function name kept for back-compat with the rest of the codebase.

    `model` defaults to OPENAI_MODEL env var, then "gpt-4o-mini".
    """
    client = _ensure_openai()
    chosen_model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    resp = client.chat.completions.create(
        model=chosen_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=temperature,
    )
    return (resp.choices[0].message.content or "").strip()


# Cleaner alias if you want to use it elsewhere
call_llm = call_gemini


# ── 2. Research tool (wraps the scraper) ────────────────────────────────────
def research(query: str, top_n: int = 7, max_parallel: int = 4) -> list[dict]:
    """
    Returns list of {"url": str, "content": str}. Calls the async scraper.
    """
    _log("info", f"Research: query='{query}' top_n={top_n}")
    articles = scrape_topic(query, top_n=top_n, max_parallel=max_parallel)
    _log("ok", f"Research: got {len(articles)} articles")
    return articles


# ── 3. File saving ──────────────────────────────────────────────────────────
def save_markdown(content: str, output_dir: str = "outputs",
                  prefix: str = "newsletter") -> str:
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(output_dir, f"{prefix}_{timestamp}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    _log("ok", f"Saved → {path}")
    return path


# ── 4. Email sending (SMTP, with markdown rendered to HTML in body) ─────────
def send_email(recipient: str, subject: str, markdown_body: str,
               attach_path: Optional[str] = None) -> bool:
    """
    Sends the markdown newsletter as an email.
    - Body: markdown rendered to HTML (so it looks pretty in the inbox).
    - Attachment: the raw .md file (so the recipient can keep the source).
    Requires GMAIL_ADDRESS + GMAIL_APP_PASSWORD in environment.
    """
    sender = os.getenv("GMAIL_ADDRESS")
    password = os.getenv("GMAIL_APP_PASSWORD")
    if not sender or not password:
        _log("error", "GMAIL_ADDRESS / GMAIL_APP_PASSWORD missing in .env")
        return False

    # Render markdown → HTML for the email body
    html_body = md_lib.markdown(markdown_body, extensions=["extra", "nl2br"])
    # Wrap in basic styled HTML so it looks decent in Gmail
    html_full = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        max-width: 680px; margin: 0 auto; padding: 24px; color: #222; line-height: 1.6; }}
h1 {{ color: #1a1a1a; border-bottom: 2px solid #eee; padding-bottom: 8px; }}
h2 {{ color: #2a4d8f; margin-top: 28px; }}
hr {{ border: none; border-top: 1px solid #eee; margin: 32px 0; }}
a {{ color: #2a4d8f; }}
code {{ background: #f4f4f4; padding: 2px 4px; border-radius: 3px; }}
</style></head><body>{html_body}</body></html>"""

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = recipient

    msg.attach(MIMEText(markdown_body, "plain", "utf-8"))
    msg.attach(MIMEText(html_full, "html", "utf-8"))

    if attach_path and os.path.exists(attach_path):
        with open(attach_path, "rb") as f:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(f.read())
        encoders.encode_base64(part)
        part.add_header(
            "Content-Disposition",
            f"attachment; filename={os.path.basename(attach_path)}",
        )
        msg.attach(part)

    try:
        ctx = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ctx) as server:
            server.login(sender, password)
            server.sendmail(sender, [recipient], msg.as_string())
        _log("ok", f"Email sent → {recipient}")
        return True
    except Exception as exc:
        _log("error", f"SMTP send failed: {exc}")
        return False