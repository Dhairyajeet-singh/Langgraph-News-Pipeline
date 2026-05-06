"""
Generalized news scraper. Adapted from the user's existing stock-news pipeline.
Key changes vs. original:
  - Accepts arbitrary topic (no fetch_symbol / stock-specific logic)
  - top_n and max_parallel are parameters (frontend "depth" control)
  - Returns articles in-memory (caller decides whether to save)
  - Final Ollama screener is optional and pluggable
"""
from __future__ import annotations

import asyncio
import os
import random
import time
from datetime import datetime
from typing import Optional
from urllib.parse import parse_qs, quote_plus, urlparse

import trafilatura
from bs4 import BeautifulSoup
from playwright.async_api import Browser, BrowserContext, Page, async_playwright

NEWS_SEARCH_URL = "https://html.duckduckgo.com/html/?q={query}"

NEWS_BLOCKED_DOMAINS: set[str] = {
    "youtube.com", "youtu.be", "facebook.com", "twitter.com", "x.com",
    "instagram.com", "linkedin.com", "tiktok.com", "pinterest.com",
    "duckduckgo.com", "google.com", "bing.com", "wikipedia.org",
}

HEADLESS = True
VIEWPORT = {"width": 1280, "height": 800}
LOCALE = "en-IN"
TIMEZONE = "Asia/Kolkata"

PAGE_TIMEOUT_MS = 25_000
SEARCH_TIMEOUT_MS = 20_000

MAX_RETRIES = 2
RETRY_DELAY = 1.0

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14.4; rv:124.0) Gecko/20100101 Firefox/124.0",
    "Mozilla/5.0 (X11; Linux x86_64; rv:124.0) Gecko/20100101 Firefox/124.0",
]


_log_callback = None

def set_log_callback(fn):
    """Caller can register a function(level, msg) to capture logs."""
    global _log_callback
    _log_callback = fn

def log(level: str, message: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] [{level.upper():5s}] {message}"
    print(line)
    if _log_callback:
        try:
            _log_callback(level, message)
        except Exception:
            pass

def _pick_user_agent() -> str:
    return random.choice(USER_AGENTS)


def _domain_of(url: str) -> str:
    try:
        host = urlparse(url).hostname or ""
        return host.lower().lstrip("www.")
    except Exception:
        return ""


def _resolve_ddg_link(href: str) -> str:
    if not href:
        return ""
    if href.startswith("//"):
        href = "https:" + href
    parsed = urlparse(href)
    if "duckduckgo.com" in (parsed.hostname or "") and parsed.path.startswith("/l/"):
        qs = parse_qs(parsed.query)
        if "uddg" in qs:
            return qs["uddg"][0]
    return href

async def _create_context(browser: Browser) -> BrowserContext:
    return await browser.new_context(
        user_agent=_pick_user_agent(),
        viewport=VIEWPORT,
        locale=LOCALE,
        timezone_id=TIMEZONE,
    )


async def _fetch_news_urls(context: BrowserContext, query: str, n: int) -> list[str]:
    search_url = NEWS_SEARCH_URL.format(query=quote_plus(query))
    log("info", f"DDG search → {query!r}")

    page = await context.new_page()
    try:
        await page.goto(search_url, timeout=SEARCH_TIMEOUT_MS, wait_until="domcontentloaded")
        await page.wait_for_timeout(1000)
        html = await page.content()
    finally:
        try:
            await page.close()
        except Exception:
            pass

    soup = BeautifulSoup(html, "html.parser")
    raw_links: list[str] = []
    for a in soup.select("a.result__a"):
        href = a.get("href", "").strip()
        if href:
            raw_links.append(href)
    if not raw_links:
        for a in soup.select("h2 a, .result__title a"):
            href = a.get("href", "").strip()
            if href:
                raw_links.append(href)

    cleaned: list[str] = []
    seen: set[str] = set()
    for raw in raw_links:
        url = _resolve_ddg_link(raw)
        if not url or not url.startswith("http"):
            continue
        if url in seen:
            continue
        d = _domain_of(url)
        if any(d == bd or d.endswith("." + bd) for bd in NEWS_BLOCKED_DOMAINS):
            continue
        seen.add(url)
        cleaned.append(url)
        if len(cleaned) >= n:
            break

    log("info", f"Got {len(cleaned)} news URLs")
    return cleaned


_BOILERPLATE_SUBSTRINGS = (
    "accept cookies", "accept all cookies", "cookie policy", "cookie settings",
    "sign in to continue", "sign in to your account", "create an account",
    "subscribe to our newsletter", "subscribe now", "follow us on",
    "skip to main content", "skip to content", "toggle navigation",
    "advertisement", "sponsored content", "sponsored by",
    "back to top", "share this article", "all rights reserved",
)


def _is_boilerplate_line(line: str) -> bool:
    low = line.lower().strip()
    if not low or len(low) < 3:
        return True
    for phrase in _BOILERPLATE_SUBSTRINGS:
        if phrase in low:
            return True
    if not any(c.isalnum() for c in low):
        return True
    return False


def _heuristic_clean(text: str) -> str:
    out_lines: list[str] = []
    blank_streak = 0
    for line in text.splitlines():
        stripped = line.strip()
        if _is_boilerplate_line(stripped) and not stripped.startswith("#"):
            if not stripped:
                blank_streak += 1
                if blank_streak <= 1:
                    out_lines.append("")
            continue
        blank_streak = 0
        out_lines.append(line.rstrip())

    deduped: list[str] = []
    prev = None
    for line in out_lines:
        if line == prev and (line.strip() == "" or len(line) > 0):
            continue
        deduped.append(line)
        prev = line

    return "\n".join(deduped).strip()


def _heuristic_extract(html: str) -> str:
    parts: list[str] = []
    try:
        extracted = trafilatura.extract(
            html,
            output_format="markdown",
            include_comments=True,
            include_tables=True,
            include_links=False,
            include_images=False,
            include_formatting=True,
            favor_recall=True,
            deduplicate=True,
        )
        if extracted and len(extracted.strip()) > 100:
            parts.append(extracted.strip())
    except Exception as exc:
        log("warn", f"trafilatura failed: {exc}")

    try:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup.find_all(["script", "style", "noscript", "head",
                                  "meta", "link", "iframe", "nav", "footer", "aside"]):
            tag.decompose()
        bs4_blocks: list[str] = []
        for el in soup.find_all(["h1", "h2", "h3"]):
            text = el.get_text(separator=" ", strip=True)
            if text and len(text) < 200:
                level = int(el.name[1])
                bs4_blocks.append(f"{'#' * level} {text}")
        if bs4_blocks:
            parts.append("\n## Additional extracted data\n\n" + "\n".join(bs4_blocks))
    except Exception as exc:
        log("warn", f"BS4 supplement failed: {exc}")

    combined = "\n\n".join(parts).strip()
    if not combined:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup.find_all(["script", "style", "noscript"]):
            tag.decompose()
        combined = soup.get_text(separator="\n", strip=True)
    return _heuristic_clean(combined)


async def _scrape_one(browser: Browser, url: str,
                      semaphore: asyncio.Semaphore) -> Optional[dict]:
    async with semaphore:
        for attempt in range(1, MAX_RETRIES + 1):
            context: Optional[BrowserContext] = None
            page: Optional[Page] = None
            try:
                log("info", f"[{attempt}/{MAX_RETRIES}] → {url}")
                context = await _create_context(browser)
                page = await context.new_page()
                await page.goto(url, timeout=PAGE_TIMEOUT_MS, wait_until="domcontentloaded")
                await page.wait_for_timeout(800)
                html = await page.content()
                cleaned = _heuristic_extract(html)
                if not cleaned or len(cleaned) < 200:
                    raise RuntimeError(f"too little content ({len(cleaned)} chars)")
                log("ok", f"  ✓ {url}  ({len(cleaned):,} chars)")
                return {"url": url, "content": cleaned}
            except Exception as exc:
                log("warn", f"  attempt {attempt} failed: {url} :: {str(exc)[:120]}")
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(RETRY_DELAY)
            finally:
                try:
                    if page: await page.close()
                except Exception:
                    pass
                try:
                    if context: await context.close()
                except Exception:
                    pass
    log("error", f"  ✗ giving up: {url}")
    return None


async def _scrape_parallel(browser: Browser, urls: list[str],
                           max_parallel: int) -> list[dict]:
    log("info", f"Scraping {len(urls)} URL(s) with concurrency={max_parallel}")
    semaphore = asyncio.Semaphore(max_parallel)
    tasks = [_scrape_one(browser, u, semaphore) for u in urls]
    raw = await asyncio.gather(*tasks, return_exceptions=True)

    results = [r for r in raw if isinstance(r, dict)]
    order = {u: i for i, u in enumerate(urls)}
    results.sort(key=lambda r: order.get(r["url"], 999))
    return results

async def _scrape_topic_async(query: str, top_n: int, max_parallel: int) -> list[dict]:
    async with async_playwright() as pw:
        log("info", "Launching Firefox …")
        browser = await pw.firefox.launch(headless=HEADLESS)
        try:
            search_ctx = await _create_context(browser)
            try:
                urls = await _fetch_news_urls(search_ctx, query, top_n)
            finally:
                await search_ctx.close()
            if not urls:
                log("error", "No news URLs found")
                return []
            t0 = time.time()
            articles = await _scrape_parallel(browser, urls, max_parallel)
            log("ok", f"Scrape done in {time.time()-t0:.1f}s — "
                     f"{len(articles)}/{len(urls)} succeeded")
            return articles
        finally:
            await browser.close()


def scrape_topic(query: str, top_n: int = 7, max_parallel: int = 4) -> list[dict]:
    """
    Synchronous wrapper. Returns list of {"url": str, "content": str}.
    This is the function the agent's research_node calls.
    """
    return asyncio.run(_scrape_topic_async(query, top_n, max_parallel))