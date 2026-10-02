"""Web tools: free DuckDuckGo search and a safe page fetcher."""
from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

MAX_REDIRECTS = 3
USER_AGENT = "Mozilla/5.0 (compatible; ResearchAgent/1.0)"


def web_search(query: str, max_results: int = 5) -> list[dict]:
    """DuckDuckGo text search via the free `ddgs` package (no API key)."""
    from ddgs import DDGS

    try:
        raw = DDGS().text(query, max_results=max_results) or []
    except Exception:  # rate limits / network: degrade to "no results" rather than crash the run
        return []
    return [
        {"title": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", "")}
        for r in raw
        if r.get("href")
    ]


def is_public_url(url: str) -> bool:
    """Reject non-http(s) schemes and hosts resolving to private, loopback or link-local addresses.

    Search results are untrusted input; without this check a poisoned result could make the
    agent fetch internal services (SSRF).
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return False
    try:
        infos = socket.getaddrinfo(parsed.hostname, None)
    except socket.gaierror:
        return False
    for info in infos:
        addr = ipaddress.ip_address(info[4][0])
        if addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_reserved or addr.is_multicast:
            return False
    return True


def html_to_text(html: str, max_chars: int = 4000) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "noscript"]):
        tag.decompose()
    blocks = [p.get_text(" ", strip=True) for p in soup.find_all(["p", "li", "h1", "h2", "h3"])]
    text = " ".join(b for b in blocks if len(b) > 40)
    return text[:max_chars]


def fetch_page(url: str, max_chars: int = 4000, timeout: int = 10) -> str:
    """Return readable text for a page, or '' when it is unsafe, unreachable or not HTML."""
    try:
        for _hop in range(MAX_REDIRECTS + 1):
            if not is_public_url(url):  # checked on every hop, so a redirect cannot reach internal hosts
                return ""
            resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout, stream=True, allow_redirects=False)
            if resp.is_redirect or resp.status_code in (301, 302, 303, 307, 308):
                url = urljoin(url, resp.headers.get("Location", ""))
                continue
            if resp.status_code != 200 or "html" not in resp.headers.get("Content-Type", ""):
                return ""
            body = resp.raw.read(1_000_000, decode_content=True)  # cap download size
            return html_to_text(body.decode(resp.encoding or "utf-8", errors="ignore"), max_chars)
        return ""  # too many redirects
    except requests.RequestException:
        return ""
