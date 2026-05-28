from __future__ import annotations

"""Google CSE based medical web search for supplementary RAG context."""

import html
import json
import os
import re
import time
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen


SEARCH_ENDPOINT = "https://www.googleapis.com/customsearch/v1"
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; VitalAI-MedicalSearch/1.0)",
}

DEFAULT_ALLOWED_DOMAINS = (
    "cdc.gov",
    "nih.gov",
    "ncbi.nlm.nih.gov",
    "medlineplus.gov",
    "mayoclinic.org",
    "clevelandclinic.org",
    "who.int",
    "nhs.uk",
    "kidney.org",
    "kdigo.org",
    "aafp.org",
    "msdmanuals.com",
    "merckmanuals.com",
    "healthdirect.gov.au",
    "betterhealth.vic.gov.au",
)

DEFAULT_BLOCKED_DOMAINS = (
    "wikipedia.org",
    "facebook.com",
    "fb.com",
    "instagram.com",
    "youtube.com",
    "youtu.be",
    "threads.net",
    "threads.com",
    "tiktok.com",
    "x.com",
    "twitter.com",
    "linkedin.com",
    "reddit.com",
    "pinterest.com",
    "snapchat.com",
    "discord.com",
    "discord.gg",
)


@dataclass(frozen=True)
class MedicalWebSearchResult:
    title: str
    url: str
    snippet: str
    content: str
    domain: str

    def to_dict(self) -> dict[str, str]:
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "content": self.content,
            "domain": self.domain,
        }


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in {"script", "style", "noscript", "svg"}:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style", "noscript", "svg"} and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        text = data.strip()
        if text:
            self._parts.append(text)

    def text(self) -> str:
        return " ".join(self._parts)


def search_medical_web(
    query: str,
    *,
    num_results: int = 3,
    max_content_chars: int = 1800,
) -> list[MedicalWebSearchResult]:
    """Return allowed medical web results only.

    This is intentionally best-effort. Missing Google credentials or network
    failures return an empty list so the internal RAG flow remains unaffected.
    """

    api_key = os.getenv("GOOGLE_API_KEY")
    cx = os.getenv("GOOGLE_CX")
    if not api_key or not cx:
        return []

    clean_query = " ".join((query or "").split())
    if not clean_query:
        return []

    allowed_domains = _domains_from_env("MEDICAL_WEB_ALLOWED_DOMAINS", DEFAULT_ALLOWED_DOMAINS)
    blocked_domains = _domains_from_env("MEDICAL_WEB_BLOCKED_DOMAINS", DEFAULT_BLOCKED_DOMAINS)
    results: list[MedicalWebSearchResult] = []
    seen_urls: set[str] = set()

    for start_index in (1, 11, 21):
        if len(results) >= num_results:
            break
        data = _google_cse_request(
            clean_query,
            api_key=api_key,
            cx=cx,
            blocked_domains=blocked_domains,
            num_results=10,
            start_index=start_index,
        )
        items = data.get("items") if isinstance(data, dict) else []
        if not items:
            break
        for item in items:
            if len(results) >= num_results:
                break
            if not isinstance(item, dict):
                continue
            url = str(item.get("link") or "").strip()
            if not url or url in seen_urls:
                continue
            domain = _hostname(url)
            if not _is_allowed_domain(domain, allowed_domains, blocked_domains):
                continue
            seen_urls.add(url)
            snippet = _clean_text(str(item.get("snippet") or ""))
            content = fetch_medical_page_text(url, max_chars=max_content_chars)
            results.append(
                MedicalWebSearchResult(
                    title=_clean_text(str(item.get("title") or domain)),
                    url=url,
                    snippet=snippet,
                    content=content or snippet,
                    domain=domain,
                )
            )
        time.sleep(0.25)
    return results[:num_results]


def fetch_medical_page_text(url: str, *, timeout: int = 12, max_chars: int = 1800) -> str:
    domain = _hostname(url)
    blocked_domains = _domains_from_env("MEDICAL_WEB_BLOCKED_DOMAINS", DEFAULT_BLOCKED_DOMAINS)
    if _domain_matches(domain, blocked_domains):
        return ""
    try:
        request = Request(url, headers=DEFAULT_HEADERS)
        with urlopen(request, timeout=timeout) as response:
            content_type = response.headers.get("Content-Type", "")
            if "text/html" not in content_type and "application/xhtml" not in content_type:
                return ""
            raw = response.read(600_000).decode("utf-8", errors="ignore")
    except Exception:
        return ""
    parser = _TextExtractor()
    try:
        parser.feed(raw)
    except Exception:
        return ""
    return _clean_text(parser.text())[:max_chars]


def _google_cse_request(
    query: str,
    *,
    api_key: str,
    cx: str,
    blocked_domains: tuple[str, ...],
    num_results: int,
    start_index: int,
) -> dict[str, Any]:
    params = {
        "key": api_key,
        "cx": cx,
        "q": _with_domain_controls(query, blocked_domains),
        "num": max(1, min(num_results, 10)),
        "start": max(1, min(start_index, 100)),
        "lr": "lang_vi|lang_en",
        "safe": "active",
    }
    url = f"{SEARCH_ENDPOINT}?{urlencode(params)}"
    try:
        request = Request(url, headers=DEFAULT_HEADERS)
        with urlopen(request, timeout=15) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception:
        return {}


def _with_domain_controls(query: str, blocked_domains: tuple[str, ...]) -> str:
    exclusions = " ".join(f"-site:{domain}" for domain in blocked_domains)
    medical_hint = "medical kidney nephrology guideline"
    return f"{query} {medical_hint} {exclusions}".strip()


def _domains_from_env(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.getenv(name, "")
    if not raw.strip():
        return default
    return tuple(item.strip().lower() for item in raw.split(",") if item.strip())


def _hostname(url: str) -> str:
    try:
        return (urlparse(url).hostname or "").lower().removeprefix("www.")
    except Exception:
        return ""


def _is_allowed_domain(
    hostname: str,
    allowed_domains: tuple[str, ...],
    blocked_domains: tuple[str, ...],
) -> bool:
    if not hostname or _domain_matches(hostname, blocked_domains):
        return False
    return _domain_matches(hostname, allowed_domains)


def _domain_matches(hostname: str, domains: tuple[str, ...]) -> bool:
    return any(hostname == domain or hostname.endswith(f".{domain}") for domain in domains)


def _clean_text(value: str) -> str:
    value = html.unescape(value or "")
    value = re.sub(r"\s+", " ", value).strip()
    return value
