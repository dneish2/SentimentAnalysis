"""Utilities for retrieving article text for sentiment analysis."""
from __future__ import annotations

import html
import re
from typing import List, Tuple
from urllib.parse import parse_qs, quote_plus, unquote, urlparse

try:
    import requests
except ImportError:  # pragma: no cover - handled via fallback in functions
    requests = None  # type: ignore[assignment]

_USER_AGENT = "Mozilla/5.0 (compatible; sentiment-service/1.0)"


def _clean_markup(raw_html: str) -> str:
    text = re.sub(r"(?is)<script.*?>.*?</script>", " ", raw_html)
    text = re.sub(r"(?is)<style.*?>.*?</style>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def fetch_article_text(url: str, timeout: int = 10) -> str:
    """Download and clean a single article."""

    if requests is None:
        return ""

    try:
        response = requests.get(url, timeout=timeout, headers={"User-Agent": _USER_AGENT})
        response.raise_for_status()
    except Exception:
        return ""
    return _clean_markup(response.text)[:4000]


def gnews_search(query: str, k: int = 3) -> List[str]:
    """Return the first ``k`` article URLs from Google News RSS."""

    if requests is None:
        return []

    rss_url = (
        "https://news.google.com/rss/search?"
        f"q={quote_plus(query)}&hl=en-US&gl=US&ceid=US:en"
    )
    try:
        response = requests.get(rss_url, timeout=10, headers={"User-Agent": _USER_AGENT})
        response.raise_for_status()
    except Exception:
        return []

    links = re.findall(r"<link>(.*?)</link>", response.text)
    cleaned: List[str] = []
    for link in links:
        link = html.unescape(link)
        if "url=" in link:
            query_params = parse_qs(urlparse(link).query)
            if "url" in query_params:
                link = unquote(query_params["url"][0])
        if link.startswith("http"):
            cleaned.append(link)
        if len(cleaned) >= k:
            break
    return cleaned


def get_texts(text_or_query: str, k: int = 3) -> List[Tuple[str, str]]:
    """Resolve either a URL or a free-text query to article content."""

    candidate = text_or_query.strip()
    if candidate.startswith("http://") or candidate.startswith("https://"):
        return [(candidate, fetch_article_text(candidate))]

    urls = gnews_search(candidate, k=k)
    return [(url, fetch_article_text(url)) for url in urls]
