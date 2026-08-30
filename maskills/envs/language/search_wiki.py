#!/usr/bin/python3

"""
    search.py

    MediaWiki API Demos
    Demo of `Search` module: Search for a text or title

    MIT License
"""

import random
import re
import time

import requests

S = requests.Session()

URL = "https://en.wikipedia.org/w/api.php"

# Exponential-backoff retry for 429 / transient 5xx responses.  Without
# this, parallel agent rollouts will hammer the MediaWiki API and most
# search calls return Search error: HTTPError: 429 — which is what the
# first eval iteration showed.
_MAX_RETRIES = 5
_RETRY_INITIAL = 1.0  # seconds
_RETRY_MAX = 16.0


def _polite_get(url: str, params: dict, headers: dict):
    """GET ``url`` with bounded exponential backoff on 429/5xx.

    Honours the ``Retry-After`` header when present.  Raises after the
    final retry so the caller can surface the error string.
    """
    backoff = _RETRY_INITIAL
    last_exc = None
    for attempt in range(_MAX_RETRIES):
        try:
            r = S.get(url=url, params=params, headers=headers, timeout=20)
        except requests.RequestException as e:  # network blip
            last_exc = e
            time.sleep(backoff + random.random() * 0.25)
            backoff = min(backoff * 2, _RETRY_MAX)
            continue
        if r.status_code == 200:
            return r
        if r.status_code in (429, 500, 502, 503, 504):
            wait = backoff
            ra = r.headers.get("Retry-After")
            if ra:
                try:
                    wait = max(wait, float(ra))
                except ValueError:
                    pass
            time.sleep(wait + random.random() * 0.25)
            backoff = min(backoff * 2, _RETRY_MAX)
            continue
        r.raise_for_status()
    if last_exc is not None:
        raise last_exc
    # Make one last attempt and let it raise normally.
    r = S.get(url=url, params=params, headers=headers, timeout=20)
    r.raise_for_status()
    return r

SEARCHPAGE = "Nelson Mandela"

PARAMS = {
    "action": "query",
    "format": "json",
    "list": "search",
    "srsearch": SEARCHPAGE,
    "srlimit": 10,
}

HEADERS = {
    "User-Agent": "maskills-agent/1.0 (https://github.com/anonymous/maskills)"
}


def strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text or "")


def search_wikipedia(query: str, limit: int = 10):
    params = {**PARAMS, "srsearch": query, "srlimit": limit}
    r = _polite_get(URL, params=params, headers=HEADERS)
    data = r.json()
    results = []
    for item in data.get("query", {}).get("search", []):
        results.append({
            "title": item.get("title"),
            "pageid": item.get("pageid"),
            "url": f"https://en.wikipedia.org/?curid={item.get('pageid')}",
            "snippet": strip_html(item.get("snippet", "")),
            "size": item.get("size"),
            "wordcount": item.get("wordcount"),
            "timestamp": item.get("timestamp"),
        })
    return results


def get_extracts(pageids, intro_only=False, max_chars=None, max_sentences=None):
    """Fetch plain-text extracts for given pageids. Returns {pageid: text}."""
    if not pageids:
        return {}
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "explaintext": 1,
        "pageids": "|".join(str(p) for p in pageids),
        "exlimit": "max",
    }
    if intro_only:
        params["exintro"] = 1
    if max_chars:
        params["exchars"] = max_chars
    if max_sentences:
        params["exsentences"] = max_sentences
    r = _polite_get(URL, params=params, headers=HEADERS)
    pages = r.json().get("query", {}).get("pages", {})
    return {int(pid): p.get("extract", "") for pid, p in pages.items()}


def search_wiki(query: str, limit: int = 10, intro_only: bool = True,
                max_chars: int = 1200, max_sentences=None):
    """High-level wrapper: search Wikipedia and attach plain-text extracts.

    Returns a list of hits, each with the fields from `search_wikipedia` plus
    an `extract` key (falls back to `snippet` if no extract is returned).
    """
    hits = search_wikipedia(query, limit=limit)
    if not hits:
        return []
    extracts = get_extracts(
        [h["pageid"] for h in hits],
        intro_only=intro_only,
        max_chars=max_chars,
        max_sentences=max_sentences,
    )
    for h in hits:
        h["extract"] = extracts.get(h["pageid"]) or h.get("snippet", "")
    return hits


if __name__ == "__main__":
    hits = search_wiki(SEARCHPAGE, intro_only=True, max_chars=1200)
    for i, h in enumerate(hits, 1):
        print(f"[{i}] {h['title']} (pageid={h['pageid']})  {h['url']}")
        print(h["extract"])
        print("-" * 80)

