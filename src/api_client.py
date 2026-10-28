"""Rate-limited, cached, usage-tracked API client.

Secrets:
- Keys are read from .env / environment at call time and only ever placed in
  request headers. They never appear in URLs, cache files, logs or errors.

Usage tracking:
- Every call (network or cache hit) is appended to data/usage/api_usage.csv
  with the provider's rate-limit headers. Network calls per UTC day are capped
  at the provider's daily budget.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from src.ingest import RAW_DIR, ROOT, USER_AGENT

USAGE_LOG = ROOT / "data" / "usage" / "api_usage.csv"
USAGE_FIELDS = [
    "timestamp_utc", "provider", "endpoint", "params", "cached",
    "status", "ratelimit_limit", "ratelimit_remaining",
]


@dataclass(frozen=True)
class Provider:
    """Static description of an API provider."""
    name: str
    base_url: str
    env_var: str
    key_header: str
    daily_budget: int | None       # max network calls per UTC day (None = no daily cap)
    min_interval_s: float          # pause between network calls
    limit_header: str
    remaining_header: str


PROVIDERS: dict[str, Provider] = {
    "highlightly": Provider(
        name="highlightly",
        base_url="https://soccer.highlightly.net",
        env_var="HIGHLIGHTLY_KEY",
        key_header="x-rapidapi-key",
        daily_budget=100,  # free Basic tier
        min_interval_s=1.0,
        limit_header="x-ratelimit-requests-limit",
        remaining_header="x-ratelimit-requests-remaining",
    ),
    "football_data_org": Provider(
        name="football_data_org",
        base_url="https://api.football-data.org/v4",
        env_var="FOOTBALL_DATA_ORG_KEY",
        key_header="X-Auth-Token",
        daily_budget=None,  # free tier is 10 calls/minute, no daily cap
        min_interval_s=6.5,
        limit_header="X-RequestCounter-Reset",
        remaining_header="X-Requests-Available-Minute",
    ),
}


class ApiKeyMissingError(RuntimeError):
    """Raised when the provider's key is not set."""


class BudgetExceededError(RuntimeError):
    """Raised when today's network-call budget is used up."""


def _get_key(p: Provider) -> str:
    """Load the key for provider p. The value is never returned to callers outside this module."""
    load_dotenv(ROOT / ".env")
    key = os.environ.get(p.env_var, "").strip()
    if not key:
        raise ApiKeyMissingError(f"{p.env_var} is not set (add it to .env)")
    return key


def _redact(text: str, key: str) -> str:
    """Remove the key from any text before it is shown or stored."""
    return text.replace(key, "***") if key else text


def _cache_path(p: Provider, endpoint: str, params: dict[str, Any]) -> Path:
    """Deterministic cache file for an endpoint + params (key is not part of it)."""
    query = urllib.parse.urlencode(sorted(params.items()))
    digest = hashlib.sha1(f"{endpoint}?{query}".encode()).hexdigest()[:12]
    slug = endpoint.strip("/").replace("/", "_") or "root"
    return RAW_DIR / p.name / f"{slug}_{digest}.json"


def _log_usage(row: dict[str, Any]) -> None:
    USAGE_LOG.parent.mkdir(parents=True, exist_ok=True)
    new = not USAGE_LOG.exists()
    with USAGE_LOG.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=USAGE_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def _read_usage() -> list[dict[str, str]]:
    if not USAGE_LOG.exists():
        return []
    with USAGE_LOG.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def calls_today(provider: str) -> int:
    """Network (non-cached) calls made to provider so far this UTC day."""
    today = datetime.now(timezone.utc).date().isoformat()
    return sum(
        1 for r in _read_usage()
        if r["provider"] == provider and r["cached"] == "False" and r["timestamp_utc"].startswith(today)
    )


_last_call: dict[str, float] = {}


def get(
    provider: str,
    endpoint: str,
    params: dict[str, Any] | None = None,
    max_age_s: float | None = None,
) -> Any:
    """GET provider endpoint as JSON, from cache when available.

    max_age_s=None: cached responses never expire (finished matches, past seasons).
    max_age_s=N: re-fetch if the cache file is older than N seconds (fixtures).
    """
    p = PROVIDERS[provider]
    params = params or {}
    cache = _cache_path(p, endpoint, params)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    log = {"timestamp_utc": now, "provider": p.name, "endpoint": endpoint,
           "params": json.dumps(params, sort_keys=True)}

    fresh = cache.exists() and (max_age_s is None or time.time() - cache.stat().st_mtime < max_age_s)
    if fresh:
        _log_usage({**log, "cached": True, "status": "cache", "ratelimit_limit": "", "ratelimit_remaining": ""})
        return json.loads(cache.read_text(encoding="utf-8"))

    if p.daily_budget is not None and calls_today(p.name) >= p.daily_budget:
        raise BudgetExceededError(f"{p.name}: daily budget of {p.daily_budget} calls used up (UTC day)")

    key = _get_key(p)
    wait = p.min_interval_s - (time.monotonic() - _last_call.get(p.name, 0.0))
    if wait > 0:
        time.sleep(wait)

    url = p.base_url + endpoint + ("?" + urllib.parse.urlencode(params) if params else "")
    req = urllib.request.Request(url, headers={p.key_header: key, "User-Agent": USER_AGENT})
    status, headers, body = 0, {}, b""
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            status, headers, body = resp.status, dict(resp.headers), resp.read()
    except urllib.error.HTTPError as e:
        status, headers, body = e.code, dict(e.headers or {}), e.read()
    finally:
        _last_call[p.name] = time.monotonic()
        lower = {k.lower(): v for k, v in headers.items()}
        _log_usage({**log, "cached": False, "status": status,
                    "ratelimit_limit": lower.get(p.limit_header.lower(), ""),
                    "ratelimit_remaining": lower.get(p.remaining_header.lower(), "")})

    if status != 200:
        msg = _redact(body.decode("utf-8", "replace")[:300], key)
        raise RuntimeError(f"{p.name} {endpoint} returned HTTP {status}: {msg}")

    data = json.loads(body)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return data


def cached(provider: str, endpoint: str, params: dict[str, Any] | None = None) -> Any | None:
    """Return the cached response for this request, or None. Never touches the network."""
    path = _cache_path(PROVIDERS[provider], endpoint, params or {})
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def usage_summary() -> str:
    """Human-readable usage report per provider."""
    rows = _read_usage()
    today = datetime.now(timezone.utc).date().isoformat()
    lines = [f"API usage (UTC day {today}), log: {USAGE_LOG.relative_to(ROOT)}"]
    for name, p in PROVIDERS.items():
        mine = [r for r in rows if r["provider"] == name]
        net_today = [r for r in mine if r["cached"] == "False" and r["timestamp_utc"].startswith(today)]
        hits_today = [r for r in mine if r["cached"] == "True" and r["timestamp_utc"].startswith(today)]
        remaining = next((r["ratelimit_remaining"] for r in reversed(net_today) if r["ratelimit_remaining"]), "?")
        budget = p.daily_budget if p.daily_budget is not None else "n/a"
        lines.append(
            f"  {name:18s} network today: {len(net_today)}/{budget}  cache hits today: {len(hits_today)}"
            f"  provider-reported remaining: {remaining}  all-time network calls: "
            f"{sum(r['cached'] == 'False' for r in mine)}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="API usage report.")
    parser.add_argument("command", choices=["usage"])
    parser.parse_args()
    print(usage_summary())


if __name__ == "__main__":
    main()
