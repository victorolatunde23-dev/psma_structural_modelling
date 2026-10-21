"""HTTP session, on-disk caching of API responses and downloads, file hashing.

Every network response and downloaded structure goes through this module, so
re-runs are fast, reproducible offline, and polite to public services.

Cache layout (under ``paths.cache``)::

    <name>.json          response body (unchanged format from the original notebook)
    <name>.json.meta     sidecar: {"status": 200|404, "fetched": ISO time, "url": ...}

Only the sidecar is new; caches written by the original notebook (without a
sidecar) are still read.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

log = logging.getLogger("psma_pipeline.cache")


class FetchError(RuntimeError):
    """Raised when a request fails after retries (the URL is in the message)."""


def make_session(retries: int = 5, backoff_factor: float = 1.0,
                 user_agent: str = "psma-structural-modelling/0.1 (python-requests)"
                 ) -> requests.Session:
    """Return a ``requests.Session`` that retries 429/5xx with exponential back-off."""
    s = requests.Session()
    retry = Retry(total=retries, backoff_factor=backoff_factor,
                  status_forcelist=(429, 500, 502, 503, 504),
                  allowed_methods=("GET", "POST"))
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.headers["User-Agent"] = user_agent
    return s


def session_from_config(cfg: Any) -> requests.Session:
    """Build a session from the ``http`` section of a :class:`~psma_pipeline.config.Config`."""
    return make_session(cfg.get("http.retries", 5), cfg.get("http.backoff_factor", 1.0),
                        cfg.get("http.user_agent", "psma-structural-modelling/0.1"))


def _meta_path(cache_file: Path) -> Path:
    return cache_file.with_name(cache_file.name + ".meta")


def _is_stale_404(cache_file: Path, refresh_404_after_days: float | None) -> bool:
    meta = _meta_path(cache_file)
    if refresh_404_after_days is None or not meta.exists():
        return False
    try:
        m = json.loads(meta.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    if m.get("status") != 404:
        return False
    fetched = _dt.datetime.fromisoformat(m["fetched"])
    return (_dt.datetime.now(_dt.timezone.utc) - fetched).days >= refresh_404_after_days


def cached_json(sess: requests.Session, url: str, cache_file: Path, method: str = "GET",
                payload: dict | None = None, allow_404: bool = False,
                timeout: float = 120, polite_delay: float = 0.15,
                refresh_404_after_days: float | None = 30) -> dict:
    """GET/POST a JSON endpoint with an on-disk cache.

    Parameters
    ----------
    allow_404:
        Treat HTTP 404 as an empty result ``{}`` (recorded in the sidecar so it
        can be refreshed after ``refresh_404_after_days``).

    Returns
    -------
    dict
        The decoded JSON body (``{}`` for an empty body or an allowed 404).

    Raises
    ------
    FetchError
        For any other HTTP error or an undecodable body.
    """
    if cache_file.exists() and not _is_stale_404(cache_file, refresh_404_after_days):
        return json.loads(cache_file.read_text())
    try:
        if method == "POST":
            r = sess.post(url, json=payload, timeout=timeout)
        else:
            r = sess.get(url, timeout=timeout)
    except requests.RequestException as ex:
        raise FetchError(f"{method} {url} failed: {ex}") from ex
    if allow_404 and r.status_code == 404:
        data: dict = {}
    else:
        try:
            r.raise_for_status()
            data = r.json() if r.content else {}
        except (requests.HTTPError, ValueError) as ex:
            raise FetchError(f"{method} {url} -> HTTP {r.status_code}: {ex}") from ex
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(data))
    _meta_path(cache_file).write_text(json.dumps({
        "status": r.status_code, "url": url, "method": method,
        "fetched": _dt.datetime.now(_dt.timezone.utc).isoformat()}))
    time.sleep(polite_delay)
    return data


def cached_download(sess: requests.Session, url: str, path: Path, timeout: float = 300,
                    polite_delay: float = 0.15) -> Path:
    """Download ``url`` to ``path`` unless it already exists; return ``path``.

    Existing files are never overwritten (original coordinates are immutable).
    """
    if path.exists():
        return path
    try:
        r = sess.get(url, timeout=timeout)
        r.raise_for_status()
    except requests.RequestException as ex:
        raise FetchError(f"GET {url} failed: {ex}") from ex
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_bytes(r.content)
    tmp.replace(path)
    time.sleep(polite_delay)
    return path


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """SHA-256 hex digest of a file (used in the reproducibility manifest)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()
