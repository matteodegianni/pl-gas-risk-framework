"""Small HTTP helper with retries, shared by all API loaders."""

from __future__ import annotations

import time
from typing import Any

import requests


def get(url: str, params: dict[str, Any] | None, http_cfg: dict[str, Any]) -> requests.Response:
    """GET ``url`` with a fixed user agent and exponential backoff.

    Raises:
        requests.HTTPError: after the configured number of retries.
    """
    headers = {"User-Agent": http_cfg["user_agent"]}
    last_exc: Exception | None = None
    for attempt in range(int(http_cfg["retries"])):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=http_cfg["timeout_s"])
            if resp.status_code == 404:
                return resp  # callers decide whether "not found" means "no data"
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            last_exc = exc
            time.sleep(float(http_cfg["backoff_s"]) * (2**attempt))
    assert last_exc is not None
    raise last_exc
