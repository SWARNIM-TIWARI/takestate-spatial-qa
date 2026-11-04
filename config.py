"""
config.py — Secret/setting lookup and paid-call protection for TakeState.

Lookup order for every setting:
  1. Environment variable (also populated from a local .env by python-dotenv)
  2. Streamlit secrets (.streamlit/secrets.toml locally, or the Secrets box on
     Streamlit Community Cloud)

Nothing in this module ever returns, logs, or formats a secret value except
get_secret(), which is called only by meta_client (Authorization header) and
by the access-code comparison. Callers must never display its result.
"""

from __future__ import annotations

import hmac
import os
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

# Defaults for the per-session paid-call budget. Override with the settings
# MAX_TAKE_CHECKS and MAX_INTERPRETATIONS (env var or Streamlit secret).
DEFAULT_MAX_TAKE_CHECKS = 20        # each check = 2 SAM requests (subject + reference)
DEFAULT_MAX_INTERPRETATIONS = 10    # each interpretation = 1 Muse Spark request


def get_secret(name: str) -> Optional[str]:
    """Return a setting from the environment, then Streamlit secrets, else None."""
    value = os.getenv(name)
    if value:
        return value
    try:
        import streamlit as st  # imported lazily so CLI smoke tests need no Streamlit

        value = st.secrets.get(name)  # raises if no secrets file exists
    except Exception:  # noqa: BLE001 — missing secrets file / not running in Streamlit
        return None
    return str(value) if value else None


def get_int_setting(name: str, default: int) -> int:
    """Integer setting with a safe default for missing or malformed values."""
    raw = get_secret(name)
    if raw is None:
        return default
    try:
        parsed = int(str(raw).strip())
    except ValueError:
        return default
    return parsed if parsed >= 0 else default


def access_code_required() -> bool:
    """True when APP_ACCESS_CODE is configured (public deployments should set it)."""
    return bool(get_secret("APP_ACCESS_CODE"))


def check_access_code(candidate: str) -> bool:
    """Constant-time comparison against APP_ACCESS_CODE. False if none configured."""
    expected = get_secret("APP_ACCESS_CODE")
    if not expected:
        return False
    return hmac.compare_digest(candidate.encode("utf-8"), expected.encode("utf-8"))


class CallBudget:
    """
    Per-session counter for paid actions.

    This limits accidental or casual overuse inside one browser session. It is
    NOT a hard cap: opening a new session resets it. For a public URL combine
    it with APP_ACCESS_CODE and a spending limit on the Meta account.
    """

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.used = 0

    @property
    def remaining(self) -> int:
        return max(self.limit - self.used, 0)

    def can_spend(self, units: int = 1) -> bool:
        return self.used + units <= self.limit

    def spend(self, units: int = 1) -> bool:
        """Consume units if available. Returns False (and consumes nothing) if not."""
        if not self.can_spend(units):
            return False
        self.used += units
        return True
