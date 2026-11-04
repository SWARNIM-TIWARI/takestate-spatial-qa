"""
meta_client.py — Meta Model API client for TakeState.

Provides:
  - call_sam_image()   → SAM 3.1 via Responses API (streaming SSE)
  - call_spark_chat()  → Muse Spark 1.3 via Chat Completions API
  - get_api_key_status() → {"configured": bool}  — NEVER exposes the key

Security rules enforced here:
  • The API key is never printed, logged, or returned to callers.
  • get_api_key_status() returns only a boolean.
  • Error messages strip the key from any exception text before propagating.
"""

from __future__ import annotations

import json
import os
from typing import Optional

import requests
from config import get_secret

_BASE_URL = "https://api.meta.ai/v1"
_SAM_MODEL = "sam-3.1"
_SPARK_MODEL = "muse-spark-1.3"
_TIMEOUT = 90  # seconds


# ---------------------------------------------------------------------------
# Key management — never expose the raw key outside this module
# ---------------------------------------------------------------------------

def _get_key() -> Optional[str]:
    """Return the raw API key (environment/.env, then Streamlit secrets), or None."""
    return get_secret("MODEL_API_KEY")


def get_api_key_status() -> dict:
    """
    Return {"configured": True} if a non-empty MODEL_API_KEY is set,
    {"configured": False} otherwise.

    The key value and any prefix are NEVER included in the return value.
    """
    return {"configured": bool(_get_key())}


def _auth_header() -> dict:
    key = _get_key()
    if not key:
        raise RuntimeError("MODEL_API_KEY is not set. Add it to .env locally, or to Streamlit secrets when deployed.")
    return {"Authorization": f"Bearer {key}"}


def _safe_error(msg: str) -> str:
    """
    Strip any accidental key leakage from an error message.
    Replace the raw key with '[REDACTED]' if it appears in the string.
    """
    key = _get_key()
    if key and key in msg:
        msg = msg.replace(key, "[REDACTED]")
    return msg


# ---------------------------------------------------------------------------
# SAM 3.1 — Responses API, streaming SSE
# ---------------------------------------------------------------------------

def call_sam_image(
    image_data_url: str,
    concept: str,
) -> tuple[list[dict], Optional[str]]:
    """
    Send one image + one concept noun phrase to SAM 3.1.

    Returns
    -------
    (events, error)
        events : list of parsed SSE event dicts (may be empty on zero matches)
        error  : human-readable error string, or None on success

    Error vs zero matches
    ---------------------
    - response.failed  → error (NOT zero matches)
    - error SSE event  → error (NOT zero matches)
    - HTTP 4xx/5xx     → error
    - response.completed received, output_text empty → zero matches (events=[])
    - No response.completed received, loop ended     → error (incomplete stream)
    """
    try:
        headers = {
            **_auth_header(),
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
    except RuntimeError as exc:
        return [], str(exc)

    payload = {
        "model": _SAM_MODEL,
        "stream": True,
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_image",
                        "image_url": image_data_url,
                    },
                    {
                        "type": "input_text",
                        "text": concept,
                    },
                ],
            }
        ],
    }

    try:
        resp = requests.post(
            f"{_BASE_URL}/responses",
            headers=headers,
            json=payload,
            stream=True,
            timeout=_TIMEOUT,
        )
    except requests.exceptions.Timeout:
        return [], "SAM request timed out after 90 s. Try a smaller image."
    except requests.exceptions.RequestException as exc:
        return [], _safe_error(f"Network error: {exc}")

    if resp.status_code == 401:
        return [], "API key rejected (HTTP 401). Check MODEL_API_KEY (.env locally, or Streamlit secrets when deployed)."
    if resp.status_code == 402:
        return [], "API quota exceeded (HTTP 402). Check your Meta AI account."
    if resp.status_code == 404:
        return [], f"Endpoint not found (HTTP 404). Check BASE_URL and model name '{_SAM_MODEL}'."
    if resp.status_code != 200:
        return [], _safe_error(f"HTTP {resp.status_code}: {resp.text[:200]}")

    events: list[dict] = []
    completed = False
    current_event_type: Optional[str] = None
    current_data_lines: list[str] = []

    def flush_event() -> Optional[dict]:
        """Parse the buffered SSE event. Returns dict or None."""
        nonlocal current_event_type, current_data_lines
        if not current_data_lines:
            current_event_type = None
            return None
        data_str = "\n".join(current_data_lines)
        current_data_lines = []
        evt_type = current_event_type
        current_event_type = None
        if not data_str.strip():
            return None
        try:
            parsed = json.loads(data_str)
        except json.JSONDecodeError:
            parsed = {"_raw": data_str}
        if evt_type:
            parsed["_event_type"] = evt_type
        return parsed

    try:
        for raw_line in resp.iter_lines(decode_unicode=True):
            if raw_line is None:
                continue
            line = raw_line.strip()

            if line == "":
                # blank line = end of one SSE event
                evt = flush_event()
                if evt is not None:
                    evt_type = evt.get("_event_type") or evt.get("type", "")

                    # Explicit failure events
                    if evt_type == "response.failed":
                        reason = (
                            evt.get("response", {})
                            .get("error", {})
                            .get("message", "unknown reason")
                        )
                        return events, _safe_error(f"SAM response failed: {reason}")

                    if evt_type == "error":
                        msg = evt.get("message") or evt.get("error", {}).get(
                            "message", "unknown error"
                        )
                        return events, _safe_error(f"SAM API error: {msg}")

                    if evt_type == "response.completed":
                        completed = True
                        events.append(evt)
                        break

                    events.append(evt)
                continue

            if line.startswith("event:"):
                current_event_type = line[len("event:"):].strip()
            elif line.startswith("data:"):
                current_data_lines.append(line[len("data:"):].strip())
            # ignore comment lines (starting with ':')

    except requests.exceptions.ChunkedEncodingError as exc:
        return events, _safe_error(f"Stream interrupted: {exc}")

    if not completed:
        return events, (
            "SAM stream ended without a response.completed event. "
            "The response may be truncated."
        )

    return events, None


# ---------------------------------------------------------------------------
# Muse Spark 1.3 — Chat Completions API
# ---------------------------------------------------------------------------

def call_spark_chat(
    system: str,
    user: str,
    response_format: Optional[dict] = None,
) -> tuple[str, Optional[str]]:
    """
    Send a system + user message to Muse Spark 1.3.

    Parameters
    ----------
    response_format : optional dict
        If provided, added to the request payload (e.g. JSON schema mode).

    Returns
    -------
    (content, error)
        content : assistant reply text (empty string on error)
        error   : human-readable error string, or None on success
    """
    try:
        headers = {
            **_auth_header(),
            "Content-Type": "application/json",
        }
    except RuntimeError as exc:
        return "", str(exc)

    payload = {
        "model": _SPARK_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    if response_format is not None:
        payload["response_format"] = response_format

    try:
        resp = requests.post(
            f"{_BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
            timeout=_TIMEOUT,
        )
    except requests.exceptions.Timeout:
        return "", "Spark request timed out after 90 s."
    except requests.exceptions.RequestException as exc:
        return "", _safe_error(f"Network error: {exc}")

    if resp.status_code == 401:
        return "", "API key rejected (HTTP 401). Check MODEL_API_KEY (.env locally, or Streamlit secrets when deployed)."
    if resp.status_code != 200:
        return "", _safe_error(f"HTTP {resp.status_code}: {resp.text[:200]}")

    try:
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        return content, None
    except (KeyError, IndexError, json.JSONDecodeError) as exc:
        return "", _safe_error(f"Unexpected Spark response format: {exc}")
