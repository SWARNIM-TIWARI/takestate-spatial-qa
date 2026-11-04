"""
test_spark_live.py — LIVE Muse Spark 1.3 smoke test. Makes THREE small paid requests.

Usage (PowerShell):
    python test_spark_live.py

Exit codes:
    0  structured JSON returned and mapped into the supported rule vocabulary
    1  at least one check failed
    2  SKIPPED — MODEL_API_KEY not configured (this is NOT a pass)

Uses the same code path as the app (interpreter.parse_requirement, JSON-schema
response_format). The API key is never printed.
"""

from __future__ import annotations

import sys

from interpreter import parse_requirement
from meta_client import get_api_key_status

CASES = [
    # (requirement text, expected (subject-contains, relation, reference-contains) or None = unsupported)
    ("The glass must be to the right of the bottle.", ("glass", "right_of", "bottle")),
    ("Keep the watch above the box.", ("watch", "above", "box")),
    ("Make the glass red.", None),   # not a spatial relation -> must NOT become a rule
]

failures = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global failures
    failures += 0 if ok else 1
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def main() -> int:
    if not get_api_key_status()["configured"]:
        print("SKIPPED: MODEL_API_KEY is not configured (.env or Streamlit secrets).")
        print("A skipped live test is not a passing live test.")
        return 2

    print("\nMuse Spark 1.3 live test (3 requests)\n")
    for text, expected in CASES:
        print(f"  requirement: {text!r}")
        parsed, err = parse_requirement(text)
        if err or parsed is None:
            check("Request returned structured JSON", False, err or "no result")
            continue
        print(f"    -> subject={parsed.subject!r} relation={parsed.relation!r} "
              f"reference={parsed.reference!r} supported={parsed.supported}")
        if expected is None:
            check("Non-spatial requirement is flagged unsupported (no rule created)",
                  not parsed.supported and parsed.to_spatial_rule() is None)
        else:
            subj, rel, ref = expected
            check("Mapped to the expected supported rule",
                  parsed.supported and parsed.relation == rel
                  and subj in parsed.subject and ref in parsed.reference)

    print(f"\nSpark live test: {'all checks passed' if failures == 0 else f'{failures} check(s) failed'}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
