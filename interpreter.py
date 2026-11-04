"""
interpreter.py — Natural-language requirement → SpatialRule via Muse Spark 1.3.

Two entry points
----------------
  parse_requirement(text) → (ParsedRule | None, error | None)
      Calls Muse Spark and parses its JSON response.
      Falls back gracefully to ParsedRule with supported=False on any failure.

  manual_rule(subject, relation, reference) → ParsedRule
      Constructs a rule directly from validated UI form fields.
      No API call is made.

Supported relations (must match rule_engine.SUPPORTED_RELATIONS)
----------------------------------------------------------------
  right_of, left_of, above, below
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

from rule_engine import SUPPORTED_RELATIONS, SpatialRule

# Structured output schema for Muse Spark — prevents hallucinated keys and
# ensures the response can be parsed as JSON without markdown fences.
_RESPONSE_FORMAT: dict = {
    "type": "json_schema",
    "json_schema": {
        "name": "spatial_rule",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "subject":            {"type": "string"},
                "relation": {
                    "type": "string",
                    "enum": ["right_of", "left_of", "above", "below", "unsupported"],
                },
                "reference":          {"type": "string"},
                "unsupported_reason": {"type": "string"},
            },
            "required": ["subject", "relation", "reference", "unsupported_reason"],
            "additionalProperties": False,
        },
    },
}

_MAX_PHRASE_LEN = 60  # sanity bound on object noun phrases sent on to SAM

# Muse Spark system prompt — instructs the model to return structured JSON only.
_SYSTEM_PROMPT = """You are a requirement parser for a product photography rule checker.
The user will give you ONE natural-language requirement about the spatial arrangement
of objects in a photograph.

Extract:
  - subject   : the object that must satisfy the constraint (noun phrase, lowercase)
  - relation  : one of exactly: right_of, left_of, above, below
  - reference : the reference object (noun phrase, lowercase)

Return ONLY valid JSON with keys "subject", "relation", "reference",
"unsupported_reason". Use an empty string for unsupported_reason when the
relation is supported. No markdown fences. No explanation. No extra keys.

Examples:
  Input: "The glass must be to the right of the bottle."
  Output: {"subject": "glass", "relation": "right_of", "reference": "bottle", "unsupported_reason": ""}

  Input: "Place the watch above the box."
  Output: {"subject": "watch", "relation": "above", "reference": "box", "unsupported_reason": ""}

If the relation cannot be mapped to right_of/left_of/above/below, return:
  {"subject": "...", "relation": "unsupported", "reference": "...",
   "unsupported_reason": "brief explanation"}
"""


@dataclass
class ParsedRule:
    subject:            str
    relation:           str
    reference:          str
    supported:          bool
    unsupported_reason: str
    raw_input:          str
    source:             str   # "spark" | "manual" | "error"

    def to_spatial_rule(self) -> Optional[SpatialRule]:
        """Convert to SpatialRule if supported, else None."""
        if not self.supported:
            return None
        return SpatialRule(
            subject=self.subject,
            relation=self.relation,
            reference=self.reference,
        )


def _extract_json(text: str) -> dict:
    """
    Extract a JSON object from model output that may contain prose or fences.
    """
    # Strip markdown code fences
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    # Find first {...}
    start = cleaned.find("{")
    end   = cleaned.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError("No JSON object found in model response.")
    return json.loads(cleaned[start:end])


def parse_requirement(
    text: str,
) -> tuple[Optional[ParsedRule], Optional[str]]:
    """
    Parse a natural-language requirement via Muse Spark 1.3.

    Returns
    -------
    (parsed_rule, error)
        parsed_rule : ParsedRule on success (supported may be False)
        error       : human-readable error string, or None on success
    """
    # Import here to avoid circular dependency at module level
    from meta_client import call_spark_chat  # noqa: PLC0415

    content, err = call_spark_chat(
        system=_SYSTEM_PROMPT,
        user=text.strip(),
        response_format=_RESPONSE_FORMAT,
    )

    if err:
        return None, f"Spark call failed: {err}"

    try:
        data = _extract_json(content)
    except (ValueError, json.JSONDecodeError) as exc:
        return None, f"Could not parse Spark response as JSON: {exc}\nRaw: {content[:300]}"

    subject   = str(data.get("subject",   "")).strip().lower()
    relation  = str(data.get("relation",  "")).strip().lower()
    reference = str(data.get("reference", "")).strip().lower()

    if not subject or not reference:
        return None, f"Incomplete JSON from Spark: {data}"

    if len(subject) > _MAX_PHRASE_LEN or len(reference) > _MAX_PHRASE_LEN:
        return None, "Spark returned an object description that is too long; please rephrase."

    supported = relation in SUPPORTED_RELATIONS
    unsupported_reason = ""
    if not supported:
        unsupported_reason = data.get(
            "unsupported_reason",
            f"Relation '{relation}' is not in the supported set: "
            + ", ".join(SUPPORTED_RELATIONS),
        )

    return ParsedRule(
        subject=subject,
        relation=relation,
        reference=reference,
        supported=supported,
        unsupported_reason=unsupported_reason,
        raw_input=text,
        source="spark",
    ), None


def manual_rule(
    subject: str,
    relation: str,
    reference: str,
) -> Optional[ParsedRule]:
    """
    Build a ParsedRule directly from validated form fields.

    Returns None if relation is not supported.
    """
    relation = relation.strip().lower()
    if relation not in SUPPORTED_RELATIONS:
        return None

    return ParsedRule(
        subject=subject.strip().lower(),
        relation=relation,
        reference=reference.strip().lower(),
        supported=True,
        unsupported_reason="",
        raw_input=f"{subject} {relation} {reference}",
        source="manual",
    )
