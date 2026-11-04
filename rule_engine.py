"""
rule_engine.py — Deterministic geometric rule evaluation for TakeState.

Measurement invariant
---------------------
PASS and FAIL use ONLY decoded SAM segmentation-mask centroids, for BOTH objects.
A verdict of PASS or FAIL requires all of:
  1. both objects identified,
  2. the intended detection unambiguous for each,
  3. both masks decoded,
  4. both masks contain foreground pixels,
  5. both mask centroids validly mapped into the original image
     (finite, inside the image, consistent with the detection's bounding box).
Otherwise the verdict is UNABLE TO VERIFY. Bounding-box centres are reported in
the evidence as separately labelled diagnostics and are NEVER substituted for a
mask centroid.

Supported spatial relations
----------------------------
  right_of  — subject centroid_x > reference centroid_x
  left_of   — subject centroid_x < reference centroid_x
  above     — subject centroid_y < reference centroid_y  (y=0 is top)
  below     — subject centroid_y > reference centroid_y

Verdicts
--------
  PASS             — rule satisfied, with centroid evidence
  FAIL             — rule violated, with centroid evidence + correction hint
  UNABLE TO VERIFY — object missing, ambiguous, or a mask is missing / undecodable /
                     empty / unmappable for either object
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from sam_parser import DetectedObject


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class Verdict(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    UNABLE_TO_VERIFY = "UNABLE TO VERIFY"


SUPPORTED_RELATIONS = ("right_of", "left_of", "above", "below")

RELATION_LABELS: dict[str, str] = {
    "right_of": "to the right of",
    "left_of":  "to the left of",
    "above":    "above",
    "below":    "below",
}


@dataclass
class SpatialRule:
    subject:   str   # noun phrase, e.g. "glass"
    relation:  str   # one of SUPPORTED_RELATIONS
    reference: str   # noun phrase, e.g. "bottle"

    def human_label(self) -> str:
        rel = RELATION_LABELS.get(self.relation, self.relation)
        return f"{self.subject} must be {rel} {self.reference}"


@dataclass
class RuleEvidence:
    subject_found:        bool
    reference_found:      bool
    subject_centroid_x:   Optional[float] = None
    subject_centroid_y:   Optional[float] = None
    reference_centroid_x: Optional[float] = None
    reference_centroid_y: Optional[float] = None
    # Bounding-box centres: diagnostic only, never used for the verdict
    subject_bbox_center_x:   Optional[float] = None
    subject_bbox_center_y:   Optional[float] = None
    reference_bbox_center_x: Optional[float] = None
    reference_bbox_center_y: Optional[float] = None
    # Signed separation: positive means subject is RIGHT-of / BELOW reference
    measured_separation: Optional[float] = None
    image_width:         Optional[int]   = None
    image_height:        Optional[int]   = None
    # "x" for right_of / left_of;  "y" for above / below
    separation_axis:     str             = "x"
    # "mask" when both centroids came from validated decoded masks, else "none"
    # (there is no bounding-box substitute)
    centroid_source:     str             = "none"

    def separation_pct(self) -> Optional[float]:
        """Separation as % of the relevant image dimension, or None."""
        if self.measured_separation is None:
            return None
        if self.separation_axis == "y" and self.image_height:
            return abs(self.measured_separation) / self.image_height * 100
        if self.separation_axis == "x" and self.image_width:
            return abs(self.measured_separation) / self.image_width * 100
        return None


@dataclass
class RuleResult:
    verdict:          Verdict
    rule:             SpatialRule
    evidence:         RuleEvidence
    explanation:      str
    correction_hint:  str = ""

    def to_dict(self) -> dict:
        return {
            "verdict":         self.verdict.value,
            "rule":            self.rule.human_label(),
            "explanation":     self.explanation,
            "correction_hint": self.correction_hint,
            "evidence": {
                "subject_found":        self.evidence.subject_found,
                "reference_found":      self.evidence.reference_found,
                "centroid_source":      self.evidence.centroid_source,
                "subject_centroid": {
                    "x": self.evidence.subject_centroid_x,
                    "y": self.evidence.subject_centroid_y,
                },
                "reference_centroid": {
                    "x": self.evidence.reference_centroid_x,
                    "y": self.evidence.reference_centroid_y,
                },
                "subject_bbox_center": {
                    "x": self.evidence.subject_bbox_center_x,
                    "y": self.evidence.subject_bbox_center_y,
                },
                "reference_bbox_center": {
                    "x": self.evidence.reference_bbox_center_x,
                    "y": self.evidence.reference_bbox_center_y,
                },
                "measured_separation_px":  self.evidence.measured_separation,
                "measured_separation_pct": self.evidence.separation_pct(),
                "image_width":  self.evidence.image_width,
                "image_height": self.evidence.image_height,
            },
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _best_detection(
    detections: list[DetectedObject],
) -> tuple[Optional[DetectedObject], bool]:
    """
    Return (detection, ambiguous).

    - No detections  → (None, False)   — not found
    - Exactly one    → (detection, False)
    - More than one  → (None, True)    — ambiguous; use selection UI or return
                                         UNABLE_TO_VERIFY
    """
    if len(detections) == 0:
        return None, False
    if len(detections) > 1:
        return None, True
    return detections[0], False


def _has_empty_mask(det: DetectedObject) -> bool:
    """
    True if a mask was decoded (mask_raster is not None) but yielded no
    centroid because it has no foreground pixels. Distinct from "no mask".
    """
    return det.mask_raster is not None and det.mask_centroid_x is None


_BBOX_TOLERANCE_PX = 2.0
_BBOX_TOLERANCE_FRAC = 0.02


def _mask_problem(det: DetectedObject) -> Optional[str]:
    """
    Return a short reason why this detection has no valid mask centroid, or
    None if its mask centroid is valid for measurement.
    """
    cx, cy = det.mask_centroid_x, det.mask_centroid_y
    if cx is None or cy is None:
        status = getattr(det, "mask_status", "none")
        if status == "decode_failed":
            return "its segmentation mask could not be decoded"
        if status == "unmappable":
            return "its segmentation mask could not be mapped into the image"
        if status == "empty" or _has_empty_mask(det):
            return "its segmentation mask contains no foreground pixels"
        return "no segmentation mask was returned for it"

    if not (math.isfinite(cx) and math.isfinite(cy)):
        return "its mask centroid is not a valid number"

    # Mapping validation: inside the original image …
    if det.image_width and det.image_height:
        if not (0.0 <= cx <= det.image_width and 0.0 <= cy <= det.image_height):
            return "its mask centroid falls outside the image (mapping is unreliable)"

    # … and consistent with the detection's own bounding box.
    tol_x = _BBOX_TOLERANCE_PX + _BBOX_TOLERANCE_FRAC * det.width
    tol_y = _BBOX_TOLERANCE_PX + _BBOX_TOLERANCE_FRAC * det.height
    if not (det.x1 - tol_x <= cx <= det.x2 + 1 + tol_x
            and det.y1 - tol_y <= cy <= det.y2 + 1 + tol_y):
        return "its mask centroid is inconsistent with its bounding box (mapping is unreliable)"
    return None


# ---------------------------------------------------------------------------
# Evaluation logic
# ---------------------------------------------------------------------------

def evaluate_rule(
    rule: SpatialRule,
    subject_detections:   list[DetectedObject],
    reference_detections: list[DetectedObject],
) -> RuleResult:
    """
    Evaluate a spatial rule given detections for subject and reference.

    Measurements use mask centroids only. A missing, undecodable, empty or
    unmappable mask on EITHER object yields UNABLE TO VERIFY.
    """
    sub, sub_ambiguous = _best_detection(subject_detections)
    ref, ref_ambiguous = _best_detection(reference_detections)

    subject_found   = sub is not None
    reference_found = ref is not None

    img_w = (sub or ref).image_width  if (sub or ref) else None
    img_h = (sub or ref).image_height if (sub or ref) else None

    evidence = RuleEvidence(
        subject_found=subject_found,
        reference_found=reference_found,
        subject_centroid_x=sub.mask_centroid_x if sub else None,
        subject_centroid_y=sub.mask_centroid_y if sub else None,
        reference_centroid_x=ref.mask_centroid_x if ref else None,
        reference_centroid_y=ref.mask_centroid_y if ref else None,
        subject_bbox_center_x=sub.bbox_center_x if sub else None,
        subject_bbox_center_y=sub.bbox_center_y if sub else None,
        reference_bbox_center_x=ref.bbox_center_x if ref else None,
        reference_bbox_center_y=ref.bbox_center_y if ref else None,
        image_width=img_w,
        image_height=img_h,
    )

    # ---- UNABLE TO VERIFY — ambiguous multiple detections ----------------
    if sub_ambiguous:
        return RuleResult(
            verdict=Verdict.UNABLE_TO_VERIFY,
            rule=rule,
            evidence=evidence,
            explanation=(
                f"Multiple '{rule.subject}' detections found — cannot "
                "determine which one to evaluate. Use the selection UI to "
                "pick the intended instance."
            ),
            correction_hint=(
                f"Frame the shot so only one {rule.subject} is clearly "
                "visible, or use a more specific noun phrase."
            ),
        )

    if ref_ambiguous:
        return RuleResult(
            verdict=Verdict.UNABLE_TO_VERIFY,
            rule=rule,
            evidence=evidence,
            explanation=(
                f"Multiple '{rule.reference}' detections found — cannot "
                "determine which one to evaluate. Use the selection UI to "
                "pick the intended instance."
            ),
            correction_hint=(
                f"Frame the shot so only one {rule.reference} is clearly "
                "visible, or use a more specific noun phrase."
            ),
        )

    # ---- UNABLE TO VERIFY — objects not detected -------------------------
    if not subject_found and not reference_found:
        return RuleResult(
            verdict=Verdict.UNABLE_TO_VERIFY,
            rule=rule,
            evidence=evidence,
            explanation=(
                f"Neither '{rule.subject}' nor '{rule.reference}' "
                "were detected in this image."
            ),
            correction_hint=(
                f"Make sure both {rule.subject} and {rule.reference} "
                "are clearly visible and not obscured."
            ),
        )

    if not subject_found:
        return RuleResult(
            verdict=Verdict.UNABLE_TO_VERIFY,
            rule=rule,
            evidence=evidence,
            explanation=f"'{rule.subject}' was not detected in this image.",
            correction_hint=(
                f"Ensure {rule.subject} is clearly visible, "
                "well-lit, and not hidden behind other objects."
            ),
        )

    if not reference_found:
        return RuleResult(
            verdict=Verdict.UNABLE_TO_VERIFY,
            rule=rule,
            evidence=evidence,
            explanation=f"'{rule.reference}' was not detected in this image.",
            correction_hint=(
                f"Ensure {rule.reference} is clearly visible, "
                "well-lit, and not hidden behind other objects."
            ),
        )

    # ---- UNABLE TO VERIFY — mask missing / undecodable / empty / unmappable --
    problems = []
    sub_problem = _mask_problem(sub)
    ref_problem = _mask_problem(ref)
    if sub_problem:
        problems.append(f"'{rule.subject}': {sub_problem}")
    if ref_problem:
        problems.append(f"'{rule.reference}': {ref_problem}")
    if problems:
        return RuleResult(
            verdict=Verdict.UNABLE_TO_VERIFY,
            rule=rule,
            evidence=evidence,
            explanation=(
                "A PASS/FAIL decision requires a valid segmentation-mask centroid "
                "for both objects, and one is not available — "
                + "; ".join(problems)
                + ". Bounding-box centres are not used as a substitute."
            ),
            correction_hint=(
                "Re-check, or retake the photo so both objects are clearly visible "
                "and well lit; try a more specific object description."
            ),
        )

    # ---- Both masks valid — evaluate geometrically -----------------------
    sx, sy = sub.mask_centroid_x, sub.mask_centroid_y
    rx, ry = ref.mask_centroid_x, ref.mask_centroid_y
    evidence.centroid_source = "mask"
    measure_label = "mask centroid"

    relation = rule.relation

    if relation in ("right_of", "left_of"):
        separation = sx - rx       # positive → subject is right of reference
        evidence.measured_separation = separation
        evidence.separation_axis = "x"
        pct = abs(separation) / img_w * 100 if img_w else None
        pct_str = f" ({pct:.1f}% of width)" if pct is not None else ""

        if relation == "right_of":
            passed = separation > 0
            if passed:
                explanation = (
                    f"'{rule.subject}' {measure_label} is {abs(separation):.1f} px "
                    f"to the right of '{rule.reference}' {measure_label}{pct_str}. ✓"
                )
                hint = ""
            else:
                explanation = (
                    f"'{rule.subject}' {measure_label} is {abs(separation):.1f} px "
                    f"to the LEFT of '{rule.reference}' {measure_label}{pct_str}. "
                    "Rule requires it to be to the RIGHT."
                )
                hint = (
                    f"Move {rule.subject} further to the right "
                    f"of {rule.reference} in the frame."
                )
        else:  # left_of
            passed = separation < 0
            if passed:
                explanation = (
                    f"'{rule.subject}' {measure_label} is {abs(separation):.1f} px "
                    f"to the left of '{rule.reference}' {measure_label}{pct_str}. ✓"
                )
                hint = ""
            else:
                explanation = (
                    f"'{rule.subject}' {measure_label} is {abs(separation):.1f} px "
                    f"to the RIGHT of '{rule.reference}' {measure_label}{pct_str}. "
                    "Rule requires it to be to the LEFT."
                )
                hint = (
                    f"Move {rule.subject} further to the left "
                    f"of {rule.reference} in the frame."
                )

    elif relation in ("above", "below"):
        separation = sy - ry       # positive → subject is BELOW reference (y=0 at top)
        evidence.measured_separation = separation
        evidence.separation_axis = "y"
        pct = abs(separation) / img_h * 100 if img_h else None
        pct_str = f" ({pct:.1f}% of height)" if pct is not None else ""

        if relation == "above":
            passed = separation < 0
            if passed:
                explanation = (
                    f"'{rule.subject}' {measure_label} is {abs(separation):.1f} px "
                    f"above '{rule.reference}' {measure_label}{pct_str}. ✓"
                )
                hint = ""
            else:
                explanation = (
                    f"'{rule.subject}' {measure_label} is {abs(separation):.1f} px "
                    f"BELOW '{rule.reference}' {measure_label}{pct_str}. "
                    "Rule requires it to be ABOVE."
                )
                hint = (
                    f"Raise {rule.subject} so it appears above "
                    f"{rule.reference} in the frame."
                )
        else:  # below
            passed = separation > 0
            if passed:
                explanation = (
                    f"'{rule.subject}' {measure_label} is {abs(separation):.1f} px "
                    f"below '{rule.reference}' {measure_label}{pct_str}. ✓"
                )
                hint = ""
            else:
                explanation = (
                    f"'{rule.subject}' {measure_label} is {abs(separation):.1f} px "
                    f"ABOVE '{rule.reference}' {measure_label}{pct_str}. "
                    "Rule requires it to be BELOW."
                )
                hint = (
                    f"Lower {rule.subject} so it appears below "
                    f"{rule.reference} in the frame."
                )

    else:
        return RuleResult(
            verdict=Verdict.UNABLE_TO_VERIFY,
            rule=rule,
            evidence=evidence,
            explanation=(
                f"Unsupported relation '{relation}'. "
                f"Supported: {', '.join(SUPPORTED_RELATIONS)}"
            ),
            correction_hint="Select a supported spatial relation.",
        )

    return RuleResult(
        verdict=Verdict.PASS if passed else Verdict.FAIL,
        rule=rule,
        evidence=evidence,
        explanation=explanation,
        correction_hint=hint,
    )
