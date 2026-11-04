"""
tests.py — Offline unit tests for TakeState. No API calls required.

Run:
    python -m pytest tests.py -v
    # or simply:
    python tests.py
"""

from __future__ import annotations

import sys
import traceback
from dataclasses import dataclass
from typing import Callable


# ---------------------------------------------------------------------------
# Minimal test harness (avoids pytest dependency for quick runs)
# ---------------------------------------------------------------------------

@dataclass
class TestResult:
    name:    str
    passed:  bool
    message: str = ""

_results: list[TestResult] = []


def test(name: str):
    """Decorator that registers and runs a test function."""
    def decorator(fn: Callable):
        try:
            fn()
            _results.append(TestResult(name=name, passed=True))
        except AssertionError as exc:
            _results.append(TestResult(name=name, passed=False, message=str(exc)))
        except Exception as exc:  # noqa: BLE001
            _results.append(
                TestResult(name=name, passed=False, message=f"{type(exc).__name__}: {exc}")
            )
        return fn
    return decorator


def assert_eq(a, b, msg=""):
    assert a == b, msg or f"Expected {b!r}, got {a!r}"


def assert_close(a: float, b: float, tol: float = 0.5, msg=""):
    assert abs(a - b) <= tol, msg or f"{a} not within {tol} of {b}"


# ---------------------------------------------------------------------------
# Imports under test
# ---------------------------------------------------------------------------
from rule_engine import (
    SpatialRule, Verdict, evaluate_rule,
    SUPPORTED_RELATIONS, RELATION_LABELS, RuleEvidence,
)
from sam_parser import DetectedObject, parse_sam_result


# ---------------------------------------------------------------------------
# Synthetic DetectedObject factory
# ---------------------------------------------------------------------------

def _det_nomask(concept: str, x1: int, y1: int, x2: int, y2: int,
                img_w: int = 1000, img_h: int = 800) -> DetectedObject:
    """A detection with a bounding box ONLY: SAM returned no mask for it."""
    return DetectedObject(
        object_id=0,
        concept=concept,
        frame=0,
        x1=x1, y1=y1, x2=x2, y2=y2,
        image_width=img_w,
        image_height=img_h,
        parse_method="synthetic",
    )


def _det(concept: str, x1: int, y1: int, x2: int, y2: int,
         img_w: int = 1000, img_h: int = 800) -> DetectedObject:
    """
    A detection with a VALID decoded mask whose centroid sits at the middle of
    its box (so the box and mask agree). Tests that need a different centroid
    overwrite mask_centroid_x/y explicitly.
    """
    import numpy as np
    d = _det_nomask(concept, x1, y1, x2, y2, img_w, img_h)
    d.mask_raster = np.ones((4, 4), dtype=np.uint8)
    d.mask_width = d.mask_height = 4
    d.mask_centroid_x = d.bbox_center_x
    d.mask_centroid_y = d.bbox_center_y
    d.mask_status = "ok"
    return d


# ---------------------------------------------------------------------------
# Tests: DetectedObject geometry
# ---------------------------------------------------------------------------

@test("bbox_center_x is midpoint of x1 and x2")
def _():
    d = _det("obj", x1=100, y1=50, x2=200, y2=150)
    assert_close(d.bbox_center_x, 150.0)


@test("bbox_center_y is midpoint of y1 and y2")
def _():
    d = _det("obj", x1=100, y1=50, x2=200, y2=150)
    assert_close(d.bbox_center_y, 100.0)


@test("bbox_center is NOT affected by mask data")
def _():
    import numpy as np
    d = _det("obj", x1=10, y1=20, x2=110, y2=120)
    # Attach a fake mask — centre should still be from bbox, not mask pixels
    d.mask_raster = np.zeros((100, 100), dtype=np.uint8)
    assert_close(d.bbox_center_x, 60.0)
    assert_close(d.bbox_center_y, 70.0)


@test("DetectedObject.width and height are correct")
def _():
    d = _det("obj", x1=10, y1=20, x2=110, y2=120)
    assert_eq(d.width, 101)
    assert_eq(d.height, 101)


@test("to_dict includes bbox_center with x and y keys")
def _():
    d = _det("obj", x1=100, y1=50, x2=200, y2=150)
    d_dict = d.to_dict()
    assert "bbox_center" in d_dict
    assert "x" in d_dict["bbox_center"]
    assert "y" in d_dict["bbox_center"]
    assert_close(d_dict["bbox_center"]["x"], 150.0)
    assert_close(d_dict["bbox_center"]["y"], 100.0)


@test("to_dict does NOT include 'centroid' key (centroid naming removed)")
def _():
    d = _det("obj", x1=10, y1=10, x2=90, y2=90)
    d_dict = d.to_dict()
    assert "centroid" not in d_dict
    assert "centroid_x" not in d_dict
    assert "centroid_y" not in d_dict


# ---------------------------------------------------------------------------
# Tests: rule_engine — PASS cases
# ---------------------------------------------------------------------------

@test("right_of: subject right of reference → PASS")
def _():
    sub = _det("glass",  x1=600, y1=100, x2=700, y2=300)  # center 650
    ref = _det("bottle", x1=200, y1=100, x2=300, y2=300)  # center 250
    rule = SpatialRule("glass", "right_of", "bottle")
    result = evaluate_rule(rule, [sub], [ref])
    assert_eq(result.verdict, Verdict.PASS)


@test("left_of: subject left of reference → PASS")
def _():
    sub = _det("cup",   x1=100, y1=100, x2=200, y2=300)  # center 150
    ref = _det("plate", x1=500, y1=100, x2=600, y2=300)  # center 550
    rule = SpatialRule("cup", "left_of", "plate")
    result = evaluate_rule(rule, [sub], [ref])
    assert_eq(result.verdict, Verdict.PASS)


@test("above: subject above reference → PASS")
def _():
    sub = _det("label",   x1=100, y1=50,  x2=300, y2=150)  # center_y=100
    ref = _det("bottle",  x1=100, y1=300, x2=300, y2=500)  # center_y=400
    rule = SpatialRule("label", "above", "bottle")
    result = evaluate_rule(rule, [sub], [ref])
    assert_eq(result.verdict, Verdict.PASS)


@test("below: subject below reference → PASS")
def _():
    sub = _det("base",   x1=100, y1=400, x2=300, y2=500)  # center_y=450
    ref = _det("bottle", x1=100, y1=100, x2=300, y2=200)  # center_y=150
    rule = SpatialRule("base", "below", "bottle")
    result = evaluate_rule(rule, [sub], [ref])
    assert_eq(result.verdict, Verdict.PASS)


# ---------------------------------------------------------------------------
# Tests: rule_engine — FAIL cases
# ---------------------------------------------------------------------------

@test("right_of: subject left of reference → FAIL")
def _():
    sub = _det("glass",  x1=100, y1=100, x2=200, y2=300)  # center 150
    ref = _det("bottle", x1=600, y1=100, x2=700, y2=300)  # center 650
    rule = SpatialRule("glass", "right_of", "bottle")
    result = evaluate_rule(rule, [sub], [ref])
    assert_eq(result.verdict, Verdict.FAIL)


@test("left_of: subject right of reference → FAIL")
def _():
    sub = _det("cup",   x1=700, y1=100, x2=800, y2=300)  # center 750
    ref = _det("plate", x1=100, y1=100, x2=200, y2=300)  # center 150
    rule = SpatialRule("cup", "left_of", "plate")
    result = evaluate_rule(rule, [sub], [ref])
    assert_eq(result.verdict, Verdict.FAIL)


@test("above: subject below reference → FAIL")
def _():
    sub = _det("label",  x1=100, y1=400, x2=300, y2=500)  # center_y=450
    ref = _det("bottle", x1=100, y1=100, x2=300, y2=200)  # center_y=150
    rule = SpatialRule("label", "above", "bottle")
    result = evaluate_rule(rule, [sub], [ref])
    assert_eq(result.verdict, Verdict.FAIL)


@test("FAIL result includes non-empty correction_hint")
def _():
    sub = _det("glass",  x1=100, y1=100, x2=200, y2=300)
    ref = _det("bottle", x1=600, y1=100, x2=700, y2=300)
    rule = SpatialRule("glass", "right_of", "bottle")
    result = evaluate_rule(rule, [sub], [ref])
    assert result.correction_hint, "FAIL should include a correction hint"


# ---------------------------------------------------------------------------
# Tests: rule_engine — UNABLE TO VERIFY
# ---------------------------------------------------------------------------

@test("UNABLE TO VERIFY when both objects not detected")
def _():
    rule = SpatialRule("glass", "right_of", "bottle")
    result = evaluate_rule(rule, [], [])
    assert_eq(result.verdict, Verdict.UNABLE_TO_VERIFY)


@test("UNABLE TO VERIFY when subject not detected")
def _():
    ref = _det("bottle", x1=300, y1=100, x2=400, y2=300)
    rule = SpatialRule("glass", "right_of", "bottle")
    result = evaluate_rule(rule, [], [ref])
    assert_eq(result.verdict, Verdict.UNABLE_TO_VERIFY)


@test("UNABLE TO VERIFY when reference not detected")
def _():
    sub = _det("glass", x1=600, y1=100, x2=700, y2=300)
    rule = SpatialRule("glass", "right_of", "bottle")
    result = evaluate_rule(rule, [sub], [])
    assert_eq(result.verdict, Verdict.UNABLE_TO_VERIFY)


@test("UNABLE TO VERIFY for unsupported relation")
def _():
    sub = _det("glass",  x1=600, y1=100, x2=700, y2=300)
    ref = _det("bottle", x1=300, y1=100, x2=400, y2=300)
    rule = SpatialRule("glass", "diagonal_of", "bottle")  # not in SUPPORTED_RELATIONS
    result = evaluate_rule(rule, [sub], [ref])
    assert_eq(result.verdict, Verdict.UNABLE_TO_VERIFY)


# ---------------------------------------------------------------------------
# Tests: rule_engine — evidence fields
# ---------------------------------------------------------------------------

@test("Evidence records measured_separation for x-axis rule")
def _():
    sub = _det("glass",  x1=600, y1=100, x2=700, y2=300)  # center_x=650
    ref = _det("bottle", x1=200, y1=100, x2=300, y2=300)  # center_x=250
    rule = SpatialRule("glass", "right_of", "bottle")
    result = evaluate_rule(rule, [sub], [ref])
    assert result.evidence.measured_separation is not None
    assert_close(result.evidence.measured_separation, 400.0, tol=1.0)


@test("Evidence records image dimensions")
def _():
    sub = _det("glass",  x1=600, y1=100, x2=700, y2=300, img_w=1280, img_h=720)
    ref = _det("bottle", x1=200, y1=100, x2=300, y2=300, img_w=1280, img_h=720)
    rule = SpatialRule("glass", "right_of", "bottle")
    result = evaluate_rule(rule, [sub], [ref])
    assert_eq(result.evidence.image_width, 1280)
    assert_eq(result.evidence.image_height, 720)


@test("RuleResult.to_dict has expected keys")
def _():
    sub = _det("glass",  x1=600, y1=100, x2=700, y2=300)
    ref = _det("bottle", x1=200, y1=100, x2=300, y2=300)
    rule = SpatialRule("glass", "right_of", "bottle")
    result = evaluate_rule(rule, [sub], [ref])
    d = result.to_dict()
    for key in ("verdict", "rule", "explanation", "correction_hint", "evidence"):
        assert key in d, f"Missing key: {key}"
    ev = d["evidence"]
    assert "subject_bbox_center" in ev
    assert "reference_bbox_center" in ev
    assert "measured_separation_px" in ev


# ---------------------------------------------------------------------------
# Tests: sam_parser — regex path (no API, no library needed)
# ---------------------------------------------------------------------------

def _make_events_with_output_text(text: str) -> list[dict]:
    """
    Minimal SSE event list that matches the real SAM wire format.
    In response.output_text.delta events, 'delta' is a PLAIN STRING,
    not {"type": "output_text", "text": ...}.
    """
    return [
        {"_event_type": "response.output_text.delta",
         "delta": text},       # ← plain string, not a nested dict
        {"_event_type": "response.completed"},
    ]


@test("Regex parser extracts box coordinates from wire format")
def _():
    wire = "1<|box;x1=100;y1=50;x2=300;y2=250;w=200;h=200|>"
    events = _make_events_with_output_text(wire)
    objects, method = parse_sam_result(events, "glass")
    assert_eq(method, "regex")
    assert len(objects) == 1, f"Expected 1 object, got {len(objects)}"
    obj = objects[0]
    assert_eq(obj.x1, 100)
    assert_eq(obj.y1, 50)
    assert_eq(obj.x2, 300)
    assert_eq(obj.y2, 250)


@test("Regex parser returns empty list for empty output")
def _():
    events = _make_events_with_output_text("")
    objects, method = parse_sam_result(events, "glass")
    assert_eq(method, "regex")
    assert_eq(len(objects), 0)


@test("Regex parser handles multiple detections")
def _():
    wire = (
        "1<|box;x1=10;y1=10;x2=100;y2=100;w=90;h=90|>"
        "2<|box;x1=200;y1=200;x2=300;y2=300;w=100;h=100|>"
    )
    events = _make_events_with_output_text(wire)
    objects, _ = parse_sam_result(events, "cup")
    assert_eq(len(objects), 2)
    assert_eq(objects[0].x1, 10)
    assert_eq(objects[1].x1, 200)


@test("Regex parser sets concept correctly")
def _():
    wire = "1<|box;x1=10;y1=10;x2=50;y2=50;w=40;h=40|>"
    events = _make_events_with_output_text(wire)
    objects, _ = parse_sam_result(events, "my_concept")
    assert_eq(objects[0].concept, "my_concept")


@test("parse_sam_result returns 'regex' or 'library' method note")
def _():
    wire = "1<|box;x1=10;y1=10;x2=50;y2=50;w=40;h=40|>"
    events = _make_events_with_output_text(wire)
    _, method = parse_sam_result(events, "x")
    assert method in ("regex", "library"), f"Unexpected method: {method}"


# ---------------------------------------------------------------------------
# Tests: sam_parser — string-delta extraction
# ---------------------------------------------------------------------------

@test("String delta extraction: plain string delta is collected correctly")
def _():
    from sam_parser import _extract_output_text
    events = [
        {"_event_type": "response.output_text.delta", "delta": "hello "},
        {"_event_type": "response.output_text.delta", "delta": "world"},
        {"_event_type": "response.completed"},
    ]
    text = _extract_output_text(events)
    assert text == "hello world", f"Expected 'hello world', got {text!r}"


@test("String delta extraction: dict delta is NOT collected (wrong format)")
def _():
    from sam_parser import _extract_output_text
    # Old wrong format — should not contribute text
    events = [
        {"_event_type": "response.output_text.delta",
         "delta": {"type": "output_text", "text": "should be ignored"}},
    ]
    text = _extract_output_text(events)
    assert text == "", f"Expected empty string, got {text!r}"


@test("String delta extraction: empty delta is skipped")
def _():
    from sam_parser import _extract_output_text
    events = [
        {"_event_type": "response.output_text.delta", "delta": ""},
        {"_event_type": "response.output_text.delta", "delta": "abc"},
    ]
    text = _extract_output_text(events)
    assert text == "abc", f"Expected 'abc', got {text!r}"


# ---------------------------------------------------------------------------
# Tests: sam_parser — SAM request payload shape
# ---------------------------------------------------------------------------

@test("SAM payload uses 'input_image' (not 'image_url') block type")
def _():
    import json, inspect
    import meta_client as mc
    # Read the source and find the payload dict — look for the literal string
    src = inspect.getsource(mc.call_sam_image)
    assert '"input_image"' in src, (
        "call_sam_image payload must use type='input_image', not 'image_url'"
    )
    assert '"image_url"' in src, (
        "call_sam_image payload must have 'image_url' key for the data URL"
    )


@test("SAM payload uses 'input_text' (not 'text') block type")
def _():
    import inspect
    import meta_client as mc
    src = inspect.getsource(mc.call_sam_image)
    assert '"input_text"' in src, (
        "call_sam_image payload must use type='input_text', not 'text'"
    )


@test("SAM payload image_url is flat string, not nested {'url':...}")
def _():
    import inspect
    import meta_client as mc
    src = inspect.getsource(mc.call_sam_image)
    # Wrong nested form: "image_url": {"url": ...}
    assert '"image_url": {"url"' not in src, (
        "image_url must be a flat string, not a nested dict with a 'url' key"
    )


# ---------------------------------------------------------------------------
# Tests: rule_engine — ambiguous detections → UNABLE TO VERIFY
# ---------------------------------------------------------------------------

@test("UNABLE TO VERIFY when multiple subject detections (ambiguous)")
def _():
    sub1 = _det("glass", x1=100, y1=100, x2=200, y2=200)
    sub2 = _det("glass", x1=500, y1=100, x2=600, y2=200)
    ref  = _det("bottle", x1=300, y1=100, x2=400, y2=200)
    rule = SpatialRule("glass", "right_of", "bottle")
    result = evaluate_rule(rule, [sub1, sub2], [ref])
    assert_eq(result.verdict, Verdict.UNABLE_TO_VERIFY)
    assert "Multiple" in result.explanation, (
        f"Explanation should mention ambiguity; got: {result.explanation}"
    )


@test("UNABLE TO VERIFY when multiple reference detections (ambiguous)")
def _():
    sub  = _det("glass",  x1=600, y1=100, x2=700, y2=200)
    ref1 = _det("bottle", x1=100, y1=100, x2=200, y2=200)
    ref2 = _det("bottle", x1=300, y1=100, x2=400, y2=200)
    rule = SpatialRule("glass", "right_of", "bottle")
    result = evaluate_rule(rule, [sub], [ref1, ref2])
    assert_eq(result.verdict, Verdict.UNABLE_TO_VERIFY)
    assert "Multiple" in result.explanation


@test("Exactly one detection on each side → PASS/FAIL (not UNABLE)")
def _():
    sub = _det("glass",  x1=600, y1=100, x2=700, y2=200)
    ref = _det("bottle", x1=200, y1=100, x2=300, y2=200)
    rule = SpatialRule("glass", "right_of", "bottle")
    result = evaluate_rule(rule, [sub], [ref])
    assert result.verdict in (Verdict.PASS, Verdict.FAIL), (
        f"One detection each should be PASS or FAIL, got {result.verdict}"
    )


# ---------------------------------------------------------------------------
# Tests: rule_engine — vertical axis separation uses image_height
# ---------------------------------------------------------------------------

@test("above rule: separation_axis is 'y'")
def _():
    sub = _det("label",  x1=100, y1=50,  x2=300, y2=150)
    ref = _det("bottle", x1=100, y1=300, x2=300, y2=500)
    rule = SpatialRule("label", "above", "bottle")
    result = evaluate_rule(rule, [sub], [ref])
    assert_eq(result.evidence.separation_axis, "y")


@test("below rule: separation_axis is 'y'")
def _():
    sub = _det("base",   x1=100, y1=400, x2=300, y2=500)
    ref = _det("bottle", x1=100, y1=100, x2=300, y2=200)
    rule = SpatialRule("base", "below", "bottle")
    result = evaluate_rule(rule, [sub], [ref])
    assert_eq(result.evidence.separation_axis, "y")


@test("right_of rule: separation_axis is 'x'")
def _():
    sub = _det("glass",  x1=600, y1=100, x2=700, y2=300)
    ref = _det("bottle", x1=200, y1=100, x2=300, y2=300)
    rule = SpatialRule("glass", "right_of", "bottle")
    result = evaluate_rule(rule, [sub], [ref])
    assert_eq(result.evidence.separation_axis, "x")


@test("separation_pct uses image_height for above/below rules")
def _():
    # above: sub center_y=100, ref center_y=400, separation = -300 px
    # image_height = 800 → pct = 300/800*100 = 37.5%
    sub = _det("label",  x1=100, y1=50,  x2=300, y2=150, img_h=800)
    ref = _det("bottle", x1=100, y1=300, x2=300, y2=500, img_h=800)
    rule = SpatialRule("label", "above", "bottle")
    result = evaluate_rule(rule, [sub], [ref])
    pct = result.evidence.separation_pct()
    assert pct is not None, "separation_pct() returned None"
    assert_close(pct, 37.5, tol=1.0)


@test("separation_pct uses image_width for right_of/left_of rules")
def _():
    # right_of: sub center_x=650, ref center_x=250, separation = 400 px
    # image_width = 1000 → pct = 400/1000*100 = 40.0%
    sub = _det("glass",  x1=600, y1=100, x2=700, y2=300, img_w=1000)
    ref = _det("bottle", x1=200, y1=100, x2=300, y2=300, img_w=1000)
    rule = SpatialRule("glass", "right_of", "bottle")
    result = evaluate_rule(rule, [sub], [ref])
    pct = result.evidence.separation_pct()
    assert pct is not None, "separation_pct() returned None"
    assert_close(pct, 40.0, tol=1.0)


@test("separation_pct does NOT use image_width for vertical rules")
def _():
    # below: sub center_y=450, ref center_y=150, separation = 300 px
    # image_height=800, image_width=1000
    # correct:   300/800*100 = 37.5%
    # incorrect: 300/1000*100 = 30.0% (would happen if axis check is wrong)
    sub = _det("base",   x1=100, y1=400, x2=300, y2=500, img_w=1000, img_h=800)
    ref = _det("bottle", x1=100, y1=100, x2=300, y2=200, img_w=1000, img_h=800)
    rule = SpatialRule("base", "below", "bottle")
    result = evaluate_rule(rule, [sub], [ref])
    pct = result.evidence.separation_pct()
    assert pct is not None, "separation_pct() returned None"
    # Must be close to 37.5% (height-based), not 30.0% (width-based)
    assert abs(pct - 37.5) < 1.0, (
        f"Expected ~37.5% (height-based), got {pct:.1f}% — "
        "axis check may be using image_width for a y-axis rule"
    )


# ---------------------------------------------------------------------------
# Tests: meta_client — key exposure prevention
# ---------------------------------------------------------------------------

@test("get_api_key_status returns only 'configured' key")
def _():
    from meta_client import get_api_key_status
    status = get_api_key_status()
    assert isinstance(status, dict), "Expected dict"
    assert set(status.keys()) == {"configured"}, (
        f"Unexpected keys in status: {set(status.keys())}"
    )
    assert isinstance(status["configured"], bool)


@test("get_api_key_status does not contain 'key', 'prefix', or 'token'")
def _():
    from meta_client import get_api_key_status
    status = get_api_key_status()
    lowered = {k.lower(): str(v).lower() for k, v in status.items()}
    for forbidden in ("key", "prefix", "token", "secret", "bearer"):
        assert forbidden not in lowered, f"Forbidden word '{forbidden}' found in status"


# ---------------------------------------------------------------------------
# Tests: interpreter — manual_rule (no API)
# ---------------------------------------------------------------------------

@test("manual_rule builds correct SpatialRule")
def _():
    from interpreter import manual_rule
    pr = manual_rule("Glass", "right_of", "Bottle")
    assert pr is not None
    assert pr.subject == "glass"
    assert pr.relation == "right_of"
    assert pr.reference == "bottle"
    assert pr.supported is True
    assert pr.source == "manual"


@test("manual_rule returns None for unsupported relation")
def _():
    from interpreter import manual_rule
    pr = manual_rule("glass", "diagonal_of", "bottle")
    assert pr is None


@test("manual_rule to_spatial_rule returns SpatialRule")
def _():
    from interpreter import manual_rule
    from rule_engine import SpatialRule
    pr = manual_rule("glass", "left_of", "bottle")
    assert pr is not None
    sr = pr.to_spatial_rule()
    assert isinstance(sr, SpatialRule)
    assert sr.subject == "glass"
    assert sr.relation == "left_of"


@test("manual_rule strips and lowercases inputs")
def _():
    from interpreter import manual_rule
    pr = manual_rule("  WINE GLASS  ", "above", "  TABLE  ")
    assert pr is not None
    assert pr.subject == "wine glass"
    assert pr.reference == "table"


# ---------------------------------------------------------------------------
# Tests: compute_mask_centroid
# ---------------------------------------------------------------------------

@test("compute_mask_centroid: centroid of all-foreground mask is center")
def _():
    from sam_parser import compute_mask_centroid
    import numpy as np
    # 4x4 mask, all pixels foreground → centroid at (1.5, 1.5) in mask coords
    # bounds: (0, 0) → (40, 40) → source centroid at (15.0, 15.0)
    mask_bytes = bytes([1] * 16)   # 4×4 all ones
    result = compute_mask_centroid(mask_bytes, 4, 4, 0.0, 0.0, 40.0, 40.0)
    assert result is not None, "Expected a centroid tuple"
    sx, sy = result
    assert_close(sx, 15.0, tol=0.5)
    assert_close(sy, 15.0, tol=0.5)


@test("compute_mask_centroid: empty mask (all zeros) returns None")
def _():
    from sam_parser import compute_mask_centroid
    mask_bytes = bytes([0] * 100)   # 10×10 all zeros
    result = compute_mask_centroid(mask_bytes, 10, 10, 0.0, 0.0, 100.0, 100.0)
    assert result is None, f"Expected None for empty mask, got {result}"


@test("compute_mask_centroid: source-image coordinate mapping is correct")
def _():
    from sam_parser import compute_mask_centroid
    import numpy as np
    # 10×10 mask with single foreground pixel at row=5, col=5
    arr = np.zeros((10, 10), dtype=np.uint8)
    arr[5, 5] = 1
    mask_bytes = arr.tobytes()
    # bounds: (100, 200) → (200, 300)  (source rect 100×100 px)
    result = compute_mask_centroid(mask_bytes, 10, 10, 100.0, 200.0, 200.0, 300.0)
    assert result is not None
    sx, sy = result
    # col=5 → source_x = 100 + 5 * (200-100) / 10 = 100 + 50 = 150
    # row=5 → source_y = 200 + 5 * (300-200) / 10 = 200 + 50 = 250
    assert_close(sx, 150.0, tol=0.5)
    assert_close(sy, 250.0, tol=0.5)


@test("compute_mask_centroid: malformed (too-short) bytes returns None")
def _():
    from sam_parser import compute_mask_centroid
    mask_bytes = bytes([1, 1, 1])   # only 3 bytes, but claims 10×10
    result = compute_mask_centroid(mask_bytes, 10, 10, 0.0, 0.0, 100.0, 100.0)
    assert result is None, f"Expected None for malformed mask, got {result}"


@test("compute_mask_centroid: single top-left pixel maps to bounds origin")
def _():
    from sam_parser import compute_mask_centroid
    import numpy as np
    arr = np.zeros((10, 10), dtype=np.uint8)
    arr[0, 0] = 1   # top-left pixel
    mask_bytes = arr.tobytes()
    result = compute_mask_centroid(mask_bytes, 10, 10, 50.0, 80.0, 150.0, 180.0)
    assert result is not None
    sx, sy = result
    # col=0 → source_x = 50 + 0 * 100/10 = 50
    # row=0 → source_y = 80 + 0 * 100/10 = 80
    assert_close(sx, 50.0, tol=0.5)
    assert_close(sy, 80.0, tol=0.5)


# ---------------------------------------------------------------------------
# Tests: DetectedObject — centroid_x/y and centroid_source
# ---------------------------------------------------------------------------

@test("DetectedObject has no bbox-substituting centroid_x/centroid_y properties")
def _():
    d = _det_nomask("obj", x1=100, y1=50, x2=200, y2=150)
    assert not hasattr(d, "centroid_x") and not hasattr(d, "centroid_y"), (
        "a property that substitutes the bbox centre for a mask centroid must not exist"
    )


@test("centroid_source is 'none' (not 'bbox') when there is no mask centroid")
def _():
    d = _det_nomask("obj", x1=100, y1=50, x2=200, y2=150)
    assert d.centroid_source == "none"


@test("centroid_source is 'mask' when mask_centroid_x is set")
def _():
    d = _det_nomask("obj", x1=100, y1=50, x2=200, y2=150)
    d.mask_centroid_x = 150.0
    d.mask_centroid_y = 100.0
    assert d.centroid_source == "mask"


@test("to_dict includes mask_centroid x and y keys")
def _():
    d = _det("obj", x1=100, y1=50, x2=200, y2=150)
    d.mask_centroid_x = 155.0
    d.mask_centroid_y = 105.0
    d_dict = d.to_dict()
    assert "mask_centroid" in d_dict
    assert_close(d_dict["mask_centroid"]["x"], 155.0)
    assert_close(d_dict["mask_centroid"]["y"], 105.0)


@test("to_dict centroid_source is 'mask' when mask_centroid set")
def _():
    d = _det("obj", x1=100, y1=50, x2=200, y2=150)
    d.mask_centroid_x = 155.0
    d.mask_centroid_y = 105.0
    d_dict = d.to_dict()
    assert d_dict["centroid_source"] == "mask"


@test("to_dict centroid_source is 'none' when no mask centroid; keeps bbox_center")
def _():
    d = _det_nomask("obj", x1=100, y1=50, x2=200, y2=150)
    d_dict = d.to_dict()
    assert d_dict["centroid_source"] == "none"
    assert d_dict["mask_centroid"] == {"x": None, "y": None}
    assert_close(d_dict["bbox_center"]["x"], 150.0)


# ---------------------------------------------------------------------------
# Tests: rule_engine — _has_empty_mask
# ---------------------------------------------------------------------------

@test("_has_empty_mask: True when mask_raster set but mask_centroid_x is None")
def _():
    from rule_engine import _has_empty_mask
    import numpy as np
    d = _det("obj", x1=100, y1=50, x2=200, y2=150)
    d.mask_raster = np.zeros((50, 50), dtype=np.uint8)   # decoded but empty
    d.mask_centroid_x = None    # no foreground pixels → centroid not computed
    assert _has_empty_mask(d), "Expected _has_empty_mask=True"


@test("_has_empty_mask: False when mask_centroid_x is set")
def _():
    from rule_engine import _has_empty_mask
    import numpy as np
    d = _det("obj", x1=100, y1=50, x2=200, y2=150)
    d.mask_raster = np.zeros((50, 50), dtype=np.uint8)
    d.mask_centroid_x = 150.0
    d.mask_centroid_y = 100.0
    assert not _has_empty_mask(d), "Expected _has_empty_mask=False when centroid known"


@test("_has_empty_mask: False when mask_raster is None (no mask at all)")
def _():
    from rule_engine import _has_empty_mask
    d = _det_nomask("obj", x1=100, y1=50, x2=200, y2=150)
    # mask_raster stays None → no mask returned by SAM (distinct from an empty mask)
    assert not _has_empty_mask(d), "Expected _has_empty_mask=False when no mask"


# ---------------------------------------------------------------------------
# Tests: rule_engine — UNABLE TO VERIFY for empty masks
# ---------------------------------------------------------------------------

@test("UNABLE TO VERIFY when subject has empty decoded mask")
def _():
    import numpy as np
    sub = _det("glass", x1=600, y1=100, x2=700, y2=300)
    sub.mask_raster = np.zeros((50, 50), dtype=np.uint8)   # empty mask
    sub.mask_centroid_x = None   # no foreground pixels
    ref = _det("bottle", x1=200, y1=100, x2=300, y2=300)
    rule = SpatialRule("glass", "right_of", "bottle")
    result = evaluate_rule(rule, [sub], [ref])
    assert_eq(result.verdict, Verdict.UNABLE_TO_VERIFY)
    assert "no foreground" in result.explanation or "empty" in result.explanation.lower()


@test("UNABLE TO VERIFY when reference has empty decoded mask")
def _():
    import numpy as np
    sub = _det("glass", x1=600, y1=100, x2=700, y2=300)
    ref = _det("bottle", x1=200, y1=100, x2=300, y2=300)
    ref.mask_raster = np.zeros((50, 50), dtype=np.uint8)   # empty mask
    ref.mask_centroid_x = None
    rule = SpatialRule("glass", "right_of", "bottle")
    result = evaluate_rule(rule, [sub], [ref])
    assert_eq(result.verdict, Verdict.UNABLE_TO_VERIFY)


# ---------------------------------------------------------------------------
# Tests: rule_engine — evaluates with mask centroid vs bbox fallback
# ---------------------------------------------------------------------------

@test("Rule engine uses the mask centroid, not the bbox centre, for the verdict")
def _():
    # Box centres say: subject (350) is LEFT of reference (550)  -> would FAIL right_of
    # Mask centroids say: subject (550) is RIGHT of reference (350) -> PASS right_of
    sub = _det("glass",  x1=100, y1=100, x2=600, y2=300)
    ref = _det("bottle", x1=300, y1=100, x2=800, y2=300)
    sub.mask_centroid_x, sub.mask_centroid_y = 550.0, 200.0
    ref.mask_centroid_x, ref.mask_centroid_y = 350.0, 200.0
    result = evaluate_rule(SpatialRule("glass", "right_of", "bottle"), [sub], [ref])
    assert_eq(result.verdict, Verdict.PASS)
    assert_eq(result.evidence.centroid_source, "mask")
    # bbox centres are still reported, separately, as diagnostics
    assert_close(result.evidence.subject_bbox_center_x, 350.0)
    assert_close(result.evidence.reference_bbox_center_x, 550.0)


@test("centroid_source in evidence is 'mask' when both objects have valid masks")
def _():
    sub = _det("glass",  x1=600, y1=100, x2=700, y2=300)
    ref = _det("bottle", x1=200, y1=100, x2=300, y2=300)
    sub.mask_centroid_x = 650.0; sub.mask_centroid_y = 200.0
    ref.mask_centroid_x = 250.0; ref.mask_centroid_y = 200.0
    result = evaluate_rule(SpatialRule("glass", "right_of", "bottle"), [sub], [ref])
    assert_eq(result.evidence.centroid_source, "mask")


@test("Explanation string mentions 'mask centroid' when mask data available")
def _():
    sub = _det("glass",  x1=600, y1=100, x2=700, y2=300)
    ref = _det("bottle", x1=200, y1=100, x2=300, y2=300)
    sub.mask_centroid_x = 650.0; sub.mask_centroid_y = 200.0
    ref.mask_centroid_x = 250.0; ref.mask_centroid_y = 200.0
    rule = SpatialRule("glass", "right_of", "bottle")
    result = evaluate_rule(rule, [sub], [ref])
    assert "mask centroid" in result.explanation, (
        f"Expected 'mask centroid' in explanation, got: {result.explanation}"
    )


# ---------------------------------------------------------------------------
# REGRESSION: mask-only verdict invariant
# PASS/FAIL requires valid decoded mask centroids for BOTH objects. Anything
# else is UNABLE TO VERIFY; bounding boxes are never a substitute.
# ---------------------------------------------------------------------------

import numpy as _np

_ALL_RELATIONS = ("right_of", "left_of", "above", "below")


def _bbox_would_pass(relation: str):
    """Two boxes (no masks) whose CENTRES clearly satisfy `relation` for glass vs bottle."""
    if relation == "right_of":
        return (600, 100, 700, 300), (200, 100, 300, 300)
    if relation == "left_of":
        return (200, 100, 300, 300), (600, 100, 700, 300)
    if relation == "above":
        return (100, 50, 300, 150), (100, 400, 300, 600)
    return (100, 400, 300, 600), (100, 50, 300, 150)   # below


def _assert_unable_no_substitute(result):
    assert_eq(result.verdict, Verdict.UNABLE_TO_VERIFY,
              f"expected UNABLE TO VERIFY, got {result.verdict}: {result.explanation}")
    assert result.evidence.measured_separation is None, "no separation may be computed"
    assert_eq(result.evidence.centroid_source, "none")
    assert "not used as a substitute" in result.explanation, result.explanation


@test("REGRESSION missing masks: boxes alone never give PASS/FAIL (all four relations)")
def _():
    for rel in _ALL_RELATIONS:
        sb, rb = _bbox_would_pass(rel)
        sub = _det_nomask("glass", *sb)
        ref = _det_nomask("bottle", *rb)
        res = evaluate_rule(SpatialRule("glass", rel, "bottle"), [sub], [ref])
        _assert_unable_no_substitute(res)
        assert "no segmentation mask was returned" in res.explanation


@test("REGRESSION missing mask on the subject only → UNABLE (reference mask valid)")
def _():
    sub = _det_nomask("glass", 600, 100, 700, 300)
    ref = _det("bottle", 200, 100, 300, 300)
    res = evaluate_rule(SpatialRule("glass", "right_of", "bottle"), [sub], [ref])
    _assert_unable_no_substitute(res)
    assert "'glass'" in res.explanation and "'bottle'" not in res.explanation.split("—", 1)[1]


@test("REGRESSION missing mask on the reference only → UNABLE (subject mask valid)")
def _():
    sub = _det("glass", 600, 100, 700, 300)
    ref = _det_nomask("bottle", 200, 100, 300, 300)
    res = evaluate_rule(SpatialRule("glass", "right_of", "bottle"), [sub], [ref])
    _assert_unable_no_substitute(res)
    assert "'bottle'" in res.explanation.split("—", 1)[1]


@test("REGRESSION undecodable masks → UNABLE, with the decode failure reported")
def _():
    for which in ("subject", "reference", "both"):
        sub = _det("glass", 600, 100, 700, 300)
        ref = _det("bottle", 200, 100, 300, 300)
        for det, name in ((sub, "subject"), (ref, "reference")):
            if which in (name, "both"):
                det.mask_centroid_x = det.mask_centroid_y = None
                det.mask_raster = None
                det.mask_status = "decode_failed"
        res = evaluate_rule(SpatialRule("glass", "right_of", "bottle"), [sub], [ref])
        _assert_unable_no_substitute(res)
        assert "could not be decoded" in res.explanation


@test("REGRESSION empty masks (no foreground pixels) → UNABLE")
def _():
    for which in ("subject", "reference", "both"):
        sub = _det("glass", 600, 100, 700, 300)
        ref = _det("bottle", 200, 100, 300, 300)
        for det, name in ((sub, "subject"), (ref, "reference")):
            if which in (name, "both"):
                det.mask_raster = _np.zeros((10, 10), dtype=_np.uint8)
                det.mask_centroid_x = det.mask_centroid_y = None
                det.mask_status = "empty"
        res = evaluate_rule(SpatialRule("glass", "right_of", "bottle"), [sub], [ref])
        _assert_unable_no_substitute(res)
        assert "no foreground pixels" in res.explanation


@test("REGRESSION one valid mask + one invalid mask (every failure type, both sides) → UNABLE")
def _():
    def broken(det, kind):
        det.mask_centroid_x = det.mask_centroid_y = None
        if kind == "none":
            det.mask_raster = None; det.mask_status = "none"
        elif kind == "decode_failed":
            det.mask_raster = None; det.mask_status = "decode_failed"
        elif kind == "empty":
            det.mask_raster = _np.zeros((4, 4), dtype=_np.uint8); det.mask_status = "empty"
        elif kind == "unmappable":
            det.mask_raster = _np.ones((4, 4), dtype=_np.uint8); det.mask_status = "unmappable"
    for kind in ("none", "decode_failed", "empty", "unmappable"):
        for bad in ("subject", "reference"):
            sub = _det("glass", 600, 100, 700, 300)
            ref = _det("bottle", 200, 100, 300, 300)
            broken(sub if bad == "subject" else ref, kind)
            res = evaluate_rule(SpatialRule("glass", "right_of", "bottle"), [sub], [ref])
            _assert_unable_no_substitute(res)


@test("REGRESSION diagnostics: bbox centres still reported while verdict is UNABLE")
def _():
    sub = _det_nomask("glass", 600, 100, 700, 300)
    ref = _det_nomask("bottle", 200, 100, 300, 300)
    res = evaluate_rule(SpatialRule("glass", "right_of", "bottle"), [sub], [ref])
    ev = res.evidence
    assert ev.subject_centroid_x is None and ev.reference_centroid_x is None
    assert_close(ev.subject_bbox_center_x, 650.0)
    assert_close(ev.reference_bbox_center_x, 250.0)


@test("REGRESSION unreliable mapping → UNABLE (outside image / NaN / inconsistent with box / bad status)")
def _():
    rule = SpatialRule("glass", "right_of", "bottle")
    def run(mutate):
        sub = _det("glass", 600, 100, 700, 300)
        ref = _det("bottle", 200, 100, 300, 300)
        mutate(sub)
        return evaluate_rule(rule, [sub], [ref])
    def outside(d): d.mask_centroid_x = 5000.0
    def nan(d): d.mask_centroid_x = float("nan")
    def inf(d): d.mask_centroid_y = float("inf")
    def far_from_box(d): d.mask_centroid_x = 900.0     # inside image, outside its own box
    def unmappable(d):
        d.mask_centroid_x = d.mask_centroid_y = None; d.mask_status = "unmappable"
    for fn in (outside, nan, inf, far_from_box, unmappable):
        res = run(fn)
        assert_eq(res.verdict, Verdict.UNABLE_TO_VERIFY, f"{fn.__name__}: {res.explanation}")


@test("REGRESSION ambiguity is still UNABLE even when every candidate has a valid mask")
def _():
    s1 = _det("glass", 100, 100, 200, 200)
    s2 = _det("glass", 500, 100, 600, 200)
    ref = _det("bottle", 300, 100, 400, 200)
    res = evaluate_rule(SpatialRule("glass", "right_of", "bottle"), [s1, s2], [ref])
    assert_eq(res.verdict, Verdict.UNABLE_TO_VERIFY)


@test("Valid masks on both objects still give PASS and FAIL for all four relations")
def _():
    for rel in _ALL_RELATIONS:
        sb, rb = _bbox_would_pass(rel)
        ok = evaluate_rule(SpatialRule("glass", rel, "bottle"),
                           [_det("glass", *sb)], [_det("bottle", *rb)])
        assert_eq(ok.verdict, Verdict.PASS, f"{rel}: {ok.explanation}")
        bad = evaluate_rule(SpatialRule("glass", rel, "bottle"),
                            [_det("glass", *rb)], [_det("bottle", *sb)])
        assert_eq(bad.verdict, Verdict.FAIL, f"{rel}: {bad.explanation}")
        assert "mask centroid" in ok.explanation


# ---------------------------------------------------------------------------
# REGRESSION: parser populates mask_status honestly (meta_sam_parser faked offline)
# ---------------------------------------------------------------------------

def _parse_with_fake_library(records, decode):
    """Run sam_parser._parse_with_library against fake parser output (no network)."""
    import meta_sam_parser as lib
    import sam_parser

    class _Finish:
        def __init__(self): self.result = type("R", (), {"records": records})()

    class _Parser:
        def push(self, delta): pass
        def finish(self, outcome): return _Finish()

    class _Fmt:
        def create_parser(self): return _Parser()

    saved = (lib.image_segmentation_format, lib.decode_mask_to_raster)
    lib.image_segmentation_format = lambda: _Fmt()
    lib.decode_mask_to_raster = decode
    try:
        events = [{"_event_type": "response.output_text.delta",
                   "delta": "<|image;w=1000;h=800|>"}]
        return sam_parser._parse_with_library(events, "glass")
    finally:
        lib.image_segmentation_format, lib.decode_mask_to_raster = saved


def _box_rec(oid="a"):
    import meta_sam_parser as lib
    return lib.SegmentationBoxRecord(order=0, object_id=oid, left=100.0, top=200.0,
                                     right=201.0, bottom=301.0)


def _mask_rec(oid="a", w=10, h=10, bounds=(100.0, 200.0, 200.0, 300.0)):
    import meta_sam_parser as lib
    return lib.SegmentationMaskRecord(
        order=1, object_id=oid, identity=None, revision=0,
        mask=lib.SegmentationMask(encoding="rle", payload="x", width=w, height=h),
        bounds=lib.SegmentationMaskBounds(*bounds))


@test("Parser: box record with no mask record → mask_status 'none', no centroid")
def _():
    out = _parse_with_fake_library([_box_rec()], decode=lambda m: b"")
    assert_eq(len(out), 1)
    assert_eq(out[0].mask_status, "none")
    assert out[0].mask_centroid_x is None and out[0].mask_raster is None


@test("Parser: undecodable mask payload → mask_status 'decode_failed', no centroid")
def _():
    def boom(mask): raise ValueError("invalid payload")
    out = _parse_with_fake_library([_box_rec(), _mask_rec()], decode=boom)
    assert_eq(out[0].mask_status, "decode_failed")
    assert out[0].mask_centroid_x is None and out[0].mask_raster is None


@test("Parser: decoded raster shorter than width*height → 'decode_failed'")
def _():
    out = _parse_with_fake_library([_box_rec(), _mask_rec()], decode=lambda m: b"\x01\x01")
    assert_eq(out[0].mask_status, "decode_failed")


@test("Parser: all-zero mask → mask_status 'empty', no centroid")
def _():
    out = _parse_with_fake_library([_box_rec(), _mask_rec()], decode=lambda m: bytes(100))
    assert_eq(out[0].mask_status, "empty")
    assert out[0].mask_centroid_x is None
    assert out[0].mask_raster is not None


@test("Parser: degenerate mask bounds → mask_status 'unmappable', no centroid")
def _():
    out = _parse_with_fake_library([_box_rec(), _mask_rec(bounds=(100.0, 200.0, 100.0, 300.0))],
                                   decode=lambda m: bytes([1]) * 100)
    assert_eq(out[0].mask_status, "unmappable")
    assert out[0].mask_centroid_x is None


@test("Parser: valid mask → centroid mapped into source-image pixels, status 'ok'")
def _():
    arr = _np.zeros((10, 10), dtype=_np.uint8)
    arr[2:6, 4:8] = 1            # rows 2..5 (mean 3.5), cols 4..7 (mean 5.5)
    out = _parse_with_fake_library([_box_rec(), _mask_rec()], decode=lambda m: arr.tobytes())
    d = out[0]
    assert_eq(d.mask_status, "ok")
    assert_close(d.mask_centroid_x, 100.0 + 5.5 * 100.0 / 10, tol=0.01)   # 155
    assert_close(d.mask_centroid_y, 200.0 + 3.5 * 100.0 / 10, tol=0.01)   # 235
    assert_eq(d.centroid_source, "mask")
    # and the whole chain agrees: box (100,200)-(200,300) contains the centroid
    ref = _det("bottle", 20, 200, 80, 300)
    res = evaluate_rule(SpatialRule("glass", "right_of", "bottle"), [d], [ref])
    assert_eq(res.verdict, Verdict.PASS)


@test("Parser: two detections pair each box with its own mask by object_id")
def _():
    a = _np.zeros((10, 10), dtype=_np.uint8); a[0:2, 0:2] = 1
    b = _np.ones((10, 10), dtype=_np.uint8)
    blobs = {"A": a.tobytes(), "B": b.tobytes()}
    recs = [_box_rec("A"), _box_rec("B"), _mask_rec("A"), _mask_rec("B")]
    calls = iter(["A", "B"])
    out = _parse_with_fake_library(recs, decode=lambda m: blobs[next(calls)])
    assert_eq(len(out), 2)
    assert_close(out[0].mask_centroid_x, 100.0 + 0.5 * 10, tol=0.01)
    assert_close(out[1].mask_centroid_x, 100.0 + 4.5 * 10, tol=0.01)


@test("Regex fallback (box-only text) yields detections with NO mask centroid")
def _():
    wire = "1<|box;x1=100;y1=50;x2=300;y2=250;w=200;h=200|>"
    objs, method = parse_sam_result(_make_events_with_output_text(wire), "glass")
    assert_eq(len(objs), 1)
    assert objs[0].mask_centroid_x is None and objs[0].centroid_source == "none"
    res = evaluate_rule(SpatialRule("glass", "right_of", "bottle"), objs,
                        parse_sam_result(_make_events_with_output_text(
                            "1<|box;x1=10;y1=50;x2=30;y2=250;w=20;h=200|>"), "bottle")[0])
    assert_eq(res.verdict, Verdict.UNABLE_TO_VERIFY)


@test("compute_mask_centroid: degenerate (zero-size) bounds → None, not a pixel-offset guess")
def _():
    from sam_parser import compute_mask_centroid
    assert compute_mask_centroid(bytes([1]) * 100, 10, 10, 5.0, 5.0, 5.0, 50.0) is None
    assert compute_mask_centroid(bytes([1]) * 100, 10, 10, 5.0, 50.0, 50.0, 5.0) is None


# ---------------------------------------------------------------------------
# Tests: SUPPORTED_RELATIONS completeness
# ---------------------------------------------------------------------------

@test("All SUPPORTED_RELATIONS have RELATION_LABELS entries")
def _():
    for rel in SUPPORTED_RELATIONS:
        assert rel in RELATION_LABELS, f"No label for relation '{rel}'"


@test("Verdict enum has PASS, FAIL, UNABLE_TO_VERIFY")
def _():
    assert Verdict.PASS.value == "PASS"
    assert Verdict.FAIL.value == "FAIL"
    assert Verdict.UNABLE_TO_VERIFY.value == "UNABLE TO VERIFY"


# ---------------------------------------------------------------------------
# Tests: config — secrets lookup, access code, call budget (no network)
# ---------------------------------------------------------------------------

import os as _os


class _Env:
    """Temporarily set/unset environment variables inside a test."""
    def __init__(self, **kv): self.kv, self.old = kv, {}
    def __enter__(self):
        for k, v in self.kv.items():
            self.old[k] = _os.environ.get(k)
            if v is None: _os.environ.pop(k, None)
            else: _os.environ[k] = v
    def __exit__(self, *a):
        for k, v in self.old.items():
            if v is None: _os.environ.pop(k, None)
            else: _os.environ[k] = v


@test("config.get_secret reads the environment")
def _():
    from config import get_secret
    with _Env(TS_TEST_SETTING="abc"):
        assert_eq(get_secret("TS_TEST_SETTING"), "abc")


@test("config.get_secret returns None when unset and no secrets file")
def _():
    from config import get_secret
    with _Env(TS_TEST_MISSING=None):
        assert get_secret("TS_TEST_MISSING") is None


@test("get_api_key_status reflects MODEL_API_KEY without exposing it")
def _():
    from meta_client import get_api_key_status
    with _Env(MODEL_API_KEY="dummy-not-a-real-key"):
        st_on = get_api_key_status()
    with _Env(MODEL_API_KEY=None):
        st_off = get_api_key_status()
    assert_eq(st_on, {"configured": True})
    assert "dummy" not in str(st_on)
    # Only meaningful when no key exists in .env/secrets either
    assert st_off["configured"] in (True, False)


@test("meta_client._safe_error redacts the key from messages")
def _():
    import meta_client as mc
    with _Env(MODEL_API_KEY="dummy-secret-123"):
        out = mc._safe_error("failed with dummy-secret-123 in text")
    assert "dummy-secret-123" not in out and "[REDACTED]" in out


@test("config.get_int_setting: default, valid, malformed, negative")
def _():
    from config import get_int_setting
    with _Env(TS_N=None): assert_eq(get_int_setting("TS_N", 7), 7)
    with _Env(TS_N="3"):  assert_eq(get_int_setting("TS_N", 7), 3)
    with _Env(TS_N="x"):  assert_eq(get_int_setting("TS_N", 7), 7)
    with _Env(TS_N="-4"): assert_eq(get_int_setting("TS_N", 7), 7)


@test("access code: not required when unset; constant-time compare when set")
def _():
    from config import access_code_required, check_access_code
    with _Env(APP_ACCESS_CODE=None):
        assert not access_code_required()
        assert not check_access_code("anything")
    with _Env(APP_ACCESS_CODE="letmein"):
        assert access_code_required()
        assert check_access_code("letmein")
        assert not check_access_code("wrong")
        assert not check_access_code("")


@test("CallBudget: spends up to the limit and then refuses without consuming")
def _():
    from config import CallBudget
    b = CallBudget(2)
    assert b.spend() and b.spend()
    assert not b.spend()
    assert_eq(b.used, 2)
    assert_eq(b.remaining, 0)


@test("CallBudget: limit 0 refuses everything")
def _():
    from config import CallBudget
    assert not CallBudget(0).spend()


# ---------------------------------------------------------------------------
# Tests: Muse Spark structured-output schema + local validation (no network)
# ---------------------------------------------------------------------------

@test("Spark schema: strict, all four fields required, relation enum, no extras")
def _():
    from interpreter import _RESPONSE_FORMAT as rf
    js = rf["json_schema"]
    assert js["strict"] is True
    sch = js["schema"]
    assert_eq(sch["additionalProperties"], False)
    assert_eq(set(sch["required"]),
              {"subject", "relation", "reference", "unsupported_reason"})
    assert_eq(set(sch["properties"]["relation"]["enum"]),
              {"right_of", "left_of", "above", "below", "unsupported"})


def _spark_returning(payload: str):
    """Patch meta_client.call_spark_chat to return a canned reply (offline)."""
    import meta_client as mc
    orig = mc.call_spark_chat
    mc.call_spark_chat = lambda system, user, response_format=None: (payload, None)
    return lambda: setattr(mc, "call_spark_chat", orig)


@test("parse_requirement: valid structured reply → supported rule")
def _():
    from interpreter import parse_requirement
    undo = _spark_returning(
        '{"subject":"Glass","relation":"right_of","reference":"Bottle","unsupported_reason":""}')
    try:
        pr, err = parse_requirement("keep the glass right of the bottle")
    finally:
        undo()
    assert err is None and pr.supported
    assert_eq((pr.subject, pr.relation, pr.reference), ("glass", "right_of", "bottle"))


@test("parse_requirement: 'unsupported' relation is never passed to the engine")
def _():
    from interpreter import parse_requirement
    undo = _spark_returning(
        '{"subject":"glass","relation":"unsupported","reference":"bottle",'
        '"unsupported_reason":"colour is not a spatial relation"}')
    try:
        pr, err = parse_requirement("make the glass red")
    finally:
        undo()
    assert err is None and not pr.supported
    assert pr.to_spatial_rule() is None


@test("parse_requirement: invented relation is rejected locally")
def _():
    from interpreter import parse_requirement
    undo = _spark_returning(
        '{"subject":"glass","relation":"diagonal_of","reference":"bottle","unsupported_reason":""}')
    try:
        pr, err = parse_requirement("x")
    finally:
        undo()
    assert pr is not None and not pr.supported and pr.to_spatial_rule() is None


@test("parse_requirement: malformed JSON → error, no rule")
def _():
    from interpreter import parse_requirement
    undo = _spark_returning("not json at all")
    try:
        pr, err = parse_requirement("x")
    finally:
        undo()
    assert pr is None and err


@test("parse_requirement: empty subject → error")
def _():
    from interpreter import parse_requirement
    undo = _spark_returning(
        '{"subject":"","relation":"above","reference":"box","unsupported_reason":""}')
    try:
        pr, err = parse_requirement("x")
    finally:
        undo()
    assert pr is None and err


@test("parse_requirement: absurdly long object phrase → error")
def _():
    from interpreter import parse_requirement
    undo = _spark_returning(
        '{"subject":"%s","relation":"above","reference":"box","unsupported_reason":""}' % ("a" * 200))
    try:
        pr, err = parse_requirement("x")
    finally:
        undo()
    assert pr is None and err


# ---------------------------------------------------------------------------
# Tests: parser — zero results, incomplete stream, mask decode failure (offline)
# ---------------------------------------------------------------------------

@test("parse_sam_result: completed stream with no output text → zero detections")
def _():
    objects, _m = parse_sam_result([{"_event_type": "response.completed"}], "glass")
    assert_eq(objects, [])


@test("compute_mask_centroid: zero-sized mask dimensions → None")
def _():
    from sam_parser import compute_mask_centroid
    assert compute_mask_centroid(b"", 0, 0, 0.0, 0.0, 10.0, 10.0) is None


@test("compute_mask_centroid: off-centre blob maps proportionally into bounds")
def _():
    from sam_parser import compute_mask_centroid
    import numpy as np
    arr = np.zeros((20, 10), dtype=np.uint8)     # height 20, width 10
    arr[10:20, 5:10] = 1                          # lower-right quadrant
    cx, cy = compute_mask_centroid(arr.tobytes(), 10, 20, 0.0, 0.0, 100.0, 200.0)
    # mean col = 7.0 → 7.0 * 100/10 = 70 ; mean row = 14.5 → 14.5 * 200/20 = 145
    assert_close(cx, 70.0, tol=0.01)
    assert_close(cy, 145.0, tol=0.01)


@test("Vertical percentage uses image height even when mask centroids are used")
def _():
    sub = _det("base", x1=90, y1=300, x2=110, y2=400, img_w=1000, img_h=500)
    ref = _det("bottle", x1=90, y1=50, x2=110, y2=150, img_w=1000, img_h=500)
    sub.mask_centroid_x, sub.mask_centroid_y = 100.0, 350.0
    ref.mask_centroid_x, ref.mask_centroid_y = 100.0, 100.0
    res = evaluate_rule(SpatialRule("base", "below", "bottle"), [sub], [ref])
    assert_eq(res.verdict, Verdict.PASS)
    assert_close(res.evidence.separation_pct(), 50.0, tol=0.01)   # 250/500


# ---------------------------------------------------------------------------
# Tests: Streamlit rerun / API-call protection and retake state (AppTest, offline)
# call_sam_image is replaced by a counter; no network, no real key.
# ---------------------------------------------------------------------------

def _run_app_with_fake_sam(max_checks: str = "20", with_masks: bool = True):
    """
    Build an AppTest with a counting fake SAM request and a fake parse result.
    with_masks=True  → detections carry valid mask centroids
    with_masks=False → boxes only (SAM returned no masks)
    Returns (at, calls, restore). No network, no real key.
    """
    import io
    import meta_client as mc
    import sam_parser
    from PIL import Image
    from streamlit.testing.v1 import AppTest

    calls: list[str] = []
    boxes = {"glass": (600, 100, 700, 300), "bottle": (200, 100, 300, 300)}

    def fake_sam(data_url, concept):
        calls.append(concept)
        return [], None

    def fake_parse(events, concept):
        box = boxes.get(concept)
        if box is None:
            return [], "fake"
        make = _det if with_masks else _det_nomask
        return [make(concept, *box, img_w=1000, img_h=800)], "fake"

    orig_sam, orig_parse = mc.call_sam_image, sam_parser.parse_sam_result
    mc.call_sam_image = fake_sam
    sam_parser.parse_sam_result = fake_parse

    buf = io.BytesIO()
    Image.new("RGB", (1000, 800), (128, 128, 128)).save(buf, format="JPEG")

    at = AppTest.from_file("app.py", default_timeout=60)
    at.session_state["take_bytes"] = buf.getvalue()
    at.session_state["take_data_url"] = "data:image/jpeg;base64,AAAA"
    at.session_state["take_w"] = 1000
    at.session_state["take_h"] = 800

    def restore():
        mc.call_sam_image = orig_sam
        sam_parser.parse_sam_result = orig_parse

    return at, calls, restore


def _btn(at, prefix):
    return [b for b in at.button if b.label.startswith(prefix)][0]


@test("App: reruns and interpretation do not call SAM; Check calls exactly twice")
def _():
    with _Env(MODEL_API_KEY="dummy-not-a-real-key", MAX_TAKE_CHECKS="20", APP_ACCESS_CODE=None):
        at, calls, restore = _run_app_with_fake_sam()
        try:
            at.run(); at.run(); at.run()
            assert_eq(len(calls), 0, "page reruns must not call SAM")
            _btn(at, "✅").click().run()
            assert_eq(sorted(calls), ["bottle", "glass"])
            at.run(); at.run()
            assert_eq(len(calls), 2, "reruns after a check must not call SAM again")
            res = at.session_state["take_result"]
            assert res is not None and res.verdict == Verdict.PASS
        finally:
            restore()


@test("App: explicit Re-check is the only way to repeat SAM calls")
def _():
    with _Env(MODEL_API_KEY="dummy-not-a-real-key", MAX_TAKE_CHECKS="20", APP_ACCESS_CODE=None):
        at, calls, restore = _run_app_with_fake_sam()
        try:
            at.run()
            _btn(at, "✅").click().run()
            assert_eq(len(calls), 2)
            assert _btn(at, "✅").disabled, "Check must be disabled once analysed"
            _btn(at, "🔄").click().run()
            assert_eq(len(calls), 4)
        finally:
            restore()


@test("App: session take-check budget blocks further SAM calls")
def _():
    with _Env(MODEL_API_KEY="dummy-not-a-real-key", MAX_TAKE_CHECKS="1", APP_ACCESS_CODE=None):
        at, calls, restore = _run_app_with_fake_sam()
        try:
            at.run()
            _btn(at, "✅").click().run()
            assert_eq(len(calls), 2)
            _btn(at, "🔄").click().run()
            assert_eq(len(calls), 2, "budget exhausted: no further SAM calls")
            assert any("limit" in e.value.lower() for e in at.error)
        finally:
            restore()


@test("App: rule is preserved and history grows across a retake")
def _():
    with _Env(MODEL_API_KEY="dummy-not-a-real-key", MAX_TAKE_CHECKS="20", APP_ACCESS_CODE=None):
        at, calls, restore = _run_app_with_fake_sam()
        try:
            at.run()
            _btn(at, "✅").click().run()
            first = at.session_state["take_result"]
            assert_eq(first.verdict, Verdict.PASS)
            assert_eq(len(at.session_state["takes"]), 1)
            rule_before = (at.session_state["subject"], at.session_state["relation"],
                           at.session_state["reference"])
            # Simulate uploading a retake: new digest, cleared analysis, same rule
            at.session_state["take_digest"] = "retake-digest"
            at.session_state["take_sub_dets"] = None
            at.session_state["take_ref_dets"] = None
            at.session_state["take_result"] = None
            at.session_state["take_annotated"] = None
            at.run()
            _btn(at, "✅").click().run()
            assert_eq(len(at.session_state["takes"]), 2)
            assert_eq(rule_before, (at.session_state["subject"],
                                    at.session_state["relation"],
                                    at.session_state["reference"]))
        finally:
            restore()


@test("App: access code gate hides the workflow until unlocked")
def _():
    with _Env(MODEL_API_KEY="dummy-not-a-real-key", APP_ACCESS_CODE="letmein"):
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file("app.py", default_timeout=60).run()
        labels = [b.label for b in at.button]
        assert "Unlock" in labels
        assert not any(l.startswith("✅") for l in labels), "workflow must be hidden"
        at.text_input(key="access_code_input").set_value("wrong")
        at.button[0].click().run()
        assert any("Incorrect" in e.value for e in at.error)
        at.text_input(key="access_code_input").set_value("letmein")
        at.button[0].click().run()
        assert any(b.label.startswith("✅") for b in at.button), "should unlock"


@test("App end-to-end: SAM boxes without masks → UNABLE TO VERIFY, never PASS/FAIL")
def _():
    with _Env(MODEL_API_KEY="dummy-not-a-real-key", MAX_TAKE_CHECKS="20", APP_ACCESS_CODE=None):
        at, calls, restore = _run_app_with_fake_sam(with_masks=False)
        try:
            at.run()
            _btn(at, "✅").click().run()
            assert_eq(len(calls), 2)
            res = at.session_state["take_result"]
            assert_eq(res.verdict, Verdict.UNABLE_TO_VERIFY)
            assert_eq(res.evidence.centroid_source, "none")
            assert res.evidence.measured_separation is None
        finally:
            restore()


@test("App: MODEL_API_KEY supplied only via Streamlit secrets is detected (no banner)")
def _():
    from streamlit.testing.v1 import AppTest
    with _Env(MODEL_API_KEY=None, APP_ACCESS_CODE=None):
        at = AppTest.from_file("app.py", default_timeout=60)
        at.secrets["MODEL_API_KEY"] = "dummy-not-a-real-key"
        at.run()
        assert not at.exception, [e.value for e in at.exception]
        assert not any("not configured" in e.value for e in at.error)
        interpret = [b for b in at.button if b.label.startswith("Interpret")][0]
        assert not interpret.disabled


@test("App: access code can also come from Streamlit secrets")
def _():
    from streamlit.testing.v1 import AppTest
    with _Env(MODEL_API_KEY="dummy-not-a-real-key", APP_ACCESS_CODE=None):
        at = AppTest.from_file("app.py", default_timeout=60)
        at.secrets["APP_ACCESS_CODE"] = "from-secrets"
        at.run()
        assert "Unlock" in [b.label for b in at.button]


@test("App: warns when a key is configured but no access code is set")
def _():
    from streamlit.testing.v1 import AppTest
    with _Env(MODEL_API_KEY="dummy-not-a-real-key", APP_ACCESS_CODE=None):
        at = AppTest.from_file("app.py", default_timeout=60).run()
        assert any("APP_ACCESS_CODE" in w.value for w in at.warning)
    with _Env(MODEL_API_KEY="dummy-not-a-real-key", APP_ACCESS_CODE="x"):
        at = AppTest.from_file("app.py", default_timeout=60).run()
        assert not any("No APP_ACCESS_CODE" in w.value for w in at.warning)


@test("Upload limits: oversize and non-image files are rejected with a clear error")
def _():
    import io
    import importlib, types
    # normalise_upload lives in app.py; load just that function without running the UI
    src = open("app.py", encoding="utf-8").read()
    start = src.index("MAX_UPLOAD_BYTES")
    end = src.index("def _take_digest")
    ns: dict = {}
    exec("import io, base64\nfrom PIL import Image, ImageOps\n" + src[start:end], ns)
    norm = ns["normalise_upload"]
    class _F:
        def __init__(self, b): self._b = b
        def getvalue(self): return self._b
    from PIL import Image
    buf = io.BytesIO(); Image.new("RGB", (3000, 1500), (9, 9, 9)).save(buf, format="PNG")
    jpeg, url, w, h = norm(_F(buf.getvalue()))
    assert max(w, h) == 2048 and url.startswith("data:image/jpeg;base64,")
    for bad in (b"not an image", b"x" * (11 * 1024 * 1024)):
        try:
            norm(_F(bad)); raise AssertionError("expected ValueError")
        except ValueError as exc:
            assert str(exc)


# ---------------------------------------------------------------------------
# Run and report
# ---------------------------------------------------------------------------

def run_all() -> int:
    """Run all registered tests and print results. Returns exit code."""
    passed = sum(1 for r in _results if r.passed)
    failed = sum(1 for r in _results if not r.passed)
    total  = len(_results)

    print(f"\n{'='*60}")
    print(f"  TakeState offline tests — {total} tests")
    print(f"{'='*60}")
    for r in _results:
        icon = "✅" if r.passed else "❌"
        print(f"  {icon}  {r.name}")
        if not r.passed and r.message:
            print(f"       ↳ {r.message}")
    print(f"{'='*60}")
    print(f"  Passed: {passed}  |  Failed: {failed}  |  Total: {total}")
    print(f"{'='*60}\n")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(run_all())
