"""
sam_parser.py — Parse SAM 3.1 streaming output into DetectedObject records.

Parse path (in order):
  1. meta_sam_parser library  (pip install meta-sam-parser==0.1.0)
  2. Regex fallback on the wire-format string

Coordinate conventions
----------------------
DetectedObject stores INCLUSIVE bounding box coordinates:
  x1, y1 = top-left pixel (inclusive)
  x2, y2 = bottom-right pixel (inclusive)

The library uses HALF-OPEN floating-point coordinates:
  right = x2 + 1,  bottom = y2 + 1

Measurement invariant
---------------------
Verdicts use ONLY the mask centroid: mask_centroid_x / mask_centroid_y, the
mean position of the decoded foreground pixels mapped into source-image
coordinates via SegmentationMaskBounds.

bbox_center_x / bbox_center_y (midpoint of the bounding box) are diagnostic
measurements for display only. They are never substituted for a mask centroid.

mask_status records why a mask centroid is absent:
  "ok"            centroid computed
  "none"          no mask record was returned for this detection (or box-only output)
  "decode_failed" a mask record was returned but could not be decoded
  "empty"         the mask decoded but has no foreground pixels
  "unmappable"    the mask could not be reliably mapped into the image
rule_engine returns UNABLE_TO_VERIFY for every status except "ok".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class DetectedObject:
    object_id: int
    concept: str
    frame: int
    x1: int
    y1: int
    x2: int
    y2: int
    image_width: int
    image_height: int
    confidence: Optional[float] = None
    mask_raster: Optional[np.ndarray] = field(default=None, repr=False)
    mask_height: Optional[int] = None
    mask_width: Optional[int] = None
    # Mask centroid in source-image pixel space (None if no mask / empty mask)
    mask_centroid_x: Optional[float] = None
    mask_centroid_y: Optional[float] = None
    # Bounds of the mask in source-image coordinates (for overlay rendering)
    mask_bounds_left:   Optional[int] = None
    mask_bounds_top:    Optional[int] = None
    mask_bounds_right:  Optional[int] = None
    mask_bounds_bottom: Optional[int] = None
    mask_status: str = "none"   # see module docstring
    parse_method: str = "unknown"

    # ------------------------------------------------------------------
    # Bounding-box geometry (always available)
    # ------------------------------------------------------------------

    @property
    def bbox_center_x(self) -> float:
        """Horizontal midpoint of bounding box. NOT a mask centroid."""
        return (self.x1 + self.x2) / 2.0

    @property
    def bbox_center_y(self) -> float:
        """Vertical midpoint of bounding box. NOT a mask centroid."""
        return (self.y1 + self.y2) / 2.0

    @property
    def width(self) -> int:
        return self.x2 - self.x1 + 1

    @property
    def height(self) -> int:
        return self.y2 - self.y1 + 1

    @property
    def centroid_source(self) -> str:
        """'mask' if a decoded-mask centroid is available, else 'none'.
        There is deliberately no bounding-box substitute."""
        return "mask" if self.mask_centroid_x is not None else "none"

    def to_dict(self) -> dict:
        return {
            "object_id": self.object_id,
            "concept": self.concept,
            "frame": self.frame,
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "bbox_center": {          # diagnostic only; never used for verdicts
                "x": self.bbox_center_x,
                "y": self.bbox_center_y,
            },
            "mask_centroid": {
                "x": self.mask_centroid_x,
                "y": self.mask_centroid_y,
            },
            "centroid_source": self.centroid_source,
            "mask_status": self.mask_status,
            "bbox_size": {"w": self.width, "h": self.height},
            "image_size": {"w": self.image_width, "h": self.image_height},
            "confidence": self.confidence,
            "has_mask": self.mask_raster is not None,
            "parse_method": self.parse_method,
        }


# ---------------------------------------------------------------------------
# Mask centroid calculation
# ---------------------------------------------------------------------------

def compute_mask_centroid(
    mask_bytes: bytes,
    mask_width: int,
    mask_height: int,
    bounds_left: float,
    bounds_top: float,
    bounds_right: float,
    bounds_bottom: float,
) -> Optional[tuple[float, float]]:
    """
    Calculate the centroid of foreground pixels in source-image coordinates.

    Parameters
    ----------
    mask_bytes   : flat bytes returned by decode_mask_to_raster(); each byte 0 or 1
    mask_width   : width of the mask in pixels
    mask_height  : height of the mask in pixels
    bounds_*     : source-image rectangle that the mask covers (from SegmentationMaskBounds)

    Returns
    -------
    (source_x, source_y)  — centroid mapped into source-image pixel coordinates,
    or None if the mask has no foreground pixels, is malformed, or the bounds
    are degenerate (zero or negative size).
    """
    if not mask_bytes or mask_width <= 0 or mask_height <= 0:
        return None

    expected = mask_width * mask_height
    if len(mask_bytes) < expected:
        return None

    arr = np.frombuffer(mask_bytes[:expected], dtype=np.uint8).reshape(mask_height, mask_width)

    rows, cols = np.where(arr > 0)
    if len(rows) == 0:
        return None

    centroid_row = float(np.mean(rows))   # 0 = top of mask
    centroid_col = float(np.mean(cols))   # 0 = left of mask

    # Map from mask-pixel space into source-image space via bounds
    bounds_w = bounds_right - bounds_left
    bounds_h = bounds_bottom - bounds_top

    if not (bounds_w > 0 and bounds_h > 0):
        return None   # cannot be mapped reliably into source-image coordinates

    source_x = bounds_left + centroid_col * bounds_w / mask_width
    source_y = bounds_top  + centroid_row * bounds_h / mask_height
    return source_x, source_y


# ---------------------------------------------------------------------------
# Regex patterns for the SAM wire format (fallback path)
# ---------------------------------------------------------------------------

_BOX_RE = re.compile(
    r"<\|box;x1=(?P<x1>\d+);y1=(?P<y1>\d+);x2=(?P<x2>\d+);y2=(?P<y2>\d+)"
    r"(?:;w=\d+)?(?:;h=\d+)?\|>"
)
_OBJ_ID_RE = re.compile(r"(\d+)\s*<\|box")
_IMG_DIM_RE = re.compile(r"<\|image;w=(?P<w>\d+);h=(?P<h>\d+)\|>")


# ---------------------------------------------------------------------------
# _extract_output_text
# ---------------------------------------------------------------------------

def _extract_output_text(events: list[dict]) -> str:
    """
    Concatenate all output text from the SAM SSE event list.

    In the real Responses API, ``response.output_text.delta`` events carry
    ``delta`` as a plain string.  A fallback also collects any ``output_text``
    key present on the event dict directly.
    """
    parts: list[str] = []
    for evt in events:
        # Primary path: response.output_text.delta — delta is a plain string
        delta = evt.get("delta", "")
        if isinstance(delta, str) and delta:
            parts.append(delta)
        # Secondary path: some event shapes embed output_text directly
        ot = evt.get("output_text", "")
        if isinstance(ot, str) and ot:
            parts.append(ot)
    return "".join(parts)


# ---------------------------------------------------------------------------
# Library parse path
# ---------------------------------------------------------------------------

def _parse_with_library(
    events: list[dict],
    concept: str,
) -> Optional[list[DetectedObject]]:
    """
    Try to parse using the official meta_sam_parser library (v0.1.0 sync API).

    Processes BOTH SegmentationBoxRecord and SegmentationMaskRecord.
    Matches box and mask records by object_id, decodes mask rasters,
    and computes mask centroids via compute_mask_centroid().

    Returns a list of DetectedObject on success (may be empty if no matches),
    or None if the library is unavailable or a parse exception occurs.
    """
    try:
        from meta_sam_parser import (  # type: ignore
            image_segmentation_format,
            CompletedOutcome,
            SegmentationBoxRecord,
            SegmentationMaskRecord,
            decode_mask_to_raster,
        )
    except ImportError:
        return None

    try:
        fmt = image_segmentation_format()
        parser = fmt.create_parser()

        for evt in events:
            delta = evt.get("delta", "")
            if isinstance(delta, str) and delta:
                parser.push(delta)

        finish = parser.finish(CompletedOutcome())

        # Extract image dimensions from the raw text stream (not in records)
        full_text = "".join(
            evt.get("delta", "")
            for evt in events
            if isinstance(evt.get("delta", ""), str)
        )
        img_w, img_h = 0, 0
        dim_m = _IMG_DIM_RE.search(full_text)
        if dim_m:
            img_w = int(dim_m.group("w"))
            img_h = int(dim_m.group("h"))

        # Separate box records from mask records
        box_records: dict[str, object] = {}
        mask_records: dict[str, object] = {}

        for rec in finish.result.records:
            if isinstance(rec, SegmentationBoxRecord):
                box_records[rec.object_id] = rec
            elif isinstance(rec, SegmentationMaskRecord):
                mask_records[rec.object_id] = rec

        objects: list[DetectedObject] = []

        for oid, box_rec in box_records.items():
            # Library uses half-open coords: right = x2+1, bottom = y2+1
            x1 = int(box_rec.left)
            y1 = int(box_rec.top)
            x2 = int(box_rec.right)  - 1
            y2 = int(box_rec.bottom) - 1

            det = DetectedObject(
                object_id=hash(oid) & 0xFFFF,   # stable int for display
                concept=concept,
                frame=int(box_rec.frame) if box_rec.frame is not None else 0,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                image_width=img_w,
                image_height=img_h,
                confidence=(
                    float(box_rec.confidence)
                    if box_rec.confidence is not None else None
                ),
                parse_method="library",
            )

            # Decode the corresponding mask record (if present)
            mask_rec = mask_records.get(oid)
            if mask_rec is not None:
                try:
                    raw_bytes = decode_mask_to_raster(mask_rec.mask)
                    mw = int(mask_rec.mask.width)
                    mh = int(mask_rec.mask.height)
                    bl = float(mask_rec.bounds.left)
                    bt = float(mask_rec.bounds.top)
                    br = float(mask_rec.bounds.right)
                    bb = float(mask_rec.bounds.bottom)

                    # Convert to numpy uint8 array for display/overlay
                    arr = np.frombuffer(raw_bytes[:mw * mh], dtype=np.uint8).reshape(mh, mw)
                except Exception:  # noqa: BLE001 — invalid payload, bad dimensions, ...
                    det.mask_status = "decode_failed"
                else:
                    det.mask_raster = arr
                    det.mask_width  = mw
                    det.mask_height = mh
                    det.mask_bounds_left   = int(bl)
                    det.mask_bounds_top    = int(bt)
                    det.mask_bounds_right  = int(br)
                    det.mask_bounds_bottom = int(bb)

                    centroid = compute_mask_centroid(raw_bytes, mw, mh, bl, bt, br, bb)
                    if centroid is not None:
                        det.mask_centroid_x, det.mask_centroid_y = centroid
                        det.mask_status = "ok"
                    elif not bool((arr > 0).any()):
                        det.mask_status = "empty"
                    else:
                        det.mask_status = "unmappable"

            objects.append(det)

        return objects

    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# Regex fallback parse path
# ---------------------------------------------------------------------------

def _parse_with_regex(
    events: list[dict],
    concept: str,
) -> list[DetectedObject]:
    """
    Parse SAM wire-format tokens from the raw event output_text.
    No mask data available in this path.
    """
    text = _extract_output_text(events)
    if not text:
        return []

    img_w, img_h = 0, 0
    dim_m = _IMG_DIM_RE.search(text)
    if dim_m:
        img_w = int(dim_m.group("w"))
        img_h = int(dim_m.group("h"))

    objects: list[DetectedObject] = []
    for match in _BOX_RE.finditer(text):
        x1 = int(match.group("x1"))
        y1 = int(match.group("y1"))
        x2 = int(match.group("x2"))
        y2 = int(match.group("y2"))

        prefix = text[: match.start()]
        id_match = _OBJ_ID_RE.search(prefix)
        obj_id = int(id_match.group(1)) if id_match else len(objects)

        objects.append(
            DetectedObject(
                object_id=obj_id,
                concept=concept,
                frame=0,
                x1=x1, y1=y1, x2=x2, y2=y2,
                image_width=img_w,
                image_height=img_h,
                parse_method="regex",
            )
        )

    return objects


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def parse_sam_result(
    events: list[dict],
    concept: str,
) -> tuple[list[DetectedObject], str]:
    """
    Parse SAM events into a list of DetectedObject instances.

    Returns
    -------
    (objects, method_note)
        objects     : list of DetectedObject (may be empty → zero detections)
        method_note : "library" | "regex" — which parse path was used
    """
    library_result = _parse_with_library(events, concept)
    if library_result:          # non-empty list → library found real matches
        return library_result, "library"

    # Library unavailable (None) or found 0 records ([]) → try regex
    regex_result = _parse_with_regex(events, concept)
    return regex_result, "regex"
