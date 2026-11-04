r"""
test_sam_live.py — LIVE SAM 3.1 smoke test. Makes ONE paid request.

Usage (PowerShell):
    python test_sam_live.py path\to\photo.jpg "glass"

Exit codes:
    0  every check passed (request completed, detection parsed, mask decoded and mapped)
    1  at least one check failed
    2  SKIPPED — MODEL_API_KEY not configured (this is NOT a pass)
    3  bad usage / unreadable image

Prints only sanitized diagnostics. The API key is never printed.
This script is evidence of live behaviour only when it is actually run by you
with your own key; nothing here is mocked.
"""

from __future__ import annotations

import base64
import io
import sys

from PIL import Image, ImageOps

from meta_client import call_sam_image, get_api_key_status
from sam_parser import parse_sam_result, _extract_output_text

results: list[tuple[str, bool]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    results.append((name, ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    return ok


def main() -> int:
    if len(sys.argv) != 3:
        print('Usage: python test_sam_live.py <photo.jpg> "<object noun phrase>"')
        return 3
    path, concept = sys.argv[1], sys.argv[2].strip()
    if not concept:
        print("Object description must not be empty.")
        return 3

    if not get_api_key_status()["configured"]:
        print("SKIPPED: MODEL_API_KEY is not configured (.env or Streamlit secrets).")
        print("A skipped live test is not a passing live test.")
        return 2

    try:
        img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        print(f"Could not open image: {type(exc).__name__}")
        return 3
    if max(img.size) > 2048:
        img.thumbnail((2048, 2048))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    data_url = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    w, h = img.size
    print(f"\nSAM 3.1 live test — image {w}x{h}px, prompt {concept!r} (1 request)\n")

    events, err = call_sam_image(data_url, concept)
    if not check("Request completed (response.completed received)", err is None, err or ""):
        return 1

    text = _extract_output_text(events)
    print(f"  info: streamed output text = {len(text)} characters")
    print(f"  info: first 300 chars = {text[:300]!r}")

    objects, method = parse_sam_result(events, concept)
    print(f"  info: parse method = {method}")
    check("At least one detection parsed", len(objects) >= 1, f"{len(objects)} found")
    if not objects:
        print("\n  Zero detections is a valid SAM answer for an absent object, but for a")
        print("  photo that clearly contains the object it points to a prompt or parsing issue.")
        return 1

    for i, d in enumerate(objects, 1):
        print(f"\n  detection #{i}")
        print(f"    bbox            : ({d.x1},{d.y1}) -> ({d.x2},{d.y2})   image {d.image_width}x{d.image_height}")
        print(f"    bbox center     : ({d.bbox_center_x:.1f}, {d.bbox_center_y:.1f})   [diagnostic only; never used for verdicts]")
        if d.mask_centroid_x is not None:
            print(f"    mask centroid   : ({d.mask_centroid_x:.1f}, {d.mask_centroid_y:.1f})   [foreground pixels]")
            print(f"    mask raster     : {d.mask_width}x{d.mask_height}, "
                  f"bounds ({d.mask_bounds_left},{d.mask_bounds_top})->({d.mask_bounds_right},{d.mask_bounds_bottom})")
        else:
            print(f"    mask centroid   : NOT AVAILABLE (status: {d.mask_status}) -> app gives UNABLE TO VERIFY")

    d0 = objects[0]
    check("Box lies inside the image",
          0 <= d0.x1 <= d0.x2 <= w + 1 and 0 <= d0.y1 <= d0.y2 <= h + 1,
          f"bbox ({d0.x1},{d0.y1},{d0.x2},{d0.y2}) vs image {w}x{h}")
    masked = [d for d in objects if d.mask_centroid_x is not None]
    if check("Segmentation mask decoded for at least one detection", bool(masked),
             "" if masked else "boxes only: the app would return UNABLE TO VERIFY "
                               "(bounding-box centres are never used for verdicts)"):
        m = masked[0]
        check("Mask centroid lies inside the image",
              0 <= m.mask_centroid_x <= w and 0 <= m.mask_centroid_y <= h,
              f"({m.mask_centroid_x:.1f}, {m.mask_centroid_y:.1f})")
        check("Mask centroid lies inside its own bounding box (+1px tolerance)",
              m.x1 - 1 <= m.mask_centroid_x <= m.x2 + 1 and m.y1 - 1 <= m.mask_centroid_y <= m.y2 + 1,
              "a large mismatch means the coordinate mapping is wrong")

    failed = sum(1 for _, ok in results if not ok)
    print(f"\nSAM live test: {len(results) - failed} passed, {failed} failed")
    print("Check the mask overlay visually in the app before trusting centroid placement.")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
