"""
app.py — TakeState: Visual Rule-Checking Workbench for Product Photography.

Workflow:  DEFINE → SEE → MEASURE → DECIDE → CORRECT → VERIFY

API calls happen ONLY when the user clicks an explicit action button.
Streamlit reruns caused by widget interaction never trigger a new API call.

Detection selection:
    When SAM returns multiple detections for a concept, the user sees all
    candidate bounding boxes and picks the intended one before evaluation.
    Selections are cached by content digest; a Re-check clears the cache.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import base64
import hashlib
import html
import io
from typing import Optional

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageOps

from config import (
    CallBudget, DEFAULT_MAX_INTERPRETATIONS, DEFAULT_MAX_TAKE_CHECKS,
    access_code_required, check_access_code, get_int_setting,
)
from meta_client import get_api_key_status, call_sam_image
from sam_parser import parse_sam_result, DetectedObject
from rule_engine import (
    SpatialRule, Verdict, evaluate_rule, SUPPORTED_RELATIONS, RELATION_LABELS,
)

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------
C_SUBJECT   = "#00d4ff"   # cyan
C_REFERENCE = "#ff6b35"   # orange
C_PASS      = "#22c55e"   # green
C_FAIL      = "#ef4444"   # red
C_UNABLE    = "#f59e0b"   # amber

VERDICT_COLORS = {
    Verdict.PASS:             C_PASS,
    Verdict.FAIL:             C_FAIL,
    Verdict.UNABLE_TO_VERIFY: C_UNABLE,
}

_COLORS: list[str] = ["#00d4ff", "#ff6b35", "#a855f7", "#fbbf24", "#34d399"]


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="TakeState",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Global CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
#MainMenu, footer, header { visibility: hidden; }

.rule-banner {
    background: linear-gradient(90deg, #141414 0%, #1a1a2e 100%);
    border-left: 3px solid #00d4ff;
    border-radius: 6px;
    padding: 10px 16px;
    margin: 8px 0 16px 0;
    font-size: 1.05rem;
    color: #e8e8e8;
    font-family: 'SF Pro Display', 'Segoe UI', sans-serif;
}
.verdict-badge {
    display: inline-block;
    padding: 6px 22px;
    border-radius: 4px;
    font-size: 1.4rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    font-family: 'SF Mono', 'Courier New', monospace;
}
.det-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.82rem;
    font-weight: 600;
    font-family: 'SF Mono', 'Courier New', monospace;
    margin: 2px;
}
.section-label {
    font-size: 0.68rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #666;
    margin-bottom: 4px;
}
.evidence-mono {
    font-family: 'SF Mono', 'Courier New', monospace;
    font-size: 0.80rem;
    color: #aaa;
    line-height: 1.8;
}
.take-row {
    background: #141414;
    border-radius: 6px;
    padding: 8px 14px;
    margin-bottom: 6px;
    border-left: 3px solid #333;
    font-size: 0.85rem;
    color: #bbb;
}
.selection-box {
    background: #1a1a2e;
    border: 1px solid #333;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 8px 0;
}
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
def _init_state() -> None:
    defaults: dict = {
        "subject":            "glass",
        "relation":           "right_of",
        "reference":          "bottle",
        # Upload
        "take_bytes":         None,
        "take_data_url":      None,
        "take_w":             0,
        "take_h":             0,
        "take_digest":        None,
        # Raw detections from SAM (lists of DetectedObject)
        "take_sub_dets":      None,    # list[DetectedObject] | None
        "take_ref_dets":      None,
        "take_parse_method":  None,
        # User selections (int index into the respective list)
        "take_sub_sel":       None,    # None = not yet selected
        "take_ref_sel":       None,
        # Final evaluated result (after selection)
        "take_result":        None,
        "take_annotated":     None,
        # History
        "takes":              [],
        # Paid-call protection (per browser session)
        "budget_checks":      CallBudget(get_int_setting("MAX_TAKE_CHECKS", DEFAULT_MAX_TAKE_CHECKS)),
        "budget_interps":     CallBudget(get_int_setting("MAX_INTERPRETATIONS", DEFAULT_MAX_INTERPRETATIONS)),
        "access_ok":          False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------

MAX_UPLOAD_BYTES = 10 * 1024 * 1024   # 10 MB
MAX_LONG_EDGE_PX = 2048               # keeps request size and latency bounded


def normalise_upload(uploaded_file) -> tuple[bytes, str, int, int]:
    """
    Decode, orient and re-encode an upload as a JPEG data URL.
    Raises ValueError with a user-safe message for oversized or unreadable files.
    """
    raw = uploaded_file.getvalue()   # not .read(): safe across Streamlit reruns
    if len(raw) > MAX_UPLOAD_BYTES:
        raise ValueError(
            f"Image is {len(raw) / 1_048_576:.1f} MB; the limit is "
            f"{MAX_UPLOAD_BYTES // 1_048_576} MB."
        )
    try:
        img = Image.open(io.BytesIO(raw))
        img = ImageOps.exif_transpose(img).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise ValueError("Could not read this file as an image (JPG, PNG or WebP).") from exc
    if max(img.size) > MAX_LONG_EDGE_PX:
        img.thumbnail((MAX_LONG_EDGE_PX, MAX_LONG_EDGE_PX))
    w, h = img.size
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    jpeg_bytes = buf.getvalue()
    b64 = base64.b64encode(jpeg_bytes).decode()
    data_url = f"data:image/jpeg;base64,{b64}"
    return jpeg_bytes, data_url, w, h


def _take_digest(take_bytes: bytes, rule: SpatialRule) -> str:
    rule_key = f"{rule.subject}|{rule.relation}|{rule.reference}"
    return hashlib.md5(take_bytes + rule_key.encode()).hexdigest()


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


# ---------------------------------------------------------------------------
# Image annotation
# ---------------------------------------------------------------------------

def _load_font(size: int = 13) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size
        )
    except Exception:
        return ImageFont.load_default()


def annotate_candidates(
    img_bytes: bytes,
    detections: list[DetectedObject],
    hex_color: str,
    concept: str,
) -> bytes:
    """
    Draw numbered bounding boxes for all candidate detections of one concept.
    Used during the selection step.
    """
    img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    r, g, b = _hex_to_rgb(hex_color)
    font = _load_font(14)

    for i, det in enumerate(detections):
        draw.rectangle(
            [det.x1, det.y1, det.x2, det.y2],
            fill=(r, g, b, 35),
            outline=(r, g, b, 220),
            width=3,
        )
        label = f" #{i + 1} {concept} "
        draw.text(
            (det.x1 + 3, max(0, det.y1 - 18)),
            label,
            fill=(r, g, b, 255),
            font=font,
        )

    img.paste(overlay, mask=overlay.split()[3])
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=88)
    return buf.getvalue()


def annotate_result(
    img_bytes: bytes,
    sub_det: Optional[DetectedObject],
    ref_det: Optional[DetectedObject],
) -> bytes:
    """
    Draw the final selected detections with mask overlays and centroid markers.
    Mask centroid (⊕) is the measurement point used for verdicts. The bounding-box
    centre (hollow square) is a separately labelled diagnostic, never a substitute.
    """
    img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")

    def _draw_detection(det: DetectedObject, hex_color: str, role: str) -> None:
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        r, g, b = _hex_to_rgb(hex_color)
        font = _load_font(13)

        # Mask overlay (if available)
        if det.mask_raster is not None and det.mask_bounds_left is not None:
            try:
                mw = det.mask_width or det.mask_raster.shape[1]
                mh = det.mask_height or det.mask_raster.shape[0]
                bl = det.mask_bounds_left
                bt = det.mask_bounds_top
                br = det.mask_bounds_right
                bb = det.mask_bounds_bottom
                bw = max(br - bl, 1)
                bh = max(bb - bt, 1)

                # Scale mask to bounds dimensions for display
                mask_img = Image.fromarray(
                    (det.mask_raster * 110).clip(0, 255).astype(np.uint8)
                ).convert("L").resize((bw, bh), Image.NEAREST)

                fill_layer = Image.new("RGBA", (bw, bh), (r, g, b, 110))
                overlay.paste(fill_layer, (bl, bt), mask=mask_img)
            except Exception:
                pass

        # Bounding box
        draw.rectangle(
            [det.x1, det.y1, det.x2, det.y2],
            fill=(r, g, b, 25),
            outline=(r, g, b, 200),
            width=3,
        )

        # Label
        label = f" {role} "
        draw.text(
            (det.x1 + 3, max(0, det.y1 - 18)),
            label,
            fill=(r, g, b, 255),
            font=font,
        )

        # Mask centroid marker: ⊕ (filled circle + crosshair)
        if det.mask_centroid_x is not None:
            cx = int(det.mask_centroid_x)
            cy = int(det.mask_centroid_y)
            draw.ellipse([cx - 6, cy - 6, cx + 6, cy + 6],
                         fill=(r, g, b, 255), outline=(255, 255, 255, 255), width=2)
            draw.line([(cx - 12, cy), (cx + 12, cy)], fill=(255, 255, 255, 220), width=1)
            draw.line([(cx, cy - 12), (cx, cy + 12)], fill=(255, 255, 255, 220), width=1)

        # Bbox centre: small hollow square, diagnostic only (not used for verdicts)
        bx = int(det.bbox_center_x)
        by = int(det.bbox_center_y)
        sq = 4
        draw.rectangle([bx - sq, by - sq, bx + sq, by + sq],
                       fill=None, outline=(r, g, b, 120), width=1)

        img.paste(overlay, mask=overlay.split()[3])

    if sub_det:
        _draw_detection(sub_det, C_SUBJECT,   "SUBJECT")
    if ref_det:
        _draw_detection(ref_det, C_REFERENCE, "REF")

    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=88)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# SAM calls — only invoked from button handlers
# ---------------------------------------------------------------------------

def run_sam_for_both(
    data_url: str,
    rule: SpatialRule,
) -> tuple[
    Optional[list[DetectedObject]],
    Optional[list[DetectedObject]],
    str,
    Optional[str],
]:
    """
    Run two SAM calls (subject + reference).
    Returns (sub_dets, ref_dets, method, error_or_None).
    """
    status = st.empty()

    status.info(f"🔍 Detecting **{rule.subject}** with SAM 3.1…")
    sub_events, sub_err = call_sam_image(data_url, rule.subject)
    if sub_err:
        status.error(f"SAM error (subject): {sub_err}")
        return None, None, "error", sub_err
    sub_dets, sub_method = parse_sam_result(sub_events, rule.subject)

    status.info(f"🔍 Detecting **{rule.reference}** with SAM 3.1…")
    ref_events, ref_err = call_sam_image(data_url, rule.reference)
    if ref_err:
        status.error(f"SAM error (reference): {ref_err}")
        return None, None, "error", ref_err
    ref_dets, ref_method = parse_sam_result(ref_events, rule.reference)

    status.empty()

    # The library path does not always recover image dimensions from the stream;
    # the dimensions of the image we actually sent are authoritative.
    for det in list(sub_dets) + list(ref_dets):
        if not det.image_width or not det.image_height:
            det.image_width = st.session_state.take_w
            det.image_height = st.session_state.take_h

    method = (
        sub_method if sub_method == ref_method else f"{sub_method}/{ref_method}"
    )
    return sub_dets, ref_dets, method, None


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------

def centroid_chip_html(
    role: str,
    det: Optional[DetectedObject],
    hex_color: str,
) -> str:
    if det is None:
        return (
            f'<span class="det-chip" style="background:{hex_color}18;color:{hex_color};'
            f'border:1px solid {hex_color}40">✗ {html.escape(role.upper())} — not detected</span>'
        )
    if det.centroid_source == "mask":
        label = (
            f"mask centroid @ ({det.mask_centroid_x:.0f}, {det.mask_centroid_y:.0f}) px"
        )
        mark, border = "✓", f"{hex_color}80"
    else:
        label = (
            f"no valid mask centroid ({det.mask_status}); "
            f"bbox center @ ({det.bbox_center_x:.0f}, {det.bbox_center_y:.0f}) px is diagnostic only"
        )
        mark, border = "⚠", C_UNABLE
    return (
        f'<span class="det-chip" style="background:{hex_color}18;color:{hex_color};'
        f'border:1px solid {border}">{mark} {html.escape(role.upper())}: '
        f'{html.escape(det.concept)} — {html.escape(label)}</span>'
    )


def verdict_html(verdict: Verdict) -> str:
    color = VERDICT_COLORS[verdict]
    return (
        f'<div class="verdict-badge" '
        f'style="background:{color}22;color:{color};border:2px solid {color};">'
        f'{html.escape(verdict.value)}</div>'
    )


# ---------------------------------------------------------------------------
# Detection selection UI
# ---------------------------------------------------------------------------

def _selection_needed(
    sub_dets: list[DetectedObject],
    ref_dets: list[DetectedObject],
) -> bool:
    return len(sub_dets) > 1 or len(ref_dets) > 1


def render_selection_ui(
    img_bytes: bytes,
    sub_dets: list[DetectedObject],
    ref_dets: list[DetectedObject],
    rule: SpatialRule,
) -> bool:
    """
    Render the candidate-selection UI when multiple detections are returned.
    Updates session state sub_sel / ref_sel.
    Returns True when selections are confirmed and evaluation should proceed.
    """
    st.markdown(
        '<div class="selection-box">'
        '<p class="section-label">Multiple detections — select the intended instance</p>',
        unsafe_allow_html=True,
    )

    col_img, col_sel = st.columns([3, 2])

    with col_img:
        if len(sub_dets) > 1:
            ann = annotate_candidates(img_bytes, sub_dets, C_SUBJECT, rule.subject)
            st.image(ann, caption=f"'{rule.subject}' candidates (cyan)", use_container_width=True)
        if len(ref_dets) > 1:
            ann = annotate_candidates(img_bytes, ref_dets, C_REFERENCE, rule.reference)
            st.image(ann, caption=f"'{rule.reference}' candidates (orange)", use_container_width=True)

    with col_sel:
        confirmed = True

        if len(sub_dets) > 1:
            options = [
                f"#{i + 1}  bbox ({d.x1},{d.y1})→({d.x2},{d.y2})"
                for i, d in enumerate(sub_dets)
            ]
            prev_sel = st.session_state.get("take_sub_sel") or 0
            sub_choice = st.radio(
                f"Which is the **{rule.subject}**?",
                options=options,
                index=prev_sel,
                key="radio_sub_sel",
            )
            st.session_state.take_sub_sel = options.index(sub_choice)
        else:
            st.session_state.take_sub_sel = 0

        if len(ref_dets) > 1:
            options_r = [
                f"#{i + 1}  bbox ({d.x1},{d.y1})→({d.x2},{d.y2})"
                for i, d in enumerate(ref_dets)
            ]
            prev_sel_r = st.session_state.get("take_ref_sel") or 0
            ref_choice = st.radio(
                f"Which is the **{rule.reference}**?",
                options=options_r,
                index=prev_sel_r,
                key="radio_ref_sel",
            )
            st.session_state.take_ref_sel = options_r.index(ref_choice)
        else:
            st.session_state.take_ref_sel = 0

        confirmed = st.button(
            "✅ Confirm selections & evaluate",
            use_container_width=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)
    return confirmed


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main() -> None:
    # ---- Header -----------------------------------------------------------
    st.markdown(
        '<h1 style="font-size:2rem;font-weight:800;letter-spacing:0.05em;'
        'margin-bottom:0;color:#e8e8e8;">🎯 TakeState</h1>'
        '<p style="color:#666;font-size:0.85rem;margin-top:2px;margin-bottom:20px;">'
        "Visual rule-checking workbench for controlled product photography</p>",
        unsafe_allow_html=True,
    )

    key_status = get_api_key_status()
    if not key_status["configured"]:
        st.error(
            "⚠️ MODEL_API_KEY not configured. "
            "Copy `.env.example` → `.env` and add your key from "
            "https://dev.meta.ai/platform"
        )

    if key_status["configured"] and not access_code_required():
        st.warning(
            "No APP_ACCESS_CODE is set: anyone who can open this page can spend this "
            "app's API credits. Set APP_ACCESS_CODE before sharing a public URL."
        )

    # ---- Optional access gate (set APP_ACCESS_CODE for public deployments) -
    if access_code_required() and not st.session_state.access_ok:
        st.markdown('<p class="section-label">Access</p>', unsafe_allow_html=True)
        code = st.text_input("Access code", type="password", key="access_code_input")
        if st.button("Unlock"):
            if check_access_code(code):
                st.session_state.access_ok = True
                st.rerun()
            else:
                st.error("Incorrect access code.")
        st.info("This deployment makes paid model calls and is protected by an access code.")
        st.stop()

    st.divider()

    # ---- Natural-language interpretation ----------------------------------
    st.markdown('<p class="section-label">Interpret Requirement (Muse Spark 1.3)</p>',
                unsafe_allow_html=True)
    col_nl, col_btn = st.columns([5, 1])
    with col_nl:
        nl_text = st.text_input(
            "NL requirement",
            value="",
            key="nl_input",
            label_visibility="collapsed",
            placeholder='e.g. "The glass must be to the right of the bottle."',
        )
    with col_btn:
        interpret_clicked = st.button(
            "Interpret with Muse Spark",
            use_container_width=True,
            disabled=not key_status["configured"],
        )

    if interpret_clicked and nl_text.strip():
        from interpreter import parse_requirement
        if not st.session_state.budget_interps.spend(1):
            parsed, err = None, (
                "Interpretation limit reached for this session. "
                "Use the manual fields below."
            )
        else:
            with st.spinner("Calling Muse Spark 1.3…"):
                parsed, err = parse_requirement(nl_text.strip())
        if err:
            st.error(f"Interpretation failed: {err}")
        elif parsed is None:
            st.error("Muse Spark returned no result.")
        elif not parsed.supported:
            st.warning(
                f"Relation '{parsed.relation}' is not supported. "
                f"{parsed.unsupported_reason} — "
                "Please use the manual fields below."
            )
        else:
            st.session_state.subject   = parsed.subject
            st.session_state.relation  = parsed.relation
            st.session_state.reference = parsed.reference
            st.session_state.takes     = []
            st.session_state.take_digest = None
            st.success(
                f"Interpreted: **{parsed.subject}** {parsed.relation} "
                f"**{parsed.reference}**"
            )
            st.rerun()

    # ---- Manual rule definition ------------------------------------------
    st.markdown('<p class="section-label">Define Requirement (manual — no model call)</p>',
                unsafe_allow_html=True)
    col_sub, col_rel, col_ref, col_apply = st.columns([3, 2, 3, 1])
    with col_sub:
        subject = st.text_input(
            "Subject",
            value=st.session_state.subject,
            key="input_subject",
            label_visibility="collapsed",
            placeholder="e.g. glass",
        )
    with col_rel:
        relation = st.selectbox(
            "Relation",
            options=list(SUPPORTED_RELATIONS),
            index=list(SUPPORTED_RELATIONS).index(st.session_state.relation),
            format_func=lambda r: RELATION_LABELS[r],
            key="input_relation",
            label_visibility="collapsed",
        )
    with col_ref:
        reference = st.text_input(
            "Reference",
            value=st.session_state.reference,
            key="input_reference",
            label_visibility="collapsed",
            placeholder="e.g. bottle",
        )
    with col_apply:
        apply_clicked = st.button("Apply rule", use_container_width=True)

    if apply_clicked:
        new_sub = subject.strip().lower()
        new_rel = relation
        new_ref = reference.strip().lower()
        if (
            new_sub != st.session_state.subject
            or new_rel != st.session_state.relation
            or new_ref != st.session_state.reference
        ):
            st.session_state.subject   = new_sub
            st.session_state.relation  = new_rel
            st.session_state.reference = new_ref
            st.session_state.takes     = []
            st.session_state.take_digest = None
            st.session_state.take_result = None

    current_rule = SpatialRule(
        subject=st.session_state.subject,
        relation=st.session_state.relation,
        reference=st.session_state.reference,
    )

    st.markdown(
        f'<div class="rule-banner">📋 {html.escape(current_rule.human_label())}</div>',
        unsafe_allow_html=True,
    )

    st.divider()

    # ---- Take upload + Check button --------------------------------------
    st.markdown('<p class="section-label">New Take</p>', unsafe_allow_html=True)

    take_upload = st.file_uploader(
        "Upload take",
        type=["jpg", "jpeg", "png", "webp"],
        key="take_uploader",
        label_visibility="collapsed",
    )

    if take_upload is not None:
        try:
            take_bytes, take_url, take_w, take_h = normalise_upload(take_upload)
        except ValueError as exc:
            st.error(str(exc))
            take_bytes = None
        incoming_digest = (
            _take_digest(take_bytes, current_rule) if take_bytes is not None else None
        )

        if incoming_digest is not None and incoming_digest != st.session_state.take_digest:
            st.session_state.take_bytes        = take_bytes
            st.session_state.take_data_url     = take_url
            st.session_state.take_w            = take_w
            st.session_state.take_h            = take_h
            st.session_state.take_digest       = incoming_digest
            st.session_state.take_sub_dets     = None
            st.session_state.take_ref_dets     = None
            st.session_state.take_parse_method = None
            st.session_state.take_sub_sel      = None
            st.session_state.take_ref_sel      = None
            st.session_state.take_result       = None
            st.session_state.take_annotated    = None

    # Preview image (annotated if result exists, raw otherwise)
    if st.session_state.take_bytes is not None:
        if st.session_state.take_annotated is not None:
            st.image(st.session_state.take_annotated, use_container_width=True)
        else:
            st.image(st.session_state.take_bytes, use_container_width=True)

    # Action buttons
    col_check, col_recheck = st.columns([2, 1])
    with col_check:
        check_clicked = st.button(
            "✅ Check this take",
            use_container_width=True,
            disabled=(
                st.session_state.take_bytes is None
                or st.session_state.take_sub_dets is not None   # already analysed: use Re-check
                or not key_status["configured"]
            ),
        )
    with col_recheck:
        recheck_clicked = st.button(
            "🔄 Re-check",
            use_container_width=True,
            disabled=(
                st.session_state.take_sub_dets is None
                or not key_status["configured"]
            ),
        )

    # ---- Run SAM (only on an explicit button click) ------------------------
    budget = st.session_state.budget_checks
    st.caption(
        f"Session usage: {budget.used}/{budget.limit} take checks "
        f"({budget.remaining} remaining; each check makes 2 SAM requests)."
    )
    if (check_clicked or recheck_clicked) and st.session_state.take_bytes is not None:
        if not budget.spend(1):
            st.error(
                "Take-check limit reached for this session. "
                "Reload the page to start a new session."
            )
        else:
            with st.spinner("Running SAM 3.1 detections…"):
                sub_dets, ref_dets, method, err = run_sam_for_both(
                    st.session_state.take_data_url,
                    current_rule,
                )
            if err is None:
                st.session_state.take_sub_dets     = sub_dets
                st.session_state.take_ref_dets     = ref_dets
                st.session_state.take_parse_method = method
                st.session_state.take_sub_sel      = None
                st.session_state.take_ref_sel      = None
                st.session_state.take_result       = None
                st.session_state.take_annotated    = None
                st.rerun()

    # ---- Detection selection or auto-evaluate ----------------------------
    sub_dets = st.session_state.take_sub_dets
    ref_dets = st.session_state.take_ref_dets

    if sub_dets is not None and ref_dets is not None:
        needs_selection = _selection_needed(sub_dets, ref_dets)

        if needs_selection and st.session_state.take_result is None:
            confirmed = render_selection_ui(
                st.session_state.take_bytes,
                sub_dets,
                ref_dets,
                current_rule,
            )
            if confirmed:
                si = st.session_state.take_sub_sel or 0
                ri = st.session_state.take_ref_sel or 0
                sel_sub = [sub_dets[si]] if si < len(sub_dets) else []
                sel_ref = [ref_dets[ri]] if ri < len(ref_dets) else []
                _finalise_result(sel_sub, sel_ref, current_rule)
                st.rerun()

        elif not needs_selection and st.session_state.take_result is None:
            # Single detection each side — evaluate immediately
            _finalise_result(sub_dets, ref_dets, current_rule)
            st.rerun()

    # ---- Result display --------------------------------------------------
    if st.session_state.take_result is not None:
        result = st.session_state.take_result
        sub_dets_s = st.session_state.take_sub_dets or []
        ref_dets_s = st.session_state.take_ref_dets or []
        si = st.session_state.take_sub_sel or 0
        ri = st.session_state.take_ref_sel or 0
        sub_det = sub_dets_s[si] if si < len(sub_dets_s) else None
        ref_det = ref_dets_s[ri] if ri < len(ref_dets_s) else None
        method  = st.session_state.take_parse_method or "?"
        ev      = result.evidence

        st.markdown(verdict_html(result.verdict), unsafe_allow_html=True)
        st.markdown(
            centroid_chip_html("subject",   sub_det, C_SUBJECT)
            + "&nbsp;"
            + centroid_chip_html("reference", ref_det, C_REFERENCE),
            unsafe_allow_html=True,
        )

        if result.correction_hint:
            st.warning(f"💡 **Correction hint:** {result.correction_hint}")

        with st.expander("📐 Geometric evidence", expanded=False):
            lines = [
                f"Rule: {current_rule.human_label()}",
                "Verdict basis: decoded segmentation-mask centroids only "
                "(bounding-box centres are diagnostics, never a substitute).",
                f"Mask centroids used: {'yes' if ev.centroid_source == 'mask' else 'no (PASS/FAIL is not possible without both)'}",
                "",
                (
                    f"subject   mask centroid = ({ev.subject_centroid_x:.1f}, "
                    f"{ev.subject_centroid_y:.1f}) px"
                    if ev.subject_centroid_x is not None
                    else "subject   mask centroid = unavailable"
                    + (f" ({sub_det.mask_status})" if sub_det else " (not detected)")
                ),
                (
                    f"reference mask centroid = ({ev.reference_centroid_x:.1f}, "
                    f"{ev.reference_centroid_y:.1f}) px"
                    if ev.reference_centroid_x is not None
                    else "reference mask centroid = unavailable"
                    + (f" ({ref_det.mask_status})" if ref_det else " (not detected)")
                ),
            ]
            if sub_det:
                lines.append(
                    f"subject   bbox center = ({sub_det.bbox_center_x:.1f}, "
                    f"{sub_det.bbox_center_y:.1f}) px  [diagnostic only]"
                )
            if ref_det:
                lines.append(
                    f"reference bbox center = ({ref_det.bbox_center_x:.1f}, "
                    f"{ref_det.bbox_center_y:.1f}) px  [diagnostic only]"
                )
            lines += [
                "",
                (
                    f"separation = {ev.measured_separation:.1f} px "
                    f"({ev.separation_pct():.1f}%)"
                    if ev.measured_separation is not None and ev.separation_pct() is not None
                    else "separation = n/a"
                ),
                (
                    f"image size = {ev.image_width} × {ev.image_height} px"
                    if ev.image_width
                    else "image size = unknown"
                ),
                "",
                f"parse method : {method}",
                "",
                result.explanation,
            ]
            st.markdown(
                '<div class="evidence-mono">'
                + "<br>".join(html.escape(ln) for ln in lines)
                + "</div>",
                unsafe_allow_html=True,
            )

    # ---- Take history ----------------------------------------------------
    takes = st.session_state.takes
    if len(takes) > 1:
        st.divider()
        st.markdown('<p class="section-label">Take History</p>', unsafe_allow_html=True)
        for take in reversed(takes[:-1]):
            v     = take["result"].verdict
            color = VERDICT_COLORS[v]
            st.markdown(
                f'<div class="take-row" style="border-left-color:{color}">'
                f'Take #{take["index"]} — '
                f'<span style="color:{color};font-weight:700">'
                f'{html.escape(v.value)}</span>'
                f' — {html.escape(take["result"].explanation[:120])}'
                f"</div>",
                unsafe_allow_html=True,
            )


def _finalise_result(
    sub_dets: list[DetectedObject],
    ref_dets: list[DetectedObject],
    rule: SpatialRule,
) -> None:
    """Evaluate rule, annotate image, store result in session_state."""
    result = evaluate_rule(rule, sub_dets, ref_dets)

    sub_det = sub_dets[0] if len(sub_dets) == 1 else None
    ref_det = ref_dets[0] if len(ref_dets) == 1 else None

    annotated = annotate_result(
        st.session_state.take_bytes, sub_det, ref_det
    )

    st.session_state.take_result    = result
    st.session_state.take_annotated = annotated

    # Append to history (avoid duplicate entries)
    last = st.session_state.takes[-1] if st.session_state.takes else None
    if last is not None and last.get("digest") == st.session_state.take_digest:
        last["annotated"] = annotated
        last["result"] = result
    else:
        st.session_state.takes.append({
            "digest":    st.session_state.take_digest,
            "annotated": annotated,
            "result":    result,
            "index":     len(st.session_state.takes) + 1,
        })


if __name__ == "__main__":
    main()
