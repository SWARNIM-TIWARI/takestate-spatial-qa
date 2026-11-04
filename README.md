# TakeState: visual rule checking for controlled product photography

TakeState checks whether each new photo of a staged shot satisfies a creator-defined spatial rule, for example *"the glass must be to the right of the bottle"*, and shows its evidence.

- **Muse Spark 1.3** turns a natural-language requirement into a structured rule (JSON schema, validated locally; only four relations are accepted).
- **SAM 3.1** segments the named objects in each photo.
- A **deterministic geometry engine** compares the **mask centroids** (centre of mass of each decoded segmentation mask, mapped into source-image pixels) and returns **PASS**, **FAIL** or **UNABLE TO VERIFY**, with the measured separation and a correction hint.

Streamlit app, single entry point: `app.py`.

> **Status:** the offline test suite passes (116 tests; see `RUNBOOK.md`). The live SAM / Muse Spark integration has **not** been verified against the real service by the author of this code; use `test_sam_live.py` and `test_spark_live.py` with your own key. See `FINAL_CHECKLIST.md`.

## Workflow

1. **Define** the requirement: type it and click *Interpret with Muse Spark*, or use the manual fields and *Apply rule* (the manual path makes no model call). The rule banner shows what will be checked.
2. **Upload** a photo (JPG/PNG/WebP, up to 10 MB). Nothing is sent to any model yet.
3. **Check this take** runs SAM twice (subject, reference). If a name matches several instances, candidate boxes are shown and you pick the intended one.
4. **Inspect** the result: verdict, mask overlay, mask-centroid coordinates (used for the verdict) and bounding-box-centre coordinates (diagnostic only, labelled separately), separation in pixels and as a percentage of image width (left/right) or height (above/below).
5. **Correct and retake**: upload the next take; the rule is preserved, and previous results stay in the history. *Re-check* deliberately repeats the analysis of the same image.

**Scope.** TakeState evaluates the spatial relationship *within each photograph* against the rule. It does **not** compare a later image with an approved reference image or enforce a position tolerance against a reference. The same rule can be re-checked on later takes, and earlier results stay in the session history (they are lost when the session ends).

## Rule semantics (image coordinates, y grows downward)

| Relation | PASS when |
|---|---|
| right_of | subject centroid x > reference centroid x |
| left_of | subject centroid x < reference centroid x |
| above | subject centroid y < reference centroid y |
| below | subject centroid y > reference centroid y |

**Mask-only invariant.** PASS or FAIL is returned only when both objects are identified, the intended detection is unambiguous for each, both segmentation masks were decoded, both masks contain foreground pixels, and both mask centroids map validly into the original image (finite, inside the image, consistent with the detection's box). Otherwise the result is `UNABLE TO VERIFY`, and the explanation says which object lacks which condition. Bounding-box centres are shown only as separately labelled diagnostics and are **never** a substitute for a mask centroid, so if SAM returns boxes without usable masks the app cannot give a PASS or FAIL.

## Quick start (Windows PowerShell)

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
notepad .env          # set MODEL_API_KEY=...
python tests.py       # offline tests, no key needed
python -m streamlit run app.py
```

Full step-by-step instructions, expected output and troubleshooting: **`RUNBOOK.md`**.

## Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit UI (entry point) |
| `rule_engine.py` | Deterministic spatial rules and evidence |
| `sam_parser.py` | SAM output parsing, mask decoding, centroid mapping |
| `meta_client.py` | Meta Model API client (SAM 3.1 Responses stream, Muse Spark chat) |
| `interpreter.py` | Natural-language requirement → validated rule |
| `config.py` | Secret lookup (env/.env, Streamlit secrets), access code, per-session call budget |
| `tests.py` | Offline tests (no key, no network) |
| `test_sam_live.py`, `test_spark_live.py` | Separate **live** smoke tests (paid, need a key) |
| `RUNBOOK.md`, `ACCEPTANCE_TESTS.md`, `DEMO_SCRIPT.md`, `SUBMISSION_DRAFT.md`, `FINAL_CHECKLIST.md`, `DEPLOY.md` | Execution, testing, demo, submission, deployment |

## Architecture

```
text requirement ──► Muse Spark 1.3 (JSON schema) ──► local validation ──► SpatialRule
photo ──► SAM 3.1 (/v1/responses, streaming) ──► meta-sam-parser (box + mask records)
      ──► decode_mask_to_raster ──► mask centroid in source-image pixels
SpatialRule + centroids ──► rule_engine ──► PASS / FAIL / UNABLE TO VERIFY + evidence
```

The models propose; plain code decides. No model-generated code is executed, and the language model never sees the verdict logic.

## Security and cost

- The key lives in `.env` (git-ignored) locally, or in Streamlit secrets when deployed. It is never displayed, logged, or included in error text (it is redacted if it appears).
- Paid calls happen only on explicit button clicks; page reruns never call a model. A per-session budget limits checks and interpretations, and `APP_ACCESS_CODE` can gate a public deployment. See `DEPLOY.md`.

## Known limitations

- Live behaviour of SAM 3.1 and Muse Spark in this app is unverified until you run the live tests.
- Requires SAM to return decodable masks. Without them the verdict is UNABLE TO VERIFY (no bounding-box fallback). Whether the live service returns masks for your photos is unverified until you run `test_sam_live.py`.
- Positions are 2-D image positions of mask centroids, not physical-world positions; camera angle changes what "right of" means in the scene.
- A mask centroid is the centre of mass of the visible mask; occlusion changes it.
- Four relations only (left/right/above/below). No colour, orientation, text, lighting or aesthetic checks.
- The per-session budget resets when a new browser session starts; it is not a hard spending cap.
