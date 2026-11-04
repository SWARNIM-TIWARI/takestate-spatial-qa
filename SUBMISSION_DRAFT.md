# SUBMISSION DRAFT — TakeState

Draft only. Bracketed items need your real data. Do not assume a demo video, public repository or live URL is mandatory: **verify the required fields, formats, links, media and deadline in the official participant portal.** Replace anything about live behaviour with what you actually observed.

## Project name
TakeState

## One-sentence pitch
TakeState checks whether each photo of a staged shot satisfies a plain-language spatial rule, using Muse Spark 1.3 to interpret the rule and SAM 3.1 segmentation masks to measure it, and returns PASS, FAIL or UNABLE TO VERIFY with its evidence.

## Problem
Staged shoots, such as product and tabletop photography, depend on arrangement staying right from take to take. A glass that ends up on the wrong side of a bottle is easy to miss while shooting and expensive to discover later. Checking is usually done by eye, with no written rule and no measurement to point to.

## What TakeState does
1. The creator writes a requirement in plain language ("the glass must be to the right of the bottle"), or enters it manually.
2. Muse Spark 1.3 turns it into a structured rule limited to four relations (left of, right of, above, below). The app validates the reply locally; unsupported requests create no rule.
3. For each uploaded photo, SAM 3.1 segments the two named objects. If a name matches several instances, the user picks the intended one on the image.
4. The app decodes the SAM masks with the official `meta-sam-parser`, computes each mask's centroid in image pixels, and a deterministic geometry engine compares them.
5. The result is PASS, FAIL (with measured separation and a correction hint) or UNABLE TO VERIFY (with the reason).
6. The same rule can be checked again on later takes; earlier results stay in the session history.

Scope: the app evaluates the spatial relationship **within each photograph**. It does not compare a later image against an approved reference image and does not enforce a reference-position tolerance.

## What makes the approach distinctive
- **Decisions rest on mask centroids.** PASS/FAIL requires valid decoded segmentation masks for both objects. If a mask is missing, undecodable, empty, or cannot be mapped reliably into the image, or the detection is ambiguous, the answer is UNABLE TO VERIFY. Bounding-box centres are shown only as labelled diagnostics and are never a substitute.
- **Models propose; deterministic code decides.** The language model only interprets the requirement into a constrained schema; the verdict is plain geometry, reproducible and explainable. No model-generated code is executed.
- **Uncertainty is a first-class result,** not a guess, and ambiguity is resolved by the user choosing the intended object on the image.
- **Cost-aware by design:** paid calls only on explicit clicks, content-hash gating against repeats, per-session limits and an optional access code.

## Technical architecture
- Python, Streamlit (`app.py`).
- `interpreter.py`: Muse Spark 1.3 through Chat Completions with a strict JSON-schema `response_format` (relation restricted to an enum), validated again locally.
- `meta_client.py`: SAM 3.1 through the Responses API with streaming; failed, errored and truncated streams are errors, not "zero matches".
- `sam_parser.py`: official `meta-sam-parser` 0.1.0 parses box and mask records, pairs them by object id, decodes masks with `decode_mask_to_raster`, and maps the centroid into source-image coordinates using the mask bounds. Records why a mask is unusable (`none`, `decode_failed`, `empty`, `unmappable`).
- `rule_engine.py`: pure geometry on mask centroids; horizontal rules use image width, vertical rules image height; centroids must lie inside the image and agree with the detection's box.
- `config.py`: secrets from environment/`.env` or Streamlit secrets, access code, session call budget.
- 116 offline tests (no key, no network) plus separate live smoke tests (`test_sam_live.py`, `test_spark_live.py`).

## Specific use of Muse Spark 1.3 and SAM 3.1
- **Muse Spark 1.3:** natural-language requirement → structured rule (subject, relation, reference) under a JSON schema.
- **SAM 3.1:** open-vocabulary segmentation from a noun phrase; its masks provide the centroids that the verdict is computed from.

## What was actually demonstrated
Offline: the 116-test suite passes in a clean Python 3.12 environment; it uses fake data and a fake SAM function, so it proves the local logic (mask maths, rules, refusal to decide without masks, rerun protection), **not** live API behaviour.

Live (to be filled in from your own runs; leave blank if not run):
- SAM 3.1 request completes and parses: [result]
- Masks returned and decoded; centroid placement checked visually: [result]
- Muse Spark structured interpretation (`test_spark_live.py`): [result]
- Acceptance scenarios A (PASS), B (FAIL then corrected), C (ambiguous), D (UNABLE TO VERIFY): [results, including any misses or false alarms]
- Number of real photos evaluated: [n]

## Known limitations
- Live behaviour of SAM 3.1 and Muse Spark in this app is unverified until the live tests are run; if SAM does not return decodable masks for a photo, the app returns UNABLE TO VERIFY.
- Positions are 2-D image positions of mask centroids, not physical distances; camera angle changes what "right of" means.
- Occlusion changes a mask and therefore its centroid.
- Four spatial relations only; no colour, orientation, text, lighting or aesthetic checks.
- No reference-image comparison or tolerance; each photo is judged against the rule only.
- Results history lives in the session only.
- Two SAM requests per check. Per-session limits are not a global billing cap.
- No accuracy figures have been measured.

## Future potential
Possibilities, not implemented: comparison against an approved reference frame with a tolerance, more relations (distance bands, alignment, containment), saved rule sets per shot, batch checking of a shoot folder, and extension to video frames.

## Items to verify in the participant portal
- [ ] Required submission fields and character limits
- [ ] Whether a repository link is required, and its visibility
- [ ] Whether a live application URL is required
- [ ] Whether a demo video is required, and length/format
- [ ] Deadline and timezone
- [ ] Any required use-of-Meta-models statement or licence/attribution
