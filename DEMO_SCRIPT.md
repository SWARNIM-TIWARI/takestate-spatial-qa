# DEMO SCRIPT — 90–120 seconds

**Rule:** everything on screen is a real run of the app with your own key. Do not use mocked output. If a step misbehaves live, show it honestly and move on; do not splice in numbers. Rehearse once with the exact photos beforehand (that uses credits), then record. Hide the terminal and any `.env` editor; never show the key.

Prepare: the app running on localhost (or the deployed URL), four photos from `ACCEPTANCE_TESTS.md` (glass right, glass right with small lighting change, glass left, glass corrected), browser zoom ~110%.

| Time | On screen | Say |
|---|---|---|
| 0:00–0:12 | App header, empty state | "When you shoot product photos, a tiny shift, like a glass drifting to the other side of a bottle, ruins continuity. You usually catch it by eye, late. TakeState checks each take against a rule you define, and shows its evidence." |
| 0:12–0:30 | Type "The glass must be to the right of the bottle." Click *Interpret with Muse Spark*. Rule banner appears. | "I write the requirement in plain language. Muse Spark 1.3 turns it into a structured rule, constrained to four relations, and the app validates it locally before using it. Only supported rules reach the checker." |
| 0:30–0:55 | Upload take 1 → *Check this take*. Show mask overlay, PASS badge, centroid chips. | "SAM 3.1 segments the glass and the bottle. TakeState decodes the masks and computes each object's mask centroid in image pixels. Here the glass centroid is at x equals [read value], the bottle at [read value]. That's a PASS, and the evidence is on screen." |
| 0:55–1:10 | Open *Geometric evidence*. | "Bounding-box centres are shown only as labelled diagnostics. The decision uses the mask centroids of both objects, and without both it refuses to decide. The verdict comes from plain geometry, not from a model's opinion." |
| 1:10–1:30 | Upload take 3 (glass left) → *Check*. FAIL badge and correction hint. | "On the next take the glass has moved left. FAIL, with the measured separation and a correction hint: move it right." |
| 1:30–1:45 | Upload corrected take 4 → *Check*. PASS; scroll to history. | "I fix the setup, shoot again, same rule, and it passes. The earlier failure stays in this session's history. Each take is checked against the rule on its own; it is not compared with a reference photo." |
| 1:45–1:55 | (Optional, only if you tested it) take with a hidden glass → amber UNABLE TO VERIFY. Or with several glasses → selection screen. | "When the evidence isn't enough, including when a mask is missing, it says UNABLE TO VERIFY instead of guessing; when several glasses match, I choose which one is meant." |
| 1:55–2:00 | Rule banner / header | "Evidence, deterministic geometry, honest uncertainty, and a correction loop. TakeState." |

## Do and don't

- Read numbers from the screen as they appear; do not state them in advance.
- Do not claim speed, accuracy or reliability figures. Say only what you observed in these runs.
- Do not say the system measures real-world distances; it measures positions in the image.
- Do not call TakeState the first or only tool of its kind.
- Keep the *Session usage* line out of focus or explain it if visible; do not click *Re-check* needlessly on camera (each check is paid).
- If live behaviour differs from this script (for example SAM returns boxes without usable masks, so the app answers UNABLE TO VERIFY), show and narrate exactly that. Do not claim mask-centroid measurements you did not see, and do not describe a box centre as the measurement.

## Capture list

Screenshots: rule banner; take 1 PASS with overlay; evidence panel; FAIL with hint; corrected PASS; history; (optional) UNABLE TO VERIFY; selection screen. Save the sanitized output of `test_sam_live.py` and `test_spark_live.py` as technical evidence.
