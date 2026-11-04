# RUNBOOK — Windows PowerShell, installation to running app

All commands run from the project folder (the one containing `app.py`). Nothing below has been run against the live Meta API by the author; "expected output" describes what a success looks like.

## 0. Prerequisites

- Python 3.10+ (`py -3 --version`).
- Hackathon credits claimed and billing checked at https://dev.meta.ai/platform (set a spending alert/limit there if offered).
- Your API key, copied from the dashboard. Never paste it into chat, screenshots, source files or Git.

## 1. Virtual environment

```powershell
cd path\to\takestate
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Expected: prompt starts with `(.venv)`.
If activation is blocked: `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`, or skip activation and prefix every `python` below with `.\.venv\Scripts\`.

## 2. Dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -c "import streamlit, meta_sam_parser, numpy, PIL; print('imports ok')"
```

Expected: `imports ok`.

## 3. API key (local)

```powershell
Copy-Item .env.example .env
notepad .env
```

Set the line `MODEL_API_KEY=` to your key, with no quotes or spaces. Save. Check that it is detected without printing it:

```powershell
python -c "from meta_client import get_api_key_status; print(get_api_key_status())"
```

Expected: `{'configured': True}`.

## 4. Offline tests (free, no key)

```powershell
python -m py_compile app.py config.py interpreter.py meta_client.py rule_engine.py sam_parser.py tests.py test_sam_live.py test_spark_live.py
python tests.py
```

Expected: no compile output, then a list of passing tests ending with `Passed: 116  |  Failed: 0  |  Total: 116`. (The AppTest-based tests take several seconds.) These tests use fake data and a fake SAM function; they prove local logic only, not live API behaviour.

## 5. Live SAM smoke test (1 paid request)

Use a real photo that clearly contains the object:

```powershell
python test_sam_live.py C:\path\to\take1.jpg "glass"
```

Expected on success: `[PASS]` lines for request completed, detection parsed, box inside the image, mask decoded, mask centroid inside the image and inside its own box; per-detection bbox, bbox centre and mask centroid coordinates; final `SAM live test: N passed, 0 failed`; exit code 0 (`echo $LASTEXITCODE`).
Exit code 2 = skipped (no key). Exit code 1 = a check failed. Exit code 3 = bad arguments or unreadable image.

If `Segmentation mask decoded` fails but boxes parse, the app will return **UNABLE TO VERIFY** for that photo: it never uses bounding-box centres for a verdict, so no PASS/FAIL is possible until masks decode. Save the sanitized output (it contains no key) before changing anything.

## 6. Live Muse Spark smoke test (3 small paid requests)

```powershell
python test_spark_live.py
```

Expected: two requirements map to `right_of` and `above`, and "Make the glass red." is flagged unsupported (no rule created); `Spark live test: all checks passed`; exit code 0.

## 7. Start the app

```powershell
python -m streamlit run app.py
```

Expected: browser opens at http://localhost:8501 showing the TakeState header and no "MODEL_API_KEY not configured" banner. Then follow `ACCEPTANCE_TESTS.md`.

## Troubleshooting

| Symptom | Likely cause and action |
|---|---|
| Banner "MODEL_API_KEY not configured" | `.env` missing, in the wrong folder, key line empty, or app started from another directory. Run step 3 check. |
| `API key rejected (HTTP 401)` | Wrong/revoked key, stray quotes or spaces in `.env`. Re-copy the key from the dashboard. |
| `API quota exceeded (HTTP 402)` | Credits not claimed or exhausted. Check Billing on the dashboard. |
| `Endpoint not found (HTTP 404)` | Model name or base URL differs for your account. Check model access in the dashboard against `_SAM_MODEL` / `_SPARK_MODEL` in `meta_client.py`. |
| HTTP 403 or "model not available" | Your account lacks access to that model; request/confirm access. |
| HTTP 429 | Rate limited. Wait and retry once; do not loop. |
| `Network error` / timeout after 90 s | Connectivity, proxy or VPN; try a smaller image. |
| `SAM stream ended without a response.completed event` | Interrupted or truncated stream. Retry once with *Re-check*. |
| Zero detections for an object clearly in the photo | Try a more specific or simpler noun phrase ("wine glass", "bottle"). Look at the `first 300 chars` output in `test_sam_live.py`. |
| Boxes but no mask (`mask centroid: NOT AVAILABLE (status: none / decode_failed / empty / unmappable)`) | The result will be UNABLE TO VERIFY by design. `none`: SAM returned no mask record for the box; `decode_failed`: the payload could not be decoded; `empty`: no foreground pixels; `unmappable`: bounds unusable. Keep the sanitized output for diagnosis. |
| Mask overlay offset from the object, or UNABLE TO VERIFY "inconsistent with its bounding box" | Mapping problem; compare the `mask raster` bounds with the bbox in `test_sam_live.py` output. The app refuses to decide when a centroid is outside the image or far from its own box. Do not demonstrate mask centroids until resolved. |
| `UNABLE TO VERIFY` naming a missing/empty/undecodable mask | The decoded mask for that object is unusable; try another photo or a more specific object description. |
| `Image is X MB; the limit is 10 MB` / "Could not read this file" | Use a JPG/PNG/WebP under 10 MB. |
| PowerShell won't activate venv | See step 1. |
| `pip` cannot install `meta-sam-parser==0.1.0` | Check `python --version` (needs 3.10+) and internet access. |

Never include the key when asking for help; the app and live tests never print it.
