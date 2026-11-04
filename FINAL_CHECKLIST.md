# FINAL CHECKLIST

Legend: ✅ done and evidenced in this package · ⬜ pending (needs you / your account / your laptop). Items marked ✅ were verified in the author's sandbox only.

## Code and offline verification
- ✅ Python compiles for all project files.
- ✅ Offline tests: 116 passed, 0 failed (`python tests.py`; clean Python 3.12 venv with `requirements.txt` incl. Streamlit), including mask-only invariant regressions (missing / undecodable / empty / unmappable / one-valid-one-not), and: mask centroid maths and source-image mapping, empty/invalid masks, all four rules, ambiguity, vertical percentages, schema validation, secret redaction, call budget, access code, and Streamlit rerun / retake protection (fake SAM function, no network).
- ✅ `.env` and `.streamlit/secrets.toml` are git-ignored (checked in a scratch Git repo); `.env.example` is tracked and contains no key.
- ✅ Key can come from `.env` or Streamlit secrets; never displayed.
- ✅ Mask-only invariant: no bounding-box fallback in verdicts (confirmed by regression tests that fail when the fallback is reintroduced).
- ✅ Docs written: README, RUNBOOK, ACCEPTANCE_TESTS, DEMO_SCRIPT, SUBMISSION_DRAFT, DEPLOY.

## Account and live verification (all pending)
- ⬜ Hackathon credits claimed.
- ⬜ Billing confirmed; spending alert/limit set.
- ⬜ API key configured securely (`.env`; never in chat, screenshots or Git).
- ⬜ Offline tests re-run on your Windows machine.
- ⬜ SAM live test passed: `python test_sam_live.py <photo> "glass"` (request completes, mask decoded, centroid inside image and object).
- ⬜ Muse Spark live test passed: `python test_spark_live.py`.
- ⬜ Mask overlay visually checked on the correct object.
- ⬜ Full app workflow passed (define → upload → check → inspect → retake → verify).
- ⬜ Acceptance A (PASS) evaluated.
- ⬜ Acceptance B (FAIL, then corrected PASS) evaluated.
- ⬜ Acceptance C (ambiguous, selection) evaluated.
- ⬜ Acceptance D (UNABLE TO VERIFY) evaluated.
- ⬜ No unnecessary repeat API calls (session usage line only changes after button clicks).
- ⬜ Misses, false alarms and ambiguous detections recorded honestly.

## Deployment (only if required or useful; see DEPLOY.md)
- ⬜ Code pushed to GitHub `main` after the DEPLOY.md Part 1 safety checks (`.env` / `secrets.toml` not tracked, no `LLM_` text).
- ⬜ Cloud build checked with NO key (banner visible, build succeeds on Python 3.12).
- ⬜ Official event requirements checked in the participant portal (deployed URL? video? repo format?).
- ⬜ GitHub repo created; verified no key in history (`git log -p` search).
- ⬜ Streamlit Community Cloud app created; secrets pasted in the Secrets box.
- ⬜ `APP_ACCESS_CODE` set before sharing a public URL.
- ⬜ Deployed app smoke-tested once.

## Submission
- ⬜ Screenshots captured (see DEMO_SCRIPT capture list).
- ⬜ Sanitized live-test output saved as evidence.
- ⬜ Demo video recorded (if requested or useful).
- ⬜ README checked against observed behaviour (limitations updated).
- ⬜ SUBMISSION_DRAFT completed with real observations only.
- ⬜ Final submission completed before the deadline (confirm exact time and timezone in the portal).
