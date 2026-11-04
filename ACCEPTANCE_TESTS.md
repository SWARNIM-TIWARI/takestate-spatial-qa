# ACCEPTANCE TESTS — real images, four scenarios

**No results are recorded in this document.** Run each scenario with your own photos and key and fill in the *Observed* lines. Unverified until you do.

Common setup: rule `glass right_of bottle` (type it and use *Interpret with Muse Spark*, or the manual fields). Use a plain background and fixed lighting. Shoot at the same camera position for A, B and C. Keep the camera roughly level with the objects. Each *Check this take* makes 2 SAM requests, so budget about 1 check per scenario plus at most one *Re-check*.

Record for every scenario: image file name, verdict, parse method, mask status for each object (ok / none / decode_failed / empty / unmappable), both centroids, separation px and %, and a screenshot.

---

## A. PASS — glass clearly right of the bottle

- **Image setup:** one bottle on the left, one glass clearly on the right, no overlap, both fully visible, at least about one object-width of gap.
- **User action:** upload; click **Check this take**.
- **Expected SAM behaviour:** one detection each for "glass" and "bottle"; a decodable, non-empty mask for each. **A box without a mask cannot produce PASS or FAIL.**
- **Expected rule evaluation:** glass mask-centroid x > bottle mask-centroid x → **PASS**; separation is positive, in px and % of image width.
- **Expected UI:** green PASS badge; overlay masks on the correct objects; subject (cyan) and reference (orange) chips with pixel coordinates; centroid marker inside each mask; bbox centre shown separately; no correction hint; history has one entry; no selection screen.
- **Failure if:** verdict is FAIL; verdict is UNABLE TO VERIFY despite visible, separate objects (record the stated reason and mask status: a missing-mask result means SAM returned no usable mask, which is a live-integration finding, not a pass); an overlay sits on the wrong object or is visibly offset; a centroid marker is outside its object; bounding-box centres are presented as the measurement; more than 2 SAM requests are made.
- **Observed:** _(fill in)_

## B. FAIL — glass clearly left of the bottle

- **Image setup:** same objects and camera, positions swapped (glass on the left).
- **User action:** upload; **Check this take**.
- **Expected SAM behaviour:** as A.
- **Expected rule evaluation:** glass centroid x < bottle centroid x → **FAIL**, explanation says the glass is N px to the LEFT and the rule requires RIGHT.
- **Expected UI:** red FAIL badge; correction hint ("Move glass further to the right of bottle…"); evidence panel with signed separation.
- **Retake loop:** move the glass back to the right, upload, **Check this take** → PASS; the rule stays unchanged and the FAIL remains in the history.
- **Failure if:** verdict is PASS; no hint; hint points the wrong way; the rule changes between takes; history entry lost.
- **Observed:** _(fill in)_

## C. AMBIGUOUS — several glasses present

- **Image setup:** the bottle and two or three glasses, all visible.
- **User action:** upload; **Check this take**; on the selection screen choose the intended glass; **Confirm selections & evaluate**.
- **Expected SAM behaviour:** more than one "glass" detection.
- **Expected rule evaluation:** nothing is evaluated until you choose; then the chosen instance only is evaluated. Choosing a different glass can change the verdict, which is correct.
- **Expected UI:** candidate boxes numbered #1, #2, … on the image with a radio list; after confirming, only the selected instance is highlighted; the chosen index is preserved while the result is shown.
- **Failure if:** the first detection is silently used; the result ignores the selection; the selection screen never appears; the highlighted instance differs from the selected one.
- **Observed:** _(fill in; if SAM returns only one detection for a multi-glass scene, record that as a SAM miss, not a pass)_

## D. UNABLE TO VERIFY — heavily obscured or unlocalisable object

- **Image setup:** hide the glass almost entirely behind the bottle or another object, or use a very dark frame; alternatively request an object that is not in the photo.
- **User action:** upload; **Check this take**.
- **Expected SAM behaviour:** zero detections for the obscured/absent object (or a doubtful one).
- **Expected rule evaluation:** missing object → **UNABLE TO VERIFY** with a message naming the missing object; no PASS/FAIL.
- **Expected UI:** amber badge; explanation and a visibility hint; no invented centroid.
- **Failure if:** a confident PASS or FAIL is shown for an object a human cannot see in the photo, or for any object whose mask was not decoded. Note: if SAM does return a mask for a partly hidden object, the verdict is legitimate for the *visible* part; record this as a limitation (centroid of visible mask), not as a pass of this scenario.
- **Observed:** _(fill in)_

---

## Cross-cutting checks

- For every PASS/FAIL the evidence panel shows mask centroids for **both** objects; bbox centres appear only labelled "diagnostic only".
- Mask-invariant cases (missing / undecodable / empty / one-valid-one-not) are covered by offline regression tests; they cannot be reliably staged with photos, so do not claim them as live results.

- Reloading the page or interacting with widgets makes **no** SAM requests (watch the "Session usage" line; it must change only after a button click).
- *Check this take* is disabled after a check; only *Re-check* repeats the call.
- Unsupported file or >10 MB shows a clear message and sends nothing.
- No key or key prefix appears anywhere in the UI, terminal output or error messages.
- Optional four-photo sequence for the demo: (1) glass right, (2) small lighting/framing change, glass still right, (3) glass moved left, (4) corrected. Report what the app actually returns, including any miss or false alarm.
