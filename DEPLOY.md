# DEPLOY — GitHub + Streamlit Community Cloud (beginner-friendly)

Target: GitHub repository, branch `main`, entry point `app.py` at the repository root, Streamlit Community Cloud, Python 3.12.

What has and has not been checked: the app and its 116 offline tests were run in a clean Python 3.12 virtual environment with `requirements.txt` installed (including `meta-sam-parser==0.1.0` and Streamlit). Nothing has been deployed to Streamlit Community Cloud, and no live Meta API call has been made by the author. Menu labels below follow Streamlit's current documentation and may differ slightly on screen.

**First confirm in the participant portal whether a live URL is required.** If it is not, a local demo recording may be enough and you can skip deployment.

---

## Part 1 — Publish the code to GitHub

You need a GitHub account and Git for Windows. Unzip `takestate.zip` into a folder; `app.py` must sit directly in that folder.

1. On github.com click **New repository**. Choose a name (for example `takestate`). Leave **Add a README / .gitignore / licence** unticked (the project already has them). Click **Create repository**. (Public or private: see "Visibility" below.)
2. In PowerShell, inside the project folder:

```powershell
cd path\to\takestate
git init -b main
git add .
git status
```

3. **Safety check before the first commit.** Each command must behave as stated:

```powershell
git ls-files                                   # must NOT list .env or .streamlit/secrets.toml
git check-ignore -v .env .streamlit/secrets.toml   # both must be reported as ignored
git grep -n -I "LLM_"                          # must print nothing (no key-like text)
```

If `.env` appears in `git ls-files`, stop: `git rm --cached .env`, then repeat. If a real key was ever committed, revoke it on the Meta dashboard; deleting the file later does not remove it from Git history.

4. Commit and push (replace the URL with your repository's):

```powershell
git commit -m "TakeState release candidate"
git remote add origin https://github.com/YOUR-USERNAME/takestate.git
git push -u origin main
```

Expected: the repository page on GitHub shows `app.py`, `requirements.txt`, `config.py`, `.streamlit/config.toml`, `.streamlit/secrets.toml.example` and the docs, and does **not** show `.env` or `.streamlit/secrets.toml`.

---

## Part 2 — Deploy WITHOUT an API key first (to check the cloud build)

Do this while credits are pending. No paid call is possible without a key.

1. Go to https://share.streamlit.io and sign in with GitHub (authorise access to the repository if asked).
2. In your workspace click **Create app** (upper right). When asked "Do you already have an app?" choose **Yup, I have an app**.
3. Fill in:
   - **Repository:** `YOUR-USERNAME/takestate`
   - **Branch:** `main`
   - **Main file path:** `app.py`
   - **App URL** (optional): pick a name.
4. Click **Advanced settings**. Set **Python version** to **3.12**. Leave **Secrets** empty for now. Click **Save**, then deploy.
5. Wait for the build (a few minutes; watch the logs on the right). Dependencies come from `requirements.txt`.

Expected result with no key: the TakeState page loads and shows the red banner "MODEL_API_KEY not configured". The **Interpret** and **Check** buttons are disabled; you can still type a rule, click **Apply rule**, and upload an image. That confirms the build, entry point, Python version and `meta-sam-parser` install.

If the build fails, open the logs. Common causes: wrong **Main file path** (must be `app.py`, no folder prefix), a pushed folder level (files inside a subfolder), or an unsupported Python version choice.

---

## Part 3 — Add secrets after credits are active

Add the access code **first**, then the key. A deployed app with a key and no access code lets anyone with the link spend your credits (the app shows a warning in that case).

1. Open your app's **Settings** (App settings) in your workspace on share.streamlit.io, then **Secrets**.
2. Paste exactly this format (TOML: names unquoted, values in double quotes, numbers unquoted), with your own values:

```toml
APP_ACCESS_CODE = "choose-a-long-random-code"
MODEL_API_KEY = "paste-your-key-here"
MAX_TAKE_CHECKS = 10
MAX_INTERPRETATIONS = 5
```

3. Save. If the app does not pick up the change, reboot it from the app menu (**Manage app**), then reload.
4. Open the app. It shows an **Access** form; enter the access code and click **Unlock**. The "not configured" banner is gone and the buttons are enabled.
5. Run one acceptance scenario from `ACCEPTANCE_TESTS.md` (this uses credits).

`.streamlit/secrets.toml.example` in the repository shows the same format. Never commit a real `secrets.toml`.

### What the protections do, and do not do

- `APP_ACCESS_CODE`: until the code is entered the app shows only the access form, so strangers cannot trigger model calls.
- `MAX_TAKE_CHECKS` (default 20; one check = 2 SAM requests) and `MAX_INTERPRETATIONS` (default 10; one = 1 Muse Spark request): limits **per browser session**.
- **These per-session limits are NOT a global billing cap.** Each new browser session starts with a fresh allowance, and anyone with the access code can open many sessions. The only real cap on spend is the limit/alert on your Meta account (set it, and watch usage). Keep the access code private, and delete the app or rotate/revoke the key after judging.
- Every paid call needs an explicit button click; page reloads and widget changes never call a model.

### Visibility

Per Streamlit's documentation, an app inherits visibility from its GitHub repository: a public repository gives a public app, a private repository gives a private app, and only one private app is allowed at a time. Change who can view it in **App settings → Sharing**. If judges need to open the app, check the portal for what they require, and share the access code only through the submission channel the portal provides.

---

## Updating, rotating, removing

- Update code: `git add . ; git commit -m "..." ; git push`. The app redeploys.
- Rotate the key: create a new key on the Meta dashboard, replace `MODEL_API_KEY` in Secrets, revoke the old one.
- Take the app down: delete it in your Streamlit workspace and revoke the key.

## Local alternatives

Local development reads `.env` (`MODEL_API_KEY=...`) or `.streamlit/secrets.toml` (copy from the `.example` file). The environment/`.env` takes precedence over Streamlit secrets.

## Troubleshooting (cloud)

| Symptom | Action |
|---|---|
| Build error installing `meta-sam-parser` | Confirm Python version is 3.12 in Advanced settings; check the logs for the exact line. |
| Banner "MODEL_API_KEY not configured" after adding the secret | Check TOML quoting, save again, reboot the app. |
| Access form keeps rejecting the code | Compare with the value in Secrets (case-sensitive, no stray spaces). |
| "limit reached for this session" | Expected; reload the page for a new session (this is why it is not a billing cap). |
| Upload rejected | Files are limited to 10 MB (server setting and app check); JPG/PNG/WebP only. |
| Result is UNABLE TO VERIFY with a mask reason | By design: no PASS/FAIL without decoded masks for both objects. See `RUNBOOK.md` troubleshooting. |
