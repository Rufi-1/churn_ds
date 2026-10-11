# Deploying so that everyone can use it

The repository already contains the pre-built demo (trained model, tables, charts, reports), so a public deployment works
immediately and visitors can also upload their own files.

## Option 1 - Streamlit Community Cloud (free, recommended)
1. Push the project to a **public** GitHub repository (see README / chat instructions). Make sure `outputs/`, `data/processed/`,
   `data/sample/`, `requirements.txt` and `.streamlit/config.toml` are committed.
2. Go to https://share.streamlit.io, sign in with GitHub, **Create app**, choose the repo, branch `main`, main file `app.py`.
3. Advanced settings -> Python 3.12. Deploy. First start takes a few minutes while packages install.
4. Share the URL. Anyone with the link can open it; no login is required for visitors.

## Option 2 - Hugging Face Spaces (free)
New Space -> SDK **Streamlit** -> push the same files. Put this at the very top of the Space's README.md:
```
---
title: Churn Intelligence
sdk: streamlit
app_file: app.py
python_version: "3.12"
---
```

## Things to know for a public app
* **Resources**: free tiers have roughly 1 GB RAM and shared CPU. Option A (train on my data) runs one analysis at a time
  per server (a lock queues the others); "Quick" mode finishes in about 1-3 minutes there. Option B (scoring) is instant.
* **Large files**: a 440,000-row file peaked at about 600 MB of memory in testing (23 s in Quick mode). Free tiers have about 1 GB, so very
  large files may fail there; cut the file to a few hundred thousand rows or use a paid tier.
* **Privacy**: uploads live only in a temporary folder per browser session, are deleted after about 6 hours or on restart, and are
  never committed or sent elsewhere. Tell visitors not to upload personal data.
* **Sleeping apps**: free apps sleep when idle and wake on the next visit (a few seconds to a minute).
* **Pinned versions**: do not loosen the pins in requirements.txt unless you retrain; the demo model is a pickle bound to scikit-learn 1.9.1.
  If the model cannot be loaded the app shows a button that rebuilds the demo from the bundled sample.
* **Prediction history** is per session workspace and not persistent.
