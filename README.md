# DarGlobal & Wasalt Assistant

RAG chatbot over public data from darglobal.co.uk and wasalt.sa (`data/*.md`, collected 2026-11-19).
FastAPI + keyword/TF-IDF retrieval + OpenRouter free models (with fallback list). Dockerised.

## Run locally
    docker build -t dg-wasalt-bot .
    docker run -p 8000:8000 -e OPENROUTER_API_KEY=sk-or-... dg-wasalt-bot

Env: `OPENROUTER_API_KEY` (required), `OPENROUTER_MODELS` (comma list, default free Llama 3.3 70B -> gpt-oss-20b -> Gemma 3 27B).

## Deploy (Render, free tier)
1. Push this folder to a GitHub repo.
2. Render -> New Web Service -> select repo, Runtime: Docker.
3. Add env var `OPENROUTER_API_KEY`. Deploy. Health check path: `/health`.
4. Open the generated `https://<name>.onrender.com` URL.

## Notes
- Scraping: pages are bot-protected (Incapsula / Cloudflare), so content was read through a normal browser session rather than scripted HTTP clients. Respect each site's terms.
- Free OpenRouter model IDs change; update `OPENROUTER_MODELS` if one is retired.

## Run locally without Docker (conda env `autoflow`)
The app falls back to `sentence-transformers` + built-in keyword vectors when `fastembed` is not installed, and to embedded Qdrant when `QDRANT_URL` is unset.

    $env:GROQ_API_KEY = "..."            # or OPENROUTER_API_KEY
    $env:STORE_DIR = "$PWD\.store"
    C:\Users\HP\anaconda3\envs\autoflow\python.exe -m uvicorn app.main:app --port 8010
    # eval (app must be running):
    C:\Users\HP\anaconda3\envs\autoflow\python.exe eval\run_eval.py http://127.0.0.1:8010

Open http://127.0.0.1:8010 . Never commit API keys.
