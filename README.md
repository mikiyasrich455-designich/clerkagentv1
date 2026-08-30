# Clerk

An evidence-first shopping research agent. Clerk queries Google Shopping live and returns a buying brief with current prices, retailer links, ratings, and cited sources.

## Run locally

1. Put `GROQ_API_KEY` and `SERP_API_KEY` in the ignored parent `.env` file, or create `clerk/.env` from `.env.example`.
2. Double-click `start.bat`.
3. Open [http://127.0.0.1:8010/](http://127.0.0.1:8010/).

Manual alternative:

```powershell
python -m uvicorn api.index:app --host 127.0.0.1 --port 8010
```

## Deploy on Vercel

The project uses Vercel's native Python convention: `api/index.py` is the serverless FastAPI function. No `vercel.json` is needed.

1. Import the repository into Vercel with **`clerk` as Root Directory**.
2. In **Project Settings → Environment Variables**, add the two exact names below for **Production**, **Preview**, and **Development**:
   - `GROQ_API_KEY`
   - `SERP_API_KEY`
3. Deploy or redeploy from the latest Git commit.
4. Open `https://YOUR-PROJECT.vercel.app/api/health`.

A correct deployment reports:

```json
{"ok": true, "providers": {"groq_configured": true, "serp_configured": true}}
```

If either provider is `false`, the variable was not saved in Vercel for the environment you deployed. Add it and redeploy; Vercel only applies new environment variables to new deployments.

## API providers

| Purpose | Provider |
| --- | --- |
| Product prices and offers | SerpAPI Google Shopping |
| Supporting source links | SerpAPI Google Search |
| Buying brief | Groq `qwen/qwen3.8-27b` |
