# Clerk

An evidence-first shopping research agent. Clerk queries Google Shopping in real time, then returns a buying brief with product prices, images, retailer links, ratings, delivery details, and cited sources.

## Deploy on Vercel

1. Import this repository into Vercel, with the repository root set to this `clerk` folder.
2. In **Project Settings → Environment Variables**, add the following for **Production**, **Preview**, and **Development**:
   - `GROQ_API_KEY`
   - `SERP_API_KEY`
3. Deploy. Vercel uses `vercel.json` to run `backend.py` as the FastAPI server and serve the single-file `index.html` frontend.

Never put actual keys in `.env.example`, source files, or Git commits.

## Run locally

```powershell
cd clerk
$env:GROQ_API_KEY="your_groq_key"
$env:SERP_API_KEY="your_serpapi_key"
python -m uvicorn backend:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`.
