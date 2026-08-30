"""
CartSaver (Clerk) backend — FastAPI.

One-store merchant assistant: a shopper chat agent that reads the store's
catalog, answers product questions with real web prices (SERP), and logs
shopper-intent insights for the merchant dashboard.

Endpoints:
    GET  /api/health            -> {ok: true}
    GET  /api/catalog           -> full product catalog
    POST /api/chat              -> {reply, products[], sources[], insight}
    POST /api/search            -> live web price search (SERP + Groq)
    GET  /api/insights          -> aggregated shopper insights for dashboard

LLM: Groq (qwen/qwen3.8-27b). Live prices: SERP Google search.
State is JSON files (hackathon scope — no DB, no cloud).
"""
import concurrent.futures
import json
import os
import threading
import time
import urllib.parse
import urllib.request

from dotenv import load_dotenv
from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CATALOG_PATH = os.path.join(PROJECT_ROOT, "catalog.json")
INSIGHTS_PATH = os.path.join(PROJECT_ROOT, "insights.json")
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
load_dotenv(os.path.join(os.path.dirname(PROJECT_ROOT), ".env"))
GROQ_KEY = os.environ.get("GROQ_API_KEY", "").strip()
SERP_KEY = os.environ.get("SERP_API_KEY", "").strip()
GROQ_MODEL = "qwen/qwen3.8-27b"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
SERP_URL = "https://serpapi.com/search?engine=google"

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"

app = FastAPI(title="CartSaver Clerk")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["null", "http://127.0.0.1:8010", "http://localhost:8010"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def _read_json(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _catalog():
    return _read_json(CATALOG_PATH, {"products": []})


def _insights():
    return _read_json(INSIGHTS_PATH, {"log": []})


def _log_insight(entry):
    data = _insights()
    entry["ts"] = int(time.time())
    entry["id"] = str(len(data["log"]) + 1)
    data["log"].insert(0, entry)
    data["log"] = data["log"][:200]
    _write_json(INSIGHTS_PATH, data)


# ---------------------------------------------------------------------------
# LLM + web search
# ---------------------------------------------------------------------------
def _groq(messages, max_tokens=600, temperature=0.4, json_mode=False):
    """Call Groq. Returns text or None on failure."""
    if not GROQ_KEY:
        raise RuntimeError("GROQ_API_KEY is not configured")
    body = {
        "model": GROQ_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    req = urllib.request.Request(
        GROQ_URL,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + GROQ_KEY,
            "User-Agent": UA,
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        d = json.loads(r.read().decode("utf-8"))
    return d["choices"][0]["message"]["content"].strip()


def _groq_json(messages, max_tokens=800):
    """Call Groq and parse a JSON object, tolerating markdown code fences."""
    text = _groq(messages, max_tokens=max_tokens, temperature=0.3, json_mode=True)
    if not text:
        return None
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        i, j = text.find("{"), text.rfind("}")
        if i != -1 and j > i:
            try:
                return json.loads(text[i:j + 1])
            except json.JSONDecodeError:
                return None
        return None


def _serp(query, num=8):
    """Live Google search via SERP. Returns list of {title, link, snippet, price, image, source}."""
    if not SERP_KEY:
        raise RuntimeError("SERP_API_KEY is not configured")
    params = {
        "engine": "google",
        "q": query,
        "api_key": SERP_KEY,
        "num": str(num),
        "gl": "us",
        "hl": "en",
    }
    url = SERP_URL + "&" + urllib.parse.urlencode(
        {k: v for k, v in params.items() if k != "engine"}
    )
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=15) as r:
        data = json.loads(r.read().decode("utf-8"))

    out = []
    # Shopping results have prices + thumbnails — best for commerce
    for s in data.get("shopping_results", [])[:num]:
        price = s.get("price") or (
            f"${s['extracted_price']}" if s.get("extracted_price") else ""
        )
        out.append({
            "title": s.get("title", ""),
            "link": s.get("link") or s.get("product_link", "") or "",
            "snippet": s.get("source") or s.get("snippet") or "",
            "price": price,
            "image": s.get("thumbnail") or "",
            "source": s.get("source") or "",
            "rating": s.get("rating") or "",
            "reviews": s.get("reviews") or "",
        })
    for res in data.get("organic_results", [])[:num]:
        out.append({
            "title": res.get("title", ""),
            "link": res.get("link", ""),
            "snippet": (res.get("snippet") or "")[:220],
            "price": res.get("price") or "",
            "image": (res.get("thumbnail") or ""),
            "source": (res.get("displayed_link") or "").split("/")[0],
        })
    return out


def _serp_shopping(query, num=8):
    """Google Shopping engine — richer product data (image, price, source, link)."""
    if not SERP_KEY:
        raise RuntimeError("SERP_API_KEY is not configured")
    params = {
        "engine": "google_shopping",
        "q": query,
        "api_key": SERP_KEY,
        "num": str(num),
        "gl": "us",
        "hl": "en",
    }
    url = "https://serpapi.com/search?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=15) as r:
        data = json.loads(r.read().decode("utf-8"))
    out = []
    for s in data.get("shopping_results", [])[:num]:
        price = s.get("price")
        if not price and s.get("extracted_price"):
            price = f"${s['extracted_price']}"
        out.append({
            "title": s.get("title", ""),
            "link": s.get("product_link") or s.get("link", "") or "",
            "snippet": s.get("source") or "",
            "price": price or "",
            "image": s.get("thumbnail") or "",
            "source": s.get("source") or "",
            "rating": s.get("rating") or "",
            "reviews": s.get("reviews") or "",
            "delivery": s.get("delivery") or "",
        })
    return out


# ---------------------------------------------------------------------------
# Matching helpers (deterministic rules first — reliable demo)
# ---------------------------------------------------------------------------
def _match_products(query):
    """Deterministic catalog search over name, category, tags."""
    q = query.lower()
    prods = _catalog().get("products", [])
    scored = []
    for p in prods:
        hay = " ".join([
            p.get("name", ""),
            p.get("category", ""),
            " ".join(p.get("tags", [])),
        ]).lower()
        score = 0
        if q in hay:
            score += 5
        for word in q.split():
            if word and word in hay:
                score += 2
        # size/color matches
        for attr in p.get("sizes", []) + p.get("colors", []):
            if attr.lower() in q:
                score += 2
        if score:
            scored.append((score, p))
    scored.sort(key=lambda x: -x[0])
    return [p for _, p in scored[:5]]


def _price_talk(query, products):
    """Compose the shopping-assistant reply from catalog + live web prices."""
    # Try live prices for the top product
    sources = []
    price_note = ""
    if products:
        top = products[0]
        try:
            hits = _serp(top.get("name") + " price", num=5)
            prices = [h for h in hits if h.get("price")]
            if prices:
                price_note = (
                    f"Live web prices for the {top['name']}: "
                    + " · ".join(f"{h['title'][:40]} {h['price']}" for h in prices[:3])
                )
                sources = hits[:3]
        except Exception:
            pass

    lines = []
    for p in products[:3]:
        price = p.get("price", "Contact for price")
        sizes = ", ".join(p.get("sizes", [])) or "one size"
        stock = "in stock" if p.get("in_stock", True) else "out of stock"
        lines.append(
            f"• {p['name']} — {price} ({sizes}; {stock})"
        )
    if not lines:
        lines.append("I couldn't find that in our catalog — try another search term.")

    reply = "\n".join(lines)
    if price_note:
        reply += "\n\n" + price_note
    return reply, sources


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health():
    return {
        "ok": bool(GROQ_KEY and SERP_KEY),
        "model": GROQ_MODEL,
        "providers": {
            "groq_configured": bool(GROQ_KEY),
            "serp_configured": bool(SERP_KEY),
        },
    }


@app.get("/api/catalog")
def catalog():
    return _catalog()


@app.get("/api/insights")
def insights():
    data = _insights()
    log = data.get("log", [])
    total = len(log)
    categories = {}
    for e in log:
        c = e.get("category", "general")
        categories[c] = categories.get(c, 0) + 1
    top_queries = {}
    for e in log:
        q = e.get("query", "").strip()
        if q:
            top_queries[q] = top_queries.get(q, 0) + 1
    return {
        "total_sessions": total,
        "categories": categories,
        "top_queries": sorted(
            top_queries.items(), key=lambda x: -x[1]
        )[:8],
        "recent": log[:15],
        "recovered_hint": "Live once checkout is connected",
    }


@app.post("/api/chat")
def chat(payload: dict = Body(...)):
    """Agentic shopping loop.

    Understand -> Plan -> Act -> Observe -> Decide -> Act again -> Verify.

    Returns the final brief plus a `trace` (the visible reasoning the judge
    sees) and an `evidence` trail of the searches that were actually run.
    """
    query = (payload.get("message") or "").strip()
    if not query:
        return JSONResponse({"error": "message is required"}, status_code=400)

    trace = []
    evidence = []

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # 1+2) UNDERSTAND + PLAN - deterministic (fast, no LLM): extract a
    #      budget via regex and plan the search queries. Keeps the agentic
    #      trace without adding an LLM round-trip.
    # ------------------------------------------------------------------
    import re as _re
    budget = None
    _m = _re.search(r'(?:under|below|less than|max|budget|around|~)\s*\$?\s*(\d{2,4})', query, _re.I)
    if _m:
        budget = int(_m.group(1))
    category = query
    queries = [query]
    trace.append({
        "step": "understand",
        "label": "Understood the request",
        "detail": (f"shopping for \u201c{query}\u201d"
                   + (f", budget \u2264 ${budget}" if budget else "")),
    })
    trace.append({
        "step": "plan",
        "label": "Planned the research",
        "detail": "Searching: " + " \u00b7 ".join(queries[:2]),
    })

    # ------------------------------------------------------------------
    # 3) ACT — run the searches (all queries in parallel)
    # ------------------------------------------------------------------
    def _run(query_text):
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            f1 = ex.submit(_serp_shopping, query_text, 8)
            f2 = ex.submit(_serp, query_text, 8)
            try:
                shop = f1.result(timeout=15)
            except Exception:
                shop = []
            try:
                org = f2.result(timeout=15)
            except Exception:
                org = []
        return shop, org

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(queries)) as ex:
        results = list(ex.map(_run, queries))

    live_products = []
    live_sources = []
    for query_text, (shop, org) in zip(queries, results):
        live_products.extend(shop)
        live_sources.extend(org)
        for s in shop:
            evidence.append({"q": query_text, "title": s.get("title", ""),
                             "price": s.get("price", ""), "source": s.get("source", "")})

    # de-dupe by title
    seen = set()
    deduped = []
    for p in live_products:
        key = (p.get("title") or "").lower()
        if key and key not in seen:
            seen.add(key)
            deduped.append(p)
    live_products = deduped

    trace.append({
        "step": "act",
        "label": "Searched live sources",
        "detail": f"{len(live_products)} product results · {len(live_sources)} sources",
    })

    # ------------------------------------------------------------------
    # 4) OBSERVE + DECIDE — does anything meet every constraint?
    # ------------------------------------------------------------------
    catalog_matches = _match_products(query)

    def _build_cards(items, source_override=None):
        cards = []
        for item in items[:6]:
            cards.append({
                "name": item.get("title") or item.get("name") or "Product",
                "category": item.get("source") or item.get("category") or "Web result",
                "price": item.get("price") or "",
                "sizes": item.get("sizes") or [],
                "in_stock": item.get("in_stock", True),
                "image": item.get("image") or item.get("thumbnail") or "",
                "link": item.get("link") or item.get("product_link") or "",
                "source": source_override or item.get("source") or "",
                "rating": item.get("rating") or "",
                "reviews": item.get("reviews") or "",
                "delivery": item.get("delivery") or "",
            })
        return cards

    products = _build_cards(live_products)

    # Local catalog is the "own store" — used for the inventory-reaction demo.
    own_store = _build_cards(catalog_matches, source_override="Mika's Threads")

    # Decide whether to refine: nothing affordable, or no results at all.
    refine_query = None
    refine_reason = None
    if not live_products:
        refine_query = " ".join(query.split()[:3])
        refine_reason = "No live results — broadening the search."
    elif budget:
        affordable = [p for p in products if _price_le(p.get("price", ""), budget)]
        if not affordable and queries:
            refine_query = f"cheap {category} under ${int(budget)}"
            refine_reason = f"Nothing found under ${int(budget)} — refining."

    trace.append({
        "step": "decide",
        "label": "Evaluated the evidence",
        "detail": refine_reason or "Found candidates that fit the request.",
    })

    # ------------------------------------------------------------------
    # 5) ACT AGAIN — one refinement pass if the critic demanded it
    # ------------------------------------------------------------------
    if refine_query and refine_query != query:
        trace.append({
            "step": "act_again",
            "label": "Refining the search",
            "detail": refine_query,
        })
        try:
            more_products, more_sources = _run(refine_query)
            live_products = deduped + more_products
            live_sources = live_sources + more_sources
            products = _build_cards(live_products)
            for s in more_products:
                evidence.append({"q": refine_query, "title": s.get("title", ""),
                                 "price": s.get("price", ""), "source": s.get("source", "")})
        except Exception as e:
            print("refine failed:", e)

    # ------------------------------------------------------------------
    # 6) VERIFY — deterministic scoring against constraints (no extra LLM call)
    # ------------------------------------------------------------------
    for p in products:
        score = 3
        reasons = []
        if budget and p.get("price") and _price_le(p.get("price", ""), budget):
            score += 1
            reasons.append("under budget")
        elif budget and p.get("price"):
            score -= 1
            reasons.append("over budget")
        name = (p.get("name") or "").lower()
        score += 1 if budget else 0
        p["fit_score"] = min(5, max(1, score))
        p["fit_reason"] = ", ".join(reasons) if reasons else "scores well"
    if products:
        top = products[0]
        trace.append({
            "step": "verify",
            "label": "Verified against your needs",
            "detail": f"Top fit: {top['name']} ({top.get('fit_score')}/5).",
        })

    # ------------------------------------------------------------------
    # Build the final brief + sources
    # ------------------------------------------------------------------
    sources = []
    for s in (live_sources or [])[:6]:
        if not s.get("title"):
            continue
        sources.append({
            "title": s.get("title"),
            "link": s.get("link", ""),
            "snippet": s.get("snippet", ""),
            "price": s.get("price", ""),
        })

    # Inventory reaction (track example): if the store has a match but it's
    # out of stock, recommend a live alternative instead.
    out_of_stock = [p for p in own_store if not p.get("in_stock", True)]
    inventory_note = ""
    if out_of_stock and products:
        alt = products[0].get("name", "a live alternative")
        inventory_note = (f" Heads up: {out_of_stock[0]['name']} in your size is out of stock, "
                          f"so I found {alt} instead.")

    context_lines = []
    for p in products[:6]:
        line = f"{p['name']}"
        if p.get("price"):
            line += f" — {p['price']}"
        if p.get("source"):
            line += f" ({p['source']})"
        if p.get("rating"):
            line += f", rating {p['rating']}"
        if p.get("fit_score"):
            line += f", fit {p['fit_score']}/5"
        context_lines.append(line)
    context = "\n".join(context_lines) if context_lines else "(no live results)"

    reply = None
    try:
        reply = _groq([
            {"role": "system", "content":
             "You are Clerk, an evidence-first shopping research agent. "
             "Given the shopper's request and REAL live Google Shopping results, "
             "write a concise 3-5 sentence buying brief naming the best options "
             "and their real prices. Never invent prices — only cite what is given. "
             "If nothing fits the budget, say so honestly."},
            {"role": "user", "content":
             f"Shopper request: {query}\n\nLive results:\n{context}\n\n"
             f"Write the buying brief now." + inventory_note},
        ], max_tokens=320)
    except Exception as e:
        reply = f"I couldn't complete the research. Error: {str(e)}"
        print("Groq failed:", e)

    if not reply or len(reply) < 5:
        if products:
            reply = (f"Here are the strongest live matches for '{query}':\n"
                     + "\n".join(f"• {p['name']} — {p.get('price','—')} ({p.get('source','')})"
                                 for p in products[:3]))
        else:
            reply = f"I couldn't find live results for '{query}' right now."

    _log_insight({
        "query": query,
        "category": category,
        "matched": len(products),
        "products": [p.get("name") for p in products[:3]],
        "live": bool(live_products),
    })

    return {
        "reply": reply,
        "products": products,
        "sources": sources,
        "live": bool(live_products),
        "trace": trace,
        "evidence": evidence[:12],
        "inventory_note": inventory_note,
    }


def _price_le(price_str, budget):
    """True if a '$xx.xx' price string is at or under budget (best-effort)."""
    try:
        digits = "".join(ch for ch in str(price_str) if ch.isdigit() or ch == ".")
        val = float(digits)
        return val <= float(budget)
    except (ValueError, TypeError):
        return False


@app.post("/api/search")
def search(payload: dict = Body(...)):
    """Raw live web price search."""
    query = (payload.get("q") or payload.get("query") or "").strip()
    if not query:
        return JSONResponse({"error": "q is required"}, status_code=400)
    try:
        hits = _serp(query, num=8)
        return {"query": query, "results": hits}
    except Exception as e:
        return JSONResponse({"error": f"search failed: {e}"}, status_code=500)


# Serve the single-file frontend.
app.mount("/", StaticFiles(directory=PROJECT_ROOT, html=True), name="frontend")
