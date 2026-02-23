"""
chatbot_logic.py  —  Intelligence layer for the Influenza Surveillance chatbot.

Architecture
────────────
1. Intent detection   — classify the query into one of 10 intents.
2. Live data evidence — pull real numbers from the CSV datasets via data_query.py.
3. RAG retrieval      — semantic search over the knowledge-base index.
4. LLM synthesis      — Groq / llama-3.1-8b-instant synthesises all evidence into
                        a concise plain-text answer.

Supported natural-language query types
───────────────────────────────────────
• "Why is Region 5 increasing?"          → trend_reason
• "Which strain dominated in 2018 in Region 3?" → strain_query
• "Compare current activity with last season."  → season_compare
• "What is the severity / forecast?"     → dashboard
• "Which age group is most at risk?"     → age_risk
• Explanations, comparisons, greetings, lookups, open-ended RAG
"""

import json
import os
import re
from pathlib import Path

import faiss
import numpy as np
from groq import Groq
from sentence_transformers import SentenceTransformer

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from utils.data_query import (
    format_evidence,
    query_age_risk,
    query_all_regions_latest,
    query_dominant_strain,
    query_recent_history,
    query_season_comparison,
    query_trend_explanation,
)

# ──────────────────────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).resolve().parent.parent
RAG_DIR        = BASE_DIR / "rag_model" / "rag_model_data"
INDEX_PATH     = RAG_DIR / "influenza.index"
KNOWLEDGE_PATH = RAG_DIR / "merged_knowledge.json"  # built by build_index.py (includes CSV chunks)
_FALLBACK_PATH = RAG_DIR / "influenza_rag_knowledge.json"  # original fallback

# ──────────────────────────────────────────────────────────────
# GROQ CLIENT
# ──────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    import warnings
    warnings.warn("GROQ_API_KEY not set — RAG chatbot unavailable. Add it to .env.")
    client = None
else:
    client = Groq(api_key=GROQ_API_KEY)

embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ──────────────────────────────────────────────────────────────
# KNOWLEDGE BASE + FAISS INDEX
# ──────────────────────────────────────────────────────────────
try:
    index = faiss.read_index(str(INDEX_PATH))
except Exception:
    index = None

try:
    with open(KNOWLEDGE_PATH, "r", encoding="utf-8") as f:
        knowledge = json.load(f)
except Exception:
    # Fall back to the original expert-narrative JSON if merged hasn't been built yet
    try:
        with open(_FALLBACK_PATH, "r", encoding="utf-8") as f:
            knowledge = json.load(f)
    except Exception:
        knowledge = []

documents = [d.get("content", "") for d in knowledge]


def build_and_load_index():
    """Builds a FAISS index from the in-memory documents list if missing."""
    global index
    if not documents:
        return
    emb = embedder.encode(documents, show_progress_bar=False).astype(np.float32)
    dim = emb.shape[1]
    idx = faiss.IndexFlatL2(dim)
    idx.add(emb)
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(idx, str(INDEX_PATH))
    index = idx


# ──────────────────────────────────────────────────────────────
# INTENT DETECTION
# ──────────────────────────────────────────────────────────────
_GREETINGS = {
    "hi", "hello", "hey", "howdy", "sup", "yo", "hiya", "greetings",
    "good morning", "good afternoon", "good evening",
}


def detect_intent(query: str) -> str:
    q = (query or "").lower().strip().rstrip("!?.,;:")

    # Greeting
    if q in _GREETINGS or (len(q.split()) <= 3 and any(q.startswith(g) for g in _GREETINGS)):
        return "greeting"

    # Trend explanation
    if any(kw in q for kw in ["why is", "why are", "why has", "why did", "reason for",
                               "increasing", "decreasing", "rising", "climbing", "dropping"]):
        return "trend_reason"

    # Historical strain lookup
    if any(kw in q for kw in ["which strain", "what strain", "dominant strain",
                               "dominated in", "dominated during",
                               "h1n1", "h3n2", "b/victoria", "b/yamagata",
                               "strain in 20", "virus in 20", "subtype"]):
        return "strain_query"

    # Season comparison
    if any(kw in q for kw in ["compare", "vs ", "versus", "last season", "prior season",
                               "previous season", "this season vs", "year over year",
                               "compared to", "how does this season", "how does current"]):
        return "season_compare"

    # Age risk
    if any(kw in q for kw in ["age group", "age risk", "most at risk", "vulnerable",
                               "children", "elderly", "seniors", "kids", "pediatric", "age-risk"]):
        return "age_risk"

    # Dashboard / forecast / severity
    if any(kw in q for kw in ["forecast", "predict", "next week", "outlook", "severity",
                               "will it", "how bad", "going to", "what is the trend",
                               "current activity", "current status"]):
        return "dashboard"

    # Jurisdiction lookup
    if any(kw in q for kw in ["which states", "which state", "states in", "list state",
                               "what states", "covered by", "includes which"]):
        return "lookup"

    # City level
    if any(kw in q for kw in ["city", "cities", "municipality", "town"]):
        return "city_lookup"

    # Explanation / general why
    if any(kw in q for kw in ["why", "explain", "how come", "what causes", "what is ili",
                               "what is influenza", "how does", "tell me about"]):
        return "explanation"

    return "rag"  # general open-ended → full RAG pipeline


# ──────────────────────────────────────────────────────────────
# REGION EXTRACTION
# ──────────────────────────────────────────────────────────────
def extract_region(query: str):
    m = re.search(r"region\s*(\d+)", (query or "").lower())
    if m:
        return f"Region {m.group(1)}"
    return None


def extract_year(query: str):
    m = re.search(r"\b(19|20)\d{2}\b", query or "")
    return int(m.group(0)) if m else None


# ──────────────────────────────────────────────────────────────
# RETRIEVAL
# ──────────────────────────────────────────────────────────────
def retrieve_chunks(query: str, k: int = 5) -> list:
    if index is None:
        build_and_load_index()
    if index is None or not documents:
        return []
    q_emb = embedder.encode([query]).astype(np.float32)
    D, I  = index.search(q_emb, k)
    chunks = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0 or idx >= len(documents):
            continue
        chunks.append({
            "text":       documents[idx],
            "confidence": round(float(1 / (1 + score)), 3),
        })
    return chunks


# ──────────────────────────────────────────────────────────────
# DATA EVIDENCE BUILDER
# ──────────────────────────────────────────────────────────────
def _gather_data_evidence(intent: str, region: str, query: str) -> str:
    """
    Pull live CSV numbers for the given intent + region.
    Returns a formatted plain-text block to inject into the LLM prompt.
    """
    evidence_parts = []

    try:
        if intent == "trend_reason":
            data = query_trend_explanation(region) if region else {}
            if not region:
                # Try to pull latest for all regions as context
                data = query_all_regions_latest()
            evidence_parts.append(format_evidence(data))

        elif intent == "strain_query":
            year = extract_year(query)
            data = query_dominant_strain(region or "Region 1", year=year)
            evidence_parts.append(format_evidence(data))

        elif intent == "season_compare":
            data = query_season_comparison(region)
            evidence_parts.append(format_evidence(data))

        elif intent == "age_risk":
            data = query_age_risk()
            evidence_parts.append(format_evidence(data))

        elif intent in ("dashboard", "explanation", "rag"):
            if region:
                evidence_parts.append(format_evidence(query_trend_explanation(region)))
            # Recent history for any region mentioned
            if region:
                hist = query_recent_history(region, n_weeks=6)
                rows = hist.get("weeks", [])
                if rows:
                    lines = [f"week {r['year']}w{r['week']}: ILI={r['ili']}%, pos={r['pos_pct']}%, A={r['flu_a']}, B={r['flu_b']}" for r in rows]
                    evidence_parts.append("Recent weekly data:\n" + "\n".join(lines))

    except Exception as e:
        evidence_parts.append(f"(Data evidence unavailable: {e})")

    return "\n\n".join(evidence_parts) if evidence_parts else ""


# ──────────────────────────────────────────────────────────────
# SNAPSHOT TEXT BUILDER
# ──────────────────────────────────────────────────────────────
def _snapshot_text(snapshot: dict) -> str:
    if not snapshot:
        return ""
    return (
        f"Region: {snapshot.get('region', 'Unknown')}\n"
        f"ILI forecast next week: {snapshot.get('ili_forecast_pct', 'N/A')}\n"
        f"Severity: {snapshot.get('severity', 'Unknown')}\n"
        f"Trend: {snapshot.get('trend', 'Unknown')}\n"
        f"Dominant virus: {snapshot.get('dominant_virus', 'Unknown')}"
    )


# ──────────────────────────────────────────────────────────────
# DETERMINISTIC DASHBOARD CARD  (no LLM, no markdown)
# ──────────────────────────────────────────────────────────────
def _dashboard_card(snapshot: dict) -> str:
    s_region = snapshot.get("region") or "the selected region"
    s_sev    = snapshot.get("severity")
    s_trend  = snapshot.get("trend")
    s_ili    = snapshot.get("ili_forecast_pct")
    s_virus  = snapshot.get("dominant_virus")

    if not any([s_sev, s_trend, s_ili]):
        return "No forecast data available for this region yet."

    sev_headline = {
        "Low":    "Low influenza activity expected next week.",
        "Medium": "Moderate influenza activity expected next week.",
        "High":   "High influenza activity expected next week.",
    }
    headline = sev_headline.get(s_sev, "Influenza forecast updated.")

    trend_text = s_trend or "stable"
    ili_str    = f"{round(float(s_ili), 1)}%" if s_ili is not None else "data unavailable"
    virus_str  = f" Dominant strain: {s_virus}." if s_virus else ""
    detail     = f"Cases are {trend_text} in {s_region} (forecast ILI: {ili_str}).{virus_str}"

    action_map = {
        "Low":    "Continue routine monitoring.",
        "Medium": "Prepare staff and supplies; watch trends closely.",
        "High":   "Activate response plans and inform stakeholders.",
    }
    action = action_map.get(s_sev, "Monitor the region closely.")

    return f"{headline} {detail} {action}"


# ──────────────────────────────────────────────────────────────
# CORE LLM CALL
# ──────────────────────────────────────────────────────────────
def _llm_call(query: str, intent: str, region: str,
              snapshot: dict, history: list,
              rag_context: str, data_evidence: str) -> str:
    if client is None:
        return "Chatbot unavailable: GROQ_API_KEY is not configured."

    snap_block = _snapshot_text(snapshot)
    history_lines = []
    for turn in (history or [])[-6:]:
        if turn.get("user"):
            history_lines.append(f"User: {turn['user']}")
        if turn.get("assistant"):
            history_lines.append(f"Assistant: {turn['assistant']}")

    system_prompt = f"""You are a helpful and knowledgeable Influenza Surveillance Expert. Your goal is to provide relatable, understandable, and accurate information about flu trends, based on the data provided.

=== YOUR PERSONA ===
- **Helpful & Professional**: You talk like a friendly data expert.
- **Relatable**: You explain what the numbers actually mean for someone looking at the dashboard.
- **Data-Driven**: You always prioritize the LIVE SNAPSHOT and DATA EVIDENCE provided.

=== DATA SOURCES (Use these for your answers) ===
1. **LIVE DASHBOARD SNAPSHOT**: (Authoritative truth for right now)
{snap_block if snap_block else "No region selected yet."}

2. **DATA EVIDENCE**: (Recent historical numbers from CDC surveillance data)
{data_evidence if data_evidence else "No additional historical data found for this specific query."}

3. **BACKGROUND KNOWLEDGE**: (General info on flu seasons, strains, and terminology)
{rag_context if rag_context else "No fallback knowledge available."}

=== RULES ===
1. **Accuracy**: Never contradict the Live Snapshot. If the user asks about a region that isn't in the snapshot, tell them you're using the data for {region} but specify that.
2. **Formatting**: Use **bolding** for key terms and bullet points if you have multiple points to share. It makes it much easier to read!
3. **Conversational Tonality**: Start your answer naturally. Don't just spit out numbers. For instance: "Looking at the records for {region or 'the region'}..." or "Based on current trends..."
4. **Empathy**: If activity is high, acknowledge it with professional concern (e.g., "Activity is quite high this week, so it's a good time to be cautious.")
5. **No Hallucination**: If you don't know something, just say "I don't have the specific data for that yet."

Intent Hints: {intent}
Current Region context: {region or "Not specified"}
Recent context: {history_lines[-2:] if history_lines else "None"}
"""

    user_prompt = f"Question: {query}"

    # We increased max_tokens to allow for a more natural response
    resp = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.4, # Slightly higher for more natural variety in speech
        max_tokens=350,
    )
    return resp.choices[0].message.content.strip()


# ──────────────────────────────────────────────────────────────
# PUBLIC API — used by app.py
# ──────────────────────────────────────────────────────────────
def region_aware_response(query: str, snapshot: dict, history=None) -> str:
    """
    Main entry point for the Flask /chat route.
    Everything now flows through the LLM for a natural chat experience.
    """
    snapshot = snapshot or {}
    history  = history or []
    intent   = detect_intent(query)

    # ── Greetings are handled naturally ──────────────────────────
    if intent == "greeting":
        # We let the LLM handle even greetings if we want, or do a quick friendly one
        region = snapshot.get("region")
        hint = f" I'm currently monitoring {region} with you." if region else " Which region would you like to look at today?"
        return f"Hi there! I'm your flu surveillance assistant. {hint} How can I help you understand the data?"

    # ── For all other questions, we use the LLM synthesize approach ──
    region   = extract_region(query) or snapshot.get("region")

    # 1. Gather real data evidence
    data_evidence = _gather_data_evidence(intent, region, query)

    # 2. Semantic retrieval from knowledge base
    chunks    = retrieve_chunks(query, k=4)
    rag_ctx   = "\n\n".join(c["text"] for c in chunks) if chunks else ""

    # 3. LLM synthesis (This provides the 'Relatable' chat experience)
    return _llm_call(query, intent, region, snapshot, history, rag_ctx, data_evidence)


# ──────────────────────────────────────────────────────────────
# LEGACY ALIAS  (kept for backward compatibility)
# ──────────────────────────────────────────────────────────────
def rag_answer(query: str, history=None, snapshot=None) -> dict:
    answer = region_aware_response(query, snapshot or {}, history=history)
    return {
        "answer":     answer,
        "intent":     detect_intent(query),
        "region":     extract_region(query) or (snapshot or {}).get("region"),
        "confidence": 0.9,
    }

