import json
import os
import re
from pathlib import Path

import faiss
import numpy as np
from groq import Groq
from sentence_transformers import SentenceTransformer

# =============================
# PATHS
# =============================
BASE_DIR = Path(__file__).resolve().parent.parent
RAG_DIR = BASE_DIR / "rag_model_data"
INDEX_PATH = RAG_DIR / "influenza.index"
KNOWLEDGE_PATH = RAG_DIR / "influenza_rag_knowledge.json"

# =============================
# LOAD COMPONENTS
# =============================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is not set. Please export/set it before running the app.")

client = Groq(api_key=GROQ_API_KEY)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Load FAISS index and knowledge base with safe fallbacks
try:
    index = faiss.read_index(str(INDEX_PATH))
except Exception:
    index = None

try:
    with open(KNOWLEDGE_PATH, "r", encoding="utf-8") as f:
        knowledge = json.load(f)
except Exception:
    knowledge = []

documents = [d.get("content", "") for d in knowledge]


def build_and_load_index():
    """Builds a FAISS index from the knowledge base if missing."""
    global index
    if not documents:
        return
    emb = embedder.encode(documents, show_progress_bar=False)
    dim = emb.shape[1]
    idx = faiss.IndexFlatL2(dim)
    idx.add(np.array(emb))
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(idx, str(INDEX_PATH))
    index = idx

# =============================
# INTENT DETECTION (FREE)
# =============================
def detect_intent(query):
    q = (query or "").lower()
    if "city" in q or "cities" in q or "municipality" in q:
        return "city_lookup"
    if any(kw in q for kw in ["which states", "which state", "states", "list states"]):
        return "lookup"
    if any(kw in q for kw in ["why", "explain", "how come", "reason"]):
        return "explanation"
    if "compare" in q or "vs " in q or "versus" in q:
        return "comparison"
    if any(kw in q for kw in ["forecast", "predict", "next week", "outlook"]):
        return "forecast"
    return "summary"

# =============================
# REGION EXTRACTION
# =============================
def extract_region(query):
    match = re.search(r"region\s*(\d+)", query.lower())
    if match:
        return f"Region {match.group(1)}"
    return None

# =============================
# RETRIEVAL
# =============================
def retrieve_chunks(query, k=6):
    if index is None:
        build_and_load_index()

    if index is None or not documents:
        return []

    q_emb = embedder.encode([query])
    D, I = index.search(q_emb, k)

    chunks = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0 or idx >= len(documents):
            continue
        confidence = float(1 / (1 + score))  # similarity confidence
        chunks.append({
            "text": documents[idx],
            "confidence": round(confidence, 3)
        })
    return chunks

# =============================
# RAG RESPONSE
# =============================
def rag_answer(query, history=None, snapshot=None):
    """
    Run retrieval and call the LLM with an explicit instruction that the
    provided live `snapshot` is authoritative. The knowledge base can only
    be used for background and explanations that do not contradict the snapshot.
    """
    intent = detect_intent(query)
    region = extract_region(query) or (snapshot.get("region") if snapshot else None)

    chunks = retrieve_chunks(query)

    # Fallback: if retrieval fails, return a safe message
    if not chunks:
        return {
            "answer": "I don't have enough indexed context yet to answer. Please ensure the RAG index is built.",
            "intent": intent,
            "region": region,
            "confidence": 0.0
        }

    history = history or []
    history_lines = []
    for turn in history[-6:]:
        u = turn.get("user")
        a = turn.get("assistant")
        if u:
            history_lines.append(f"User: {u}")
        if a:
            history_lines.append(f"Assistant: {a}")

    context_text = "\n\n".join([c["text"] for c in chunks])

    # Snapshot enforcement text
    snapshot_text = ""
    if snapshot:
        # Only include the most relevant, authoritative fields
        snapshot_text = (
            f"LIVE DASHBOARD SNAPSHOT:\n"
            f"- Region: {snapshot.get('region', 'Unknown')}\n"
            f"- ILI forecast (next week): {snapshot.get('ili_forecast_pct', 'N/A')}\n"
            f"- Severity: {snapshot.get('severity', 'Unknown')}\n"
            f"- Trend: {snapshot.get('trend', 'Unknown')}\n"
            f"- Dominant virus: {snapshot.get('dominant_virus', 'Unknown')}\n"
        )

    system_prompt = f"""
You are an AI Influenza Surveillance Assistant.

Rules (strict):
- Treat the LIVE DASHBOARD SNAPSHOT below as authoritative truth for any local values (region, ILI, severity, trend, dominant virus).
- If static knowledge from the knowledge base conflicts with the live dashboard snapshot, always follow the live dashboard snapshot and state that you used the dashboard.
- Use the retrieved knowledge only for background explanations that do not contradict the snapshot.
- Keep answers concise and public-facing when the question pertains to the dashboard (max 4-6 lines, plain language).
- If the user asks for details not present in the snapshot, clearly state you don't have that local detail and offer general background from the knowledge base.

Snapshot:
{snapshot_text}

Intent: {intent}
Region: {region if region else '(Not specified)'}

Recent chat history:
{chr(10).join(history_lines) if history_lines else '(none)'}
"""

    user_prompt = f"""
Context (retrieved from knowledge base):
{context_text}

Question:
{query}
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3,
        max_tokens=300
    )

    answer = response.choices[0].message.content

    avg_confidence = round(
        sum(c["confidence"] for c in chunks) / len(chunks), 2
    )

    return {
        "answer": answer,
        "intent": intent,
        "region": region,
        "confidence": avg_confidence
    }


def summarize_to_two_points(text):
    """Returns a concise snippet: up to 4 bullets if present, otherwise 3 sentences."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    bullet_lines = [ln for ln in lines if re.match(r"^[\-•\u2022]|^\d+\.\s", ln)]
    if bullet_lines:
        return "\n".join(bullet_lines[:4])

    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    return " ".join(sentences[:3])


def region_aware_response(query, snapshot, history=None):
    """
    Flask-facing entrypoint that enforces use of a live dashboard `snapshot`.

    Behavior:
      - Show a Dashboard Snapshot block (authoritative)
      - For dashboard-style questions, return a short (4-line) public-facing card
        built deterministically from the snapshot (no contradiction allowed).
      - For out-of-scope questions, call the RAG pipeline with the snapshot so
        the LLM can provide background; always prefix the reply with the
        Dashboard Snapshot and clearly state when local details are missing.
    """
    history = history or []

    # Normalize snapshot fields
    region = snapshot.get('region') if snapshot else None
    ili = snapshot.get('ili_forecast_pct') if snapshot else None
    severity = snapshot.get('severity') if snapshot else None
    trend = snapshot.get('trend') if snapshot else None
    dominant = snapshot.get('dominant_virus') if snapshot else None
    age_risks = snapshot.get('age_risks') if snapshot else None
    jurisdictions = snapshot.get('jurisdictions') if snapshot else None

    intent = detect_intent(query)

    # Handlers
    def short_dashboard_summary(snap):
        # Four-line public-facing summary
        s_region = snap.get('region') or 'the region'
        s_sev = snap.get('severity')
        s_trend = snap.get('trend')
        s_ili = snap.get('ili_forecast_pct')

        # Missing-data behaviour
        if not any([s_sev, s_trend, s_ili]):
            return "**Recent data for this region is limited, so trends are uncertain.**\nPlease check back when more weekly data is available.\nContinue monitoring and avoid making region-wide changes based on limited data.\nConfidence is lower due to limited recent data."

        if s_sev == 'Low':
            headline = "**Low sickness expected next week.**"
        elif s_sev == 'Medium':
            headline = "**Some increase in sick visits expected.**"
        elif s_sev == 'High':
            headline = "**Significant increase in sick visits expected.**"
        else:
            headline = "**Forecast update for the requested region.**"

        trend_text = s_trend or 'showing no clear recent trend'
        whats = f"Recent data suggests cases are {trend_text} in {s_region}."
        if snap.get('dominant_virus'):
            whats = whats.rstrip('.') + f" (Dominant: {snap.get('dominant_virus')})."

        if s_sev == 'Low':
            action = "No extra steps — continue routine monitoring."
        elif s_sev == 'Medium':
            action = "Prepare staff and supplies; monitor trends closely."
        elif s_sev == 'High':
            action = "Activate response plans; inform stakeholders and prepare messaging."
        else:
            action = "Monitor the region and request more detailed local data if available."

        confidence = "High confidence (short-term forecast)." if s_ili is not None else "Confidence is lower due to limited recent data."
        return "\n".join([headline, whats, action, confidence])

    def region_lookup_answer(q, snap):
        # list states/jurisdictions for region
        if snap.get('jurisdictions'):
            states_list = ", ".join(snap.get('jurisdictions'))
            return f"Region {snap.get('region')} includes:\n{states_list}."
        return f"Region {snap.get('region')} jurisdiction list is not available."

    def city_fallback_answer(snap):
        if snap.get('jurisdictions'):
            states_list = ", ".join(snap.get('jurisdictions'))
            return f"**City-level data aren't available for {snap.get('region')}.**\nThe dashboard provides regional summaries; city-level figures are not presented.\nI can list the states in this region instead: {states_list}.\nConfidence is lower due to limited city-level data."
        return "**City-level data aren't available for the requested region.**\nThe dashboard provides regional summaries; city-level figures are not included.\nIf you need city-level detail, provide a data source or upload city data to the knowledge base.\nConfidence is lower due to limited city-level data."

    def explanation_answer(q, snap, history=None):
        # Use RAG for explanation, but enforce snapshot authority in prompt
        rag = rag_answer(q, history=history, snapshot=snap)
        ans = rag.get('answer') or "I don't have an explanation available from the knowledge base."
        # Keep short: return the RAG answer summarized
        brief = summarize_to_two_points(ans)
        return brief

    def comparison_answer(q, snap, history=None):
        # Defer to RAG for comparisons; ensure snapshot presence
        rag = rag_answer(q, history=history, snapshot=snap)
        ans = rag.get('answer') or "I don't have a comparison available from the knowledge base."
        return summarize_to_two_points(ans)

    def rag_fallback_answer(q, snap, history=None):
        rag = rag_answer(q, history=history, snapshot=snap)
        ans = rag.get('answer') or "I don't have an answer from the knowledge base."
        return summarize_to_two_points(ans)

    # Route by intent
    if intent == 'summary':
        return short_dashboard_summary(snapshot)
    if intent == 'lookup':
        return region_lookup_answer(query, snapshot)
    if intent == 'city_lookup':
        return city_fallback_answer(snapshot)
    if intent == 'explanation':
        return explanation_answer(query, snapshot, history=history)
    if intent == 'comparison':
        return comparison_answer(query, snapshot, history=history)

    # default fallback uses RAG
    return rag_fallback_answer(query, snapshot, history=history)
