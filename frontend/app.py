"""
app.py — Streamlit frontend for the GDPR RAG system.

Calls the live Render-hosted FastAPI backend (/ask, /health) and renders
grounded answers with [Article X] citations. No secrets live here — the
backend URL is public and GEMINI_API_KEY / Upstash credentials stay
server-side on Render (never touched by this file).
"""

import os
import time

import requests
import streamlit as st

# ============================================================
# Config
# ============================================================

# Public API URL — not a secret, so no st.secrets needed. Overridable via
# env var for local testing against a different backend (e.g. ngrok tunnel
# to a Colab-hosted dev server) without touching code.
API_BASE_URL = os.environ.get("GDPR_RAG_API_URL", "https://gdpr-rag-api.onrender.com")

# Render free-tier cold start is officially documented as ~1 minute as of
# this writing (verified fresh for Phase 8; was previously assumed 30-50s
# based on older info — flagging in case this drifts again).
COLD_START_THRESHOLD_SECONDS = 50
REQUEST_TIMEOUT_SECONDS = 90  # hard ceiling — beyond this, treat as genuinely down

ABSTENTION_MARKERS = [
    "don't have enough information",
    "do not have enough information",
    "cannot answer",
    "not covered in the provided context",
]


# ============================================================
# Backend calls
# ============================================================

def call_ask_api(question: str, top_k: int = 5) -> dict:
    """
    Calls POST /ask on the backend. Returns a dict with either:
      {"ok": True, "data": <response json>, "elapsed": float}
      {"ok": False, "error_type": "cold_start" | "down" | "http_error", "detail": str, "elapsed": float}
    """
    start = time.time()
    try:
        resp = requests.post(
            f"{API_BASE_URL}/ask",
            json={"query": question, "top_k": top_k},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        elapsed = time.time() - start

        if resp.status_code == 200:
            return {"ok": True, "data": resp.json(), "elapsed": elapsed}
        else:
            return {
                "ok": False,
                "error_type": "http_error",
                "detail": f"API returned status {resp.status_code}: {resp.text[:300]}",
                "elapsed": elapsed,
            }

    except requests.exceptions.Timeout:
        elapsed = time.time() - start
        # We only reach here if even REQUEST_TIMEOUT_SECONDS (90s) wasn't enough —
        # genuinely treat this as down, not just cold-starting.
        return {
            "ok": False,
            "error_type": "down",
            "detail": "Request timed out after 90s — backend is likely down, not just cold-starting.",
            "elapsed": elapsed,
        }
    except requests.exceptions.ConnectionError as e:
        elapsed = time.time() - start
        return {
            "ok": False,
            "error_type": "down",
            "detail": f"Could not connect to the API: {e}",
            "elapsed": elapsed,
        }


def check_health() -> dict | None:
    """Best-effort health check, used only for the sidebar status indicator."""
    try:
        resp = requests.get(f"{API_BASE_URL}/health", timeout=10)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


def is_abstention(answer: str) -> bool:
    answer_lower = answer.lower()
    return any(marker in answer_lower for marker in ABSTENTION_MARKERS)


# ============================================================
# UI
# ============================================================

st.set_page_config(page_title="GDPR Q&A", page_icon="⚖️", layout="centered")

st.title("⚖️ GDPR Q&A")
st.caption("Ask a question about the GDPR (Regulation EU 2016/679). Answers are grounded in the regulation text with article citations.")

# --- Sidebar: live backend status (nice-to-have, not critical path) ---
with st.sidebar:
    st.subheader("Backend status")
    health = check_health()
    if health:
        st.success("API is live")
        st.caption(f"Strategy: {health.get('strategy', 'n/a')}")
        st.caption(f"Cache enabled: {health.get('cache_enabled', 'n/a')}")
    else:
        st.warning("API status unknown (may be cold-starting)")

# --- Golden question dropdown (demo convenience) ---
golden_questions_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden_questions.json")
golden_questions = []
if os.path.exists(golden_questions_path):
    import json
    with open(golden_questions_path, "r") as f:
        golden_questions = json.load(f)

selected_question = ""
if golden_questions:
    options = ["— Type my own question —"] + [q["question"] for q in golden_questions]
    choice = st.selectbox("Quick-select a sample question:", options)
    if choice != "— Type my own question —":
        selected_question = choice

# --- Question input ---
question = st.text_area(
    "Your question:",
    value=selected_question,
    height=100,
    placeholder="e.g. What are the core principles relating to processing of personal data?",
)

submit = st.button("Ask", type="primary")

# ============================================================
# Handle submission
# ============================================================

if submit:
    if not question or len(question.strip()) < 3:
        st.error("Please enter a question (at least 3 characters).")
    else:
        placeholder = st.empty()
        placeholder.info("⏳ Sending request... if the backend has been idle, this may take up to a minute to wake up.")

        result = call_ask_api(question.strip())
        placeholder.empty()

        if result["ok"]:
            data = result["data"]
            answer = data["answer"]
            cited_articles = data.get("cited_articles", [])
            cache_hit = data.get("cache_hit", False)
            latency = data.get("latency_seconds", result["elapsed"])

            # --- Cache badge (confirmed: show it) ---
            if cache_hit:
                st.success(f"⚡ Served from cache — {latency:.2f}s")
            else:
                st.caption(f"Fresh answer generated in {latency:.1f}s")

            # --- Abstention vs normal answer ---
            if is_abstention(answer):
                st.warning("ℹ️ The system could not find enough information in the GDPR text to answer this confidently.")
                st.write(answer)
            else:
                st.markdown("### Answer")
                st.write(answer)

                if cited_articles:
                    st.markdown("**Cited articles:** " + ", ".join(f"Article {a}" for a in cited_articles))

        else:
            # Three distinct, deliberate error states, branched on error_type
            # (which call_ask_api() already classified correctly) — not on
            # elapsed time, which was a self-review bug fix (elapsed-based
            # branching contradicted the function's own classification).
            if result["error_type"] == "down" and result["elapsed"] < COLD_START_THRESHOLD_SECONDS:
                st.error(
                    "❌ The API appears to be down (connection failed quickly, not a timeout). "
                    "This is likely a real outage, not a cold start."
                )
            elif result["error_type"] == "down":
                # Reached only via a 90s timeout with no response at all
                st.error(
                    "⏳ No response after 90 seconds. This usually means the backend is down, "
                    "though an unusually slow cold start is also possible. Please try again shortly."
                )
            else:
                st.error(f"❌ Something went wrong: {result['detail']}")

st.divider()
st.caption("Backend hosted on Render (free tier) · Frontend hosted on Streamlit Community Cloud")
