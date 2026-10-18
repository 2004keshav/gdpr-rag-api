# GDPR RAG API

A "chat with your documents" RAG system over GDPR (Regulation EU 2016/679), with a
Phase 4-validated evaluation harness confirming `structure` chunking as the best-performing
retrieval strategy (recall@5 = 1.000, faithfulness = 0.997, zero hallucinated citations).

## Endpoints
- `GET /health` - liveness check
- `POST /ask` - ask a GDPR question, get a grounded answer with article citations

## Setup (local)
```bash
cd api
pip install -r requirements.txt
export GEMINI_API_KEY=your_key_here
export EMBEDDING_BACKEND=fastembed
uvicorn main:app --reload
```

## Deployment
Deployed on Render free tier. Required environment variables:
- `GEMINI_API_KEY` (secret, set in Render dashboard)
- `EMBEDDING_BACKEND=fastembed` (required - keeps RAM usage under the 512MB free-tier limit)

## Live URLs
- **API:** https://gdpr-rag-api.onrender.com (`/health`, `/ask`)
- **Frontend:** https://gdpr-rag-api-zcxzo8b4nv3q7zxtgzpzzi.streamlit.app

## Frontend
A Streamlit chat UI lives in `frontend/` and calls the live API above. It includes:
- A dropdown of the 27 golden evaluation questions (from Phase 4) for quick demoing
- Grounded answers with `[Article X]` citations
- A "served from cache" badge when a response comes from the Phase 6 cache
- Handling for three distinct states: a slow/cold-starting backend, a genuinely down backend, and a valid abstention answer

Deployed on Streamlit Community Cloud (free tier) from this same repo, main file path `frontend/app.py`.

Run locally:
```bash
cd frontend
pip install -r requirements.txt
streamlit run app.py
```

## CORS
`api/main.py` restricts cross-origin requests to the deployed frontend's origin only
(`https://gdpr-rag-api-zcxzo8b4nv3q7zxtgzpzzi.streamlit.app`). If you deploy your own
copy of the frontend at a different URL, update `allow_origins` in `api/main.py` to match.
