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
