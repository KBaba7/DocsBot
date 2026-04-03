# DocsQA Smart Research Assistant

This is my take-home submission for the ABSTRABIT AI/ML Engineer assignment: a RAG-powered assistant where users upload PDFs, ask questions, and get grounded answers with citations.

## Live Project

- Live app (Railway): `https://docsbot-web-production.up.railway.app`
- GitHub: `https://github.com/KBaba7/DocsBot`
- Loom walkthrough: _add your link here_

## What I Built

The app supports authentication, PDF upload (up to 5 files and 10 pages per file), document chunking + vector indexing, and a chat experience that answers from uploaded documents first.  
If the uploaded documents are not enough, the agent falls back to web search and cites those sources too.

## Stack

- FastAPI + SQLAlchemy
- LangGraph agent
- Groq chat model
- Supabase Postgres + `pgvector`
- Railway deployment

## How Retrieval Works

Uploaded PDFs are parsed page by page and split into chunks.  
Each chunk is stored with metadata (document, page number, chunk index) and embedded into `pgvector`.

At question time:
1. The app searches relevant chunks from the user’s accessible documents.
2. The agent answers from those chunks when possible.
3. If evidence is weak, the agent uses web search and cites external URLs.

## Chunking Strategy

- Chunk size: `1200`
- Overlap: `200`

Why this setup:
- Long, structured documents need enough contiguous context.
- Overlap helps avoid missing content around chunk boundaries.
- It gives a practical quality/cost balance for retrieval.

## Retrieval Approach

I use cosine similarity search in `pgvector` (no reranker yet).  
The top matches are turned into readable citations (document name + page + snippet), and those are shown per answer in the UI.

## Agent Routing Logic

The agent is prompted to prefer document context first.

- If retrieved document context is sufficient: answer from documents with citations.
- If not sufficient: clearly say docs are insufficient and use web search tool.

This is implemented as tool-based behavior in LangGraph rather than a static fallback message.

## Source Citations

Each turn stores/returns source metadata separately from the answer body.

- Vector source cards include:
  - document name
  - page number
  - excerpt (short snippet from retrieved chunk)
- Web source cards include:
  - title
  - URL

## Conversation Memory

Conversation history is maintained within session scope, so follow-ups like “tell me more about that” work as expected.

## Bonus Feature

I added hash-based deduplicated ingestion:

- If the same PDF is uploaded again, processing/indexing is reused.
- Access control is still user-scoped via ownership mapping.

Why I chose this:
- saves compute/time,
- avoids duplicate indexing,
- keeps retrieval secure per user.

## Challenges I Ran Into

1. Heavy embedding dependencies made deployment images too large.
   - I switched to lightweight embeddings for deployment and added Jina API embedding support.
2. Source rendering got messy across multiple chat turns.
   - I separated answer text from source payloads and extracted sources per turn.
3. Intermittent DB DNS/pooler issues during deployment.
   - I improved connection handling and standardized Supabase transaction-pooler config.

## If I Had More Time

- Add reranking (cross-encoder) for better precision on long multi-doc queries.
- Add automated citation-faithfulness checks.
- Add Alembic migrations for cleaner schema evolution.
- Add stronger eval/observability for routing and retrieval quality.

## Local Setup

```bash
cp .env.example .env
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
uvicorn app.main:app --reload
```

Open: `http://127.0.0.1:8000`

## Important Environment Variables

Required:
- `GROQ_API_KEY`
- `SECRET_KEY`
- `DATABASE_URL`

Embeddings (recommended):
- `JINA_API_KEY`
- `JINA_API_BASE` (default: `https://api.jina.ai/v1/embeddings`)
- `JINA_EMBEDDING_MODEL` (default: `jina-embeddings-v3`)
- `EMBEDDING_DIMENSIONS` (default: `1024`)

Storage:
- `STORAGE_BACKEND=local|supabase`
- `SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE_KEY`
- `SUPABASE_STORAGE_BUCKET`
- `SUPABASE_STORAGE_PREFIX`

Web search:
- `WEB_SEARCH_PROVIDER=duckduckgo|tavily`
- `TAVILY_API_KEY` (if using Tavily)

## API Endpoints

- `POST /register`
- `POST /login`
- `POST /logout`
- `POST /upload`
- `GET /documents`
- `DELETE /documents/{document_id}`
- `GET /documents/{document_id}/pdf`
- `POST /ask`

## Sample Documents

As requested in the assignment, sample PDFs are included in `test_documents/`.

## Railway Deployment

```bash
railway login
railway link
railway up
```

Set the same env vars in Railway service settings before deploying.
