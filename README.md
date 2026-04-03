# DocsQA LangGraph Assignment

RAG-powered research assistant with:

- Auth (register/login/logout) using HTTP-only cookie sessions
- Multi-file PDF upload (up to 5 files/request, max 10 pages/file)
- Duplicate detection by SHA-256 hash with cross-user document reuse
- Vector indexing in Supabase Postgres + `pgvector`
- LangGraph agent with document retrieval + web search fallback
- Session conversation memory for follow-up questions
- Source citations in answers for both document and web evidence
- Chat-style UI with markdown rendering

## Architecture

- Backend: FastAPI + SQLAlchemy
- Agent: LangGraph ReAct agent
- LLM: Groq chat model
- Vector store: Supabase Postgres with `pgvector`
- Search fallback: Tavily (preferred) or DuckDuckGo when available

## Chunking Strategy

- Splitter: recursive character splitter (`chunk_size=1200`, `chunk_overlap=200`)
- Why:
  - 1200 keeps enough local context for legal/business clauses
  - 200 overlap reduces boundary loss between adjacent chunks
  - good balance for retrieval accuracy vs. embedding cost
- Indexing is page-aware: each stored chunk carries `page_number` metadata.

## Retrieval Approach

- Retrieval method: cosine similarity search in `pgvector`
- Pipeline:
  - determine relevant user-owned document hashes
  - embed query
  - retrieve top-k chunks across selected docs
- Returned evidence includes:
  - document filename
  - page number
  - excerpt text
- Final assistant answer is instructed to cite these in a human-friendly source section.

## Agent Routing Logic

- Default behavior: prefer `vector_search` for questions answerable from uploaded docs.
- If document evidence is insufficient, agent can call `web_search` tool.
- Web search output is normalized to citation-friendly rows (title, URL, snippet).
- Prompt requires:
  - vector citations: document + page + excerpt
  - web citations: website title + URL

## Bonus Feature

**Implemented bonus:** User-scoped retrieval with automatic document dedup reuse.

- If two users upload the same file, processing/indexing is reused by file hash.
- Ownership is still enforced via `user_documents` mapping, so retrieval stays user-scoped.
- Why chosen: materially improves performance/cost while preserving access boundaries.

## Problems Faced and Fixes

- Dependency mismatch (`transformers`/`sentence-transformers`/`torch`) causing startup errors.
  - Added robust local fallback embedding path to keep app functional.
- Optional web-search dependency (`ddgs`) missing.
  - Added graceful web tool fallback and Tavily direct tool support.
- Passlib bcrypt backend issues.
  - Switched new password hashing to `pbkdf2_sha256` while retaining bcrypt verify compatibility.
- Template/render and response UX issues.
  - Reworked frontend into a stable chat-style UI with clean result handling.

## If I Had More Time

- Add proper migration tooling (Alembic) instead of startup `ALTER TABLE`.
- Add reranking for higher retrieval precision on long multi-document queries.
- Add persistent server-side conversation storage (Redis/Postgres) for multi-worker deployments.
- Add automated evaluation suite for citation faithfulness and retrieval quality.

## Environment Setup

```bash
cp .env.example .env
```

Required:

- `GROQ_API_KEY`
- `SECRET_KEY`
- `DATABASE_URL` (Supabase transaction pooler recommended)

Optional:

- `TAVILY_API_KEY` (for Tavily web search)

Storage (optional, recommended for deployment):

- `STORAGE_BACKEND=local` or `supabase`
- `SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE_KEY`
- `SUPABASE_STORAGE_BUCKET` (default: `documents`)
- `SUPABASE_STORAGE_PREFIX` (default: `docsqa`)

Recommended `DATABASE_URL` format:

`postgresql+psycopg://<user>:<password>@<pooler-host>:6543/postgres?sslmode=require`

## Install and Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
uvicorn app.main:app --reload
```

Open: `http://127.0.0.1:8000`

## File Storage Mode

- Local dev default: `STORAGE_BACKEND=local` (writes under `UPLOAD_DIRECTORY`).
- Deployment recommendation: `STORAGE_BACKEND=supabase` to store PDFs in Supabase Storage instead of local disk.

## API Endpoints

- `POST /register`
- `POST /login`
- `POST /logout`
- `POST /upload`
- `GET /documents`
- `POST /ask`

## test_documents

Sample PDFs used during development are in `test_documents/`.

## Deployment and Loom

- Live deployed URL: _add your deployed link here_
- Loom walkthrough (<5 min): _add your Loom link here_

## Deploy on Render

This repo now includes a `render.yaml` Blueprint.

1. Push the latest `main` branch to GitHub.
2. In Render, click **New +** -> **Blueprint**.
3. Connect GitHub and select this repository.
4. Render will detect `render.yaml` and create a `docsbot` web service.
5. Set required secret env vars in Render:
   - `SECRET_KEY`
   - `DATABASE_URL`
   - `GROQ_API_KEY`
   - `SUPABASE_URL`
   - `SUPABASE_SERVICE_ROLE_KEY`
   - optionally `TAVILY_API_KEY`
6. Deploy and open the generated Render URL.

Render uses:
- Build command: `pip install -e .`
- Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`

## Deploy on Fly.io

This repo includes `Dockerfile` and `fly.toml`.

1. Install Fly CLI:
   - macOS: `brew install flyctl`
2. Login:
   - `fly auth login`
3. If app name `docsbot-kbaba7` is unavailable, change `app` in `fly.toml`.
4. Create app (first time only):
   - `fly apps create docsbot-kbaba7`
5. Set secrets:
   - `fly secrets set SECRET_KEY=...`
   - `fly secrets set DATABASE_URL=...`
   - `fly secrets set GROQ_API_KEY=...`
   - `fly secrets set SUPABASE_URL=...`
   - `fly secrets set SUPABASE_SERVICE_ROLE_KEY=...`
   - optional: `fly secrets set TAVILY_API_KEY=...`
6. Deploy:
   - `fly deploy`
7. Open app:
   - `fly open`
