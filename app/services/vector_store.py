import json
from typing import Any

import requests
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.models import DocumentChunk


class JinaEmbeddings:
    def __init__(self, *, api_key: str, base_url: str, model: str, dimensions: int) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.dimensions = dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts=texts, task="retrieval.passage")

    def embed_query(self, text: str) -> list[float]:
        vectors = self._embed(texts=[text], task="retrieval.query")
        return vectors[0] if vectors else [0.0] * self.dimensions

    def _embed(self, *, texts: list[str], task: str) -> list[list[float]]:
        if not texts:
            return []

        response = requests.post(
            self.base_url,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            json={
                "model": self.model,
                "task": task,
                "embedding_type": "float",
                "normalized": True,
                "input": texts,
            },
            timeout=60,
        )
        response.raise_for_status()
        data = response.json().get("data", [])
        vectors = [row.get("embedding", []) for row in data]

        validated: list[list[float]] = []
        for vector in vectors:
            if len(vector) != self.dimensions:
                raise ValueError(
                    f"Jina embedding dimension mismatch: got {len(vector)}, expected {self.dimensions}. "
                    "Adjust EMBEDDING_DIMENSIONS or switch embedding model."
                )
            validated.append(vector)
        return validated


class JinaReranker:
    def __init__(self, *, api_key: str, base_url: str, model: str) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.model = model

    def rerank(self, *, query: str, documents: list[str], top_n: int) -> list[dict[str, Any]]:
        if not documents:
            return []

        response = requests.post(
            self.base_url,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            json={
                "model": self.model,
                "query": query,
                "top_n": top_n,
                "documents": documents,
                "return_documents": False,
            },
            timeout=60,
        )
        response.raise_for_status()
        return response.json().get("results", [])


class VectorStoreService:
    def __init__(self) -> None:
        self.settings = get_settings()
        if not self.settings.jina_api_key:
            raise RuntimeError("JINA_API_KEY is required for document embedding and retrieval.")

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            separators=[
                "\n\n",
                "\n",
                ". ",
                "? ",
                "! ",
                "; ",
                ", ",
                " ",
                "",
            ],
            keep_separator=True,
        )
        self.embeddings = JinaEmbeddings(
            api_key=self.settings.jina_api_key,
            base_url=self.settings.jina_api_base,
            model=self.settings.jina_embedding_model,
            dimensions=self.settings.embedding_dimensions,
        )
        self.retrieval_router = (
            ChatGroq(
                api_key=self.settings.groq_api_key,
                model=self.settings.model_name,
                temperature=0,
            )
            if self.settings.groq_api_key
            else None
        )
        self.reranker = JinaReranker(
            api_key=self.settings.jina_api_key,
            base_url=self.settings.jina_reranker_api_base,
            model=self.settings.jina_reranker_model,
        )

    def _get_embeddings(self) -> Any:
        return self.embeddings

    def _choose_retrieval_sizes(
        self,
        *,
        db: Session,
        query: str,
        file_hashes: list[str],
        requested_k: int,
    ) -> tuple[int, int]:
        available_chunks = db.scalar(
            select(func.count())
            .select_from(DocumentChunk)
            .where(DocumentChunk.file_hash.in_(file_hashes))
        ) or 0
        if available_chunks <= 0:
            return 0, 0

        if self.retrieval_router is None:
            raise RuntimeError("GROQ_API_KEY is required for LLM-based retrieval size selection.")

        prompt = (
            "You are a retrieval planner for a RAG system.\n"
            "Choose how many chunks to keep after reranking and how many vector candidates to send to the reranker.\n"
            "Return only valid JSON with this exact schema:\n"
            '{"final_k": 4, "candidate_k": 12}\n\n'
            "Rules:\n"
            f"- final_k must be between 1 and {min(8, available_chunks)}\n"
            f"- candidate_k must be between final_k and {min(30, available_chunks)}\n"
            "- candidate_k should usually be around 2x to 4x final_k\n"
            "- Use larger values for broad, comparative, or synthesis-heavy queries\n"
            "- Use smaller values for narrow fact lookup queries\n\n"
            f"Query: {query}\n"
            f"Selected documents: {len(file_hashes)}\n"
            f"Available chunks: {available_chunks}\n"
            f"Requested final_k hint: {requested_k}\n"
            f"Configured minimum final_k: {self.settings.retrieval_k}\n"
            f"Configured minimum candidate_k: {self.settings.rerank_candidate_k}\n"
        )

        response = self.retrieval_router.invoke(prompt)
        content = response.content if isinstance(response.content, str) else str(response.content)
        if "```json" in content:
            content = content.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in content:
            content = content.split("```", 1)[1].split("```", 1)[0].strip()
        data = json.loads(content)
        final_k = int(data["final_k"])
        candidate_k = int(data["candidate_k"])

        final_k = max(1, min(final_k, available_chunks, 8))
        candidate_floor = max(final_k, self.settings.rerank_candidate_k)
        candidate_k = max(final_k, candidate_k)
        candidate_k = min(max(candidate_floor, candidate_k), available_chunks, 30)
        return final_k, candidate_k

    def _rerank_matches(self, *, query: str, matches: list[dict[str, Any]], top_n: int) -> list[dict[str, Any]]:
        if self.reranker is None or not matches:
            return matches[:top_n]

        try:
            results = self.reranker.rerank(
                query=query,
                documents=[match["content"] for match in matches],
                top_n=min(top_n, len(matches)),
            )
        except requests.RequestException:
            return matches[:top_n]

        reranked: list[dict[str, Any]] = []
        for item in results:
            index = item.get("index")
            if not isinstance(index, int) or index < 0 or index >= len(matches):
                continue
            match = dict(matches[index])
            score = item.get("relevance_score")
            if isinstance(score, (int, float)):
                match["rerank_score"] = float(score)
            reranked.append(match)

        return reranked or matches[:top_n]

    def add_document(self, *, db: Session, document_id: int, file_hash: str, filename: str, pages: list[tuple[int, str]]) -> None:
        chunk_rows: list[tuple[int | None, str]] = []
        for page_number, page_text in pages:
            if not page_text.strip():
                continue
            page_chunks = self.splitter.split_text(page_text)
            chunk_rows.extend((page_number, chunk) for chunk in page_chunks if chunk.strip())
        chunks = [chunk for _, chunk in chunk_rows]
        if not chunks:
            return
        embeddings_client = self._get_embeddings()
        embeddings = embeddings_client.embed_documents(chunks)
        db.execute(delete(DocumentChunk).where(DocumentChunk.document_id == document_id))
        rows = [
            DocumentChunk(
                document_id=document_id,
                file_hash=file_hash,
                filename=filename,
                chunk_index=index,
                page_number=page_number,
                content=chunk,
                embedding=embedding,
            )
            for index, ((page_number, chunk), embedding) in enumerate(zip(chunk_rows, embeddings, strict=False))
        ]
        db.add_all(rows)
        db.flush()

    def similarity_search(self, *, db: Session, query: str, file_hashes: list[str], k: int = 4) -> list[dict[str, Any]]:
        if not file_hashes:
            return []
        final_k, candidate_k = self._choose_retrieval_sizes(
            db=db,
            query=query,
            file_hashes=file_hashes,
            requested_k=k,
        )
        if final_k == 0:
            return []
        query_embedding = self._get_embeddings().embed_query(query)
        stmt = (
            select(
                DocumentChunk.document_id,
                DocumentChunk.content,
                DocumentChunk.filename,
                DocumentChunk.file_hash,
                DocumentChunk.chunk_index,
                DocumentChunk.page_number,
                DocumentChunk.embedding.cosine_distance(query_embedding).label("distance"),
            )
            .where(DocumentChunk.file_hash.in_(file_hashes))
            .order_by(DocumentChunk.embedding.cosine_distance(query_embedding))
            .limit(candidate_k)
        )
        results = db.execute(stmt).all()
        matches: list[dict[str, Any]] = []
        for row in results:
            matches.append(
                {
                    "content": row.content,
                    "metadata": {
                        "document_id": row.document_id,
                        "filename": row.filename,
                        "file_hash": row.file_hash,
                        "chunk_index": row.chunk_index,
                        "page_number": row.page_number,
                    },
                    "distance": row.distance,
                }
            )
        return self._rerank_matches(query=query, matches=matches, top_n=final_k)
