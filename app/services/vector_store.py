import hashlib
import math
import re
from typing import Any

import requests
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.models import DocumentChunk


class SimpleTextSplitter:
    def __init__(self, *, chunk_size: int, chunk_overlap: int) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> list[str]:
        normalized = text.strip()
        if not normalized:
            return []
        if len(normalized) <= self.chunk_size:
            return [normalized]

        chunks: list[str] = []
        start = 0
        step = max(1, self.chunk_size - self.chunk_overlap)
        text_length = len(normalized)
        while start < text_length:
            end = min(text_length, start + self.chunk_size)
            chunk = normalized[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= text_length:
                break
            start += step
        return chunks


class LocalHashEmbeddings:
    def __init__(self, dimensions: int) -> None:
        self.dimensions = dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_text(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_text(text)

    def _embed_text(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = re.findall(r"\w+", text.lower())
        if not tokens:
            return vector

        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]


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


class VectorStoreService:
    def __init__(self) -> None:
        self.splitter = SimpleTextSplitter(chunk_size=1200, chunk_overlap=200)
        settings = get_settings()
        if settings.jina_api_key:
            self.embeddings = JinaEmbeddings(
                api_key=settings.jina_api_key,
                base_url=settings.jina_api_base,
                model=settings.jina_embedding_model,
                dimensions=settings.embedding_dimensions,
            )
        else:
            # Lightweight fallback when hosted embedding credentials are not configured.
            self.embeddings = LocalHashEmbeddings(settings.embedding_dimensions)

    def _get_embeddings(self) -> Any:
        return self.embeddings

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
            .limit(k)
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
        return matches
