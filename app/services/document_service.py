import hashlib
import json
import re

from fastapi import UploadFile
from langchain_groq import ChatGroq
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.models import Document, DocumentChunk, User, UserDocument
from app.services.pdf_utils import extract_pdf_pages_from_bytes, extract_pdf_text_from_bytes
from app.services.storage_service import StorageService
from app.services.vector_store import VectorStoreService


class DocumentService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.storage = StorageService()
        self.vector_store = VectorStoreService()
        self.summarizer = None
        self.matcher_llm = None

    async def save_upload(self, upload: UploadFile) -> tuple[bytes, str]:
        content = await upload.read()
        file_hash = hashlib.sha256(content).hexdigest()
        return content, file_hash

    def get_or_create_document(self, *, db: Session, user: User, upload: UploadFile, content: bytes, file_hash: str) -> tuple[Document, bool, bool]:
        existing_document = db.scalar(select(Document).where(Document.file_hash == file_hash))
        created = False
        processed = False

        if existing_document is None:
            file_path = self.storage.save_pdf(
                file_hash=file_hash,
                filename=upload.filename or "document.pdf",
                content=content,
            )
            preview_text, page_count = extract_pdf_text_from_bytes(content, max_pages=10)
            full_pages, _ = extract_pdf_pages_from_bytes(content)
            summary = self._summarize_preview(preview_text, upload.filename or "document.pdf")
            existing_document = Document(
                filename=upload.filename or "document.pdf",
                file_hash=file_hash,
                file_path=file_path,
                page_count=page_count,
                summary=summary,
                extracted_preview=preview_text[:8000],
                processing_status="completed",
            )
            db.add(existing_document)
            db.flush()
            self.vector_store.add_document(
                db=db,
                document_id=existing_document.id,
                file_hash=file_hash,
                filename=existing_document.filename,
                pages=full_pages,
            )
            created = True
            processed = True
        else:
            needs_page_reindex = db.scalar(
                select(DocumentChunk.id)
                .where(DocumentChunk.document_id == existing_document.id, DocumentChunk.page_number.is_(None))
                .limit(1)
            )
            if needs_page_reindex:
                content_bytes = self.storage.read_file_bytes(file_path=existing_document.file_path)
                full_pages, _ = extract_pdf_pages_from_bytes(content_bytes)
                self.vector_store.add_document(
                    db=db,
                    document_id=existing_document.id,
                    file_hash=existing_document.file_hash,
                    filename=existing_document.filename,
                    pages=full_pages,
                )
                processed = True

        link = db.scalar(
            select(UserDocument).where(
                UserDocument.user_id == user.id,
                UserDocument.document_id == existing_document.id,
            )
        )
        if link is None:
            db.add(UserDocument(user_id=user.id, document_id=existing_document.id))
            db.flush()

        return existing_document, created, processed

    def list_user_documents(self, db: Session, user: User) -> list[Document]:
        stmt = (
            select(Document)
            .join(UserDocument, UserDocument.document_id == Document.id)
            .where(UserDocument.user_id == user.id)
            .order_by(Document.created_at.desc())
        )
        return list(db.scalars(stmt))

    def delete_user_document(self, db: Session, *, user: User, document_id: int) -> dict[str, str | bool]:
        link = db.scalar(
            select(UserDocument).where(
                UserDocument.user_id == user.id,
                UserDocument.document_id == document_id,
            )
        )
        if link is None:
            raise ValueError("Document not found for this user.")

        document = db.get(Document, document_id)
        if document is None:
            raise ValueError("Document does not exist.")

        db.delete(link)
        db.flush()

        remaining_links = db.scalar(select(func.count()).select_from(UserDocument).where(UserDocument.document_id == document_id)) or 0
        deleted_shared_document = False

        if remaining_links == 0:
            db.delete(document)
            db.flush()
            self.storage.delete_file(file_path=document.file_path)
            deleted_shared_document = True

        return {
            "filename": document.filename,
            "deleted_shared_document": deleted_shared_document,
        }

    def resolve_relevant_document_hashes(self, db: Session, *, user: User, query: str, limit: int = 5) -> list[str]:
        docs = self.list_user_documents(db, user)
        if not docs:
            return []

        # Send all documents to LLM for semantic matching
        matched_hashes = self._llm_filter_documents(query=query, candidates=docs, limit=limit)
        print("Documents Matched ----->", matched_hashes)
        return matched_hashes

    def _llm_filter_documents(self, *, query: str, candidates: list[Document], limit: int) -> list[str]:
        if not self.settings.groq_api_key or not candidates:
            return []
        if self.matcher_llm is None:
            self.matcher_llm = ChatGroq(api_key=self.settings.groq_api_key, model=self.settings.model_name, temperature=0)

        payload = []
        for doc in candidates:
            payload.append(
                {
                    "file_hash": doc.file_hash,
                    "filename": doc.filename,
                    "summary": (doc.summary or "")[:1000],
                    "preview": (doc.extracted_preview or "")[:1200],
                }
            )

        prompt = (
            "You are a document relevance filter. Analyze the user query and select ONLY the truly relevant documents.\n"
            "Consider semantic similarity, topic alignment, and document purpose.\n\n"
            "IMPORTANT: Only include documents that are actually relevant to answering the query.\n"
            "It's better to return fewer relevant documents than to include irrelevant ones.\n"
            f"You may return anywhere from 0 to {limit} documents.\n\n"
            "Return ONLY valid JSON with this exact schema:\n"
            '{"file_hashes": ["<hash1>", "<hash2>", ...]}\n\n'
            f"User query: {query}\n\n"
            f"Available documents:\n{json.dumps(payload, ensure_ascii=True, indent=2)}"
        )
        try:
            response = self.matcher_llm.invoke(prompt)
            content = response.content if isinstance(response.content, str) else str(response.content)
            
            # Handle markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            data = json.loads(content)
            hashes = data.get("file_hashes", [])
            valid = {item.get("file_hash", "") for item in payload}
            return [value for value in hashes if isinstance(value, str) and value in valid][:limit]
        except Exception:
            # Fallback: return first N documents
            return [doc.file_hash for doc in candidates[:limit]]

    def ensure_page_metadata_for_user(self, *, db: Session, user: User) -> None:
        docs = self.list_user_documents(db, user)
        changed = False
        for doc in docs:
            needs_page_reindex = db.scalar(
                select(DocumentChunk.id)
                .where(DocumentChunk.document_id == doc.id, DocumentChunk.page_number.is_(None))
                .limit(1)
            )
            if not needs_page_reindex:
                continue
            try:
                content_bytes = self.storage.read_file_bytes(file_path=doc.file_path)
            except Exception:
                continue
            full_pages, _ = extract_pdf_pages_from_bytes(content_bytes)
            self.vector_store.add_document(
                db=db,
                document_id=doc.id,
                file_hash=doc.file_hash,
                filename=doc.filename,
                pages=full_pages,
            )
            changed = True
        if changed:
            db.commit()

    def _summarize_preview(self, preview_text: str, filename: str) -> str:
        if not preview_text.strip():
            return f"No text could be extracted from the first pages of {filename}."
        if not self.settings.groq_api_key:
            return preview_text[:1200]
        if self.summarizer is None:
            self.summarizer = ChatGroq(api_key=self.settings.groq_api_key, model=self.settings.model_name, temperature=0)
        prompt = (
            "Summarize the following document preview in 6-8 concise bullet-style sentences. "
            "Focus on purpose, key topics, and likely use cases.\n\n"
            f"Filename: {filename}\n\nPreview:\n{preview_text[:16000]}"
        )
        response = self.summarizer.invoke(prompt)
        return response.content if isinstance(response.content, str) else str(response.content)
