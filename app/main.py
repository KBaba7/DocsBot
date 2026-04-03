import re
from typing import Any

from fastapi import Cookie, Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage, ToolMessage
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.exc import OperationalError
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import get_db, init_db
from app.models import Document, User, UserDocument
from app.schemas import AskRequest, AskResponse, UserCreate, UserLogin
from app.security import create_access_token, decode_access_token, hash_password, verify_password
from app.services.agent import build_agent
from app.services.document_service import DocumentService
from app.services.pdf_utils import count_pdf_pages_from_bytes
from app.services.storage_service import StorageService


init_db()

app = FastAPI(title="DocsQA Assignment")
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")
document_service = DocumentService()
storage_service = StorageService()
MAX_UPLOAD_FILES = 5
MAX_PDF_PAGES = 10


def _message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text_value = item.get("text")
                if isinstance(text_value, str):
                    chunks.append(text_value)
            else:
                chunks.append(str(item))
        return "\n".join(chunks)
    return str(content)


def _parse_vector_sources(tool_output: str) -> list[dict[str, str]]:
    lines = tool_output.splitlines()
    sources: list[dict[str, str]] = []
    current_document_id = ""
    current_doc = ""
    current_page = ""

    for line in lines:
        match = re.match(r"^\s*\d+\.\s+document_id=(.*?)\s+\|\s+document=(.*?)\s+\|\s+page=(.*?)\s+\|\s+distance=", line)
        if match:
            current_document_id = match.group(1).strip()
            current_doc = match.group(2).strip()
            current_page = match.group(3).strip()
            continue
        excerpt_match = re.match(r"^\s*excerpt:\s*(.*)$", line)
        if excerpt_match and current_doc:
            excerpt = excerpt_match.group(1).strip()
            sources.append(
                {
                    "document_id": current_document_id,
                    "document": current_doc,
                    "page": current_page or "unknown",
                    "excerpt": excerpt,
                }
            )
            current_document_id = ""
            current_doc = ""
            current_page = ""
    return sources


def _parse_web_sources(tool_output: str) -> list[dict[str, str]]:
    lines = tool_output.splitlines()
    sources: list[dict[str, str]] = []
    current_title = ""
    current_url = ""

    for line in lines:
        title_match = re.match(r"^\s*\d+\.\s+title:\s*(.*)$", line)
        if title_match:
            current_title = title_match.group(1).strip()
            current_url = ""
            continue
        url_match = re.match(r"^\s*url:\s*(.*)$", line)
        if url_match and current_title:
            current_url = url_match.group(1).strip()
            sources.append({"title": current_title, "url": current_url})
            current_title = ""
            current_url = ""
    return sources


def _extract_current_turn_tool_messages(messages: list[Any]) -> list[ToolMessage]:
    turn_tools_reversed: list[ToolMessage] = []
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            break
        if isinstance(message, ToolMessage):
            turn_tools_reversed.append(message)
    return list(reversed(turn_tools_reversed))


def _extract_sources_from_messages(messages: list[Any]) -> dict[str, list[dict[str, str]]]:
    vector_sources: list[dict[str, str]] = []
    web_sources: list[dict[str, str]] = []

    for message in _extract_current_turn_tool_messages(messages):
        tool_name = (message.name or "").strip()
        text = _message_content_to_text(message.content)
        if tool_name == "vector_search":
            vector_sources.extend(_parse_vector_sources(text))
        elif tool_name == "web_search":
            web_sources.extend(_parse_web_sources(text))

    seen_vector: set[tuple[str, str, str]] = set()
    deduped_vector: list[dict[str, str]] = []
    for item in vector_sources:
        key = (item.get("document", ""), item.get("page", ""), item.get("excerpt", ""))
        if key in seen_vector:
            continue
        seen_vector.add(key)
        deduped_vector.append(item)

    seen_web: set[tuple[str, str]] = set()
    deduped_web: list[dict[str, str]] = []
    for item in web_sources:
        key = (item.get("title", ""), item.get("url", ""))
        if key in seen_web:
            continue
        seen_web.add(key)
        deduped_web.append(item)

    return {"vector": deduped_vector, "web": deduped_web}


def _strip_sources_from_answer(answer: str) -> str:
    # Remove any trailing "Sources" section and source-status lines from model text.
    cleaned = re.sub(r"(?is)\n+\s*(?:#+\s*)?sources\s*:?\s*\n.*$", "", answer).strip()
    lines = cleaned.splitlines()
    filtered = []
    for line in lines:
        if re.match(r"^\s*source(s)?\s*:", line, flags=re.IGNORECASE):
            continue
        if re.match(r"^\s*no sources were used for this response\.?\s*$", line, flags=re.IGNORECASE):
            continue
        if re.match(r"^\s*no citations available for this turn\.?\s*$", line, flags=re.IGNORECASE):
            continue
        filtered.append(line)
    return "\n".join(filtered).strip()


def get_current_user(
    access_token: str | None = Cookie(default=None),
    db: Session = Depends(get_db),
) -> User:
    if not access_token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    user_id = decode_access_token(access_token)
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    try:
        user = db.get(User, int(user_id))
    except OperationalError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database is temporarily unavailable. Please try again in a moment.",
        ) from exc
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return user


@app.get("/", response_class=HTMLResponse)
def home(request: Request, access_token: str | None = Cookie(default=None), db: Session = Depends(get_db)):
    user = None
    documents = []
    db_unavailable = False
    if access_token:
        user_id = decode_access_token(access_token)
        if user_id:
            try:
                user = db.get(User, int(user_id))
                if user:
                    documents = document_service.list_user_documents(db, user)
            except OperationalError:
                # Keep homepage responsive during transient DNS/DB outages.
                db_unavailable = True
                user = None
                documents = []
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"request": request, "user": user, "documents": documents, "db_unavailable": db_unavailable},
    )


@app.post("/register")
def register(email: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    payload = UserCreate(email=email, password=password)
    existing = db.scalar(select(User).where(User.email == payload.email))
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    user = User(email=payload.email, password_hash=hash_password(payload.password))
    db.add(user)
    db.commit()
    token = create_access_token(str(user.id))
    response = JSONResponse(
        status_code=status.HTTP_201_CREATED,
        content={"message": "Registered successfully", "email": user.email},
    )
    response.set_cookie("access_token", token, httponly=True, samesite="lax", path="/")
    return response


@app.post("/login")
def login(email: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    payload = UserLogin(email=email, password=password)
    user = db.scalar(select(User).where(User.email == payload.email))
    if not user or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=400, detail="Invalid credentials")
    token = create_access_token(str(user.id))
    response = JSONResponse(content={"message": "Login successful", "email": user.email})
    response.set_cookie("access_token", token, httponly=True, samesite="lax", path="/")
    return response


@app.post("/logout")
def logout(access_token: str | None = Cookie(default=None)):
    response = JSONResponse(content={"message": "Logged out successfully"})
    response.delete_cookie("access_token", path="/")
    return response


@app.post("/upload")
async def upload_document(
    files: list[UploadFile] = File(..., alias="file"),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    if not files:
        raise HTTPException(status_code=400, detail="At least one PDF file is required")
    if len(files) > MAX_UPLOAD_FILES:
        raise HTTPException(status_code=400, detail=f"Upload supports up to {MAX_UPLOAD_FILES} files at a time")

    results = []
    for file in files:
        filename = file.filename or ""
        if not filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"Only PDF files are supported: {filename or '<unnamed file>'}")

        content, file_hash = await document_service.save_upload(file)
        page_count = count_pdf_pages_from_bytes(content)
        if page_count > MAX_PDF_PAGES:
            raise HTTPException(
                status_code=400,
                detail=f"{filename} has {page_count} pages. Maximum allowed is {MAX_PDF_PAGES} pages per file.",
            )

        document, created, processed = document_service.get_or_create_document(
            db=db,
            user=user,
            upload=file,
            content=content,
            file_hash=file_hash,
        )
        results.append(
            {
                "filename": document.filename,
                "created": created,
                "processed": processed,
                "page_count": document.page_count,
            }
        )

    db.commit()
    return {
        "message": "Upload handled successfully",
        "count": len(results),
        "documents": results,
    }


@app.get("/documents")
def list_documents(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    documents = document_service.list_user_documents(db, user)
    return [
        {
            "id": document.id,
            "filename": document.filename,
            "file_hash": document.file_hash,
            "page_count": document.page_count,
            "summary": document.summary,
        }
        for document in documents
    ]


@app.get("/documents/{document_id}/pdf")
def get_document_pdf(document_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    link = db.scalar(
        select(UserDocument.id).where(UserDocument.document_id == document_id, UserDocument.user_id == user.id)
    )
    if not link:
        raise HTTPException(status_code=404, detail="Document not found for this user.")

    document = db.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found.")

    try:
        content = storage_service.read_file_bytes(file_path=document.file_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Unable to load document content.") from exc

    headers = {"Content-Disposition": f'inline; filename="{document.filename}"'}
    return StreamingResponse(iter([content]), media_type="application/pdf", headers=headers)


@app.delete("/documents/{document_id}")
def delete_document(document_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    try:
        result = document_service.delete_user_document(db, user=user, document_id=document_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    db.commit()
    return {
        "message": f"Removed {result['filename']} from your account.",
        "deleted_shared_document": result["deleted_shared_document"],
    }


@app.post("/ask", response_model=AskResponse)
def ask_question(
    payload: AskRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    access_token: str | None = Cookie(default=None),
    x_session_id: str | None = Header(default=None, alias="X-Session-Id"),
):
    document_service.ensure_page_metadata_for_user(db=db, user=user)
    agent = build_agent(db=db, user=user)
    
    # Use session ID from header if provided, otherwise fall back to access token or user ID
    if x_session_id:
        session_key = f"user:{user.id}:session:{x_session_id}"
    else:
        session_key = access_token or f"user:{user.id}"
    
    config = {"configurable": {"thread_id": session_key}}
    print(f"[Agent] thread_id: {session_key}")
    previous_messages: list[Any] = []
    try:
        state = agent.get_state(config)
        values = getattr(state, "values", {}) or {}
        maybe_messages = values.get("messages", [])
        if isinstance(maybe_messages, list):
            previous_messages = maybe_messages
    except Exception:
        # If state read fails, continue safely and parse from result fallback.
        previous_messages = []

    result = agent.invoke({"messages": [("user", payload.query)]}, config=config)
    final_message = result["messages"][-1].content
    answer = final_message if isinstance(final_message, str) else str(final_message)
    answer = _strip_sources_from_answer(answer)
    all_messages = result.get("messages", [])
    if isinstance(all_messages, list) and len(all_messages) >= len(previous_messages):
        current_turn_messages = all_messages[len(previous_messages):]
    else:
        current_turn_messages = all_messages if isinstance(all_messages, list) else []
    sources = _extract_sources_from_messages(current_turn_messages)
    return AskResponse(answer=answer, sources=sources)
