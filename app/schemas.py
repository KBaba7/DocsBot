from pydantic import BaseModel, EmailStr, Field


class UserCreate(BaseModel):
    email: EmailStr
    password: str


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class TokenPayload(BaseModel):
    sub: str


class AskRequest(BaseModel):
    query: str


class AskResponse(BaseModel):
    answer: str
    sources: dict[str, list[dict[str, str]]] = Field(default_factory=lambda: {"vector": [], "web": []})


class DocumentResponse(BaseModel):
    id: int
    filename: str
    file_hash: str
    summary: str
    page_count: int

    class Config:
        from_attributes = True
