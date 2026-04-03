from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "DocsQA Assignment"
    secret_key: str = "change-me"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 720
    database_url: str = "postgresql+psycopg://postgres:postgres@localhost:5432/postgres"
    upload_directory: str = "./uploads"
    storage_backend: str = "local"
    supabase_url: str | None = None
    supabase_service_role_key: str | None = None
    supabase_storage_bucket: str = "documents"
    supabase_storage_prefix: str = "docsqa"
    model_name: str = "llama-3.1-8b-instant"
    embedding_model: str = "mixedbread-ai/mxbai-embed-large-v1"
    embedding_dimensions: int = 1024
    jina_api_key: str | None = None
    jina_api_base: str = "https://api.jina.ai/v1/embeddings"
    jina_embedding_model: str = "jina-embeddings-v3"
    jina_reranker_api_base: str = "https://api.jina.ai/v1/rerank"
    jina_reranker_model: str = "jina-reranker-v3"
    retrieval_k: int = 4
    rerank_candidate_k: int = 12
    groq_api_key: str | None = None
    web_search_provider: str = "tavily"
    tavily_api_key: str | None = None

    @property
    def upload_path(self) -> Path:
        return Path(self.upload_directory)


@lru_cache
def get_settings() -> Settings:
    return Settings()
