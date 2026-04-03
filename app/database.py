from collections.abc import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import get_settings


settings = get_settings()

engine = create_engine(settings.database_url, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


def init_db() -> None:
    try:
        with engine.begin() as connection:
            connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            connection.execute(text("ALTER TABLE IF EXISTS document_chunks ADD COLUMN IF NOT EXISTS page_number INTEGER"))
        Base.metadata.create_all(bind=engine)
    except OperationalError as exc:
        host = make_url(settings.database_url).host or "<unknown>"
        raise RuntimeError(
            "Database connection failed for host "
            f"'{host}'. If you are using Supabase, use the Transaction Pooler URL "
            "(port 6543) from the Supabase dashboard with sslmode=require."
        ) from exc


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
