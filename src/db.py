from __future__ import annotations

import os
from typing import Iterable

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from .models import Base


def get_database_url() -> str:
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is not set")
    return database_url


_ENGINE = None


def get_engine():
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = create_engine(get_database_url(), pool_pre_ping=True)
    return _ENGINE


def get_session_factory():
    return sessionmaker(autocommit=False, autoflush=False, bind=get_engine())


def ensure_extensions(engine, extensions: Iterable[str]) -> list[str]:
    created = []
    with engine.begin() as connection:
        for ext in extensions:
            try:
                connection.execute(text(f"CREATE EXTENSION IF NOT EXISTS {ext}"))
                created.append(ext)
            except Exception:
                # Extension may not be available; ignore and continue
                continue
    return created


def init_db() -> list[str]:
    engine = get_engine()
    created = ensure_extensions(engine, ["vector", "pg_trgm"])
    Base.metadata.create_all(bind=engine)

    with engine.begin() as connection:
        if "vector" in created:
            _safe_exec(
                connection,
                "ALTER TABLE senses ALTER COLUMN embedding TYPE vector(384)",
            )

        _safe_exec(
            connection,
            "CREATE INDEX IF NOT EXISTS idx_senses_term_de ON senses (term_de)",
        )

        if "pg_trgm" in created:
            _safe_exec(
                connection,
                "CREATE INDEX IF NOT EXISTS idx_senses_term_de_trgm ON senses USING GIN (term_de gin_trgm_ops)",
            )

        if "vector" in created:
            _safe_exec(
                connection,
                "CREATE INDEX IF NOT EXISTS idx_senses_embedding_hnsw ON senses USING hnsw (embedding vector_l2_ops)",
            )

    return created


def _safe_exec(connection, statement: str) -> None:
    try:
        connection.execute(text(statement))
    except Exception:
        pass
