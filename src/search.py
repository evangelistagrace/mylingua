from __future__ import annotations

from typing import List

from sqlalchemy import func, text
from sqlalchemy.orm import Session

from .models import Sense


def is_extension_enabled(session: Session, ext_name: str) -> bool:
    result = session.execute(
        text("SELECT 1 FROM pg_extension WHERE extname = :ext"), {"ext": ext_name}
    ).fetchone()
    return result is not None


def exact_prefix_search(session: Session, query: str, limit: int = 20) -> List[Sense]:
    q = query.strip()
    if not q:
        return []

    exact = (
        session.query(Sense)
        .filter(func.lower(Sense.term_de) == q.lower())
        .order_by(Sense.term_de.asc())
        .limit(limit)
        .all()
    )

    prefix = (
        session.query(Sense)
        .filter(Sense.term_de.ilike(f"{q}%"))
        .order_by(Sense.term_de.asc())
        .limit(limit)
        .all()
    )

    seen = set()
    merged: list[Sense] = []
    for item in exact + prefix:
        if item.id in seen:
            continue
        seen.add(item.id)
        merged.append(item)

    return merged[:limit]


def fuzzy_search(session: Session, query: str, limit: int = 10) -> List[Sense]:
    if not is_extension_enabled(session, "pg_trgm"):
        return []

    return (
        session.query(Sense)
        .filter(Sense.term_de.op("%")(query))
        .order_by(func.similarity(Sense.term_de, query).desc())
        .limit(limit)
        .all()
    )


def semantic_search(
    session: Session, query_embedding: list[float], limit: int = 1
) -> List[tuple[Sense, float]]:
    distance = Sense.embedding.l2_distance(query_embedding)

    rows = (
        session.query(Sense, distance.label("distance"))
        .order_by(distance.asc())
        .limit(limit)
        .all()
    )
    return rows
