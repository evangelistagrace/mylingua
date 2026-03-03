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


def exact_prefix_search_en(session: Session, query: str, limit: int = 20) -> List[Sense]:
    q = query.strip()
    if not q:
        return []

    exact = (
        session.query(Sense)
        .filter(Sense.translation_en.isnot(None))
        .filter(func.lower(Sense.translation_en) == q.lower())
        .order_by(Sense.translation_en.asc())
        .limit(limit)
        .all()
    )

    prefix = (
        session.query(Sense)
        .filter(Sense.translation_en.isnot(None))
        .filter(Sense.translation_en.ilike(f"{q}%"))
        .order_by(Sense.translation_en.asc())
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


def semantic_search_dual_english_first(
    session: Session,
    query_embedding: list[float],
    limit: int = 10,
    max_distance: float = 0.95,
) -> List[tuple[Sense, float]]:
    distance_en = Sense.embedding_en.l2_distance(query_embedding)
    distance_de = Sense.embedding_de.l2_distance(query_embedding)

    rows_en = (
        session.query(Sense, distance_en.label("distance"))
        .filter(Sense.embedding_en.isnot(None))
        .order_by(distance_en.asc())
        .limit(limit)
        .all()
    )
    rows_de = (
        session.query(Sense, distance_de.label("distance"))
        .filter(Sense.embedding_de.isnot(None))
        .order_by(distance_de.asc())
        .limit(limit)
        .all()
    )

    merged: list[tuple[Sense, float]] = []
    seen: set = set()

    for sense, distance in rows_en:
        if distance > max_distance or sense.id in seen:
            continue
        seen.add(sense.id)
        merged.append((sense, distance))

    for sense, distance in rows_de:
        if distance > max_distance or sense.id in seen:
            continue
        seen.add(sense.id)
        merged.append((sense, distance))

    return merged[:limit]
