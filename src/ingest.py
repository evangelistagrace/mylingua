from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from io import StringIO
from typing import Iterable, List

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .cohere_client import embed_document_cached, embed_documents_batch
from .models import Sense


@dataclass
class IngestRow:
    term_de: str
    definition_de: str | None
    translation_en: str | None
    definition_en: str | None
    pos: str | None
    source: str | None

    def meaning_blob(self) -> str:
        return (
            f"DE: {self.term_de}. Definition: {self.definition_de or ''}. "
            f"EN: translation: {self.translation_en or ''}. Gloss: {self.definition_en or ''}."
        )

    def dedupe_key(self) -> tuple[str, str | None, str | None]:
        return (self.term_de, self.definition_de, self.translation_en)


def parse_csv(file_bytes: bytes) -> List[IngestRow]:
    text = file_bytes.decode("utf-8", errors="replace")
    reader = csv.DictReader(StringIO(text))
    rows: List[IngestRow] = []
    for raw in reader:
        rows.append(
            IngestRow(
                term_de=(raw.get("term_de") or "").strip(),
                definition_de=(raw.get("definition_de") or "").strip() or None,
                translation_en=(raw.get("translation_en") or "").strip() or None,
                definition_en=(raw.get("definition_en") or "").strip() or None,
                pos=(raw.get("pos") or "").strip() or None,
                source=(raw.get("source") or "").strip() or None,
            )
        )
    return [r for r in rows if r.term_de]


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def ingest_rows(
    session: Session,
    rows: Iterable[IngestRow],
    batch_size: int = 96,
) -> tuple[int, int]:
    created = 0
    skipped = 0
    meaning_cache: dict[str, list[float]] = {}

    buffer: list[IngestRow] = []
    for row in rows:
        buffer.append(row)
        if len(buffer) >= batch_size:
            c, s = _flush_batch(session, buffer, meaning_cache)
            created += c
            skipped += s
            buffer = []

    if buffer:
        c, s = _flush_batch(session, buffer, meaning_cache)
        created += c
        skipped += s

    return created, skipped


def _flush_batch(
    session: Session,
    batch: list[IngestRow],
    meaning_cache: dict[str, list[float]],
) -> tuple[int, int]:
    created = 0
    skipped = 0

    to_embed: list[str] = []
    embed_index: list[str] = []

    for row in batch:
        meaning = row.meaning_blob()
        meaning_hash = _hash_text(meaning)
        if meaning_hash not in meaning_cache:
            to_embed.append(meaning)
            embed_index.append(meaning_hash)

    if to_embed:
        embeddings = embed_documents_batch(to_embed, input_type="search_document")
        for meaning_hash, embedding in zip(embed_index, embeddings):
            meaning_cache[meaning_hash] = embedding

    for row in batch:
        meaning = row.meaning_blob()
        meaning_hash = _hash_text(meaning)
        embedding = meaning_cache.get(meaning_hash) or embed_document_cached(meaning)

        sense = Sense(
            term_de=row.term_de,
            definition_de=row.definition_de,
            translation_en=row.translation_en,
            definition_en=row.definition_en,
            pos=row.pos,
            source=row.source,
            embedding=embedding,
        )
        session.add(sense)
        try:
            session.commit()
            created += 1
        except IntegrityError:
            session.rollback()
            skipped += 1

    return created, skipped
