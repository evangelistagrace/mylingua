from __future__ import annotations

import csv
import hashlib
import re
from dataclasses import dataclass
from io import StringIO
from typing import Iterable, List

from sqlalchemy import func
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .cohere_client import embed_document_cached, embed_documents_batch
from .models import Sense


def _strip_reference_numbers(value: str | None) -> str | None:
    if not value:
        return None
    cleaned = re.sub(r"\[\d+\]", "", value)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned or None


def _normalize_artikel_token(value: str | None) -> str | None:
    if not value:
        return None
    first_token = value.strip().split(" ", 1)[0].lower()
    if first_token in {"der", "die", "das"}:
        return first_token
    return None


@dataclass
class IngestRow:
    term_de: str
    artikel_nominativ: str | None
    definition_de: str | None
    sample_sentences_de: str | None
    translation_en: str | None
    definition_en: str | None
    pos: str | None
    source: str | None

    def meaning_blob_en(self) -> str:
        return (
            f"EN translation: {self.translation_en or ''}. "
            f"EN definition: {self.definition_en or ''}."
        )

    def meaning_blob_de(self) -> str:
        return (
            f"DE term: {self.term_de}. "
            f"DE definition: {self.definition_de or ''}."
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
                artikel_nominativ=(raw.get("artikel_nominativ") or "").strip() or None,
                definition_de=(raw.get("definition_de") or "").strip() or None,
                sample_sentences_de=(raw.get("sample_sentences_de") or "").strip() or None,
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
    updated = 0
    meaning_cache: dict[str, list[float]] = {}

    buffer: list[IngestRow] = []
    for row in rows:
        buffer.append(row)
        if len(buffer) >= batch_size:
            c, u = _flush_batch(session, buffer, meaning_cache)
            created += c
            updated += u
            buffer = []

    if buffer:
        c, u = _flush_batch(session, buffer, meaning_cache)
        created += c
        updated += u

    return created, updated


def _flush_batch(
    session: Session,
    batch: list[IngestRow],
    meaning_cache: dict[str, list[float]],
) -> tuple[int, int]:
    created = 0
    updated = 0

    to_embed_en: list[str] = []
    embed_index_en: list[str] = []
    to_embed_de: list[str] = []
    embed_index_de: list[str] = []

    for row in batch:
        en_text = row.meaning_blob_en()
        en_hash = _hash_text(f"en::{en_text}")
        if en_hash not in meaning_cache:
            to_embed_en.append(en_text)
            embed_index_en.append(en_hash)

        de_text = row.meaning_blob_de()
        de_hash = _hash_text(f"de::{de_text}")
        if de_hash not in meaning_cache:
            to_embed_de.append(de_text)
            embed_index_de.append(de_hash)

    if to_embed_en:
        embeddings = embed_documents_batch(to_embed_en, input_type="search_document")
        for meaning_hash, embedding in zip(embed_index_en, embeddings):
            meaning_cache[meaning_hash] = embedding

    if to_embed_de:
        embeddings = embed_documents_batch(to_embed_de, input_type="search_document")
        for meaning_hash, embedding in zip(embed_index_de, embeddings):
            meaning_cache[meaning_hash] = embedding

    for row in batch:
        en_text = row.meaning_blob_en()
        en_hash = _hash_text(f"en::{en_text}")
        embedding_en = meaning_cache.get(en_hash) or embed_document_cached(en_text)

        de_text = row.meaning_blob_de()
        de_hash = _hash_text(f"de::{de_text}")
        embedding_de = meaning_cache.get(de_hash) or embed_document_cached(de_text)
        artikel = _normalize_artikel_token(row.artikel_nominativ)
        definition_de = _strip_reference_numbers(row.definition_de)
        sample_sentences_de = _strip_reference_numbers(row.sample_sentences_de)

        existing = (
            session.query(Sense)
            .filter(func.lower(Sense.term_de) == row.term_de.lower())
            .order_by(Sense.created_at.desc())
            .first()
        )

        try:
            if existing:
                existing.term_de = row.term_de
                existing.artikel_nominativ = artikel
                existing.definition_de = definition_de
                existing.sample_sentences_de = sample_sentences_de
                existing.translation_en = row.translation_en
                existing.definition_en = row.definition_en
                existing.pos = row.pos
                existing.source = row.source
                existing.embedding_en = embedding_en
                existing.embedding_de = embedding_de
                updated += 1
            else:
                sense = Sense(
                    term_de=row.term_de,
                    artikel_nominativ=artikel,
                    definition_de=definition_de,
                    sample_sentences_de=sample_sentences_de,
                    translation_en=row.translation_en,
                    definition_en=row.definition_en,
                    pos=row.pos,
                    source=row.source,
                    embedding_en=embedding_en,
                    embedding_de=embedding_de,
                )
                session.add(sense)
                created += 1
            session.commit()
        except IntegrityError:
            session.rollback()
            continue

    return created, updated
