from __future__ import annotations

import csv
import hashlib
import re
from dataclasses import dataclass
from io import BytesIO, StringIO
from typing import Iterable, List

from openpyxl import load_workbook
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
    text = _decode_csv_text(file_bytes)
    sample = text[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel

    reader = csv.DictReader(StringIO(text), dialect=dialect)
    return _rows_from_dict_reader(reader)


def _decode_csv_text(file_bytes: bytes) -> str:
    # Try common spreadsheet/export encodings before lossy replacement.
    encodings = ["utf-8-sig", "utf-16", "cp1252", "latin-1"]
    for encoding in encodings:
        try:
            return file_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue
    return file_bytes.decode("utf-8", errors="replace")


def parse_xlsx(file_bytes: bytes) -> List[IngestRow]:
    workbook = load_workbook(filename=BytesIO(file_bytes), read_only=True, data_only=True)
    sheet = workbook.active
    values = sheet.values

    try:
        headers_raw = next(values)
    except StopIteration:
        return []

    if not headers_raw:
        return []

    headers = [_canonical_header(h) for h in headers_raw]
    rows: List[IngestRow] = []
    for record in values:
        raw = {
            headers[idx]: _cell_to_str(record[idx]) if idx < len(record) else ""
            for idx in range(len(headers))
        }
        rows.append(_row_from_raw(raw))
    return [r for r in rows if r.term_de]


def parse_tabular_file(filename: str, file_bytes: bytes) -> List[IngestRow]:
    lower = filename.lower()
    if lower.endswith(".xlsx"):
        return parse_xlsx(file_bytes)
    return parse_csv(file_bytes)


def _canonical_header(value: object) -> str:
    text = str(value or "").strip().lower().replace("\ufeff", "")
    text = re.sub(r"\s+", "_", text)
    return text


def _cell_to_str(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _rows_from_dict_reader(reader: csv.DictReader) -> List[IngestRow]:
    rows: List[IngestRow] = []
    for source_raw in reader:
        raw = {_canonical_header(k): (v or "") for k, v in source_raw.items() if k is not None}
        rows.append(_row_from_raw(raw))
    return [r for r in rows if r.term_de]


def _row_from_raw(raw: dict[str, str]) -> IngestRow:
    def val(key: str) -> str:
        return (raw.get(key) or "").strip()

    return IngestRow(
        term_de=val("term_de"),
        artikel_nominativ=val("artikel_nominativ") or None,
        definition_de=val("definition_de") or None,
        sample_sentences_de=val("sample_sentences_de") or None,
        translation_en=val("translation_en") or None,
        definition_en=val("definition_en") or None,
        pos=val("pos") or None,
        source=val("source") or None,
    )


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
