from __future__ import annotations

import uuid

from sqlalchemy import Column, DateTime, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import declarative_base
from pgvector.sqlalchemy import Vector

EMBEDDING_DIM = 384  # Cohere embed-multilingual-light-v3.0 dimension

Base = declarative_base()


class Sense(Base):
    __tablename__ = "senses"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    term_de = Column(Text, nullable=False)
    artikel_nominativ = Column(Text)
    definition_de = Column(Text)
    sample_sentences_de = Column(Text)
    translation_en = Column(Text)
    synonyms_en = Column(Text)
    definition_en = Column(Text)
    pos = Column(Text)
    source = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    embedding_en = Column(Vector(EMBEDDING_DIM))
    embedding_de = Column(Vector(EMBEDDING_DIM))

    __table_args__ = (
        UniqueConstraint(
            "term_de",
            "definition_de",
            "translation_en",
            name="uq_senses_term_def_trans",
        ),
    )
