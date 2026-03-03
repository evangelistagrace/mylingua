# MyLingua

A language-learning vocabulary app for English speakers learning German. It uses Streamlit for the UI, PostgreSQL with pgvector for storage/vector search, and Cohere embeddings.

![alt text](image.png)

## Features

- Search flow optimized for learners:
  - Single-word queries: lexical first (`translation_en`, then `term_de`)
  - Phrase queries / fallback: semantic search
  - Single-word no-match path: quick action to jump to Add Word with prefilled context
- Manual Add Word form:
  - Required: `term_de`
  - Optional: `artikel_nominativ` (`der|die|das`), `definition_de`, `translation_en`, `sample_sentences_de`, `pos`
  - Missing fields auto-filled from external sources
- CSV ingest with the same missing-field autofill behavior
- Upsert by `term_de`:
  - existing entries are updated with newest values instead of skipped
- Dual-vector semantic retrieval:
  - `embedding_en` (English-facing fields)
  - `embedding_de` (German-facing fields)
  - English-first semantic ranking with distance threshold filtering
- Result cards include:
  - POS badge
  - noun gender badge (`masculine`/`feminine`/`neuter`) derived from article

## Project structure

- app.py — Streamlit entry
- src/db.py — database engine, extensions, indexes
- src/models.py — SQLAlchemy model
- src/cohere_client.py — Cohere embeddings wrapper
- src/autofill.py — Wiktionary/Wikidata/Tatoeba enrichment
- src/ingest.py — CSV + manual-entry ingest/upsert logic
- src/search.py — lexical + dual-vector semantic search
- docker-compose.yml — Postgres + pgvector
- .env.example — environment variables template
- data/sample_vocab.csv — sample data

## Setup

1. Copy .env.example to .env and set COHERE_API_KEY.
2. Start Postgres with Docker Compose. `docker-compose up -d`
3. The app will auto-create required extensions (pgvector, optional pg_trgm) and indexes on startup.
4. Create a virtual environment. `python -m venv venv && source venv/bin/activate` (Unix) or `python -m venv venv && venv\Scripts\activate` (Windows)
5. Install dependencies from requirements.txt. `pip install -r requirements.txt`
6. Run the app with Streamlit using app.py. `streamlit run app.py`

## Notes

- DATABASE_URL should use the SQLAlchemy psycopg driver format, for example: postgresql+psycopg://postgres:postgres@localhost:5432/mylingua
- Auto-fill priority (Add Word and CSV ingest):
  - User-entered values always win
  - Missing `definition_de`, `definition_en`, `translation_en`, `pos`, `artikel_nominativ` are filled from Wiktionary first
  - Missing English definition/translation can fall back to Wikidata
  - Missing sample sentence is pulled from Tatoeba first; if unavailable, Wiktionary example is used
- Sample sentence autofill requires at least 5 words.
- `artikel_nominativ` is normalized to article token only: `der`, `die`, or `das`.
- Semantic threshold is configurable in `app.py` via `SEMANTIC_MAX_DISTANCE`.

## Sample data

Use data/sample_vocab.csv in the Ingest page to quickly test the app.
CSV headers supported by ingest:
- `term_de` (required)
- `artikel_nominativ`, `definition_de`, `translation_en`, `definition_en`, `sample_sentences_de`, `pos`, `source` (optional)
