# MyLingua

A POC language-learning vocabulary app that lets users search German words by natural-language meaning in English or German. It uses Streamlit for the UI, PostgreSQL with pgvector for storage and vector search, and Cohere embeddings.

![alt text](image.png)

## Features

- CSV ingestion with embeddings using Cohere embed-multilingual-light-v3.0
- Manual single-word form entry (German word required, other fields optional)
  - Missing fields can be auto-filled from Wiktionary, Wikidata, and Tatoeba
  - Supports POS and nominative article 
- Exact and prefix search on German terms
- Optional fuzzy search via pg_trgm
- Semantic search via pgvector similarity
- Simple Streamlit UI with Search, Ingest, and Add Word pages

## Project structure

- app.py — Streamlit entry
- src/db.py — database engine, extensions, indexes
- src/models.py — SQLAlchemy model
- src/cohere_client.py — Cohere embeddings wrapper
- src/ingest.py — CSV + manual-entry ingest logic
- src/search.py — exact, fuzzy, semantic search
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
- Vector index creation is attempted automatically; if your pgvector version does not support HNSW, the app will continue without it.
- Fuzzy search requires the pg_trgm extension; if unavailable, the UI will show a message.
- Auto-fill priority in Add Word form:
  - User-entered values always win
  - Missing definition/translation/POS/article are pulled from Wiktionary first
  - Missing translation can fall back to Wikidata
  - Missing sample sentence is pulled from Tatoeba; if unavailable, Wiktionary example is used

## Sample data

Use data/sample_vocab.csv in the Ingest page to quickly test the app.
You can optionally include `sample_sentences_de` and `artikel_nominativ` columns in CSV files.
