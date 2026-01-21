from __future__ import annotations

import streamlit as st
from dotenv import load_dotenv

from src.cohere_client import embed_query_cached
from src.db import get_session_factory, init_db
from src.ingest import ingest_rows, parse_csv
from src.search import exact_prefix_search, fuzzy_search, is_extension_enabled, semantic_search

load_dotenv()

st.set_page_config(page_title="MyLingua", page_icon="📚", layout="wide")

st.title("What's the word? 💬")

with st.sidebar:
    st.header("Navigation")
    page = st.radio("Go to", ["Search", "Ingest"], index=0)

    st.divider()
    st.caption("Settings")
    display_lang = st.selectbox("Display hint", ["Auto", "EN", "DE"], index=0)


@st.cache_resource
def _session_factory():
    return get_session_factory()


@st.cache_resource
def _init_db():
    return init_db()


try:
    created_exts = _init_db()
except Exception as exc:
    st.error(f"Database init failed: {exc}")
    st.stop()

if created_exts:
    st.sidebar.success(f"Extensions ready: {', '.join(created_exts)}")


def render_sense(sense, distance: float | None = None) -> None:
    st.markdown(f"**{sense.term_de}** — {sense.translation_en or ''}")
    if display_lang == "DE":
        if sense.definition_de:
            st.caption(sense.definition_de)
        if sense.definition_en:
            st.caption(sense.definition_en)
    elif display_lang == "EN":
        if sense.definition_en:
            st.caption(sense.definition_en)
        if sense.definition_de:
            st.caption(sense.definition_de)
    else:
        if sense.definition_de:
            st.caption(sense.definition_de)
        if sense.definition_en:
            st.caption(sense.definition_en)

    if distance is not None:
        st.caption(f"distance: {distance:.4f}")
    st.divider()


if page == "Ingest":
    st.subheader("Ingest CSV")
    st.write("Upload a CSV with headers: term_de, definition_de, translation_en, definition_en, pos, source")

    uploaded = st.file_uploader("CSV file", type=["csv"])
    if uploaded is not None:
        rows = parse_csv(uploaded.getvalue())
        st.write(f"Parsed {len(rows)} rows")
        if rows:
            st.dataframe(
                {
                    "term_de": [r.term_de for r in rows[:10]],
                    "definition_de": [r.definition_de for r in rows[:10]],
                    "translation_en": [r.translation_en for r in rows[:10]],
                    "definition_en": [r.definition_en for r in rows[:10]],
                }
            )

        if st.button("Ingest now", type="primary", disabled=not rows):
            session_factory = _session_factory()
            with session_factory() as session:
                with st.spinner("Embedding and storing..."):
                    try:
                        created, skipped = ingest_rows(session, rows)
                        st.success(f"Done. Added {created}, skipped {skipped} duplicates.")
                    except Exception as exc:
                        st.error(f"Ingest failed: {exc}")


if page == "Search":
    st.subheader("Search by meaning")
    st.caption(f"Display hint: {display_lang}")

    query = st.text_input("Search", placeholder="e.g. a place where you borrow books")
    force_word_search = st.checkbox("Force word search (exact/prefix)", value=False)

    if query:
        session_factory = _session_factory()
        with session_factory() as session:
            col1, col2 = st.columns([1, 1])

            with col1:
                st.markdown("### Exact / Prefix matches")
                results = []
                if force_word_search or (" " not in query.strip()):
                    results = exact_prefix_search(session, query, limit=20)
                    if is_extension_enabled(session, "pg_trgm"):
                        fuzzy = fuzzy_search(session, query, limit=10)
                    else:
                        fuzzy = []
                else:
                    fuzzy = []

                if results or fuzzy:
                    for sense in results:
                        render_sense(sense)

                    if fuzzy:
                        st.markdown("#### Fuzzy matches")
                        for sense in fuzzy:
                            render_sense(sense)
                    elif force_word_search and not is_extension_enabled(session, "pg_trgm"):
                        st.info("pg_trgm extension not available; fuzzy search is disabled.")
                else:
                    st.info("No exact or prefix matches.")

            with col2:
                st.markdown("### Semantic matches")
                with st.spinner("Embedding query..."):
                    try:
                        embedding = embed_query_cached(query)
                        semantic_results = semantic_search(session, embedding, limit=1)
                    except Exception as exc:
                        st.error(f"Semantic search failed: {exc}")
                        semantic_results = []

                if semantic_results:
                    for sense, distance in semantic_results:
                        render_sense(sense, distance=distance)
                else:
                    st.info("No semantic matches yet. Try ingesting data.")

    else:
        st.info("Enter a query to search.")
