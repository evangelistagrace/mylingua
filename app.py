from __future__ import annotations

import re

import streamlit as st
from dotenv import load_dotenv

from src.autofill import autofill_missing_fields
from src.cohere_client import embed_query_cached
from src.db import get_session_factory, init_db
from src.ingest import IngestRow, ingest_rows, parse_tabular_file
from src.models import Sense
from src.search import (
    exact_prefix_search_en,
    exact_prefix_search_synonyms_en,
    semantic_search_dual_english_first,
    tolerant_german_word_search,
)

SEMANTIC_MAX_DISTANCE = 1.1
POS_OPTIONS = ["", "noun", "verb", "adjective", "preposition", "adverb"]

load_dotenv()

st.set_page_config(page_title="MyLingua", page_icon="📚", layout="wide")

st.title("What's that (German) word? 💬")


@st.cache_resource
def _session_factory():
    return get_session_factory()


@st.cache_resource
def _init_db():
    return init_db()


try:
    _init_db()
except Exception as exc:
    st.error(f"Database init failed: {exc}")
    st.stop()

def render_sense(sense, distance: float | None = None) -> None:
    def _strip_reference_numbers(value: str | None) -> str | None:
        if not value:
            return None
        cleaned = re.sub(r"\[\d+\]", "", value)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned or None

    title = sense.term_de
    if sense.artikel_nominativ:
        title = f"({sense.artikel_nominativ}) {sense.term_de}"
    gender_label = None
    if sense.pos == "noun" and sense.artikel_nominativ:
        article = sense.artikel_nominativ.strip().lower()
        if article == "der":
            gender_label = "masculine"
        elif article == "die":
            gender_label = "feminine"
        elif article == "das":
            gender_label = "neuter"

    badge_parts: list[str] = []
    if sense.pos:
        badge_parts.append(
            "<span style='display:inline-block;padding:2px 10px;border-radius:999px;"
            "font-size:0.85rem;font-weight:600;background:#e2e8f0;color:#1e3a8a;margin-right:6px;'>"
            f"{sense.pos}</span>"
        )
    if gender_label:
        gender_styles = {
            "masculine": "background:#dbeafe;color:#1d4ed8;",
            "feminine": "background:#fee2e2;color:#b91c1c;",
            "neuter": "background:#dcfce7;color:#15803d;",
        }
        badge_parts.append(
            "<span style='display:inline-block;padding:2px 10px;border-radius:999px;"
            f"font-size:0.85rem;font-weight:600;{gender_styles.get(gender_label, '')}'>"
            f"{gender_label}</span>"
        )
    definition_de = _strip_reference_numbers(sense.definition_de)
    definition_en = _strip_reference_numbers(sense.definition_en)
    sample_de = _strip_reference_numbers(sense.sample_sentences_de)

    with st.container(border=True):
        # left, right = st.columns([0.82, 0.18])
        # with left:
        st.markdown(f"**{title}**")
        en_parts = []
        if sense.translation_en:
            en_parts.append(sense.translation_en)
        if getattr(sense, "synonyms_en", None):
            en_parts.append(sense.synonyms_en)
        if en_parts:
            st.caption(", ".join(en_parts))
        # with right:
        if badge_parts:
            st.markdown("".join(badge_parts), unsafe_allow_html=True)

        if definition_de:
            st.write(definition_de)
        elif definition_en:
            st.write(definition_en)
        if sample_de:
            st.markdown(f"*{sample_de}*")

        if distance is not None:
            if distance <= 1:
                st.caption("High match")
            elif distance <= 1.2:
                st.caption("Medium match")
            else:
                st.caption("Weak match")


def _render_footer() -> None:
    st.markdown("---")
    st.caption(
        "Data sources: Wiktionary, Wikidata, Tatoeba. Embeddings by Cohere."
    )


def _detect_single_word_language(word: str) -> str:
    if re.search(r"[äöüßÄÖÜ]", word):
        return "German"
    return "English"


def _save_word_entry(
    term_de: str,
    artikel_nominativ: str,
    definition_de: str,
    translation_en: str,
    synonyms_en: str,
    sample_sentences_de: str,
    pos: str,
    success_prefix: str = "Saved",
) -> None:
    cleaned_term = term_de.strip()
    if not cleaned_term:
        st.error("German word is required.")
        return

    user_definition = definition_de.strip() or None
    user_translation = translation_en.strip() or None
    user_synonyms = synonyms_en.strip() or None
    user_sample = sample_sentences_de.strip() or None
    user_pos = pos or None
    user_artikel = artikel_nominativ.strip() or None

    with st.spinner("Auto-filling missing fields..."):
        (
            auto_term_de,
            auto_definition,
            auto_definition_en,
            auto_translation,
            auto_synonyms,
            auto_sample,
            auto_pos,
            auto_artikel,
        ) = autofill_missing_fields(
            term_de=cleaned_term,
            definition_de=user_definition,
            definition_en=None,
            translation_en=user_translation,
            synonyms_en=user_synonyms,
            sample_sentences_de=user_sample,
            pos=user_pos,
            artikel_nominativ=user_artikel,
        )

    row = IngestRow(
        term_de=auto_term_de,
        artikel_nominativ=auto_artikel,
        definition_de=auto_definition,
        sample_sentences_de=auto_sample,
        translation_en=auto_translation,
        synonyms_en=auto_synonyms,
        definition_en=auto_definition_en,
        pos=auto_pos,
        source="manual_form",
    )
    session_factory = _session_factory()
    with session_factory() as session:
        with st.spinner("Embedding and storing..."):
            try:
                created, updated = ingest_rows(session, [row], batch_size=1)
                sources_used: list[str] = []
                if (not user_definition and auto_definition) or (not user_pos and auto_pos) or (
                    not user_artikel and auto_artikel
                ):
                    sources_used.append("Wiktionary")
                if (not user_translation and auto_translation) or auto_definition_en:
                    sources_used.append("Wiktionary/Wikidata")
                if not user_synonyms and auto_synonyms:
                    sources_used.append("Wiktionary")
                if not user_sample and auto_sample:
                    sources_used.append("Tatoeba/Wiktionary")

                if created:
                    st.success(f"{success_prefix}. The entry is now searchable.")
                    if auto_term_de != cleaned_term:
                        st.caption(f"Saved as canonical German term: {auto_term_de}")
                    if sources_used:
                        st.caption(f"Auto-filled from: {', '.join(sources_used)}")
                elif updated:
                    st.success("Updated existing entry for this term.")
                    if auto_term_de != cleaned_term:
                        st.caption(f"Saved as canonical German term: {auto_term_de}")
                    if sources_used:
                        st.caption(f"Auto-filled from: {', '.join(sources_used)}")
            except Exception as exc:
                st.error(f"Save failed: {exc}")


def render_ingest_page() -> None:
    st.subheader("Ingest File")
    st.write(
        "Upload a CSV or XLSX with headers: term_de, artikel_nominativ, definition_de, "
        "sample_sentences_de, translation_en, synonyms_en, definition_en, pos, source"
    )

    uploaded = st.file_uploader("File", type=["csv", "xlsx"])
    if uploaded is not None:
        rows = parse_tabular_file(uploaded.name, uploaded.getvalue())
        st.write(f"Parsed {len(rows)} rows")
        if rows:
            st.dataframe(
                {
                    "term_de": [r.term_de for r in rows[:10]],
                    "artikel_nominativ": [r.artikel_nominativ for r in rows[:10]],
                    "definition_de": [r.definition_de for r in rows[:10]],
                    "sample_sentences_de": [r.sample_sentences_de for r in rows[:10]],
                    "translation_en": [r.translation_en for r in rows[:10]],
                    "synonyms_en": [r.synonyms_en for r in rows[:10]],
                    "definition_en": [r.definition_en for r in rows[:10]],
                }
            )

        if st.button("Ingest now", type="primary", disabled=not rows):
            enriched_rows: list[IngestRow] = []
            autofilled_rows = 0
            with st.spinner("Auto-filling missing CSV values..."):
                for row in rows:
                    needs_autofill = any(
                        [
                            not row.definition_de,
                            not row.definition_en,
                            not row.translation_en,
                            not row.sample_sentences_de,
                            not row.pos,
                            not row.artikel_nominativ,
                            not row.synonyms_en,
                        ]
                    )
                    if not needs_autofill:
                        enriched_rows.append(row)
                        continue

                    (
                        auto_term_de,
                        auto_definition_de,
                        auto_definition_en,
                        auto_translation_en,
                        auto_synonyms_en,
                        auto_sample_de,
                        auto_pos,
                        auto_artikel,
                    ) = autofill_missing_fields(
                        term_de=row.term_de,
                        definition_de=row.definition_de,
                        definition_en=row.definition_en,
                        translation_en=row.translation_en,
                        synonyms_en=row.synonyms_en,
                        sample_sentences_de=row.sample_sentences_de,
                        pos=row.pos,
                        artikel_nominativ=row.artikel_nominativ,
                    )

                    updated_row = IngestRow(
                        term_de=auto_term_de,
                        artikel_nominativ=auto_artikel,
                        definition_de=auto_definition_de,
                        sample_sentences_de=auto_sample_de,
                        translation_en=auto_translation_en,
                        synonyms_en=auto_synonyms_en,
                        definition_en=auto_definition_en,
                        pos=auto_pos,
                        source=row.source,
                    )
                    enriched_rows.append(updated_row)

                    if (
                        updated_row.definition_de != row.definition_de
                        or updated_row.definition_en != row.definition_en
                        or updated_row.translation_en != row.translation_en
                        or updated_row.synonyms_en != row.synonyms_en
                        or updated_row.sample_sentences_de != row.sample_sentences_de
                        or updated_row.pos != row.pos
                        or updated_row.artikel_nominativ != row.artikel_nominativ
                    ):
                        autofilled_rows += 1

            session_factory = _session_factory()
            with session_factory() as session:
                with st.spinner("Embedding and storing..."):
                    try:
                        created, updated = ingest_rows(session, enriched_rows)
                        st.success(f"Done. Added {created}, updated {updated} existing terms.")
                        if autofilled_rows:
                            st.caption(f"Auto-filled missing values for {autofilled_rows} CSV rows.")
                    except Exception as exc:
                        st.error(f"Ingest failed: {exc}")
    _render_footer()


def render_add_word_page() -> None:
    st.subheader("Add a word manually")
    st.write("`German word` is required. All other fields are optional. Missing fields will be auto-filled where possible.")

    if "add_word_term_de_prefill" in st.session_state:
        st.session_state["add_word_term_de"] = st.session_state.pop("add_word_term_de_prefill")
    if "add_word_translation_en_prefill" in st.session_state:
        st.session_state["add_word_translation_en"] = st.session_state.pop(
            "add_word_translation_en_prefill"
        )

    with st.form("add_word_form", clear_on_submit=True):
        term_de = st.text_input("German word (required)", key="add_word_term_de")
        artikel_nominativ = st.text_input("Artikel (der/die/das, optional)")
        translation_en = st.text_input(
            "Translation in English (optional)",
            key="add_word_translation_en",
        )
        synonyms_en = st.text_input("English synonyms (optional)")
        pos = st.selectbox(
            "Part of speech (optional)",
            POS_OPTIONS,
            index=0,
        )
        definition_de = st.text_area("Meaning in German (optional)")
        sample_sentences_de = st.text_area("Sample sentences in German (optional)")
        submitted = st.form_submit_button("Save word", type="primary")

    if submitted:
        _save_word_entry(
            term_de=term_de,
            artikel_nominativ=artikel_nominativ,
            definition_de=definition_de,
            translation_en=translation_en,
            synonyms_en=synonyms_en,
            sample_sentences_de=sample_sentences_de,
            pos=pos,
        )
    _render_footer()


def render_words_page() -> None:
    st.subheader("Search")

    search_query = st.text_input(
        "Search by meaning or word",
        placeholder="e.g. a place where you borrow books",
    )

    if search_query:
        q = search_query.strip()
        is_single_word = " " not in q
        session_factory = _session_factory()
        with session_factory() as session:
            seen = set()
            merged_results: list[tuple[object, float | None]] = []

            lexical_results = []
            if is_single_word:
                # For one-word queries, prioritize English lexical lookup before German.
                lexical_results.extend(exact_prefix_search_en(session, q, limit=5))
                lexical_results.extend(exact_prefix_search_synonyms_en(session, q, limit=5))
                lexical_results.extend(tolerant_german_word_search(session, q, limit=10))

                for sense in lexical_results:
                    if sense.id in seen:
                        continue
                    seen.add(sense.id)
                    merged_results.append((sense, None))

            if not merged_results:
                with st.spinner("Finding best matches..."):
                    try:
                        embedding = embed_query_cached(q)
                        semantic_results = semantic_search_dual_english_first(
                            session,
                            embedding,
                            limit=10,
                            max_distance=SEMANTIC_MAX_DISTANCE,
                        )
                    except Exception as exc:
                        st.error(f"Semantic search failed: {exc}")
                        semantic_results = []

                for sense, distance in semantic_results:
                    if sense.id in seen:
                        continue
                    seen.add(sense.id)
                    merged_results.append((sense, distance))

            if merged_results:
                st.markdown("Best match(es):")
                for sense, distance in merged_results:
                    render_sense(sense, distance=distance)
            else:
                with st.container(border=True):
                    st.markdown("**No strong matches found.**")
                    st.caption("Try a shorter phrase, a synonym, or add this as a new word.")
                if is_single_word:
                    detected_language = _detect_single_word_language(q)
                    language_choice = st.radio(
                        "Query language",
                        ["English", "German"],
                        index=0 if detected_language == "English" else 1,
                        horizontal=True,
                        key=f"search_add_lang_{q}",
                    )
                    if st.button(f"Add word *{q}*?", type="primary"):
                        st.session_state["add_word_term_de_prefill"] = (
                            q if language_choice == "German" else ""
                        )
                        st.session_state["add_word_translation_en_prefill"] = (
                            q if language_choice == "English" else ""
                        )
                        st.switch_page(PAGE_ADD_WORD)
    # else:
    #     st.info("Enter a query to search.")

    _render_footer()


@st.fragment
def render_all_words_fragment() -> None:
    st.subheader("All words")
    list_filter = st.text_input(
        "Filter word list",
        placeholder="Filter by German term, translation, or synonym",
        key="words_list_filter",
    )
    session_factory = _session_factory()
    with session_factory() as session:
        rows = session.query(Sense).order_by(Sense.term_de.asc()).all()

    if list_filter:
        q = list_filter.strip().lower()
        rows = [
            r
            for r in rows
            if q in (r.term_de or "").lower()
            or q in (r.translation_en or "").lower()
            or q in (r.synonyms_en or "").lower()
        ]

    st.caption(f"{len(rows)} word(s)")
    if not rows:
        st.info("No words found.")
        return

    table_data = {
        "term_de": [r.term_de for r in rows],
        "artikel_nominativ": [r.artikel_nominativ for r in rows],
        "translation_en": [r.translation_en for r in rows],
        "synonyms_en": [r.synonyms_en for r in rows],
        "pos": [r.pos for r in rows],
    }
    event = st.dataframe(
        table_data,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="words_table",
    )

    selected = None
    selected_rows = event.selection.rows if event and event.selection else []
    if selected_rows:
        row_idx = selected_rows[0]
        if 0 <= row_idx < len(rows):
            selected = rows[row_idx]

    if selected:
        render_sense(selected)
    else:
        st.caption("Click a row to view full details.")


def render_all_words_page() -> None:
    render_all_words_fragment()
    _render_footer()



PAGE_WORDS = st.Page(render_words_page, title="Search", icon="🔎", default=True)
PAGE_ALL_WORDS = st.Page(render_all_words_page, title="All Words", icon="📖")
PAGE_INGEST = st.Page(render_ingest_page, title="Ingest", icon="📥")
PAGE_ADD_WORD = st.Page(render_add_word_page, title="Add Word", icon="➕")

navigation = st.navigation([PAGE_WORDS, PAGE_ALL_WORDS, PAGE_INGEST, PAGE_ADD_WORD])
navigation.run()
