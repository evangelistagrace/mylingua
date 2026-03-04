from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote
from urllib.request import Request, urlopen

USER_AGENT = "MyLingua/1.0 (autofill)"
MIN_SAMPLE_WORDS = 5
MAX_SAMPLE_SENTENCES = 1
ARTICLES = ("der", "die", "das")


@dataclass
class AutoFillResult:
    resolved_term_de: str | None = None
    definition_de: str | None = None
    definition_en: str | None = None
    translation_en: str | None = None
    synonyms_en: str | None = None
    sample_sentences_de: str | None = None
    pos: str | None = None
    artikel_nominativ: str | None = None
    wiktionary_example_de: str | None = None


def _http_get_json(url: str, timeout: int = 8) -> dict[str, Any] | None:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = response.read().decode("utf-8", errors="replace")
            return json.loads(payload)
    except Exception:
        return None


def _clean_wiki_markup(text: str) -> str:
    cleaned = text
    cleaned = re.sub(r"<ref[^>]*>.*?</ref>", "", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"<[^>]+>", "", cleaned)
    cleaned = re.sub(r"\[\[(?:[^|\]]+\|)?([^\]]+)\]\]", r"\1", cleaned)
    cleaned = re.sub(r"\{\{[^{}]*\|([^{}|]+)\}\}", r"\1", cleaned)
    cleaned = re.sub(r"\{\{[^{}]+\}\}", "", cleaned)
    cleaned = re.sub(r"\[\d+\]", "", cleaned)
    cleaned = re.sub(r"''+", "", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip(" .;")


def _word_count(text: str) -> int:
    tokens = re.findall(r"\b[\wÄÖÜäöüß-]+\b", text, flags=re.UNICODE)
    return len(tokens)


def _sentence_count(text: str) -> int:
    parts = [p.strip() for p in re.split(r"[.!?]+", text) if p.strip()]
    return len(parts)


def _normalize_sample_sentence(text: str | None) -> str | None:
    if not text:
        return None
    cleaned = re.sub(r"\s+", " ", text).strip()
    if _word_count(cleaned) < MIN_SAMPLE_WORDS:
        return None
    if _sentence_count(cleaned) > MAX_SAMPLE_SENTENCES:
        return None
    return cleaned


def _normalize_artikel_token(value: str | None) -> str | None:
    if not value:
        return None
    first_token = value.strip().split(" ", 1)[0].lower()
    if first_token in ARTICLES:
        return first_token
    return None


def _fetch_de_wiktionary_wikitext_by_title(title_text: str) -> str | None:
    title = quote(title_text, safe="")
    url = (
        "https://de.wiktionary.org/w/api.php?action=query&prop=revisions&rvslots=main"
        f"&rvprop=content&format=json&titles={title}"
    )
    data = _http_get_json(url)
    if not data:
        return None

    pages = (data.get("query", {}) or {}).get("pages", {}) or {}
    for page in pages.values():
        if isinstance(page, dict) and page.get("missing") is not None:
            continue
        revision = ((page.get("revisions") or [{}])[0]) if isinstance(page, dict) else {}
        slots = revision.get("slots", {}) if isinstance(revision, dict) else {}
        main = slots.get("main", {}) if isinstance(slots, dict) else {}
        content = main.get("*") if isinstance(main, dict) else None
        if isinstance(content, str) and content.strip():
            return content
    return None


def _search_first_de_wiktionary_match(term_de: str) -> str | None:
    query = quote(term_de, safe="")
    url = (
        "https://de.wiktionary.org/w/api.php?action=query&list=search&format=json"
        f"&srsearch={query}&srlimit=1&srnamespace=0"
    )
    data = _http_get_json(url)
    if not data:
        return None
    results = (data.get("query", {}) or {}).get("search", []) or []
    if not results:
        return None
    title = (results[0] or {}).get("title")
    if isinstance(title, str):
        value = title.strip()
        return value or None
    return None


def _fetch_en_wiktionary_wikitext(term_en: str) -> str | None:
    title = quote(term_en, safe="")
    url = (
        "https://en.wiktionary.org/w/api.php?action=query&prop=revisions&rvslots=main"
        f"&rvprop=content&format=json&titles={title}"
    )
    data = _http_get_json(url)
    if not data:
        return None

    pages = (data.get("query", {}) or {}).get("pages", {}) or {}
    for page in pages.values():
        revision = ((page.get("revisions") or [{}])[0]) if isinstance(page, dict) else {}
        slots = revision.get("slots", {}) if isinstance(revision, dict) else {}
        main = slots.get("main", {}) if isinstance(slots, dict) else {}
        content = main.get("*") if isinstance(main, dict) else None
        if isinstance(content, str) and content.strip():
            return content
    return None


def _extract_first_section_line(wikitext: str, section_marker: str) -> str | None:
    try:
        section_idx = wikitext.index(section_marker)
    except ValueError:
        return None

    tail = wikitext[section_idx:].splitlines()
    for line in tail[1:]:
        if line.strip().startswith("{{") and section_marker not in line:
            break
        if line.strip().startswith(":"):
            cleaned = _clean_wiki_markup(line.lstrip(":").strip())
            if cleaned:
                return cleaned
    return None


def _extract_section_lines(wikitext: str, section_marker: str) -> list[str]:
    try:
        section_idx = wikitext.index(section_marker)
    except ValueError:
        return []

    lines: list[str] = []
    tail = wikitext[section_idx:].splitlines()
    for line in tail[1:]:
        if line.strip().startswith("{{") and section_marker not in line:
            break
        if line.strip().startswith(":"):
            cleaned = _clean_wiki_markup(line.lstrip(":").strip())
            if cleaned:
                lines.append(cleaned)
    return lines


def _extract_en_translation(wikitext: str) -> str | None:
    en_block = re.search(r"\*\s*\{\{en\}\}\s*:\s*(.+)", wikitext)
    if not en_block:
        return None

    line = en_block.group(1)
    candidates = re.findall(r"\{\{Ü\|en\|([^}|]+)", line)
    if candidates:
        return _clean_wiki_markup(candidates[0])
    return _clean_wiki_markup(line) or None


def _extract_en_synonyms_from_wikitext(wikitext: str) -> str | None:
    lines = wikitext.splitlines()
    in_synonyms = False
    candidates: list[str] = []

    for line in lines:
        stripped = line.strip()
        if re.match(r"^====+\s*Synonyms\s*====+$", stripped, flags=re.IGNORECASE):
            in_synonyms = True
            continue

        if in_synonyms and re.match(r"^====+[^=].*====+$", stripped):
            break

        if not in_synonyms:
            continue

        if stripped.startswith("*"):
            cleaned = _clean_wiki_markup(stripped.lstrip("*").strip())
            if not cleaned:
                continue
            for token in re.split(r",|;|/| or ", cleaned):
                item = token.strip()
                if item:
                    candidates.append(item)

    unique: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)

    if not unique:
        return None
    return ", ".join(unique[:12])


def fetch_english_synonyms_for_translation(translation_en: str | None) -> str | None:
    if not translation_en:
        return None
    wikitext = _fetch_en_wiktionary_wikitext(translation_en)
    if not wikitext:
        return None
    return _extract_en_synonyms_from_wikitext(wikitext)


def _extract_pos(wikitext: str) -> str | None:
    matches = re.findall(r"\{\{Wortart\|([^|}]+)\|Deutsch", wikitext)
    if not matches:
        return None

    mapping = {
        "Substantiv": "noun",
        "Verb": "verb",
        "Adjektiv": "adjective",
        "Präposition": "preposition",
        "Adverb": "adverb",
    }
    for match in matches:
        pos = mapping.get(match.strip())
        if pos:
            return pos
    return None


def _extract_artikel_nominativ(wikitext: str, _term_de: str) -> str | None:
    nominative = re.search(r"\|Nominativ Singular(?:\s*\d+)?=([^\n|]+)", wikitext)
    if nominative:
        phrase = _clean_wiki_markup(nominative.group(1))
        article = _normalize_artikel_token(phrase)
        if article:
            return article

    genus = re.search(r"\|Genus(?:\s*\d+)?=([mfn])", wikitext)
    if not genus:
        return None

    article_map = {"m": "der", "f": "die", "n": "das"}
    article = article_map.get(genus.group(1).strip().lower())
    if not article:
        return None
    return article


def fetch_wiktionary_data(term_de: str) -> AutoFillResult:
    resolved_term = _search_first_de_wiktionary_match(term_de) or term_de
    wikitext = _fetch_de_wiktionary_wikitext_by_title(resolved_term)
    if not wikitext and resolved_term != term_de:
        wikitext = _fetch_de_wiktionary_wikitext_by_title(term_de)
        resolved_term = term_de
    if not wikitext:
        return AutoFillResult(resolved_term_de=term_de)

    definition_de = _extract_first_section_line(wikitext, "{{Bedeutungen}}")
    sample_candidates = _extract_section_lines(wikitext, "{{Beispiele}}")
    sample_de = next(
        (
            normalized
            for normalized in (_normalize_sample_sentence(s) for s in sample_candidates)
            if normalized
        ),
        None,
    )
    translation_en = _extract_en_translation(wikitext)
    synonyms_en = fetch_english_synonyms_for_translation(translation_en)
    definition_en = translation_en
    pos = _extract_pos(wikitext)
    artikel_nominativ = _extract_artikel_nominativ(wikitext, term_de)
    return AutoFillResult(
        resolved_term_de=resolved_term,
        definition_de=definition_de,
        definition_en=definition_en,
        translation_en=translation_en,
        synonyms_en=synonyms_en,
        sample_sentences_de=sample_de,
        pos=pos,
        artikel_nominativ=artikel_nominativ,
        wiktionary_example_de=sample_de,
    )


def fetch_wikidata_gloss_en(term_de: str) -> str | None:
    term = quote(term_de, safe="")
    url = (
        "https://www.wikidata.org/w/api.php?action=wbsearchentities&format=json"
        f"&type=lexeme&language=de&uselang=en&limit=1&search={term}"
    )
    data = _http_get_json(url)
    if not data:
        return None
    results = data.get("search") or []
    if not results:
        return None
    description = (results[0] or {}).get("description")
    if isinstance(description, str):
        desc = description.strip()
        return desc or None
    return None


def fetch_tatoeba_example_de(term_de: str) -> str | None:
    term = quote(term_de, safe="")
    url = (
        "https://tatoeba.org/en/api_v0/search?from=de&query="
        f"{term}&sort=relevance&orphans=no&unapproved=no&limit=20"
    )
    data = _http_get_json(url)
    if not data:
        return None

    results = data.get("results")
    if not isinstance(results, list) or not results:
        return None

    for row in results:
        text = (row or {}).get("text")
        if isinstance(text, str):
            normalized = _normalize_sample_sentence(text)
            if normalized:
                return normalized
    return None


def autofill_missing_fields(
    term_de: str,
    definition_de: str | None,
    definition_en: str | None,
    translation_en: str | None,
    synonyms_en: str | None,
    sample_sentences_de: str | None,
    pos: str | None,
    artikel_nominativ: str | None,
) -> tuple[str, str | None, str | None, str | None, str | None, str | None, str | None, str | None]:
    wiktionary = fetch_wiktionary_data(term_de)
    final_term_de = (wiktionary.resolved_term_de or term_de).strip() or term_de

    final_definition = definition_de or wiktionary.definition_de

    final_translation = translation_en or wiktionary.translation_en
    if not final_translation:
        final_translation = fetch_wikidata_gloss_en(term_de)
    final_synonyms = synonyms_en or wiktionary.synonyms_en

    final_definition_en = definition_en or wiktionary.definition_en
    if not final_definition_en:
        final_definition_en = fetch_wikidata_gloss_en(term_de) or final_translation

    final_sample = sample_sentences_de or fetch_tatoeba_example_de(term_de)
    if not final_sample:
        final_sample = _normalize_sample_sentence(wiktionary.wiktionary_example_de)

    final_pos = pos or wiktionary.pos
    normalized_user_artikel = _normalize_artikel_token(artikel_nominativ)
    final_artikel = normalized_user_artikel or wiktionary.artikel_nominativ
    if final_pos != "noun":
        final_artikel = normalized_user_artikel

    return (
        final_term_de,
        final_definition,
        final_definition_en,
        final_translation,
        final_synonyms,
        final_sample,
        final_pos,
        final_artikel,
    )
