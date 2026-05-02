"""
Normalization helpers for external identifiers and field names.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


_CANONICAL_EXTERNAL_ID_KEYS = {
    "doi": "doi",
    "arxiv": "arxiv",
    "acl": "acl",
    "pubmed": "pubmed",
    "pmid": "pubmed",
    "pmcid": "pmcid",
    "mag": "mag",
    "dblp": "dblp",
    "openalex": "openalex",
    "corpusid": "corpusid",
    "s2corpusid": "corpusid",
    "semanticscholar": "semanticscholar",
}

_EXTERNAL_ID_KEY_CANDIDATES = {
    "doi": ["doi", "DOI", "DoI"],
    "arxiv": ["arxiv", "ArXiv", "arXiv"],
    "acl": ["acl", "ACL"],
    "pubmed": ["pubmed", "PubMed", "PMID", "pmid"],
    "pmcid": ["pmcid", "PMCID"],
    "mag": ["mag", "MAG"],
    "dblp": ["dblp", "DBLP"],
    "openalex": ["openalex", "OpenAlex"],
    "corpusid": ["corpusid", "CorpusId", "corpusId"],
    "semanticscholar": ["semanticscholar", "SemanticScholar", "semanticScholar"],
}


def _normalize_external_id_key(key: str) -> str:
    key_norm = re.sub(r"[^a-z0-9]", "", key.strip().lower())
    return _CANONICAL_EXTERNAL_ID_KEYS.get(key_norm, key.strip())


def _normalize_external_id_value(key: str, value: Any) -> Optional[str]:
    if value is None:
        return None

    normalized = str(value).strip()
    if not normalized:
        return None

    if key == "doi":
        normalized = re.sub(r"^https?://(dx\.)?doi\.org/", "", normalized, flags=re.IGNORECASE)
        return normalized.lower()

    if key == "openalex":
        normalized = re.sub(r"^https?://openalex\.org/", "", normalized, flags=re.IGNORECASE)
        return normalized

    return normalized


def normalize_external_ids(external_ids: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Normalize external IDs to canonical key casing and clean values."""
    if not external_ids:
        return {}

    normalized: Dict[str, Any] = {}
    for raw_key, raw_value in external_ids.items():
        if not isinstance(raw_key, str):
            continue

        canonical_key = _normalize_external_id_key(raw_key)
        canonical_value = _normalize_external_id_value(canonical_key, raw_value)

        if canonical_value is None:
            continue

        if canonical_key not in normalized:
            normalized[canonical_key] = canonical_value

    return normalized


def external_id_key_candidates(canonical_key: str) -> List[str]:
    """Return known key aliases for backward-compatible DB lookups."""
    key_norm = re.sub(r"[^a-z0-9]", "", canonical_key.strip().lower())
    canonical = _CANONICAL_EXTERNAL_ID_KEYS.get(key_norm, key_norm)
    return _EXTERNAL_ID_KEY_CANDIDATES.get(canonical, [canonical_key])


def normalize_fields_of_study(fields: Optional[List[str]]) -> Optional[List[str]]:
    """Normalize field-of-study list by trimming and case-insensitive dedupe. Could be not needed
    """
    if not fields:
        return None

    result: List[str] = []
    seen = set()

    for field in fields:
        if field is None:
            continue

        cleaned = re.sub(r"\s+", " ", str(field).strip())
        if not cleaned:
            continue

        key = cleaned.lower()
        if key in seen:
            continue

        seen.add(key)
        result.append(cleaned)

    return result or None


def normalize_s2_fields_of_study(
    fields: Optional[List[Dict[str, Any]]],
) -> Optional[List[Dict[str, Any]]]:
    """Normalize S2 enriched field metadata with dedupe by category/source."""
    if not fields:
        return None

    result: List[Dict[str, Any]] = []
    seen = set()

    for entry in fields:
        if not isinstance(entry, dict):
            continue

        category_raw = entry.get("category")
        source_raw = entry.get("source")

        category = re.sub(r"\s+", " ", str(category_raw).strip()) if category_raw else ""
        source = re.sub(r"\s+", " ", str(source_raw).strip()) if source_raw else ""

        if not category:
            continue

        dedupe_key = (category.lower(), source.lower())
        if dedupe_key in seen:
            continue

        seen.add(dedupe_key)
        normalized_entry = dict(entry)
        normalized_entry["category"] = category
        if source:
            normalized_entry["source"] = source
        result.append(normalized_entry)

    return result or None
