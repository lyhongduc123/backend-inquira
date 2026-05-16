"""Filter parsing helpers shared by search workflows."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.search.filter_options import SearchFilterOptions


def get_filter_value(filters: Optional[Dict[str, Any]], snake_key: str) -> Any:
    """Read a filter value from either snake_case or camelCase key."""
    if not filters:
        return None

    if snake_key in filters:
        return filters.get(snake_key)

    parts = snake_key.split("_")
    camel_key = parts[0] + "".join(part.capitalize() for part in parts[1:])
    return filters.get(camel_key)


def _parse_fields_of_study(filters: Optional[Dict[str, Any]]) -> Optional[List[str]]:
    field_of_study = get_filter_value(filters, "field_of_study")
    if isinstance(field_of_study, str) and field_of_study.strip():
        return [field_of_study.strip()]
    if isinstance(field_of_study, list):
        return [str(field).strip() for field in field_of_study if str(field).strip()]

    fields = get_filter_value(filters, "fields_of_study")
    if isinstance(fields, list):
        return [str(field).strip() for field in fields if str(field).strip()]

    return None


def parse_search_filter_options(
    filters: Optional[Dict[str, Any]],
) -> SearchFilterOptions:
    """Convert request/filter dictionaries into repository search filter options."""
    return SearchFilterOptions(
        author_name=get_filter_value(filters, "author_name")
        or get_filter_value(filters, "author"),
        year_min=get_filter_value(filters, "year_min"),
        year_max=get_filter_value(filters, "year_max"),
        venue=get_filter_value(filters, "venue"),
        min_citation_count=get_filter_value(filters, "min_citation_count")
        or get_filter_value(filters, "min_citations"),
        max_citation_count=get_filter_value(filters, "max_citation_count")
        or get_filter_value(filters, "max_citations"),
        journal_quartile=get_filter_value(filters, "journal_quartile")
        or get_filter_value(filters, "journal_rank"),
        field_of_study=_parse_fields_of_study(filters),
    )
