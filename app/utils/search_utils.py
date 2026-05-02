import re
from typing import List

def build_paradedb_query(query: str, fields: List[str]) -> str:
    """
    Builds a safe ParadeDB (Tantivy) BM25 query.
    
    Args:
        query: The raw search string from the user.
        fields: A list of default fields to search if the user doesn't specify one. 
                Defaults to ["title", "abstract"].
    """
    if fields is None:
        raise ValueError("Fields list cannot be None. Provide at least one field to search.")
        
    query = query.strip()
    if not query or not fields:
        return ""

    escaped_fields = [re.escape(f) for f in fields]
    field_regex = r"\b(" + "|".join(escaped_fields) + r"):"
    
    has_valid_field = bool(re.search(field_regex, query))

    if has_valid_field:
        return query
    else:
        safe_q = re.sub(r'[+\-!(){}\[\]^"~*?:\\/]', ' ', query)
        safe_q = re.sub(r"\s+", " ", safe_q).strip()
        
        if not safe_q:
            return ""

        query_parts = [f"{field}:({safe_q})" for field in fields]
        return " OR ".join(query_parts)