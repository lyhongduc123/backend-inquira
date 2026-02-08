from typing import Dict, Any, List, Optional
from app.models.papers import DBPaper
from app.retriever.paper_schemas import Paper, PaperResponse, Author
from app.retriever.schemas import NormalizedResult
from datetime import datetime


def parse_date(date_str: Optional[str]) -> Optional[datetime]:
    """
    Parse an ISO 8601 date string into a datetime, falling back to None.
    Accepts full timestamps or date-only strings like 'YYYY-MM-DD' and
    handles a trailing 'Z' timezone designator by stripping it.
    """
    if not date_str:
        return None
    try:
        # datetime.fromisoformat handles most ISO 8601 forms except a trailing 'Z'
        if date_str.endswith("Z"):
            date_str = date_str[:-1]
        return datetime.fromisoformat(date_str)
    except ValueError:
        # try a couple of common date-only formats as a fallback
        for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
            try:
                return datetime.strptime(date_str, fmt)
            except Exception:
                continue
    return None