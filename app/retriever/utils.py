from typing import Dict, Any, List, Optional
from app.models.papers import DBPaper
from app.retriever.paper_schemas import Paper, Author
from app.retriever.provider.base_schemas import NormalizedResult
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


def normalized_to_paper(result: NormalizedResult) -> Paper:
    """
    Convert a provider NormalizedResult dict to a Paper Pydantic model.
    """

    authors_data = result.get("authors", [])
    authors = [Author(**a) for a in authors_data]

    pub_date_str = result.get("publication_date")
    publication_date = parse_date(pub_date_str) if pub_date_str else None

    external_ids = result.get("external_ids") or {}

    return Paper(
        paper_id=result.get("paper_id", ""),
        title=result.get("title", ""),
        abstract=result.get("abstract"),
        authors=authors,
        publication_date=publication_date,
        venue=result.get("venue"),
        url=result.get("url"),
        pdf_url=result.get("pdf_url"),
        is_open_access=result.get("is_open_access", False),
        open_access_pdf=result.get("open_access_pdf"),
        citation_count=result.get("citation_count", 0),
        influential_citation_count=result.get("influential_citation_count", 0),
        reference_count=result.get("reference_count", 0),
        source=result.get("source", ""),
        external_ids=external_ids,
        is_processed=False,
        processing_status="pending",
    )


def batch_normalized_to_papers(results: List[NormalizedResult]) -> List[Paper]:
    """
    Convert a list of NormalizedResult dicts to a list of Paper Pydantic models.
    """
    papers = []
    for result in results:
        try:
            paper = normalized_to_paper(result)
            papers.append(paper)
        except Exception as e:
            # Log error and skip problematic entry
            print(f"Error converting normalized result to Paper: {e}")
            continue
    return papers


def dbpaper_to_paper(db_paper) -> Paper:
    """
    Convert a database paper ORM object to a Paper Pydantic model.
    Assumes db_paper has attributes matching Paper fields.
    """
    authors = []
    if hasattr(db_paper, "authors") and db_paper.authors:
        for author in db_paper.authors:
            if isinstance(author, dict):
                authors.append(
                    Author(
                        name=author.get("name", ""),
                        author_id=author.get("author_id"),
                        citation_count=author.get("citation_count"),
                        h_index=author.get("h_index"),
                    )
                )
            else:
                authors.append(
                    Author(
                        name=author.name,
                        author_id=author.author_id,
                        citation_count=author.citation_count,
                        h_index=author.h_index,
                    )
                )

    publication_date = None
    if hasattr(db_paper, "publication_date") and db_paper.publication_date:
        publication_date = db_paper.publication_date

    return Paper(
        paper_id=str(db_paper.paper_id),
        title=db_paper.title,
        abstract=db_paper.abstract,
        authors=authors,
        publication_date=publication_date,
        venue=db_paper.venue,
        url=db_paper.url,
        pdf_url=db_paper.pdf_url,
        is_open_access=db_paper.is_open_access,
        open_access_pdf=db_paper.open_access_pdf,
        citation_count=db_paper.citation_count,
        influential_citation_count=db_paper.influential_citation_count,
        reference_count=db_paper.reference_count,
        source=db_paper.source,
        external_ids=db_paper.external_ids,
        is_processed=db_paper.is_processed,
        processing_status=db_paper.processing_status,
    )


def batch_dbpaper_to_papers(db_papers: List[DBPaper]) -> List[Paper]:
    """
    Convert a list of database paper ORM objects to a list of Paper Pydantic models.
    """
    papers = []
    for db_paper in db_papers:
        try:
            paper = dbpaper_to_paper(db_paper)
            papers.append(paper)
        except Exception as e:
            # Log error and skip problematic entry
            print(f"Error converting DB paper to Paper: {e}")
            continue
    return papers


def paper_to_dict(paper: Paper) -> Dict[str, Any]:
    """
    Convert a Paper Pydantic model to a dictionary suitable for database insertion.
    """
    return paper.model_dump()


def batch_paper_to_dicts(papers: List[Paper]) -> List[Dict[str, Any]]:
    """
    Convert a list of Paper Pydantic models to a list of dictionaries.
    """
    dicts = []
    for paper in papers:
        try:
            paper_dict = paper_to_dict(paper)
            dicts.append(paper_dict)
        except Exception as e:
            # Log error and skip problematic entry
            print(f"Error converting Paper to dict: {e}")
            continue
    return dicts
