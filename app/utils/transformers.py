"""
Utility functions for data transformation between different formats.
Moved from TransformerService to promote functional programming.
"""
from typing import Optional, List
from app.core.dtos.paper import PaperEnrichedDTO, PaperDTO
from app.core.dtos.author import AuthorDTO
from app.retriever.schemas import NormalizedPaperResult
from app.extensions.logger import create_logger

logger = create_logger(__name__)


def normalize_issn(issn: Optional[str]) -> Optional[str]:
    """
    Normalize ISSN by removing hyphens and whitespace.
    
    Args:
        issn: Raw ISSN string
    Returns:
        Normalized ISSN string or None
    """
    if not issn:
        return None
    issn = issn.strip().upper().replace("-", "")
    return issn if len(issn) == 8 else None


def normalized_to_paper(result: NormalizedPaperResult) -> PaperEnrichedDTO:
    """
    Convert a provider NormalizedPaperResult dict to a PaperEnrichedDTO Pydantic model.
    
    Stores complex nested data (biblio, primary_location, locations, authorships)
    in openalex_data or semantic_data JSONB fields to match DBPaper schema.
    """
    # Convert AuthorSchema objects to AuthorDTO
    authors_dto = []
    for author in result.authors:
        # Convert AuthorSchema to dict, then to AuthorDTO
        if isinstance(author, dict):
            author_dto = AuthorDTO(**author)
            authors_dto.append(author_dto)
        else:
            # It's an AuthorSchema object, convert to dict first
            author_dto = AuthorDTO(**author.model_dump())
            authors_dto.append(author_dto)
    
    # Debug: Check if names are present
    if authors_dto:
        author_names = [a.name for a in authors_dto]
        logger.debug(f"Converted {len(authors_dto)} authors for paper {result.paper_id}: {author_names}")
    
    # Convert result to dict and replace authors with AuthorDTO list
    result_dict = result.model_dump()
    result_dict['authors'] = authors_dto
    
    # Extract tldr text from S2's tldr dict structure: {"model": "...", "text": "..."}
    if result_dict.get('tldr') and isinstance(result_dict['tldr'], dict):
        result_dict['tldr'] = result_dict['tldr'].get('text')
    
    # Ensure has_content is a dict, not None
    if result_dict.get('has_content') is None:
        result_dict['has_content'] = {}
    
    paper = PaperEnrichedDTO.model_validate(result_dict)
    locations = [result.primary_location, result.best_oa_location, result.locations]
    for loc in locations:
        if not loc:
            continue

        source = loc.get("source")
        if not paper.issn:
            issn_set = set()
            for i in source.get("issn", []) or []:
                norm = normalize_issn(i)
                if norm:
                    issn_set.add(norm)
            paper.issn = list(issn_set) if issn_set else None

        if not paper.issn_l:
            norm_l = normalize_issn(source.get("issn_l"))
            paper.issn_l = norm_l if norm_l else None

        if paper.issn and paper.issn_l:
            break

    return paper


def batch_normalized_to_papers(results: List[NormalizedPaperResult]) -> List[PaperEnrichedDTO]:
    """
    Convert multiple NormalizedPaperResult to PaperEnrichedDTO.
    
    Args:
        results: List of normalized paper results
    Returns:
        List of PaperEnrichedDTO
    """
    papers = []
    for result in results:
        try:
            paper = normalized_to_paper(result)
            papers.append(paper)
        except Exception as e:
            # Log error and skip problematic entry
            logger.error(f"Error converting normalized result to Paper: {e}")
            continue
    return papers