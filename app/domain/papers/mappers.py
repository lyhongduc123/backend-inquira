"""Paper mapping helpers between ORM/internal/API contracts."""

from __future__ import annotations

from typing import Iterable, List

from app.domain.papers.schemas import PaperMetadata
from app.domain.papers.types import PaperDTO


def paper_dto_from_db_model(db_paper) -> PaperDTO:
    """Map a DBPaper ORM instance to the internal paper transfer type."""

    return PaperDTO.from_db_model(db_paper)


def paper_dtos_from_db_models(db_papers: Iterable) -> List[PaperDTO]:
    """Map DBPaper ORM instances to internal paper transfer types."""

    return PaperDTO.batch_from_db_models(list(db_papers))


def paper_metadata_from_db_model(db_paper) -> PaperMetadata:
    """Map a DBPaper ORM instance to the public paper metadata response."""

    return PaperMetadata.from_db_model(db_paper)


def paper_metadata_from_ranked_paper(ranked_paper) -> PaperMetadata:
    """Map a ranked paper result to the public paper metadata response."""

    return PaperMetadata.from_ranked_paper(ranked_paper)
