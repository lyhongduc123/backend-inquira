"""
Paper DTOs - Clean data transfer objects following single responsibility principle
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field
from app.core.dtos.author import AuthorDTO


class PaperDTO(BaseModel):
    """
    Internal paper DTO for data transfer between layers.
    Aligned with DBPaper model structure.
    Source of truth for paper data transfer.
    """

    # Core identifiers
    paper_id: str
    title: str
    abstract: Optional[str] = None
    embedding: Optional[List[float]] = None
    authors: List[AuthorDTO] = []
    publication_date: Optional[datetime] = None
    venue: Optional[str] = None
    issn: Optional[List[str]] = None
    issn_l: Optional[str] = None
    url: Optional[str] = None
    pdf_url: Optional[str] = None
    is_open_access: bool = False
    open_access_pdf: Optional[Dict[str, Any]] = None
    source: str = "SemanticScholar"
    external_ids: Optional[Dict[str, Any]] = None
    tldr: Optional[str] = None  # Renamed from summary
    tldr_embedding: Optional[List[float]] = None  # Renamed from summary_embedding
    citation_count: Optional[int] = 0
    influential_citation_count: Optional[int] = 0
    reference_count: Optional[int] = 0
    citation_styles: Optional[Dict[str, str]] = None
    topics: Optional[List[Dict[str, Any]]] = None
    keywords: Optional[List[Dict[str, Any]]] = None
    concepts: Optional[List[Dict[str, Any]]] = None
    mesh_terms: Optional[List[Dict[str, Any]]] = None
    citation_percentile: Optional[Dict[str, Any]] = None
    fwci: Optional[float] = None
    author_trust_score: Optional[float] = None
    institutional_trust_score: Optional[float] = None
    network_diversity_score: Optional[float] = None
    journal_id: Optional[int] = None
    is_retracted: bool = False
    language: Optional[str] = None
    year: Optional[int] = None  # NEW: S2 publication year
    fields_of_study: Optional[List[str]] = None  # NEW: S2 fields
    s2_fields_of_study: Optional[List[Dict[str, str]]] = None  # NEW: S2 enriched fields
    corresponding_author_ids: Optional[List[str]] = None
    institutions_distinct_count: Optional[int] = None
    countries_distinct_count: Optional[int] = None
    is_processed: bool = False
    processing_status: str = "pending"
    processing_error: Optional[str] = None

    # Database fields (optional, populated when reading from DB)
    id: Optional[int] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    last_accessed_at: Optional[datetime] = None

    class Config:
        from_attributes = True
        extra = "ignore"

    @classmethod
    def from_db_model(cls, db_paper) -> "PaperDTO":
        """
        Convert a database paper ORM object to a PaperDTO model.
        Maps all DBPaper fields to PaperDTO schema.
        """
        from app.core.dtos.author import AuthorDTO

        authors = []
        if hasattr(db_paper, "authors") and db_paper.authors:
            for author in db_paper.authors:
                if isinstance(author, dict):
                    authors.append(
                        AuthorDTO(
                            name=author.get("name", ""),
                            author_id=author.get("author_id"),
                            citation_count=author.get("citation_count"),
                            h_index=author.get("h_index"),
                        )
                    )
                else:
                    authors.append(
                        AuthorDTO(
                            name=author.name,
                            author_id=author.author_id,
                            citation_count=author.citation_count,
                            h_index=author.h_index,
                        )
                    )

        publication_date = None
        if hasattr(db_paper, "publication_date") and db_paper.publication_date:
            publication_date = db_paper.publication_date

        return cls(
            id=db_paper.id if hasattr(db_paper, "id") else None,
            paper_id=str(db_paper.paper_id),
            title=db_paper.title,
            authors=authors,
            abstract=db_paper.abstract,
            publication_date=publication_date,
            venue=db_paper.venue if hasattr(db_paper, "venue") else None,
            issn=db_paper.issn if hasattr(db_paper, "issn") else None,
            issn_l=db_paper.issn_l if hasattr(db_paper, "issn_l") else None,
            url=db_paper.url if hasattr(db_paper, "url") else None,
            pdf_url=db_paper.pdf_url if hasattr(db_paper, "pdf_url") else None,
            is_open_access=(
                db_paper.is_open_access
                if hasattr(db_paper, "is_open_access")
                else False
            ),
            open_access_pdf=(
                db_paper.open_access_pdf
                if hasattr(db_paper, "open_access_pdf")
                else None
            ),
            source=db_paper.source,
            external_ids=(
                db_paper.external_ids if hasattr(db_paper, "external_ids") else {}
            ),
            tldr=db_paper.tldr if hasattr(db_paper, "tldr") else None,
            tldr_embedding=(
                db_paper.tldr_embedding if hasattr(db_paper, "tldr_embedding") else None
            ),
            year=db_paper.year if hasattr(db_paper, "year") else None,
            fields_of_study=(
                db_paper.fields_of_study
                if hasattr(db_paper, "fields_of_study")
                else None
            ),
            s2_fields_of_study=(
                db_paper.s2_fields_of_study
                if hasattr(db_paper, "s2_fields_of_study")
                else None
            ),
            citation_count=(
                db_paper.citation_count if hasattr(db_paper, "citation_count") else 0
            ),
            influential_citation_count=(
                db_paper.influential_citation_count
                if hasattr(db_paper, "influential_citation_count")
                else 0
            ),
            reference_count=(
                db_paper.reference_count if hasattr(db_paper, "reference_count") else 0
            ),
            topics=db_paper.topics if hasattr(db_paper, "topics") else None,
            keywords=db_paper.keywords if hasattr(db_paper, "keywords") else None,
            concepts=db_paper.concepts if hasattr(db_paper, "concepts") else None,
            mesh_terms=db_paper.mesh_terms if hasattr(db_paper, "mesh_terms") else None,
            citation_percentile=(
                db_paper.citation_percentile
                if hasattr(db_paper, "citation_percentile")
                else None
            ),
            fwci=db_paper.fwci if hasattr(db_paper, "fwci") else None,
            author_trust_score=(
                db_paper.author_trust_score
                if hasattr(db_paper, "author_trust_score")
                else None
            ),
            institutional_trust_score=(
                db_paper.institutional_trust_score
                if hasattr(db_paper, "institutional_trust_score")
                else None
            ),
            network_diversity_score=(
                db_paper.network_diversity_score
                if hasattr(db_paper, "network_diversity_score")
                else None
            ),
            journal_id=db_paper.journal_id if hasattr(db_paper, "journal_id") else None,
            is_retracted=(
                db_paper.is_retracted if hasattr(db_paper, "is_retracted") else False
            ),
            language=db_paper.language if hasattr(db_paper, "language") else None,
            corresponding_author_ids=(
                db_paper.corresponding_author_ids
                if hasattr(db_paper, "corresponding_author_ids")
                else None
            ),
            institutions_distinct_count=(
                db_paper.institutions_distinct_count
                if hasattr(db_paper, "institutions_distinct_count")
                else None
            ),
            countries_distinct_count=(
                db_paper.countries_distinct_count
                if hasattr(db_paper, "countries_distinct_count")
                else None
            ),
            is_processed=(
                db_paper.is_processed if hasattr(db_paper, "is_processed") else False
            ),
            processing_status=(
                db_paper.processing_status
                if hasattr(db_paper, "processing_status")
                else "pending"
            ),
            processing_error=(
                db_paper.processing_error
                if hasattr(db_paper, "processing_error")
                else None
            ),
            created_at=db_paper.created_at if hasattr(db_paper, "created_at") else None,
            updated_at=db_paper.updated_at if hasattr(db_paper, "updated_at") else None,
            last_accessed_at=(
                db_paper.last_accessed_at
                if hasattr(db_paper, "last_accessed_at")
                else None
            ),
        )

    @classmethod
    def batch_from_db_models(cls, db_papers: List) -> List["PaperDTO"]:
        """
        Convert a list of database paper ORM objects to a list of Paper Pydantic models.
        """
        papers = []
        for db_paper in db_papers:
            try:
                paper = cls.from_db_model(db_paper)
                papers.append(paper)
            except Exception as e:
                # Log error and skip problematic entry
                from app.extensions.logger import create_logger

                logger = create_logger(__name__)
                logger.error(f"Error converting DB paper to Paper: {e}")
                continue
        return papers


class PaperEnrichedDTO(PaperDTO):
    """
    Extended paper DTO with enrichment data for preprocessing.
    Used in retrieval and processing pipeline.
    Authors field contains merged data from both Semantic Scholar and OpenAlex.
    """

    has_content: Dict[str, bool] = Field(default_factory=dict, exclude=True)
    references: Optional[List[Dict[str, Any]]] = None
