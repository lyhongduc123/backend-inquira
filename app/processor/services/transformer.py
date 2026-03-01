from typing import Dict, Any, Optional, List, Sequence
from app.core.dtos.paper import PaperEnrichedDTO, PaperDTO
from app.core.dtos.author import AuthorDTO
from app.retriever.schemas import NormalizedPaperResult, NormalizedAuthorResult
from app.retriever.utils import parse_date
from app.papers.schemas import PaperMetadata, SJRMetadata
from app.authors.schemas import AuthorMetadata
from app.processor.schemas import RankedPaper
from app.models.papers import DBPaper
from app.models.authors import DBAuthorPaper


class TransformerService:
    def normalized_to_paper(self, result: NormalizedPaperResult) -> PaperEnrichedDTO:
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
            from app.extensions.logger import create_logger
            logger = create_logger(__name__)
            author_names = [a.name for a in authors_dto]
            logger.debug(f"Converted {len(authors_dto)} authors for paper {result.paper_id}: {author_names}")
        
        # Convert result to dict and replace authors with AuthorDTO list
        result_dict = result.model_dump()
        result_dict['authors'] = authors_dto
        
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
                    norm = self.normalize_issn(i)
                    if norm:
                        issn_set.add(norm)
                paper.issn = list(issn_set) if issn_set else None

            if not paper.issn_l:
                norm_l = self.normalize_issn(source.get("issn_l"))
                paper.issn_l = norm_l if norm_l else None

            if paper.issn and paper.issn_l:
                break

        return paper

    def batch_normalized_to_papers(
        self, results: List[NormalizedPaperResult]
    ) -> List[PaperEnrichedDTO]:
        papers = []
        for result in results:
            try:
                paper = self.normalized_to_paper(result)
                papers.append(paper)
            except Exception as e:
                # Log error and skip problematic entry
                print(f"Error converting normalized result to Paper: {e}")
                continue
        return papers

    def dbpaper_to_paper(self, db_paper) -> PaperDTO:
        """
        Convert a database paper ORM object to a PaperDTO model.
        Maps all DBPaper fields to PaperDTO schema.
        """
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

        return PaperDTO(
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
            summary=db_paper.summary if hasattr(db_paper, "summary") else None,
            summary_embedding=(
                db_paper.summary_embedding
                if hasattr(db_paper, "summary_embedding")
                else None
            ),
            # relevance_score=(
            #     db_paper.relevance_score
            #     if hasattr(db_paper, "relevance_score")
            #     else None
            # ),
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

    def batch_dbpaper_to_papers(self, db_papers: List[DBPaper]) -> List[PaperDTO]:
        """
        Convert a list of database paper ORM objects to a list of Paper Pydantic models.
        """
        papers = []
        for db_paper in db_papers:
            try:
                paper = self.dbpaper_to_paper(db_paper)
                papers.append(paper)
            except Exception as e:
                # Log error and skip problematic entry
                print(f"Error converting DB paper to Paper: {e}")
                continue
        return papers

    def paper_to_dict(self, paper: PaperDTO) -> Dict[str, Any]:
        """
        Convert PaperDTO Pydantic model to a dictionary for JSON serialization or database storage.
        """
        return paper.model_dump()

    def batch_paper_to_dicts(self, papers: Sequence[PaperDTO | DBPaper]) -> List[Dict[str, Any]]:
        """
        Convert a list of PaperDTO Pydantic models or DBPaper models to a list of dictionaries.
        """
        from app.models.papers import DBPaper
        
        dicts = []
        for paper in papers:
            try:
                # Convert DBPaper to PaperDTO if needed
                if isinstance(paper, DBPaper):
                    paper_dto = PaperDTO.model_validate(paper)
                    paper_dict = self.paper_to_dict(paper_dto)
                else:
                    paper_dict = self.paper_to_dict(paper)
                dicts.append(paper_dict)
            except Exception as e:
                # Log error and skip problematic entry
                print(f"Error converting Paper to dict: {e}")
                continue
        return dicts
    
    def dbpaper_to_metadata(self, db_paper: DBPaper) -> PaperMetadata:
        """Convert a DBPaper to PaperMetadata"""
        if not db_paper:
            raise ValueError("DBPaper is None")
        authors = [AuthorMetadata.model_validate(author_paper.author) for author_paper in db_paper.paper_authors]
        year = db_paper.publication_date.year if db_paper.publication_date else None
        
        paper_metadata = PaperMetadata.model_validate(db_paper)
        paper_metadata.authors = authors
        paper_metadata.year = year
        paper_metadata.sjr_data = SJRMetadata.model_validate(db_paper.journal) if db_paper.journal else None
        return paper_metadata

    def ranked_paper_to_metadata(self, ranked_paper: RankedPaper) -> PaperMetadata:
        """
        Convert paper DBPaper to consistent metadata dictionary.
        This is the SINGLE SOURCE OF TRUTH for paper metadata format.
        Used for both streaming and snapshot storage in messages.

        Args:
            ranked_paper: RankedPaper model

        Returns:
            Consistent paper metadata dictionary with all fields
        """
        # Extract data from either Pydantic schema or SQLAlchemy model
        paper = ranked_paper.paper
        year = paper.publication_date.year if paper.publication_date else None
        
        # Convert DBAuthorPaper ORM objects to dicts for validation
        authors = []
        for author_paper in paper.paper_authors:
            author_dict = {
                "author_id": author_paper.author.author_id if author_paper.author else None,
                "name": author_paper.author.name if author_paper.author else None,
                "author_position": author_paper.author_position,
            }
            authors.append(AuthorMetadata.model_validate(author_dict))
        
        # Validate paper but exclude the journal ORM relationship to avoid serialization issues
        paper_metadata = PaperMetadata.model_validate(paper, from_attributes=True)
        paper_metadata.journal = None  # Clear ORM object
        paper_metadata.authors = authors
        paper_metadata.year = year
        paper_metadata.relevance_score = ranked_paper.relevance_score
        paper_metadata.ranking_scores = ranked_paper.ranking_scores
        if not paper.journal:
            paper_metadata.sjr_data = None
        else:
            paper_metadata.sjr_data = SJRMetadata.model_validate(paper.journal)
        return paper_metadata

    def normalize_issn(self, issn: Optional[str]) -> Optional[str]:
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

    def extract_authors_from_normalized(
        self, result: NormalizedPaperResult
    ) -> List[Dict[str, Any]]:
        """
        Extract enriched author data from NormalizedPaperResult.
        Authors field contains merged data from Semantic Scholar and OpenAlex.

        Args:
            result: NormalizedPaperResult with merged authors

        Returns:
            List of author dicts ready for AuthorService.upsert_from_openalex
        """
        enriched_authors = []

        # Authors are already merged with S2 stats and OA institutions
        for author_data in result.authors:
            if isinstance(author_data, dict):
                author_info = author_data
            else:
                # If it's an AuthorSchema object, convert to dict
                author_info = (
                    author_data.model_dump()
                    if hasattr(author_data, "model_dump")
                    else author_data.__dict__
                )

            # Convert to authorship format expected by downstream services
            authorship = {
                "author": {
                    "id": author_info.get("author_id", ""),
                    "display_name": author_info.get("name", ""),
                    "orcid": author_info.get("orcid"),
                },
                "institutions": author_info.get("institutions", []),
            }

            # S2 stats are already in the author data
            s2_stats = {
                "author_id": author_info.get("author_id"),
                "h_index": author_info.get("h_index"),
                "citation_count": author_info.get("citation_count"),
                "paper_count": author_info.get("paper_count"),
            }

            enriched_author = {
                "authorship": authorship,
                "s2_stats": s2_stats,
            }
            enriched_authors.append(enriched_author)

        return enriched_authors

    def extract_institutions_from_normalized(
        self, result: NormalizedPaperResult
    ) -> List[Dict[str, Any]]:
        """
        Extract unique institution data from NormalizedPaperResult.
        Authors field contains merged institution data from OpenAlex.

        Args:
            result: NormalizedPaperResult with merged authors containing institution data

        Returns:
            List of unique institution dicts for InstitutionService.upsert_from_openalex
        """
        institutions_map = {}  # Deduplicate by institution ID

        for author_data in result.authors:
            if isinstance(author_data, dict):
                author_info = author_data
            else:
                author_info = (
                    author_data.model_dump()
                    if hasattr(author_data, "model_dump")
                    else author_data.__dict__
                )

            institutions = author_info.get("institutions", [])
            for institution in institutions:
                inst_id_url = institution.get("id")
                if not inst_id_url:
                    continue

                # Use ID as deduplication key
                if inst_id_url not in institutions_map:
                    institutions_map[inst_id_url] = institution

        return list(institutions_map.values())

    def build_author_institution_links(
        self,
        result: NormalizedPaperResult,
        author_db_ids: Dict[str, int],
        institution_db_ids: Dict[str, int],
    ) -> List[Dict[str, Any]]:
        """
        Build author-institution link data from merged author information.

        Args:
            result: NormalizedPaperResult with merged authors
            author_db_ids: Mapping of author_id to database ID
            institution_db_ids: Mapping of institution_id to database ID

        Returns:
            List of dicts for creating DBAuthorInstitution records
        """
        links = []

        if not result.authors:
            return links

        for author_data in result.authors:
            if isinstance(author_data, dict):
                author_info = author_data
            else:
                author_info = (
                    author_data.model_dump()
                    if hasattr(author_data, "model_dump")
                    else author_data.__dict__
                )

            author_id = author_info.get("author_id", "")
            if not author_id:
                continue

            # Extract OpenAlex author ID
            openalex_author_id = (
                author_id.split("/")[-1] if "/" in author_id else author_id
            )

            # Find corresponding DB author ID
            db_author_id = author_db_ids.get(openalex_author_id)
            if not db_author_id:
                continue

            # Link to each institution
            institutions = author_info.get("institutions", [])
            for institution in institutions:
                inst_id_url = institution.get("id")
                if not inst_id_url:
                    continue

                inst_id = (
                    inst_id_url.split("/")[-1] if "/" in inst_id_url else inst_id_url
                )
                db_inst_id = institution_db_ids.get(inst_id)

                if db_inst_id:
                    links.append(
                        {
                            "author_id": db_author_id,
                            "institution_id": db_inst_id,
                            "is_current": True,  # Assume current unless we have temporal data
                            "confidence": 1.0,
                        }
                    )

        return links

    def s2_author_paper_to_paper_dto(
        self, paper_data: Dict[str, Any]
    ) -> PaperEnrichedDTO:
        """
        Convert Semantic Scholar author paper response to PaperEnrichedDTO.

        Args:
            paper_data: Raw paper data from S2 /author/{id}/papers endpoint

        Returns:
            PaperEnrichedDTO object
        """
        # Extract authors
        authors = []
        for author in paper_data.get("authors", []):
            authors.append(
                AuthorDTO(
                    name=author.get("name", "Unknown"),
                    author_id=author.get("authorId"),
                    h_index=author.get("hIndex"),
                    citation_count=author.get("citationCount"),
                    paper_count=author.get("paperCount"),
                )
            )

        return PaperEnrichedDTO(
            paper_id=paper_data.get("paperId") or "",
            title=paper_data.get("title", "Untitled"),
            abstract=paper_data.get("abstract", ""),
            authors=authors,
            publication_date=paper_data.get("publicationDate"),
            venue=paper_data.get("venue"),
            citation_count=paper_data.get("citationCount", 0),
            reference_count=paper_data.get("referenceCount", 0),
            is_open_access=paper_data.get("isOpenAccess", False),
            open_access_pdf=paper_data.get("openAccessPdf"),
            external_ids=paper_data.get("externalIds", {}),
            source="semanticscholar",
        )

    def openalex_author_work_to_paper_dto(
        self, paper_data: Dict[str, Any]
    ) -> PaperEnrichedDTO:
        """
        Convert OpenAlex author work response to PaperEnrichedDTO.

        Args:
            paper_data: Raw work data from OpenAlex /works?filter=author.id:{id}

        Returns:
            PaperEnrichedDTO object
        """
        # Extract authors
        authors = []
        for authorship in paper_data.get("authorships", []):
            author_info = authorship.get("author", {})
            authors.append(
                AuthorDTO(
                    name=author_info.get("display_name", "Unknown"),
                    author_id=author_info.get("id", "").split("/")[-1],
                    orcid=author_info.get("orcid"),
                )
            )

        # Extract OpenAlex ID
        openalex_id = paper_data.get("id", "").split("/")[-1]

        # Build external IDs
        external_ids = {}
        if openalex_id:
            external_ids["openalex"] = openalex_id
        if paper_data.get("doi"):
            external_ids["DOI"] = paper_data["doi"]

        return PaperEnrichedDTO(
            paper_id=openalex_id,
            title=paper_data.get("title", "Untitled"),
            abstract=paper_data.get("abstract", ""),
            authors=authors,
            publication_date=paper_data.get("publication_date"),
            venue=paper_data.get("primary_location", {})
            .get("source", {})
            .get("display_name"),
            citation_count=paper_data.get("cited_by_count", 0),
            is_open_access=paper_data.get("open_access", {}).get("is_oa", False),
            external_ids=external_ids,
            source="openalex",
        )

    def author_api_response_to_paper_dto(
        self, paper_data: Dict[str, Any], source: Optional[str] = None
    ) -> PaperEnrichedDTO:
        """
        Convert author API response (S2 or OpenAlex) to PaperEnrichedDTO.
        Auto-detects source if not specified.

        Args:
            paper_data: Raw paper data from API
            source: Optional source hint ('semanticscholar' or 'openalex')

        Returns:
            PaperEnrichedDTO object
        """
        # Auto-detect source if not provided
        if not source:
            source = "semanticscholar" if "paperId" in paper_data else "openalex"

        if source == "semanticscholar":
            return self.s2_author_paper_to_paper_dto(paper_data)
        else:
            return self.openalex_author_work_to_paper_dto(paper_data)

    def batch_author_papers_to_dtos(
        self, papers_data: List[Dict[str, Any]], source: Optional[str] = None
    ) -> List[PaperEnrichedDTO]:
        """
        Convert batch of author papers from API to PaperEnrichedDTO list.

        Args:
            papers_data: List of raw paper data from API
            source: Optional source hint ('semanticscholar' or 'openalex')

        Returns:
            List of PaperEnrichedDTO objects
        """
        papers = []
        for paper_data in papers_data:
            try:
                paper_dto = self.author_api_response_to_paper_dto(paper_data, source)
                papers.append(paper_dto)
            except Exception as e:
                # Log error and skip problematic entry
                print(f"Error converting author paper to DTO: {e}")
                continue
        return papers
