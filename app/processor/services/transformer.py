from typing import Dict, Any, Optional, List
from app.retriever.paper_schemas import Paper, Author, PaperResponse
from app.retriever.schemas import NormalizedResult
from app.retriever.utils import parse_date

from app.models.papers import DBPaper


class TransformerService:
    def normalized_to_paper(self, result: NormalizedResult) -> Paper:
        """
        Convert a provider NormalizedResult dict to a Paper Pydantic model.

        Stores complex nested data (biblio, primary_location, locations, authorships)
        in openalex_data or semantic_data JSONB fields to match DBPaper schema.
        """
        paper = Paper.model_validate(result)
        locations = [result.primary_location, result.best_oa_location, result.locations]
        for loc in locations:
            source = loc.get("source") if loc else None
            if not paper.issn:
                paper.issn = source.get("issn") if source else None
            if not paper.issn_l:
                paper.issn_l = source.get("issn_l") if source else None
                
            if paper.issn and paper.issn_l:
                break
            
        return paper

    def batch_normalized_to_papers(
        self, results: List[NormalizedResult]
    ) -> List[Paper]:
        """
        Convert a list of NormalizedResult dicts to a list of Paper Pydantic models.
        """
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

    def dbpaper_to_paper(self, db_paper) -> Paper:
        """
        Convert a database paper ORM object to a Paper Pydantic model.
        Maps all DBPaper fields to Paper schema.
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
            relevance_score=(
                db_paper.relevance_score
                if hasattr(db_paper, "relevance_score")
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
            authorships=None,  # Not stored in DBPaper
            created_at=db_paper.created_at if hasattr(db_paper, "created_at") else None,
            updated_at=db_paper.updated_at if hasattr(db_paper, "updated_at") else None,
            last_accessed_at=(
                db_paper.last_accessed_at
                if hasattr(db_paper, "last_accessed_at")
                else None
            ),
            semantic_authors=authors,
        )

    def batch_dbpaper_to_papers(self, db_papers: List[DBPaper]) -> List[Paper]:
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

    def paper_to_dict(self, paper: Paper) -> Dict[str, Any]:
        """
        Convert a Paper Pydantic model to a dictionary suitable for database insertion.
        Only includes fields that exist in the DBPaper model.
        """
        # Fields that exist in DBPaper model
        db_fields = {
            "paper_id",
            "title",
            "abstract",
            "authors",
            "publication_date",
            "venue",
            "issn",
            "issn_l",
            "url",
            "pdf_url",
            "is_open_access",
            "open_access_pdf",
            "source",
            "external_ids",
            "summary",
            "summary_embedding",
            "relevance_score",
            "citation_count",
            "influential_citation_count",
            "reference_count",
            "topics",
            "keywords",
            "concepts",
            "mesh_terms",
            "citation_percentile",
            "fwci",
            "author_trust_score",
            "institutional_trust_score",
            "network_diversity_score",
            "journal_id",
            "is_retracted",
            "language",
            "corresponding_author_ids",
            "institutions_distinct_count",
            "countries_distinct_count",
            "is_processed",
            "processing_status",
            "processing_error",
        }

        # Convert to dict and filter to only DB fields
        paper_dict = paper.model_dump()
        return {k: v for k, v in paper_dict.items() if k in db_fields}

    def batch_paper_to_dicts(self, papers: List[Paper]) -> List[Dict[str, Any]]:
        """
        Convert a list of Paper Pydantic models to a list of dictionaries.
        """
        dicts = []
        for paper in papers:
            try:
                paper_dict = self.paper_to_dict(paper)
                dicts.append(paper_dict)
            except Exception as e:
                # Log error and skip problematic entry
                print(f"Error converting Paper to dict: {e}")
                continue
        return dicts

    def paper_to_response(self, paper: Paper) -> PaperResponse:
        """
        Convert internal Paper DTO to PaperResponse DTO for frontend API.

        Extracts year and external IDs for API consumption.
        """
        # Extract external IDs for convenient access
        external_ids = paper.external_ids or {}
        openalex_id = external_ids.get("OpenAlex") or external_ids.get("openalex")
        semantic_scholar_id = external_ids.get("CorpusId") or external_ids.get(
            "semanticscholar"
        )
        doi = external_ids.get("DOI") or external_ids.get("doi")
        pmid = external_ids.get("PubMed") or external_ids.get("pmid")
        arxiv_id = external_ids.get("arXiv") or external_ids.get("arxiv")

        # Extract year from publication_date
        year = None
        if paper.publication_date:
            year = (
                paper.publication_date.year
                if hasattr(paper.publication_date, "year")
                else None
            )

        return PaperResponse(
            paper_id=paper.paper_id,
            title=paper.title,
            authors=paper.authors,
            abstract=paper.abstract,
            publication_date=paper.publication_date,
            venue=paper.venue,
            year=year,
            url=paper.url,
            pdf_url=paper.pdf_url,
            is_open_access=paper.is_open_access,
            source=paper.source,
            openalex_id=openalex_id,
            doi=doi,
            citation_count=paper.citation_count,
            influential_citation_count=paper.influential_citation_count,
            relevance_score=paper.relevance_score,
            topics=paper.topics,
            keywords=paper.keywords,
            concepts=paper.concepts,
            fwci=paper.fwci,
            is_retracted=paper.is_retracted,
        )

    def batch_papers_to_responses(self, papers: List[Paper]) -> List[PaperResponse]:
        """
        Convert list of Paper DTOs to PaperResponse DTOs for frontend API.
        """
        responses = []
        for paper in papers:
            try:
                response = self.paper_to_response(paper)
                responses.append(response)
            except Exception as e:
                print(f"Error converting Paper to PaperResponse: {e}")
                continue
        return responses

    def extract_authors_from_normalized(
        self, result: NormalizedResult
    ) -> List[Dict[str, Any]]:
        """
        Extract enriched author data from NormalizedResult.
        
        Combines Semantic Scholar author stats with OpenAlex authorship metadata.
        Semantic Scholar provides: h_index, citation_count, paper_count
        OpenAlex provides: institutions, ORCID, position, affiliations
        
        Args:
            result: NormalizedResult with both semantic_authors and authorships
            
        Returns:
            List of author dicts ready for AuthorService.upsert_from_openalex
        """
        enriched_authors = []
        
        # Build lookup for Semantic Scholar stats by author name (case-insensitive)
        s2_stats_by_name = {}
        if result.semantic_authors:
            for s2_author in result.semantic_authors:
                name = s2_author.get("name", "").lower().strip()
                if name:
                    s2_stats_by_name[name] = {
                        "author_id": s2_author.get("author_id"),
                        "h_index": s2_author.get("h_index"),
                        "citation_count": s2_author.get("citation_count"),
                        "paper_count": s2_author.get("paper_count"),
                    }
        
        # Process OpenAlex authorships (richer metadata)
        if result.authorships:
            for authorship in result.authorships:
                author_info = authorship.get("author", {})
                if not author_info:
                    continue
                
                # Match with Semantic Scholar stats by name
                author_name = author_info.get("display_name", "")
                s2_stats = s2_stats_by_name.get(author_name.lower().strip())
                
                # Build enriched author with both S2 stats and OA metadata
                enriched_author = {
                    "authorship": authorship,  # Full OpenAlex authorship object
                    "s2_stats": s2_stats,  # Semantic Scholar stats (if matched)
                }
                enriched_authors.append(enriched_author)
        elif result.authors:
            # Fallback: if no authorships but have basic authors, use those
            for author in result.authors:
                author_name = author.name
                s2_stats = s2_stats_by_name.get(author_name.lower().strip())
                
                # Create minimal authorship-like structure
                authorship = {
                    "author": {
                        "id": author.author_id if hasattr(author, 'author_id') else None,
                        "display_name": author_name,
                        "orcid": author.orcid if hasattr(author, 'orcid') else None,
                    },
                    "institutions": author.institutions if hasattr(author, 'institutions') else [],
                    "author_position": None,
                    "is_corresponding": False,
                }
                
                enriched_author = {
                    "authorship": authorship,
                    "s2_stats": s2_stats,
                }
                enriched_authors.append(enriched_author)
        
        return enriched_authors
    
    def extract_institutions_from_normalized(
        self, result: NormalizedResult
    ) -> List[Dict[str, Any]]:
        """
        Extract unique institution data from NormalizedResult authorships.
        
        OpenAlex provides institution data in each authorship's institutions array.
        We deduplicate by institution ID and return list ready for InstitutionService.
        
        Args:
            result: NormalizedResult with authorships containing institution data
            
        Returns:
            List of unique institution dicts for InstitutionService.upsert_from_openalex
        """
        institutions_map = {}  # Deduplicate by institution ID
        
        if result.authorships:
            for authorship in result.authorships:
                institutions = authorship.get("institutions", [])
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
        result: NormalizedResult,
        author_db_ids: Dict[str, int],
        institution_db_ids: Dict[str, int]
    ) -> List[Dict[str, Any]]:
        """
        Build author-institution link data from authorship information.
        
        Args:
            result: NormalizedResult with authorships
            author_db_ids: Mapping of author_id (primary) to database ID
            institution_db_ids: Mapping of institution_id to database ID
            
        Returns:
            List of dicts for creating DBAuthorInstitution records
        """
        links = []
        
        if not result.authorships:
            return links
        
        for authorship in result.authorships:
            author_info = authorship.get("author", {})
            author_id_url = author_info.get("id")
            if not author_id_url:
                continue
            
            # Extract OpenAlex author ID
            openalex_author_id = author_id_url.split("/")[-1] if "/" in author_id_url else author_id_url
            
            # Find corresponding DB author ID (could be S2 ID or OpenAlex ID)
            db_author_id = author_db_ids.get(openalex_author_id)
            if not db_author_id:
                continue
            
            # Link to each institution
            institutions = authorship.get("institutions", [])
            for institution in institutions:
                inst_id_url = institution.get("id")
                if not inst_id_url:
                    continue
                
                inst_id = inst_id_url.split("/")[-1] if "/" in inst_id_url else inst_id_url
                db_inst_id = institution_db_ids.get(inst_id)
                
                if db_inst_id:
                    links.append({
                        "author_id": db_author_id,
                        "institution_id": db_inst_id,
                        "is_current": True,  # Assume current unless we have temporal data
                        "confidence": 1.0,
                    })
        
        return links
