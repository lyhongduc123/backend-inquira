"""Paper service for business logic"""

from typing import List, Optional, Dict, Union, TYPE_CHECKING, Any, Tuple
from .schemas import (
    PaperUpdateRequest,
    PaperDetailResponse,
)
from app.models.papers import DBPaper
from app.core.dtos.paper import PaperDTO, PaperEnrichedDTO
from app.extensions.logger import create_logger
from app.extensions.bibliography import bibtex_to_multiple_styles
from sqlalchemy import select
from app.core.exceptions import NotFoundException
from app.retriever.provider.semantic_scholar_provider import SemanticScholarProvider
from datetime import datetime, date
from sqlalchemy.dialects.postgresql import insert as pg_insert
from app.models.authors import DBAuthor

if TYPE_CHECKING:
    from .repository import PaperRepository, LoadOptions

logger = create_logger(__name__)


class PaperService:
    """Service for paper operations"""

    def __init__(
        self,
        repository: "PaperRepository",
        retriever_service,
        author_service=None,
        institution_service=None,
        journal_service=None,
    ):
        self.repository = repository
        self.retriever_service = retriever_service

        self._author_service = author_service
        self._institution_service = institution_service
        self._journal_service = journal_service

    @property
    def author_service(self):
        """Lazy load author service"""
        if self._author_service is None:
            from app.domain.authors.service import AuthorService

            self._author_service = AuthorService(self.repository.db)
        return self._author_service

    @property
    def institution_service(self):
        """Lazy load institution service"""
        if self._institution_service is None:
            from app.domain.institutions.service import InstitutionService

            self._institution_service = InstitutionService(db=self.repository.db)
        return self._institution_service

    @property
    def journal_service(self):
        """Lazy load journal service"""
        if self._journal_service is None:
            from app.domain.papers.journal_service import JournalService

            self._journal_service = JournalService(self.repository.db)
        return self._journal_service

    async def batch_check_existing_papers(
        self, paper_ids: List[str]
    ) -> Dict[str, bool]:
        """
        Efficiently check which papers already exist in database.
        Reduces N queries to 1 query.

        Args:
            paper_ids: List of paper IDs to check

        Returns:
            Dict mapping paper_id -> exists (True/False)
        """
        if not paper_ids:
            return {}

        stmt = select(DBPaper.paper_id).where(DBPaper.paper_id.in_(paper_ids))
        result = await self.repository.db.execute(stmt)
        existing_ids = set(result.scalars().all())

        return {pid: pid in existing_ids for pid in paper_ids}

    async def enrich_paper_with_authors_institutions(
        self, db_paper: DBPaper, authors: List[Dict]
    ):
        """
        Enrich paper with author and institution data.
        Orchestrates author and institution services to link them to the paper.

        Args:
            db_paper: Database paper entity
            authors: List of merged author objects with S2 stats and OA institutions
        """
        if not authors:
            logger.debug(f"No authors data for paper {db_paper.paper_id}")
            return

        # Extract publication year for author-institution tracking
        pub_year = None
        if db_paper.publication_date:
            if isinstance(db_paper.publication_date, (datetime, date)):
                pub_year = db_paper.publication_date.year
            elif isinstance(db_paper.publication_date, int):
                pub_year = db_paper.publication_date

        for position, author_data in enumerate(authors, start=1):
            # Directly upsert from merged author data
            db_author = await self.author_service.upsert_from_merged_author(author_data)
            if not db_author:
                continue

            # Process institutions for this author
            institutions = author_data.get("institutions", [])
            institution_id = None

            if institutions:
                # Process first institution (primary affiliation)
                primary_institution = institutions[0]
                db_institution = await self.institution_service.upsert_from_openalex(
                    primary_institution
                )

                if db_institution:
                    institution_id = db_institution.id

                    # Link author to institution
                    await self.author_service.link_author_to_institution(
                        author=db_author,
                        institution_id=db_institution.id,
                        year=pub_year,
                        is_current=False,
                    )

            # Link author to paper with position
            await self.author_service.link_author_to_paper(
                author=db_author,
                paper_id=db_paper.id,
                author_data=author_data,
                institution_id=institution_id,
                author_position=position,
            )

        logger.info(
            f"Enriched paper {db_paper.paper_id} with {len(authors)} authors and their institutions"
        )

    async def enrich_paper_with_journal(self, db_paper: DBPaper) -> None:
        """
        Enrich paper with journal data by linking to SJR database.

        Args:
            db_paper: Database paper entity to enrich
        """
        try:
            journal = await self.journal_service.enrich_paper_with_journal(
                paper=db_paper,
                venue=db_paper.venue,
                issn=(
                    db_paper.issn[0]
                    if db_paper.issn and len(db_paper.issn) > 0
                    else None
                ),
                issn_l=db_paper.issn_l,
            )

            if journal:
                logger.info(
                    f"Paper {db_paper.paper_id} linked to journal: {journal.title} (Q{journal.sjr_best_quartile}, SJR: {journal.sjr_score})"
                )
            else:
                logger.debug(f"No journal match for paper {db_paper.paper_id}")

        except Exception as e:
            logger.error(f"Error enriching paper {db_paper.paper_id} with journal: {e}")

    async def batch_check_processed_papers(
        self, paper_ids: List[str]
    ) -> Dict[str, bool]:
        """
        Efficiently check which papers are already processed.
        Reduces N queries to 1 query.

        Args:
            paper_ids: List of paper IDs to check

        Returns:
            Dict mapping paper_id -> is_processed (True/False)
        """
        if not paper_ids:
            return {}

        stmt = select(DBPaper.paper_id, DBPaper.is_processed).where(
            DBPaper.paper_id.in_(paper_ids)
        )
        result = await self.repository.db.execute(stmt)
        processed_map = {row[0]: row[1] for row in result.all()}

        # Return False for papers that don't exist yet
        return {pid: processed_map.get(pid, False) for pid in paper_ids}

    async def get_paper(
        self, paper_id: str, load_options: Optional["LoadOptions"] = None
    ) -> Optional[PaperDetailResponse]:
        """Get a single paper by paper_id"""
        paper = await self.repository.get_single_paper(
            paper_id, load_options=load_options
        )
        if not paper:
            return None
        return PaperDetailResponse.model_validate(paper)

    async def get_paper_by_external_ids(
        self, external_ids: dict, source: str
    ) -> Optional[DBPaper]:
        """
        Check if paper exists in database by external IDs and source.
        Returns raw DBPaper model for repository-level operations.

        Args:
            external_ids: Dictionary of external identifiers (DOI, ArXiv, etc.)
            source: Source name (e.g., 'SemanticScholar', 'OpenAlex')

        Returns:
            DBPaper if exists, None otherwise
        """
        return await self.repository.get_paper_by_external_ids(external_ids, source)

    async def get_paper_by_db_id(self, id: int) -> Optional[PaperDetailResponse]:
        """Get a single paper by database ID"""
        paper = await self.repository.get_paper_by_db_id(id)
        if not paper:
            return None
        return PaperDetailResponse.model_validate(paper)

    async def list_papers(
        self,
        page: int = 1,
        page_size: int = 20,
        processed_only: bool = False,
        source: Optional[str] = None,
        load_options: Optional["LoadOptions"] = None,
    ) -> tuple[List[PaperDetailResponse], int]:
        """List papers with pagination"""
        skip = (page - 1) * page_size
        papers, total = await self.repository.get_papers(
            skip=skip,
            limit=page_size,
            processed_only=processed_only,
            source=source,
            load_options=load_options,
        )

        paper_summaries = [PaperDetailResponse.model_validate(p) for p in papers]
        return paper_summaries, total

    async def update_paper(
        self, paper_id: str, update_data: PaperUpdateRequest
    ) -> Optional[PaperDetailResponse]:
        """Update a paper"""
        # Filter out None values
        data = update_data.model_dump(exclude_unset=True)
        if not data:
            return await self.get_paper(paper_id)

        paper = await self.repository.update_paper(paper_id, data)
        if not paper:
            return None

        return PaperDetailResponse.model_validate(paper)

    async def delete_paper(self, paper_id: str) -> bool:
        """
        Delete a paper.
        Note: Chunks should be deleted separately via ChunkService before calling this.
        """
        return await self.repository.delete_paper(paper_id)

    async def create_paper_from_schema(
        self, paper: Union[PaperDTO, PaperEnrichedDTO], defer_enrichment: bool = False
    ) -> Optional[DBPaper]:
        """
        Create paper from Paper DTO (skips if already exists).
        Handles DTO-to-DBPaper model transformation and enrichment orchestration.

        Enrichment steps:
        1. Save paper to database (skip if exists)
        2. Enrich with authors and institutions (can be deferred)
        3. Enrich with journal data (can be deferred)

        Args:
            paper: PaperDTO or PaperEnrichedDTO from retriever
            defer_enrichment: If True, skip author/journal enrichment (for batch operations)

        Returns:
            Created DBPaper object, or None if paper already exists
        """

        try:
            create_schema = self._dto_to_model(paper)
            db_paper = await self.repository.save_paper(create_schema)

            if db_paper and not defer_enrichment:
                if paper.authors:
                    try:
                        authors_dict: List[Dict] = [
                            (
                                author.model_dump()
                                if not isinstance(author, dict)
                                else author
                            )
                            for author in paper.authors
                        ]
                        await self.enrich_paper_with_authors_institutions(
                            db_paper, authors_dict
                        )
                    except Exception as e:
                        logger.error(
                            f"Failed to enrich paper {paper.paper_id} with authors: {e}"
                        )
                else:
                    logger.warning(
                        f"No authors found for paper {paper.paper_id} during enrichment"
                    )

                # Enrich with journal data
                try:
                    await self.enrich_paper_with_journal(db_paper)
                except Exception as e:
                    logger.error(
                        f"Failed to enrich paper {paper.paper_id} with journal: {e}"
                    )

            return db_paper

        except Exception as e:
            logger.error(f"Error creating paper {paper.paper_id}: {e}")
            try:
                await self.repository.rollback()
            except Exception as rollback_error:
                logger.error(f"Failed to rollback transaction: {rollback_error}")
            raise e

    async def batch_enrich_papers_with_authors_journals(
        self, papers_with_metadata: List[Tuple[DBPaper, List[Dict[str, Any]]]]
    ) -> Dict[str, int]:
        """
        Batch enrich multiple papers with authors, institutions, and journals efficiently.
        Reduces database calls from O(papers * authors * institutions) to O(1) per entity type.

        Args:
            papers_with_metadata: List of tuples (DBPaper, authors_list)
                where authors_list contains merged author dicts with institutions

        Returns:
            Dict with enrichment statistics
        """
        if not papers_with_metadata:
            return {"papers": 0, "authors": 0, "institutions": 0, "journals": 0}

        stats: Dict[str, int] = {
            "papers": len(papers_with_metadata),
            "authors": 0,
            "institutions": 0,
            "journals": 0,
        }

        try:
            # Step 1: Collect all unique authors and institutions
            all_authors_data: List[Dict[str, Any]] = []
            all_institutions_data: List[Dict[str, Any]] = []
            seen_author_ids: set[str] = set()
            seen_institution_ids: set[str] = set()

            for db_paper, authors in papers_with_metadata:
                if not authors:
                    continue

                for author_data in authors:
                    author_id = author_data.get("author_id")
                    if author_id and author_id not in seen_author_ids:
                        all_authors_data.append(author_data)
                        seen_author_ids.add(author_id)

                    # Collect institutions
                    institutions = author_data.get("institutions") or []
                    for institution in institutions:
                        inst_id_url = institution.get("id")
                        if inst_id_url:
                            inst_id = self.institution_service.extract_institution_id_from_url(
                                inst_id_url
                            )
                            if inst_id not in seen_institution_ids:
                                all_institutions_data.append(institution)
                                seen_institution_ids.add(inst_id)

            # Step 2: Batch upsert all authors
            author_map: Dict[str, DBAuthor] = {}
            if all_authors_data:
                author_map = await self.author_service.batch_upsert_authors(
                    all_authors_data
                )
                stats["authors"] = len(author_map)
                logger.info(f"Batch enrichment: upserted {len(author_map)} authors")

            # Step 3: Batch upsert all institutions
            institution_map: Dict[str, Any] = {}
            if all_institutions_data:
                institution_map = (
                    await self.institution_service.batch_upsert_institutions(
                        all_institutions_data
                    )
                )
                stats["institutions"] = len(institution_map)
                logger.info(
                    f"Batch enrichment: upserted {len(institution_map)} institutions"
                )

            # Step 4: Batch lookup journals
            journal_lookup_data: List[Dict[str, Any]] = []
            for db_paper, authors in papers_with_metadata:
                pub_year = (
                    db_paper.publication_date.year
                    if db_paper.publication_date
                    else None
                )
                journal_lookup_data.append(
                    {
                        "paper_id": db_paper.paper_id,
                        "venue": db_paper.venue,
                        "issn": (
                            db_paper.issn[0]
                            if db_paper.issn and len(db_paper.issn) > 0
                            else None
                        ),
                        "issn_l": db_paper.issn_l,
                        "year": pub_year,
                    }
                )

            journal_map: Dict[str, Any] = {}
            if journal_lookup_data:
                journal_map = await self.journal_service.batch_lookup_journals(
                    journal_lookup_data
                )
                stats["journals"] = sum(
                    1 for j in journal_map.values() if j is not None
                )
                logger.info(f"Batch enrichment: matched {stats['journals']} journals")

            # Step 5: Link authors to papers and institutions
            from app.models.authors import DBAuthorPaper

            author_paper_links: List[Dict[str, Any]] = []
            author_institution_links: List[Dict[str, Any]] = []

            for db_paper, authors in papers_with_metadata:
                if not authors:
                    continue

                # Extract publication year
                pub_year = None
                if db_paper.publication_date:
                    if isinstance(db_paper.publication_date, (datetime, date)):
                        pub_year = db_paper.publication_date.year
                    elif isinstance(db_paper.publication_date, int):
                        pub_year = db_paper.publication_date

                for position, author_data in enumerate(authors, start=1):
                    author_id = author_data.get("author_id")
                    if not author_id or author_id not in author_map:
                        continue

                    db_author = author_map[author_id]

                    # Get primary institution
                    institutions = author_data.get("institutions", [])
                    institution_db_id = None

                    if institutions:
                        primary_institution = institutions[0]
                        inst_id_url = primary_institution.get("id")
                        if inst_id_url:
                            inst_id = self.institution_service.extract_institution_id_from_url(
                                inst_id_url
                            )
                            if inst_id in institution_map:
                                institution_db_id = institution_map[inst_id].id

                                # Prepare author-institution link
                                author_institution_links.append(
                                    {
                                        "author_id": db_author.id,
                                        "institution_id": institution_db_id,
                                        "year": pub_year,
                                        "is_current": False,
                                    }
                                )

                    # Extract raw affiliation strings
                    affiliations = author_data.get("affiliations", [])
                    institution_raw = (
                        affiliations[0]
                        if affiliations and isinstance(affiliations[0], str)
                        else None
                    )
                    author_string = author_data.get("name")

                    # Prepare author-paper link with position
                    author_paper_links.append(
                        {
                            "author_id": db_author.id,
                            "paper_id": db_paper.id,
                            "author_position": position,
                            "is_corresponding": False,
                            "institution_id": institution_db_id,
                            "institution_raw": institution_raw,
                            "author_string": author_string,
                        }
                    )

            # Batch insert author-paper links
            if author_paper_links:
                stmt = (
                    pg_insert(DBAuthorPaper)
                    .values(author_paper_links)
                    .on_conflict_do_nothing()
                )
                await self.repository.db.execute(stmt)
                logger.info(
                    f"Batch enrichment: created {len(author_paper_links)} author-paper links"
                )

            # Batch insert author-institution links
            if author_institution_links:
                from app.models.authors import DBAuthorInstitution

                stmt = (
                    pg_insert(DBAuthorInstitution)
                    .values(author_institution_links)
                    .on_conflict_do_nothing(
                        index_elements=["author_id", "institution_id", "year"]
                    )
                )
                await self.repository.db.execute(stmt)
                logger.info(
                    f"Batch enrichment: created {len(author_institution_links)} author-institution links"
                )

            # Step 6: Link journals to papers
            if journal_map:
                for db_paper, _ in papers_with_metadata:
                    journal = journal_map.get(db_paper.paper_id)
                    if journal:
                        db_paper.journal_id = journal.id

            await self.repository.db.commit()

            logger.info(
                f"Batch enrichment complete: {stats['papers']} papers, "
                f"{stats['authors']} authors, {stats['institutions']} institutions, "
                f"{stats['journals']} journals"
            )

            return stats

        except Exception as e:
            logger.error(f"Batch enrichment failed: {e}", exc_info=True)
            await self.repository.db.rollback()
            raise e

    def _dto_to_model(self, paper: Union[PaperDTO, PaperEnrichedDTO]):
        """
        Transform Paper DTO to DBPaper model.
        Business logic for data transformation.

        Args:
            paper: PaperDTO or PaperEnrichedDTO

        Returns:
            DBPaper model instance
        """
        if paper.citation_styles is not None:
            bibtex = paper.citation_styles.get("bibtex", None)
            if bibtex:
                try:
                    res = bibtex_to_multiple_styles(bibtex)
                    paper.citation_styles = {**paper.citation_styles, **res}
                except Exception as e:
                    logger.error(
                        f"Failed to convert BibTeX for paper {paper.paper_id}: {e}"
                    )
                    pass
        create_schema = PaperDTO.model_dump(
            paper,
            exclude={
                "id",
                "authors",
                "created_at",
                "updated_at",
                "last_accessed_at",
                "references",
                "has_content",
            },
        )

        if create_schema.get("citation_count") is None:
            create_schema["citation_count"] = 0
        if create_schema.get("influential_citation_count") is None:
            create_schema["influential_citation_count"] = 0
        if create_schema.get("reference_count") is None:
            create_schema["reference_count"] = 0

        return create_schema

    async def update_processing_status(
        self, paper_id: str, status: str, error: Optional[str] = None
    ):
        """Update paper processing status"""
        await self.repository.update_paper_processing_status(paper_id, status, error)

    async def search_similar_papers(
        self, query_embedding: List[float], limit: int = 10
    ) -> List[DBPaper]:
        """
        Search for similar papers using summary embeddings.

        Args:
            query_embedding: Query embedding vector
            limit: Number of results to return

        Returns:
            List of similar papers ordered by similarity
        """
        return await self.repository.search_similar_papers(query_embedding, limit)
    
    async def hybrid_search_papers(
        self,
        query: str,
        limit: int = 100,
        bm25_weight: float = 0.3,
        semantic_weight: float = 0.7,
    ) -> List[tuple[DBPaper, float]]:
        """
        Hybrid BM25 + semantic search on papers.
        
        Handles:
        - Query embedding generation
        - Weight normalization
        - Calls repository for data access
        
        Args:
            query: Search query text
            limit: Maximum number of results
            bm25_weight: Weight for BM25 score (0-1)
            semantic_weight: Weight for semantic score (0-1)
            
        Returns:
            List of tuples (paper, combined_score) sorted by relevance
        """
        from app.processor.services.embeddings import get_embedding_service
        
        # Generate query embedding
        embedding_service = get_embedding_service()
        query_embedding = await embedding_service.create_embedding(query, task="search_query")
        
        if not query_embedding:
            logger.error("Failed to generate query embedding for papers")
            return []
        
        # Normalize weights
        total_weight = bm25_weight + semantic_weight
        normalized_bm25 = bm25_weight / total_weight
        normalized_semantic = semantic_weight / total_weight
        
        # Call repository for data access
        papers_with_scores = await self.repository.hybrid_search_papers(
            query=query,
            query_embedding=query_embedding,
            limit=limit,
            bm25_weight=normalized_bm25,
            semantic_weight=normalized_semantic,
        )
        
        logger.info(f"Hybrid paper search returned {len(papers_with_scores)} papers")
        return papers_with_scores
    
    async def hybrid_search_papers_with_filters(
        self,
        query: str,
        limit: int = 100,
        bm25_weight: float = 0.3,
        semantic_weight: float = 0.7,
        author_name: Optional[str] = None,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
        venue: Optional[str] = None,
        min_citation_count: Optional[int] = None,
        max_citation_count: Optional[int] = None,
    ) -> List[tuple[DBPaper, float]]:
        """
        Hybrid BM25 + semantic search with filters.
        
        Handles:
        - Query embedding generation
        - Weight normalization
        - Calls repository with filters
        
        Args:
            query: Search query text
            limit: Maximum number of results
            bm25_weight: Weight for BM25 score (0-1)
            semantic_weight: Weight for semantic score (0-1)
            author_name: Filter by author name
            year_min: Minimum publication year
            year_max: Maximum publication year
            venue: Filter by venue
            min_citation_count: Minimum citation count
            max_citation_count: Maximum citation count
            
        Returns:
            List of tuples (paper, combined_score) sorted by relevance
        """
        from app.processor.services.embeddings import get_embedding_service
        
        # Generate query embedding
        embedding_service = get_embedding_service()
        query_embedding = await embedding_service.create_embedding(query, task="search_query")
        
        if not query_embedding:
            logger.error("Failed to generate query embedding for filtered search")
            return []
        
        # Normalize weights
        total_weight = bm25_weight + semantic_weight
        normalized_bm25 = bm25_weight / total_weight
        normalized_semantic = semantic_weight / total_weight
        
        # Call repository with filters
        papers_with_scores = await self.repository.hybrid_search_papers_with_filters(
            query=query,
            query_embedding=query_embedding,
            limit=limit,
            bm25_weight=normalized_bm25,
            semantic_weight=normalized_semantic,
            author_name=author_name,
            year_min=year_min,
            year_max=year_max,
            venue=venue,
            min_citation_count=min_citation_count,
            max_citation_count=max_citation_count,
        )
        
        logger.info(f"Hybrid filtered search returned {len(papers_with_scores)} papers")
        return papers_with_scores

    async def batch_create_papers_from_schema(
        self, papers: List[Union[PaperDTO, PaperEnrichedDTO]], enrich: bool = True
    ) -> List[DBPaper]:
        """
        Batch create multiple papers efficiently with optional enrichment.

        This method optimizes paper creation by:
        1. Batch checking existing papers (N queries -> 1 query)
        2. Batch saving papers (N queries -> 1 query)
        3. Batch enriching authors/institutions/journals (N*M queries -> 3 queries)

        Args:
            papers: List of PaperDTO or PaperEnrichedDTO from retriever
            enrich: If True, enrich with authors/institutions/journals

        Returns:
            List of created DBPaper objects (newly created only)
        """
        if not papers:
            return []

        try:
            # Step 1: Batch check which papers already exist
            paper_ids = [p.paper_id for p in papers]
            existing_map = await self.batch_check_existing_papers(paper_ids)
            new_papers = [p for p in papers if not existing_map.get(p.paper_id, False)]

            if not new_papers:
                logger.info(f"All {len(papers)} papers already exist, skipping")
                return []

            logger.info(
                f"Creating {len(new_papers)} new papers (skipped {len(papers) - len(new_papers)} existing)"
            )

            # Step 2: Batch save all papers without enrichment
            created_papers = []
            papers_with_metadata = []  # For batch enrichment later

            for paper in new_papers:
                try:
                    # Create paper without enrichment
                    db_paper = await self.create_paper_from_schema(
                        paper, defer_enrichment=True
                    )
                    if db_paper:
                        created_papers.append(db_paper)

                        # Store for batch enrichment if requested
                        if enrich and hasattr(paper, "authors") and paper.authors:
                            authors_dict = [
                                (
                                    author.model_dump()
                                    if not isinstance(author, dict)
                                    else author
                                )
                                for author in paper.authors
                            ]
                            papers_with_metadata.append((db_paper, authors_dict))

                except Exception as e:
                    logger.error(f"Failed to create paper {paper.paper_id}: {e}")
                    continue

            # Step 3: Batch enrich authors, institutions, and journals
            if enrich and papers_with_metadata:
                stats = await self.batch_enrich_papers_with_authors_journals(
                    papers_with_metadata
                )
                logger.info(
                    f"Batch enrichment stats: {stats['authors']} authors, "
                    f"{stats['institutions']} institutions, {stats['journals']} journals"
                )

            logger.info(f"Successfully created {len(created_papers)} papers")
            return created_papers

        except Exception as e:
            logger.error(f"Batch paper creation failed: {e}", exc_info=True)
            try:
                await self.repository.rollback()
            except Exception as rollback_error:
                logger.error(f"Failed to rollback: {rollback_error}")
            raise e

    async def get_paper_citations(
        self, paper_id: str, offset: int = 0, limit: int = 100
    ) -> Dict[str, Any]:
        """Get papers that cite the given paper"""
        paper = await self.get_paper(paper_id)
        if not paper:
            raise NotFoundException(f"Paper {paper_id} not found")

        citations_data = await self.retriever_service.get_paper_citations(
            paper_id, limit=limit, offset=offset
        )

        if "data" in citations_data:
            citations_data["data"] = [
                ref
                for ref in citations_data["data"]
                if ref.get("citingPaper", {}).get("paperId") is not None
            ]

        return citations_data

    async def get_paper_references(
        self, paper_id: str, offset: int = 0, limit: int = 100
    ) -> Dict[str, Any]:
        """Get papers that are cited by the given paper"""
        paper = await self.get_paper(paper_id)
        if not paper:
            raise NotFoundException(f"Paper {paper_id} not found")

        references_data = await self.retriever_service.get_paper_references(
            paper_id, limit=limit, offset=offset
        )

        logger.debug(f"Fetched {len(references_data.get('data', []))} references for paper {paper_id} before filtering")
        logger.debug(f"References data: {references_data}")
        if "data" in references_data:
            references_data["data"] = [
                ref
                for ref in references_data["data"]
                if ref.get("citedPaper", {}).get("paperId") is not None
            ]        

        return references_data
