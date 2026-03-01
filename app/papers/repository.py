from typing import List, Optional, Dict, Any, TYPE_CHECKING, Tuple
from dataclasses import dataclass, field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, and_
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import selectinload, joinedload
from datetime import datetime, date
from app.models.papers import DBPaper, DBPaperChunk
from app.models.authors import DBAuthor
from app.authors.service import AuthorService
from app.institutions.service import InstitutionService
from app.papers.journal_service import JournalService
from app.extensions.logger import create_logger

logger = create_logger(__name__)


@dataclass
class LoadOptions:
    """Options for eager loading paper relationships"""
    authors: bool = False
    journal: bool = False
    citations: bool = False
    institutions: bool = False
    
    @classmethod
    def all(cls) -> 'LoadOptions':
        """Load all relationships"""
        return cls(authors=True, journal=True, citations=True, institutions=True)
    
    @classmethod
    def with_authors(cls) -> 'LoadOptions':
        """Load only authors"""
        return cls(authors=True)
    
    @classmethod
    def with_journal(cls) -> 'LoadOptions':
        """Load only journal"""
        return cls(journal=True)
    
    @classmethod
    def with_citations(cls) -> 'LoadOptions':
        """Load only citations"""
        return cls(citations=True)
    
    @classmethod
    def none(cls) -> 'LoadOptions':
        """Load no relationships (default)"""
        return cls()


class PaperRepository:
    """Repository for paper database operations"""
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.author_service = AuthorService(db)
        self.institution_service = InstitutionService(db)
        self.journal_service = JournalService(db)
    
    async def save_paper(self, paper_data) -> Optional[DBPaper]:
        """
        Save a paper to database using INSERT ON CONFLICT DO NOTHING.
        More efficient than checking existence first - handles duplicates at DB level.
        
        Args:
            paper_data: PaperDTO with all required defaults
            
        Returns:
            Created DBPaper object, or None if paper already exists
        """

        stmt = (
            pg_insert(DBPaper)
            .values(**paper_data)
            .on_conflict_do_update(
                index_elements=[DBPaper.paper_id],
                set_={
                    "last_accessed_at": datetime.now()
                }
            )
            .returning(DBPaper)
        )
        
        result = await self.db.execute(stmt)
        await self.db.commit()
        
        created_paper = result.scalar_one_or_none()
        
        if created_paper:
            logger.info(f"Created new paper {created_paper.paper_id}")
        else:
            paper_id = paper_data.get('paper_id')
            if paper_id:
                logger.debug(f"Paper {paper_id} already exists, skipped")
                existing = await self.get_paper_by_id(paper_id)
                return existing
            else:
                logger.error("Paper creation failed and no paper_id available")
                return None
        
        return created_paper
    
    async def enrich_paper_with_authors_institutions(
        self, 
        db_paper: DBPaper, 
        authors: List[Dict]
    ):
        """
        Enrich paper with author and institution data.
        Authors contain merged data from Semantic Scholar and OpenAlex.
        Creates DBAuthor, DBInstitution entities and links them to the paper.
        
        Args:
            db_paper: Database paper entity
            authors: List of merged author objects with S2 stats and OA institutions
            
        Example merged author:
        {
            "name": "Frederick Sanger",
            "author_id": "A5114007683",
            "orcid": "https://orcid.org/0000-0002-5926-4032",
            "h_index": 45,
            "citation_count": 12000,
            "paper_count": 150,
            "institutions": [
                {
                    "id": "https://openalex.org/I170203145",
                    "display_name": "MRC Laboratory of Molecular Biology",
                    "ror": "https://ror.org/00tw3jy02",
                    "country_code": "GB",
                    "type": "facility"
                }
            ],
            "affiliations": ["MRC Laboratory of Molecular Biology"]
        }
        """
        if not authors:
            logger.debug(f"No authors data for paper {db_paper.paper_id}")
            return
        
        # Extract publication year for author-institution tracking
        # publication_date is a datetime.date object from SQLAlchemy Date column
        pub_year = None
        if db_paper.publication_date:
            if isinstance(db_paper.publication_date, (datetime, date)):
                pub_year = db_paper.publication_date.year
            elif isinstance(db_paper.publication_date, int):
                pub_year = db_paper.publication_date
        
        for author_data in authors:  
            # Directly upsert from merged author data (no re-mapping needed)
            db_author = await self.author_service.upsert_from_merged_author(author_data)
            if not db_author:
                continue
            
            # Process institutions for this author
            institutions = author_data.get("institutions", [])
            institution_id = None
            
            if institutions:
                # Process first institution (primary affiliation)
                primary_institution = institutions[0]
                db_institution = await self.institution_service.upsert_from_openalex(primary_institution)
                
                if db_institution:
                    institution_id = db_institution.id
                    
                    # Link author to institution
                    await self.author_service.link_author_to_institution(
                        author=db_author,
                        institution_id=db_institution.id,
                        year=pub_year,
                        is_current=False  # We can't determine if it's current from paper data
                    )
            
            # Link author to paper
            await self.author_service.link_author_to_paper(
                author=db_author,
                paper_id=db_paper.id,
                author_data=author_data,
                institution_id=institution_id
            )
        
        logger.info(f"Enriched paper {db_paper.paper_id} with {len(authors)} authors and their institutions")
    
    async def enrich_paper_with_journal(self, db_paper: DBPaper) -> None:
        """
        Enrich paper with journal data by linking to SJR database.
        
        Looks up journal using ISSN-L (preferred), ISSN, or venue name.
        Sets journal_id on the paper if a match is found.
        
        Args:
            db_paper: Database paper entity to enrich
        """
        try:
            journal = await self.journal_service.enrich_paper_with_journal(
                paper=db_paper,
                venue=db_paper.venue,
                issn=db_paper.issn[0] if db_paper.issn and len(db_paper.issn) > 0 else None,
                issn_l=db_paper.issn_l
            )
            
            if journal:
                logger.info(f"Paper {db_paper.paper_id} linked to journal: {journal.title} (Q{journal.sjr_best_quartile}, SJR: {journal.sjr_score})")
            else:
                logger.debug(f"No journal match for paper {db_paper.paper_id}")
                
        except Exception as e:
            logger.error(f"Error enriching paper {db_paper.paper_id} with journal: {e}")
        
    async def get_papers(
        self,
        skip: int = 0,
        limit: int = 20,
        processed_only: bool = False,
        source: Optional[str] = None,
        paper_ids: Optional[List[str]] = None,
        load_options: Optional[LoadOptions] = None
    ) -> tuple[List[DBPaper], int]:
        """
        Get papers with pagination and optional filtering
        
        Args:
            skip: Number of records to skip
            limit: Maximum number of records to return
            processed_only: If True, only return processed papers
            source: Optional source filter (e.g., 'SemanticScholar', 'OpenAlex')
            paper_ids: Optional list of paper_ids to filter by (internal paper identifiers)
            load_options: LoadOptions for eager loading relationships
            
        Returns:
            Tuple of (papers list, total count)
        """
        if load_options is None:
            load_options = LoadOptions()
        
        # Build base query
        query = select(DBPaper)
        count_query = select(DBPaper)
        
        # Add eager loading options
        if load_options.authors:
            from app.models.authors import DBAuthorPaper
            query = query.options(
                selectinload(DBPaper.paper_authors).selectinload(DBAuthorPaper.author)
            )
        if load_options.journal:
            from app.models.journals import DBJournal
            query = query.options(joinedload(DBPaper.journal))
        if load_options.citations:
            query = query.options(
                selectinload(DBPaper.citations_made),
                selectinload(DBPaper.citations_received)
            )
        
        # Apply filters
        if paper_ids:
            query = query.where(DBPaper.paper_id.in_(paper_ids))
            count_query = count_query.where(DBPaper.paper_id.in_(paper_ids))
        
        if processed_only:
            query = query.where(DBPaper.is_processed == True)
            count_query = count_query.where(DBPaper.is_processed == True)
        
        if source:
            query = query.where(DBPaper.source == source)
            count_query = count_query.where(DBPaper.source == source)
        
        # Get total count
        from sqlalchemy import func
        count_result = await self.db.execute(
            select(func.count()).select_from(count_query.subquery())
        )
        total = count_result.scalar_one()
        
        # Apply pagination and ordering
        query = query.order_by(DBPaper.created_at.desc()).offset(skip).limit(limit)
        
        # Execute query
        result = await self.db.execute(query)
        papers = list(result.unique().scalars().all())
        
        return papers, total
    
    async def get_single_paper(
        self, 
        id: str,
        load_options: Optional[LoadOptions] = None
    ) -> Optional[DBPaper]:
        """
        Get single paper by paper_id or database ID
        
        Args:
            id: Paper ID (internal paper_id) or database ID (if numeric)
            load_options: LoadOptions for eager loading relationships
        """
        if load_options is None:
            load_options = LoadOptions()
        
        if id.isdigit():
            return await self.get_paper_by_db_id(int(id), load_options)
        else:
            return await self.get_paper_by_id(id, load_options)
        
    async def get_paper_by_id(
        self, 
        paper_id: str,
        load_options: Optional[LoadOptions] = None
    ) -> Optional[DBPaper]:
        """
        Get paper by internal paper_id
        
        Args:
            paper_id: Internal paper identifier
            load_options: LoadOptions for eager loading relationships
        """
        if load_options is None:
            load_options = LoadOptions()
        
        query = select(DBPaper).where(DBPaper.paper_id == paper_id)
        
        # Add eager loading options
        if load_options.authors:
            from app.models.authors import DBAuthorPaper
            query = query.options(
                selectinload(DBPaper.paper_authors).selectinload(DBAuthorPaper.author)
            )
        if load_options.journal:
            from app.models.journals import DBJournal
            query = query.options(joinedload(DBPaper.journal))
        if load_options.citations:
            query = query.options(
                selectinload(DBPaper.citations_made),
                selectinload(DBPaper.citations_received)
            )
        
        result = await self.db.execute(query)
        return result.unique().scalar_one_or_none()
    
    async def get_paper_by_db_id(
        self, 
        id: int,
        load_options: Optional[LoadOptions] = None
    ) -> Optional[DBPaper]:
        """
        Get paper by database ID
        
        Args:
            id: Database primary key ID
            load_options: LoadOptions for eager loading relationships
        """
        if load_options is None:
            load_options = LoadOptions()
        
        query = select(DBPaper).where(DBPaper.id == id)
        
        # Add eager loading options
        if load_options.authors:
            from app.models.authors import DBAuthorPaper
            query = query.options(
                selectinload(DBPaper.paper_authors).selectinload(DBAuthorPaper.author)
            )
        if load_options.journal:
            from app.models.journals import DBJournal
            query = query.options(joinedload(DBPaper.journal))
        if load_options.citations:
            query = query.options(
                selectinload(DBPaper.citations_made),
                selectinload(DBPaper.citations_received)
            )
        
        result = await self.db.execute(query)
        return result.unique().scalar_one_or_none()
    
    async def get_paper_by_external_ids(self, external_ids: Dict[str, str], source: str) -> Optional[DBPaper]:
        """Get paper by external IDs and source"""
        # Try to find by DOI first, then other IDs
        doi = external_ids.get('DOI')
        if doi:
            result = await self.db.execute(
                select(DBPaper).where(
                    and_(
                        DBPaper.external_ids['DOI'].astext == doi,
                        DBPaper.source == source
                    )
                )
            )
            paper = result.scalar_one_or_none()
            if paper:
                return paper
        
        # Fallback to other IDs
        for key, value in external_ids.items():
            if value:
                result = await self.db.execute(
                    select(DBPaper).where(
                        and_(
                            DBPaper.external_ids[key].astext == value,
                            DBPaper.source == source
                        )
                    )
                )
                paper = result.scalar_one_or_none()
                if paper:
                    return paper
        
        return None
    
    async def update_paper_processing_status(
        self,
        paper_id: str,
        status: str,
        error: Optional[str] = None
    ):
        """Update paper processing status"""
        update_data = {
            "processing_status": status,
            "updated_at": datetime.utcnow()
        }
        
        if status == "completed":
            update_data["is_processed"] = True
        
        if error:
            update_data["processing_error"] = error
        
        await self.db.execute(
            update(DBPaper)
            .where(DBPaper.paper_id == paper_id)
            .values(**update_data)
        )
        await self.db.commit()
    
    async def update_paper(self, paper_id: str, update_data: Dict[str, Any]) -> Optional[DBPaper]:
        """Update paper with provided data"""
        await self.db.execute(
            update(DBPaper)
            .where(DBPaper.paper_id == paper_id)
            .values(**update_data, updated_at=datetime.utcnow())
        )
        await self.db.commit()
        return await self.get_paper_by_id(paper_id)
    
    async def delete_paper(self, paper_id: str) -> bool:
        """
        Delete paper from database.
        Note: Chunks should be deleted separately via ChunkRepository before calling this.
        """
        from sqlalchemy import delete as sql_delete
        
        # Delete paper
        result = await self.db.execute(
            sql_delete(DBPaper).where(DBPaper.paper_id == paper_id)
        )
        await self.db.commit()
        
        return result.rowcount > 0 # type: ignore
    
    async def update_paper_summary(
        self,
        paper_id: str,
        summary: str,
        summary_embedding: List[float]
    ):
        """Update paper summary and embedding"""
        await self.db.execute(
            update(DBPaper)
            .where(DBPaper.paper_id == paper_id)
            .values(
                summary=summary,
                summary_embedding=summary_embedding,
                updated_at=datetime.utcnow()
            )
        )
        await self.db.commit()
    
    async def update_paper_embedding(
        self,
        paper_id: str,
        embedding: List[float]
    ):
        """Update paper title+abstract embedding"""
        await self.db.execute(
            update(DBPaper)
            .where(DBPaper.paper_id == paper_id)
            .values(
                embedding=embedding,
                updated_at=datetime.utcnow()
            )
        )
        await self.db.commit()
    
    async def bulk_update_paper_embeddings(
        self,
        paper_embeddings: Dict[str, List[float]]
    ):
        """
        Bulk update paper embeddings for multiple papers.
        
        Args:
            paper_embeddings: Dict mapping paper_id to embedding vector
        """
        if not paper_embeddings:
            return
        
        for paper_id, embedding in paper_embeddings.items():
            await self.db.execute(
                update(DBPaper)
                .where(DBPaper.paper_id == paper_id)
                .values(
                    embedding=embedding,
                    updated_at=datetime.utcnow()
                )
            )
        
        await self.db.commit()
        logger.info(f"Bulk updated embeddings for {len(paper_embeddings)} papers")
    
    async def search_similar_papers(
        self,
        query_embedding: List[float],
        limit: int = 10
    ) -> List[DBPaper]:
        """
        Search for similar papers using summary embeddings
        
        Args:
            query_embedding: Query embedding vector
            limit: Number of results to return
            
        Returns:
            List of similar papers ordered by similarity
        """
        result = await self.db.execute(
            select(DBPaper)
            .where(DBPaper.summary_embedding.isnot(None))
            .order_by(DBPaper.summary_embedding.cosine_distance(query_embedding))
            .limit(limit)
        )
        return list(result.scalars().all())
    
    async def update_last_accessed(self, paper_id: str):
        """Update last accessed timestamp"""
        await self.db.execute(
            update(DBPaper)
            .where(DBPaper.paper_id == paper_id)
            .values(last_accessed_at=datetime.utcnow())
        )
        await self.db.commit()
    
    async def get_paper_authors(self, paper_id: str):
        """
        Get all authors for a paper with their affiliations
        
        Args:
            paper_id: Internal paper identifier
            
        Returns:
            List of DBAuthorPaper objects with author and institution data loaded
        """
        from app.models.authors import DBAuthorPaper
        
        query = (
            select(DBAuthorPaper)
            .where(DBAuthorPaper.paper_id == paper_id)
            .options(
                selectinload(DBAuthorPaper.author),
                selectinload(DBAuthorPaper.institution)
            )
            .order_by(DBAuthorPaper.author_position)
        )
        
        result = await self.db.execute(query)
        return list(result.scalars().all())
    
    async def get_paper_journal(self, paper_id: str):
        """
        Get journal (SJR) data for a paper
        
        Args:
            paper_id: Internal paper identifier
            
        Returns:
            DBJournal object with SJR ranking data, or None if no journal linked
        """
        from app.models.journals import DBJournal
        
        # Get the paper with journal loaded
        query = (
            select(DBPaper)
            .where(DBPaper.paper_id == paper_id)
            .options(joinedload(DBPaper.journal))
        )
        
        result = await self.db.execute(query)
        paper = result.unique().scalar_one_or_none()
        
        return paper.journal if paper else None

    async def rollback(self):
        """Rollback the current transaction"""
        await self.db.rollback()
    
    async def batch_enrich_papers_with_authors_journals(
        self,
        papers_with_metadata: List[Tuple[DBPaper, List[Dict[str, Any]]]]
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
        
        stats: Dict[str, int] = {"papers": len(papers_with_metadata), "authors": 0, "institutions": 0, "journals": 0}
        
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
                            inst_id = self.institution_service.extract_institution_id_from_url(inst_id_url)
                            if inst_id not in seen_institution_ids:
                                all_institutions_data.append(institution)
                                seen_institution_ids.add(inst_id)
            
            # Step 2: Batch upsert all authors
            author_map: Dict[str, DBAuthor] = {}
            if all_authors_data:
                author_map = await self.author_service.batch_upsert_authors(all_authors_data)
                stats["authors"] = len(author_map)
                logger.info(f"Batch enrichment: upserted {len(author_map)} authors")
            
            # Step 3: Batch upsert all institutions
            institution_map: Dict[str, Any] = {}
            if all_institutions_data:
                institution_map = await self.institution_service.batch_upsert_institutions(all_institutions_data)
                stats["institutions"] = len(institution_map)
                logger.info(f"Batch enrichment: upserted {len(institution_map)} institutions")
            
            # Step 4: Batch lookup journals
            journal_lookup_data: List[Dict[str, Any]] = []
            for db_paper, authors in papers_with_metadata:
                pub_year = db_paper.publication_date.year if db_paper.publication_date else None
                journal_lookup_data.append({
                    "paper_id": db_paper.paper_id,
                    "venue": db_paper.venue,
                    "issn": db_paper.issn[0] if db_paper.issn and len(db_paper.issn) > 0 else None,
                    "issn_l": db_paper.issn_l,
                    "year": pub_year
                })
            
            journal_map: Dict[str, Any] = {}
            if journal_lookup_data:
                journal_map = await self.journal_service.batch_lookup_journals(journal_lookup_data)
                stats["journals"] = sum(1 for j in journal_map.values() if j is not None)
                logger.info(f"Batch enrichment: matched {stats['journals']} journals")
            
            # Step 5: Link authors to papers and institutions
            from app.models.authors import DBAuthorPaper
            from sqlalchemy.dialects.postgresql import insert as pg_insert
            
            author_paper_links: List[Dict[str, Any]] = []
            author_institution_links: List[Dict[str, Any]] = []
            
            for db_paper, authors in papers_with_metadata:
                if not authors:
                    continue
                
                # Get publication year for author-institution tracking
                pub_year = None
                if db_paper.publication_date:
                    if isinstance(db_paper.publication_date, (datetime, date)):
                        pub_year = db_paper.publication_date.year
                    elif isinstance(db_paper.publication_date, int):
                        pub_year = db_paper.publication_date
                
                for author_data in authors:
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
                            inst_id = self.institution_service.extract_institution_id_from_url(inst_id_url)
                            if inst_id in institution_map:
                                institution_db_id = institution_map[inst_id].id
                                
                                # Prepare author-institution link
                                author_institution_links.append({
                                    "author_id": db_author.id,
                                    "institution_id": institution_db_id,
                                    "year": pub_year,
                                    "is_current": False
                                })
                    
                    # Extract raw affiliation strings
                    affiliations = author_data.get("affiliations", [])
                    institution_raw = affiliations[0] if affiliations and isinstance(affiliations[0], str) else None
                    author_string = author_data.get("name")
                    
                    # Prepare author-paper link
                    author_paper_links.append({
                        "author_id": db_author.id,
                        "paper_id": db_paper.id,
                        "author_position": None,
                        "is_corresponding": False,
                        "institution_id": institution_db_id,
                        "institution_raw": institution_raw,
                        "author_string": author_string
                    })
            
            # Batch insert author-paper links
            if author_paper_links:
                stmt = pg_insert(DBAuthorPaper).values(author_paper_links).on_conflict_do_nothing()
                await self.db.execute(stmt)
                logger.info(f"Batch enrichment: created {len(author_paper_links)} author-paper links")
            
            # Batch insert author-institution links
            if author_institution_links:
                from app.models.authors import DBAuthorInstitution
                stmt = pg_insert(DBAuthorInstitution).values(author_institution_links).on_conflict_do_nothing(
                    index_elements=['author_id', 'institution_id', 'year']
                )
                await self.db.execute(stmt)
                logger.info(f"Batch enrichment: created {len(author_institution_links)} author-institution links")
            
            # Step 6: Link journals to papers
            if journal_map:
                for db_paper, _ in papers_with_metadata:
                    journal = journal_map.get(db_paper.paper_id)
                    if journal:
                        db_paper.journal_id = journal.id
            
            await self.db.commit()
            
            logger.info(
                f"Batch enrichment complete: {stats['papers']} papers, "
                f"{stats['authors']} authors, {stats['institutions']} institutions, "
                f"{stats['journals']} journals"
            )
            
            return stats
            
        except Exception as e:
            logger.error(f"Batch enrichment failed: {e}", exc_info=True)
            await self.db.rollback()
            raise e
