from typing import List, Optional, Dict, Any, TYPE_CHECKING, Tuple
from dataclasses import dataclass, field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, and_, literal_column
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import selectinload, joinedload
from datetime import datetime, date
from app.models.papers import DBPaper, DBPaperChunk
from app.models.authors import DBAuthor
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
    
    async def upsert_paper(self, paper_data: Dict[str, Any]) -> Tuple[DBPaper, bool]:
        """
        Create or update a paper in database.
        Updates all metadata if paper exists.
        
        Args:
            paper_data: Dictionary with paper fields
            
        Returns:
            Tuple of (DBPaper object, is_new: bool)
        """
        # Prepare update fields (exclude paper_id and id)
        update_fields = {k: v for k, v in paper_data.items() if k not in ['paper_id', 'id']}
        update_fields['last_accessed_at'] = datetime.now()
        
        stmt = (
            pg_insert(DBPaper)
            .values(**paper_data)
            .on_conflict_do_update(
                index_elements=[DBPaper.paper_id],
                set_=update_fields
            )
            .returning(DBPaper)
        )
        
        result = await self.db.execute(stmt)
        await self.db.commit()
        
        paper = result.scalar_one()
        
        # Check if it was an update or insert by comparing updated_at
        is_new = paper.created_at == paper.updated_at if hasattr(paper, 'created_at') else True
        
        if is_new:
            logger.info(f"Created new paper {paper.paper_id}")
        else:
            logger.info(f"Updated existing paper {paper.paper_id}")
        
        return paper, is_new
    
    async def paper_exists(self, paper_id: str) -> bool:
        """
        Check if a paper exists in the database.
        
        Args:
            paper_id: Paper identifier
            
        Returns:
            True if paper exists, False otherwise
        """
        from sqlalchemy import exists
        stmt = select(exists().where(DBPaper.paper_id == paper_id))
        result = await self.db.scalar(stmt)
        return result or False
    
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
                selectinload(DBPaper.authors).selectinload(DBAuthorPaper.author)
            )
        if load_options.journal:
            from app.models.journals import DBJournal
            query = query.options(joinedload(DBPaper.journal))
        if load_options.citations:
            query = query.options(
                selectinload(DBPaper.citations),
                selectinload(DBPaper.references)
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
                selectinload(DBPaper.authors).selectinload(DBAuthorPaper.author)
            )
        if load_options.journal:
            from app.models.journals import DBJournal
            query = query.options(joinedload(DBPaper.journal))
        if load_options.citations:
            query = query.options(
                selectinload(DBPaper.citations),
                selectinload(DBPaper.references)
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
                selectinload(DBPaper.authors).selectinload(DBAuthorPaper.author)
            )
        if load_options.journal:
            from app.models.journals import DBJournal
            query = query.options(joinedload(DBPaper.journal))
        if load_options.citations:
            query = query.options(
                selectinload(DBPaper.citations),
                selectinload(DBPaper.references)
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
    
    async def update_paper_tldr(
        self,
        paper_id: str,
        tldr: str,
        tldr_embedding: List[float]
    ):
        """Update paper TLDR and embedding"""
        await self.db.execute(
            update(DBPaper)
            .where(DBPaper.paper_id == paper_id)
            .values(
                tldr=tldr,
                tldr_embedding=tldr_embedding,
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
    
    async def get_paper_embeddings(
        self,
        paper_ids: List[str]
    ) -> Dict[str, Optional[List[float]]]:
        """
        Get embeddings for multiple papers.
        
        Args:
            paper_ids: List of paper IDs
            
        Returns:
            Dict mapping paper_id to embedding (or None if no embedding)
        """
        if not paper_ids:
            return {}
        
        result = await self.db.execute(
            select(DBPaper.paper_id, DBPaper.embedding)
            .where(DBPaper.paper_id.in_(paper_ids))
        )
        
        return {row[0]: row[1] for row in result.all()}
    
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
            .where(DBPaper.tldr_embedding.isnot(None))
            .order_by(DBPaper.tldr_embedding.cosine_distance(query_embedding))
            .limit(limit)
        )
        return list(result.scalars().all())
    
    async def hybrid_search_papers(
        self,
        query: str,
        query_embedding: List[float],
        limit: int = 100,
        bm25_weight: float = 0.3,
        semantic_weight: float = 0.7,
    ) -> List[tuple[DBPaper, float]]:
        """
        Hybrid BM25 + semantic search on papers using title + abstract.
        
        Pure data access layer - executes SQL query with provided parameters.
        Uses PostgreSQL ts_vector for BM25 keyword search and pgvector for semantic search.
        Combines both scores with pre-normalized weights.
        
        Args:
            query: Search query text for BM25
            query_embedding: Pre-computed query embedding vector
            limit: Maximum number of results
            bm25_weight: Normalized weight for BM25 score
            semantic_weight: Normalized weight for semantic score
            
        Returns:
            List of tuples (paper, combined_score) sorted by relevance
        """
        from sqlalchemy import func, text
        
        # Create tsvector from title + abstract on the fly
        ts_query = func.plainto_tsquery('english', query)
        ts_vector = func.to_tsvector('english', 
            func.coalesce(DBPaper.title, '') + ' ' + func.coalesce(DBPaper.abstract, '')
        )
        
        bm25_score = func.ts_rank(ts_vector, ts_query)
        semantic_score = 1 - DBPaper.embedding.cosine_distance(query_embedding)
        
        # Use COALESCE to ensure BM25 score is 0 if no keyword match (instead of filtering out)
        combined_score = (func.coalesce(bm25_score, 0) * bm25_weight * 10) + (semantic_score * semantic_weight)

        query_stmt = (
            select(DBPaper, combined_score.label('relevance_score'))
            .where(
                DBPaper.embedding.isnot(None)  # Must have embedding
            )
            .order_by(text('relevance_score DESC'))
            .limit(limit)
        )
        
        result = await self.db.execute(query_stmt)
        papers_with_scores = result.all()
        
        return [(paper, float(score)) for paper, score in papers_with_scores]
    
    async def hybrid_search_papers_with_filters(
        self,
        query: str,
        query_embedding: List[float],
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
        Hybrid BM25 + semantic search with additional filtering options.
        
        Pure data access layer - executes filtered SQL query with provided parameters.
        
        Args:
            query: Search query text for BM25
            query_embedding: Pre-computed query embedding vector
            limit: Maximum number of results
            bm25_weight: Normalized weight for BM25 score
            semantic_weight: Normalized weight for semantic score
            author_name: Filter by author name (partial match)
            year_min: Minimum publication year
            year_max: Maximum publication year
            venue: Filter by venue name (partial match)
            min_citation_count: Minimum citation count
            max_citation_count: Maximum citation count
            
        Returns:
            List of tuples (paper, combined_score) sorted by relevance
        """
        from sqlalchemy import func, text, and_, or_
        from app.models.authors import DBAuthor, DBAuthorPaper
        
        # Create tsvector from title + abstract on the fly
        ts_query = func.plainto_tsquery('english', query)
        ts_vector = func.to_tsvector('english', 
            func.coalesce(DBPaper.title, '') + ' ' + func.coalesce(DBPaper.abstract, '')
        )
        
        bm25_score = func.ts_rank(ts_vector, ts_query)
        semantic_score = 1 - DBPaper.embedding.cosine_distance(query_embedding)
        
        # Use COALESCE to ensure BM25 score is 0 if no keyword match
        combined_score = (func.coalesce(bm25_score, 0) * bm25_weight * 10) + (semantic_score * semantic_weight)

        # Build base query
        query_stmt = (
            select(DBPaper, combined_score.label('relevance_score'))
            .where(DBPaper.embedding.isnot(None))
        )
        
        # Apply filters
        filter_conditions = []
        
        # Author filter - requires join
        if author_name:
            query_stmt = query_stmt.join(
                DBAuthorPaper,
                DBPaper.paper_id == DBAuthorPaper.paper_id
            ).join(
                DBAuthor,
                DBAuthorPaper.author_id == DBAuthor.author_id
            )
            filter_conditions.append(
                DBAuthor.name.ilike(f"%{author_name}%")
            )
        
        # Year filters
        if year_min is not None:
            filter_conditions.append(DBPaper.year >= year_min)
        if year_max is not None:
            filter_conditions.append(DBPaper.year <= year_max)
        
        # Venue filter
        if venue:
            filter_conditions.append(
                DBPaper.venue.ilike(f"%{venue}%")
            )
        
        # Citation count filters
        if min_citation_count is not None:
            filter_conditions.append(DBPaper.citation_count >= min_citation_count)
        if max_citation_count is not None:
            filter_conditions.append(DBPaper.citation_count <= max_citation_count)
        
        # Apply all filters
        if filter_conditions:
            query_stmt = query_stmt.where(and_(*filter_conditions))
        
        # Add ordering and limit
        query_stmt = query_stmt.order_by(text('relevance_score DESC')).limit(limit)
        
        # If author filter was applied, we need distinct to avoid duplicates
        if author_name:
            query_stmt = query_stmt.distinct(DBPaper.paper_id)
        
        result = await self.db.execute(query_stmt)
        papers_with_scores = result.all()
        
        return [(paper, float(score)) for paper, score in papers_with_scores]

    async def bm25_search_papers_with_filters(
        self,
        query: str,
        limit: int = 100,
        author_name: Optional[str] = None,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
        venue: Optional[str] = None,
        min_citation_count: Optional[int] = None,
        max_citation_count: Optional[int] = None,
    ) -> List[tuple[DBPaper, float]]:
        """
        BM25-style lexical search with filters.

        Returns rank-ready candidates with lexical scores only.
        """
        from sqlalchemy import func, text, and_
        from app.models.authors import DBAuthor, DBAuthorPaper
        
        ts_query = func.websearch_to_tsquery("english", query)

        title_vector = func.setweight(
            func.to_tsvector("english", func.coalesce(DBPaper.title, "")), 
            literal_column("'A'::\"char\"")
        )
        abstract_vector = func.setweight(
            func.to_tsvector("english", func.coalesce(DBPaper.abstract, "")), 
            literal_column("'B'::\"char\"")
        )
        ts_vector = title_vector.op('||')(abstract_vector)
        bm25_score = func.ts_rank_cd(ts_vector, ts_query)

        query_stmt = select(DBPaper, bm25_score.label("bm25_score")).where(
            func.coalesce(bm25_score, 0) > 0
        )

        filter_conditions = []

        if author_name:
            query_stmt = query_stmt.join(
                DBAuthorPaper, DBPaper.paper_id == DBAuthorPaper.paper_id
            ).join(DBAuthor, DBAuthorPaper.author_id == DBAuthor.author_id)
            filter_conditions.append(DBAuthor.name.ilike(f"%{author_name}%"))

        if year_min is not None:
            filter_conditions.append(DBPaper.year >= year_min)
        if year_max is not None:
            filter_conditions.append(DBPaper.year <= year_max)
        if venue:
            filter_conditions.append(DBPaper.venue.ilike(f"%{venue}%"))
        if min_citation_count is not None:
            filter_conditions.append(DBPaper.citation_count >= min_citation_count)
        if max_citation_count is not None:
            filter_conditions.append(DBPaper.citation_count <= max_citation_count)

        if filter_conditions:
            query_stmt = query_stmt.where(and_(*filter_conditions))

        query_stmt = query_stmt.order_by(text("bm25_score DESC")).limit(limit)

        if author_name:
            query_stmt = query_stmt.distinct(DBPaper.paper_id)

        result = await self.db.execute(query_stmt)
        return [(paper, float(score)) for paper, score in result.all()]

    async def semantic_search_papers_with_filters(
        self,
        query_embedding: List[float],
        limit: int = 100,
        author_name: Optional[str] = None,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
        venue: Optional[str] = None,
        min_citation_count: Optional[int] = None,
        max_citation_count: Optional[int] = None,
    ) -> List[tuple[DBPaper, float]]:
        """
        Pure semantic (vector) search with filters.

        Returns rank-ready candidates with semantic scores only.
        """
        from sqlalchemy import text, and_
        from app.models.authors import DBAuthor, DBAuthorPaper

        semantic_score = 1 - DBPaper.embedding.cosine_distance(query_embedding)

        query_stmt = select(DBPaper, semantic_score.label("semantic_score")).where(
            DBPaper.embedding.isnot(None)
        )

        filter_conditions = []

        if author_name:
            query_stmt = query_stmt.join(
                DBAuthorPaper, DBPaper.paper_id == DBAuthorPaper.paper_id
            ).join(DBAuthor, DBAuthorPaper.author_id == DBAuthor.author_id)
            filter_conditions.append(DBAuthor.name.ilike(f"%{author_name}%"))

        if year_min is not None:
            filter_conditions.append(DBPaper.year >= year_min)
        if year_max is not None:
            filter_conditions.append(DBPaper.year <= year_max)
        if venue:
            filter_conditions.append(DBPaper.venue.ilike(f"%{venue}%"))
        if min_citation_count is not None:
            filter_conditions.append(DBPaper.citation_count >= min_citation_count)
        if max_citation_count is not None:
            filter_conditions.append(DBPaper.citation_count <= max_citation_count)

        if filter_conditions:
            query_stmt = query_stmt.where(and_(*filter_conditions))

        query_stmt = query_stmt.order_by(text("semantic_score DESC")).limit(limit)

        if author_name:
            query_stmt = query_stmt.distinct(DBPaper.paper_id)

        result = await self.db.execute(query_stmt)
        return [(paper, float(score)) for paper, score in result.all()]
    
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
