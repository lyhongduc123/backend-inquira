from typing import List, Optional, Dict, Any, TYPE_CHECKING, Tuple
from dataclasses import dataclass
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, and_, literal_column, func, desc, bindparam
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import selectinload, joinedload
from datetime import datetime, date
from app.models.papers import DBPaper, DBPaperChunk
from app.models.citations import DBCitation
from app.models.authors import DBAuthor
from app.extensions.logger import create_logger
from app.utils.identifier_normalization import (
    normalize_external_ids,
    external_id_key_candidates,
)
from app.utils.search_utils import build_paradedb_query

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

@dataclass
class SearchFilterOptions:
    """Unified filter options for paper search."""

    author_name: Optional[str] = None
    year_min: Optional[int] = None
    year_max: Optional[int] = None
    venue: Optional[str] = None
    min_citation_count: Optional[int] = None
    max_citation_count: Optional[int] = None
    journal_quartile: Optional[str] = None
    field_of_study: Optional[List[str]] = None

class PaperRepository:
    """Repository for paper database operations"""
    
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_paper_citations_from_db(
        self,
        paper_id: str,
        offset: int = 0,
        limit: int = 100,
    ) -> Dict[str, Any]:
        """Get citation list from local database for a paper."""
        source_paper = await self.get_paper_by_id(paper_id)
        if not source_paper:
            return {"offset": offset, "next": None, "total": 0, "data": []}

        total_result = await self.db.execute(
            select(func.count(DBCitation.id)).where(
                DBCitation.cited_paper_id == source_paper.id
            )
        )
        total = int(total_result.scalar_one() or 0)

        result = await self.db.execute(
            select(DBCitation, DBPaper)
            .join(DBPaper, DBPaper.id == DBCitation.citing_paper_id)
            .where(DBCitation.cited_paper_id == source_paper.id)
            .order_by(desc(DBCitation.created_at), desc(DBCitation.id))
            .offset(offset)
            .limit(limit)
        )

        rows = result.all()
        data: List[Dict[str, Any]] = []

        for citation, citing_paper in rows:
            data.append(
                {
                    "citingPaper": {
                        "paper_id": citing_paper.paper_id,
                        "title": citing_paper.title,
                        "abstract": citing_paper.abstract,
                        "authors": [],
                        "year": citing_paper.year,
                        "venue": citing_paper.venue,
                        "citation_count": citing_paper.citation_count,
                    },
                }
            )

        next_offset = offset + limit if (offset + limit) < total else None
        return {
            "offset": offset,
            "next": next_offset,
            "total": total,
            "data": data,
        }

    async def get_paper_references_from_db(
        self,
        paper_id: str,
        offset: int = 0,
        limit: int = 100,
    ) -> Dict[str, Any]:
        """Get reference list from local database for a paper."""
        source_paper = await self.get_paper_by_id(paper_id)
        if not source_paper:
            return {"offset": offset, "next": None, "total": 0, "data": []}

        total_result = await self.db.execute(
            select(func.count(DBCitation.id)).where(
                DBCitation.citing_paper_id == source_paper.id
            )
        )
        total = int(total_result.scalar_one() or 0)

        result = await self.db.execute(
            select(DBCitation, DBPaper)
            .join(DBPaper, DBPaper.id == DBCitation.cited_paper_id)
            .where(DBCitation.citing_paper_id == source_paper.id)
            .order_by(desc(DBCitation.created_at), desc(DBCitation.id))
            .offset(offset)
            .limit(limit)
        )

        rows = result.all()
        data: List[Dict[str, Any]] = []

        for citation, cited_paper in rows:
            data.append(
                {
                    "citedPaper": {
                        "paper_id": cited_paper.paper_id,
                        "corpus_id": None,
                        "title": cited_paper.title,
                        "abstract": cited_paper.abstract,
                        "authors": [],
                        "year": cited_paper.year,
                        "venue": cited_paper.venue,
                        "citation_count": cited_paper.citation_count,
                    },
                    "isInfluential": citation.is_influential,
                    "contexts": None,
                    "intents": [citation.intent] if citation.intent else [],
                }
            )

        next_offset = offset + limit if (offset + limit) < total else None
        return {
            "offset": offset,
            "next": next_offset,
            "total": total,
            "data": data,
        }
    
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
            source: Optional source filter (e.g., 'semantic_scholar', 'open_alex')
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
        from sqlalchemy import func

        normalized_external_ids = normalize_external_ids(external_ids)

        # Try to find by DOI first, then other IDs
        doi = normalized_external_ids.get('doi')
        if doi:
            doi_candidates = external_id_key_candidates('doi')
            for doi_key in doi_candidates:
                result = await self.db.execute(
                    select(DBPaper).where(
                        and_(
                            func.lower(DBPaper.external_ids[doi_key].astext) == str(doi).lower(),
                            DBPaper.source == source,
                        )
                    )
                )
                paper = result.scalar_one_or_none()
                if paper:
                    return paper
        
        # Fallback to other IDs
        for key, value in normalized_external_ids.items():
            if value:
                for key_candidate in external_id_key_candidates(key):
                    result = await self.db.execute(
                        select(DBPaper).where(
                            and_(
                                DBPaper.external_ids[key_candidate].astext == str(value),
                                DBPaper.source == source,
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

    async def update_paper_embeddings_bulk(
        self,
        paper_embeddings: Dict[str, List[float]],
    ) -> int:
        """Bulk update title+abstract embeddings for multiple papers in one transaction."""
        if not paper_embeddings:
            return 0

        updated = 0
        for paper_id, embedding in paper_embeddings.items():
            if not embedding:
                continue

            await self.db.execute(
                update(DBPaper)
                .where(DBPaper.paper_id == paper_id)
                .values(
                    embedding=embedding,
                    updated_at=datetime.utcnow(),
                )
            )
            updated += 1

        if updated <= 0:
            return 0

        await self.db.commit()
        return updated
    
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

    def apply_filters(
        self,
        query_stmt,
        *,
        filter_options: Optional[SearchFilterOptions] = None,
    ):
        """
        Apply reusable paper-search filters to a SQLAlchemy query.

        Returns:
            tuple(query_stmt, has_author_filter)
        """
        from sqlalchemy import and_, or_, cast, String, func
        from app.models.authors import DBAuthor, DBAuthorPaper
        from app.models.journals import DBJournal

        options = filter_options or SearchFilterOptions()

        filter_conditions = []
        has_author_filter = False

        if options.author_name:
            query_stmt = query_stmt.join(
                DBAuthorPaper,
                DBPaper.id == DBAuthorPaper.paper_id,
            ).join(
                DBAuthor,
                DBAuthorPaper.author_id == DBAuthor.id,
            )
            filter_conditions.append(
                or_(
                    DBAuthor.name.ilike(f"%{options.author_name}%"),
                    DBAuthor.display_name.ilike(f"%{options.author_name}%"),
                    DBAuthorPaper.author_string.ilike(f"%{options.author_name}%"),
                    cast(DBAuthor.external_ids, String).ilike(f"%{options.author_name}%"),
                )
            )
            has_author_filter = True

        if options.year_min is not None:
            filter_conditions.append(DBPaper.year >= options.year_min)
        if options.year_max is not None:
            filter_conditions.append(DBPaper.year <= options.year_max)
        if options.venue:
            filter_conditions.append(DBPaper.venue.ilike(f"%{options.venue}%"))
        if options.min_citation_count is not None:
            filter_conditions.append(DBPaper.citation_count >= options.min_citation_count)
        if options.max_citation_count is not None:
            filter_conditions.append(DBPaper.citation_count <= options.max_citation_count)

        if options.journal_quartile:
            query_stmt = query_stmt.join(DBJournal, DBPaper.journal_id == DBJournal.id)
            filter_conditions.append(
                func.upper(cast(DBJournal.sjr_best_quartile, String))
                == options.journal_quartile.upper()
            )

        if options.field_of_study:
            fos_terms = [term.strip() for term in options.field_of_study if term and term.strip()]
            if fos_terms:
                filter_conditions.append(
                    or_(
                        *[
                            cast(DBPaper.fields_of_study, String).ilike(f"%{term}%")
                            for term in fos_terms
                        ]
                    )
                )

        if filter_conditions:
            query_stmt = query_stmt.where(and_(*filter_conditions))

        return query_stmt, has_author_filter

    async def hybrid_search(
        self,
        query: str,
        query_embedding: List[float],
        limit: int = 100,
        bm25_weight: float = 0.3,
        semantic_weight: float = 0.7,
        rrf_only: bool = False,
        filter_options: Optional[SearchFilterOptions] = None,
    ) -> List[tuple[DBPaper, float]]:
        """
        Hybrid search using ParadeDB BM25 + semantic vector retrieval.

        Fusion strategy:
        - Always applies Reciprocal Rank Fusion (RRF) on BM25/semantic ranks.
        - Optionally adds weighted point-score fusion (normalized BM25 + semantic)
          when ``rrf_only`` is False.
        
        Args:
            query: Search query text
            query_embedding: Pre-computed query embedding vector
            limit: Maximum number of results
            bm25_weight: Weight for BM25 score in point-score fusion
            semantic_weight: Weight for semantic score in point-score fusion
            rrf_only: If True, use pure RRF only (no point-score addition)
            
        Returns:
            List of tuples (paper, combined_score) sorted by relevance
        """
        # Pull broader candidate sets from each ranker, then fuse and cut to limit.
        candidate_limit = max(limit * 3, 200)
        rrf_k = 60.0

        bm25_candidates = await self.bm25_search(
            query=query,
            limit=candidate_limit,
            filter_options=filter_options,
        )
        semantic_candidates = await self.semantic_search(
            query_embedding=query_embedding,
            limit=candidate_limit,
            filter_options=filter_options,
        )

        if not bm25_candidates and not semantic_candidates:
            return []

        def _to_rank_map(candidates: List[tuple[DBPaper, float]]) -> Dict[str, int]:
            return {
                str(paper.paper_id): rank
                for rank, (paper, _score) in enumerate(candidates, start=1)
            }

        def _to_score_map(candidates: List[tuple[DBPaper, float]]) -> Dict[str, float]:
            return {str(paper.paper_id): float(score) for paper, score in candidates}

        bm25_rank = _to_rank_map(bm25_candidates)
        semantic_rank = _to_rank_map(semantic_candidates)
        bm25_score_map = _to_score_map(bm25_candidates)
        semantic_score_map = _to_score_map(semantic_candidates)

        bm25_max = max(bm25_score_map.values()) if bm25_score_map else 0.0
        semantic_max = max(semantic_score_map.values()) if semantic_score_map else 0.0

        paper_map: Dict[str, DBPaper] = {}
        for paper, _ in bm25_candidates:
            paper_map[str(paper.paper_id)] = paper
        for paper, _ in semantic_candidates:
            paper_map[str(paper.paper_id)] = paper

        fused_scores: List[tuple[DBPaper, float]] = []
        for paper_id, paper in paper_map.items():
            rank_bm25 = bm25_rank.get(paper_id)
            rank_semantic = semantic_rank.get(paper_id)

            rrf_score = 0.0
            if rank_bm25 is not None:
                rrf_score += 1.0 / (rrf_k + rank_bm25)
            if rank_semantic is not None:
                rrf_score += 1.0 / (rrf_k + rank_semantic)

            final_score = rrf_score
            if not rrf_only:
                bm25_component = (
                    (bm25_score_map.get(paper_id, 0.0) / bm25_max) if bm25_max > 0 else 0.0
                )
                semantic_component = (
                    (semantic_score_map.get(paper_id, 0.0) / semantic_max)
                    if semantic_max > 0
                    else 0.0
                )
                final_score += (bm25_component * bm25_weight) + (
                    semantic_component * semantic_weight
                )

            fused_scores.append((paper, float(final_score)))

        fused_scores.sort(key=lambda item: item[1], reverse=True)
        return fused_scores[:limit]

    async def _bm25_search_tsrank(
        self,
        query: str,
        limit: int = 100,
        filter_options: Optional[SearchFilterOptions] = None,
    ) -> List[tuple[DBPaper, float]]:
        """
        BM25-style lexical search with filters.

        Returns rank-ready candidates with lexical scores only.
        """
        from sqlalchemy import cast, func, text, String
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
        author_string_vector = func.setweight(
            func.to_tsvector("english", func.coalesce(DBAuthorPaper.author_string, "")),
            literal_column("'A'::\"char\""),
        )
        author_name_vector = func.setweight(
            func.to_tsvector("english", func.coalesce(DBAuthor.name, "")),
            literal_column("'A'::\"char\""),
        )
        author_display_name_vector = func.setweight(
            func.to_tsvector("english", func.coalesce(DBAuthor.display_name, "")),
            literal_column("'B'::\"char\""),
        )
        author_external_ids_vector = func.setweight(
            func.to_tsvector("english", func.coalesce(cast(DBAuthor.external_ids, String), "")),
            literal_column("'D'::\"char\""),
        )

        ts_vector = (
            title_vector
            .op('||')(abstract_vector)
            .op('||')(author_string_vector)
            .op('||')(author_name_vector)
            .op('||')(author_display_name_vector)
            .op('||')(author_external_ids_vector)
        )
        bm25_score = func.ts_rank_cd(ts_vector, ts_query)

        query_stmt = (
            select(DBPaper, bm25_score.label("bm25_score"))
            .outerjoin(DBAuthorPaper, DBPaper.id == DBAuthorPaper.paper_id)
            .outerjoin(DBAuthor, DBAuthorPaper.author_id == DBAuthor.id)
            .where(func.coalesce(bm25_score, 0) > 0)
        )
        
        query_stmt, has_author_filter = self.apply_filters(
            query_stmt,
            filter_options=filter_options,
        )

        query_stmt = query_stmt.order_by(text("bm25_score DESC")).limit(limit)

        if has_author_filter:
            query_stmt = query_stmt.distinct(DBPaper.paper_id)

        result = await self.db.execute(query_stmt)
        ranked_rows = result.all()

        # Joins with author tables can duplicate papers; keep the best lexical score.
        deduped: Dict[int, tuple[DBPaper, float]] = {}
        for paper, score in ranked_rows:
            row_id = int(paper.id)
            row_score = float(score)
            existing = deduped.get(row_id)
            if existing is None or row_score > existing[1]:
                deduped[row_id] = (paper, row_score)

        return sorted(deduped.values(), key=lambda item: item[1], reverse=True)[:limit]

    async def _search_author_lexical_candidates(
        self,
        *,
        query: str,
        limit: int,
    ) -> List[tuple[int, float]]:
        """Lexical author-side search on `author_papers` and `authors` tables."""
        from sqlalchemy import cast, func, text, String
        from app.models.authors import DBAuthor, DBAuthorPaper

        ts_query = func.websearch_to_tsquery("english", query)
        author_string_vector = func.setweight(
            func.to_tsvector("english", func.coalesce(DBAuthorPaper.author_string, "")),
            literal_column("'A'::\"char\""),
        )
        author_name_vector = func.setweight(
            func.to_tsvector("english", func.coalesce(DBAuthor.name, "")),
            literal_column("'A'::\"char\""),
        )
        author_display_name_vector = func.setweight(
            func.to_tsvector("english", func.coalesce(DBAuthor.display_name, "")),
            literal_column("'B'::\"char\""),
        )
        author_external_ids_vector = func.setweight(
            func.to_tsvector("english", func.coalesce(cast(DBAuthor.external_ids, String), "")),
            literal_column("'D'::\"char\""),
        )

        author_ts_vector = (
            author_string_vector
            .op('||')(author_name_vector)
            .op('||')(author_display_name_vector)
            .op('||')(author_external_ids_vector)
        )
        bm25_score = func.ts_rank_cd(author_ts_vector, ts_query)

        stmt = (
            select(
                DBAuthorPaper.paper_id.label("id"),
                func.max(bm25_score).label("bm25_score"),
            )
            .join(DBAuthor, DBAuthorPaper.author_id == DBAuthor.id)
            .where(func.coalesce(bm25_score, 0) > 0)
            .group_by(DBAuthorPaper.paper_id)
            .order_by(text("bm25_score DESC"))
            .limit(limit)
        )

        rows = await self.db.execute(stmt)
        return [(int(row.id), float(row.bm25_score)) for row in rows.fetchall() if row.id is not None]

    async def bm25_search(
        self,
        query: str,
        limit: int = 100,
        filter_options: Optional[SearchFilterOptions] = None,
    ) -> List[tuple[DBPaper, float]]:
        """
        BM25 lexical search via ParadeDB `pg_search` index.

        Falls back to legacy `ts_rank_cd` BM25 when ParadeDB is unavailable
        or query execution fails.
        """
        import paradedb
        from sqlalchemy import text

        paper_id_col = DBPaper.__table__.c.id
        paradedb_query = build_paradedb_query(query, ["title", "abstract"])

        try:
            candidate_limit = max(limit * 3, 200)

            raw_hits = await self.db.execute(
                select(
                    paper_id_col.label("id"),
                    paradedb.score(paper_id_col).label("bm25_score"),  # type: ignore[attr-defined]
                )
                .where(paper_id_col.op("@@@")(paradedb_query))
                .order_by(text("bm25_score DESC"))
                .limit(candidate_limit)
            )

            hits = raw_hits.fetchall()
            author_hits = await self._search_author_lexical_candidates(
                query=query,
                limit=candidate_limit,
            )

            if not hits and not author_hits:
                return []

            ordered_ids: List[int] = [int(row.id) for row in hits if row.id is not None]
            score_map: Dict[int, float] = {
                int(row.id): float(row.bm25_score)
                for row in hits
                if row.id is not None
            }

            # Merge author-side lexical hits so BM25 also covers author metadata.
            for author_paper_id, author_score in author_hits:
                existing_score = score_map.get(author_paper_id)
                if existing_score is None:
                    score_map[author_paper_id] = author_score
                    ordered_ids.append(author_paper_id)
                elif author_score > existing_score:
                    score_map[author_paper_id] = author_score

            ordered_ids = sorted(
                set(ordered_ids),
                key=lambda row_id: score_map.get(row_id, 0.0),
                reverse=True,
            )

            query_stmt = select(DBPaper).where(DBPaper.id.in_(ordered_ids))

            query_stmt, has_author_filter = self.apply_filters(
                query_stmt,
                filter_options=filter_options,
            )

            if has_author_filter:
                query_stmt = query_stmt.distinct(DBPaper.paper_id)

            result = await self.db.execute(query_stmt)
            filtered_papers = result.scalars().all()
            paper_map = {int(p.id): p for p in filtered_papers}

            ranked: List[tuple[DBPaper, float]] = []
            for row_id in ordered_ids:
                paper = paper_map.get(row_id)
                if not paper:
                    continue
                ranked.append((paper, score_map.get(row_id, 0.0)))
                if len(ranked) >= limit:
                    break

            return ranked
        except Exception as exc:
            logger.warning(
                f"ParadeDB BM25 search failed, falling back to ts_rank BM25: {exc}"
            )
            # ParadeDB parse/execution errors can leave the current transaction
            # in an aborted state on asyncpg. Clear it before issuing fallback SQL.
            try:
                await self.db.rollback()
            except Exception:
                logger.debug("Failed to rollback after ParadeDB BM25 error", exc_info=True)
            return await self._bm25_search_tsrank(
                query=query,
                limit=limit,
                filter_options=filter_options,
            )

    async def semantic_search(
        self,
        query_embedding: List[float],
        limit: int = 100,
        filter_options: Optional[SearchFilterOptions] = None,
    ) -> List[tuple[DBPaper, float]]:
        """
        Pure semantic (vector) search with filters.

        Returns rank-ready candidates with semantic scores only.
        """
        from sqlalchemy import text

        semantic_score = 1 - DBPaper.embedding.cosine_distance(query_embedding)

        query_stmt = select(DBPaper, semantic_score.label("semantic_score")).where(
            DBPaper.embedding.isnot(None)
        )

        query_stmt, has_author_filter = self.apply_filters(
            query_stmt,
            filter_options=filter_options,
        )

        query_stmt = query_stmt.order_by(text("semantic_score DESC")).limit(limit)

        if has_author_filter:
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
