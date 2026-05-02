"""
Chunk repository for database operations
"""
import re
import unicodedata
from typing import List, Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.models.papers import DBPaperChunk
from app.extensions.logger import create_logger
from app.utils.search_utils import build_paradedb_query

logger = create_logger(__name__)

def sanitize_text(text: Optional[str]) -> Optional[str]:
    """Remove null bytes and other problematic characters from text for PostgreSQL.
    
    PostgreSQL doesn't support null bytes (\x00 or \u0000) in text fields.
    This function removes them to prevent insertion errors.
    
    Args:
        text: Input text that may contain null bytes
        
    Returns:
        Sanitized text with null bytes removed, or None if input is None
    """
    if text is None:
        return None
    # Remove null bytes and strip any resulting whitespace
    return text.replace('\x00', '').replace('\u0000', '')

class ChunkRepository:
    """Repository for chunk database operations"""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def create_chunk(
        self,
        chunk_id: str,
        paper_id: str,
        text: str,
        token_count: int,
        chunk_index: int,
        embedding: List[float],
        section_title: Optional[str] = None,
        page_number: Optional[int] = None,
        label: Optional[str] = None,
        level: Optional[int] = None,
        char_start: Optional[int] = None,
        char_end: Optional[int] = None,
        docling_metadata: Optional[Dict[str, Any]] = None,
        defer_commit: bool = False,
    ) -> DBPaperChunk:
        """Create a paper chunk
        
        Args:
            defer_commit: If True, don't commit immediately (for batch operations)
        """
        # Sanitize text fields to remove null bytes
        sanitized_text = sanitize_text(text)
        sanitized_section_title = sanitize_text(section_title)
        
        db_chunk = DBPaperChunk(
            chunk_id=chunk_id,
            paper_id=paper_id,
            text=sanitized_text,
            token_count=token_count,
            chunk_index=chunk_index,
            section_title=sanitized_section_title,
            page_number=page_number,
            label=label,
            level=level,
            char_start=char_start,
            char_end=char_end,
            docling_metadata=docling_metadata,
            embedding=embedding,
        )
        
        self.db.add(db_chunk)
        
        if not defer_commit:
            await self.db.commit()
            await self.db.refresh(db_chunk)
        
        return db_chunk
    
    async def get_chunks_by_paper_id(self, paper_id: str) -> List[DBPaperChunk]:
        """Get all chunks for a paper"""
        result = await self.db.execute(
            select(DBPaperChunk)
            .where(DBPaperChunk.paper_id == paper_id)
            .order_by(DBPaperChunk.chunk_index)
        )
        return list(result.scalars().all())
    
    async def get_chunk_by_id(self, chunk_id: str) -> Optional[DBPaperChunk]:
        """Get a single chunk by chunk_id"""
        result = await self.db.execute(
            select(DBPaperChunk).where(DBPaperChunk.chunk_id == chunk_id)
        )
        return result.scalar_one_or_none()
    
    async def search_similar_chunks(
        self,
        query_embedding: List[float],
        limit: int = 40,
        paper_ids: Optional[List[str]] = None
    ) -> List[tuple[DBPaperChunk, float]]:
        """
        Search for similar chunks using vector similarity
        
        Args:
            query_embedding: Query embedding vector
            limit: Number of results to return
            paper_ids: Optional list of paper IDs to restrict search
            
        Returns:
            List of tuples (chunk, similarity_score) ordered by similarity descending
            Similarity scores range from 0 to 1, where 1 is most similar
        """
        # Calculate similarity as (1 - cosine_distance)
        similarity = 1 - DBPaperChunk.embedding.cosine_distance(query_embedding)
        
        query = select(DBPaperChunk, similarity.label('similarity_score')).order_by(
            DBPaperChunk.embedding.cosine_distance(query_embedding)
        ).limit(limit)
        
        # Filter by paper IDs if provided
        if paper_ids:
            query = query.where(DBPaperChunk.paper_id.in_(paper_ids))
        
        result = await self.db.execute(query)
        chunks_with_scores = result.all()
        
        # Return as list of tuples (chunk, score)
        return [(chunk, float(score)) for chunk, score in chunks_with_scores]
    
    async def hybrid_search_chunks(
        self,
        query: str,
        query_embedding: List[float],
        limit: int = 40,
        paper_ids: Optional[List[str]] = None,
        bm25_weight: float = 0.4,
        semantic_weight: float = 0.6,
    ) -> List[tuple[DBPaperChunk, float]]:
        """
        Hybrid BM25 + semantic search on chunks.

        Uses database-level weighted Reciprocal Rank Fusion (RRF):
        - BM25 rank from ParadeDB (`pg_search`)
        - Semantic rank from pgvector cosine similarity
        - Combined in SQL as weighted RRF score

        Falls back to legacy in-memory fusion path if ParadeDB query fails.
        
        Args:
            query: Search query text for BM25
            query_embedding: Pre-computed query embedding vector
            limit: Maximum number of results
            paper_ids: Optional list of paper IDs to restrict search
            bm25_weight: Normalized weight for BM25 score
            semantic_weight: Normalized weight for semantic score
            
        Returns:
            List of tuples (chunk, combined_score) sorted by relevance
        """
        import paradedb
        from sqlalchemy import func, literal

        candidate_limit = max(limit * 3, 200)
        rrf_k = 60.0

        paradedb_query = build_paradedb_query(query, ["text", "section_title"])

        chunk_id_col = DBPaperChunk.__table__.c.id

        bm25_score_expr: Any = paradedb.score(chunk_id_col)  # type: ignore[attr-defined]
        bm25_hits = (
            select(
                chunk_id_col.label("id"),
                func.row_number()
                .over(order_by=bm25_score_expr.desc())
                .label("bm25_rank"),
            )
            .where(chunk_id_col.op("@@@")(paradedb_query))
            .limit(candidate_limit)
        )
        if paper_ids:
            bm25_hits = bm25_hits.where(DBPaperChunk.paper_id.in_(paper_ids))
        bm25_cte = bm25_hits.cte("bm25_hits")

        semantic_score_expr = 1 - DBPaperChunk.embedding.cosine_distance(query_embedding)
        semantic_hits = (
            select(
                DBPaperChunk.id.label("id"),
                func.row_number()
                .over(order_by=semantic_score_expr.desc())
                .label("semantic_rank"),
            )
            .where(DBPaperChunk.embedding.isnot(None))
            .limit(candidate_limit)
        )
        if paper_ids:
            semantic_hits = semantic_hits.where(DBPaperChunk.paper_id.in_(paper_ids))
        semantic_cte = semantic_hits.cte("semantic_hits")

        try:
            combined_id = func.coalesce(bm25_cte.c.id, semantic_cte.c.id)
            rrf_score = (
                func.coalesce(
                    bm25_weight
                    * (
                        literal(1.0)
                        / (literal(rrf_k) + bm25_cte.c.bm25_rank)
                    ),
                    literal(0.0),
                )
                + func.coalesce(
                    semantic_weight
                    * (
                        literal(1.0)
                        / (literal(rrf_k) + semantic_cte.c.semantic_rank)
                    ),
                    literal(0.0),
                )
            )

            query_stmt = (
                select(DBPaperChunk, rrf_score.label("relevance_score"))
                .select_from(
                    bm25_cte.join(
                        semantic_cte,
                        bm25_cte.c.id == semantic_cte.c.id,
                        full=True,
                    )
                )
                .join(DBPaperChunk, DBPaperChunk.id == combined_id)
                .order_by(rrf_score.desc())
                .limit(limit)
            )

            result = await self.db.execute(query_stmt)
            return [(chunk, float(score)) for chunk, score in result.all()]
        except Exception as exc:
            logger.warning(
                f"Database RRF chunk search failed, falling back to in-memory fusion: {exc}"
            )
            try:
                await self.db.rollback()
            except Exception:
                logger.debug("Failed to rollback after chunk DB-RRF error", exc_info=True)

            bm25_candidates = await self.bm25_search(
                query=query,
                limit=candidate_limit,
                paper_ids=paper_ids,
            )
            semantic_candidates = await self.search_similar_chunks(
                query_embedding=query_embedding,
                limit=candidate_limit,
                paper_ids=paper_ids,
            )

            if not bm25_candidates and not semantic_candidates:
                return []

            bm25_rank = {
                str(chunk.chunk_id): rank
                for rank, (chunk, _score) in enumerate(bm25_candidates, start=1)
            }
            semantic_rank = {
                str(chunk.chunk_id): rank
                for rank, (chunk, _score) in enumerate(semantic_candidates, start=1)
            }

            chunk_map: Dict[str, DBPaperChunk] = {}
            for chunk, _ in bm25_candidates:
                chunk_map[str(chunk.chunk_id)] = chunk
            for chunk, _ in semantic_candidates:
                chunk_map[str(chunk.chunk_id)] = chunk

            fused_scores: List[tuple[DBPaperChunk, float]] = []
            for chunk_id, chunk in chunk_map.items():
                rank_bm25 = bm25_rank.get(chunk_id)
                rank_semantic = semantic_rank.get(chunk_id)
                score = 0.0
                if rank_bm25 is not None:
                    score += bm25_weight * (1.0 / (rrf_k + rank_bm25))
                if rank_semantic is not None:
                    score += semantic_weight * (1.0 / (rrf_k + rank_semantic))
                fused_scores.append((chunk, float(score)))

            fused_scores.sort(key=lambda item: item[1], reverse=True)
            return fused_scores[:limit]

    async def _bm25_search_tsrank(
        self,
        query: str,
        limit: int = 40,
        paper_ids: Optional[List[str]] = None,
    ) -> List[tuple[DBPaperChunk, float]]:
        """
        Fallback BM25-style lexical search via PostgreSQL full-text ranking.
        """
        from sqlalchemy import func, text

        ts_query = func.websearch_to_tsquery("english", query)

        text_vector = func.setweight(
            func.to_tsvector("english", func.coalesce(DBPaperChunk.text, "")),
            text("'A'::\"char\""),
        )
        section_vector = func.setweight(
            func.to_tsvector("english", func.coalesce(DBPaperChunk.section_title, "")),
            text("'B'::\"char\""),
        )
        ts_vector = text_vector.op("||")(section_vector)
        bm25_score = func.ts_rank_cd(ts_vector, ts_query)

        query_stmt = select(DBPaperChunk, bm25_score.label("bm25_score")).where(
            func.coalesce(bm25_score, 0) > 0
        )

        if paper_ids:
            query_stmt = query_stmt.where(DBPaperChunk.paper_id.in_(paper_ids))

        query_stmt = query_stmt.order_by(text("bm25_score DESC")).limit(limit)

        result = await self.db.execute(query_stmt)
        return [(chunk, float(score)) for chunk, score in result.all()]

    async def bm25_search(
        self,
        query: str,
        limit: int = 40,
        paper_ids: Optional[List[str]] = None,
    ) -> List[tuple[DBPaperChunk, float]]:
        """
        BM25 lexical search via ParadeDB `pg_search` index.

        Falls back to legacy ts_rank BM25 when ParadeDB is unavailable
        or query execution fails.
        """
        import paradedb
        from sqlalchemy import text

        chunk_id_col = DBPaperChunk.__table__.c.id
        paradedb_query = build_paradedb_query(query, ["text", "section_title"])

        try:
            candidate_limit = max(limit * 3, 200)

            query_stmt = (
                select(
                    chunk_id_col.label("id"),
                    paradedb.score(chunk_id_col).label("bm25_score"),  # type: ignore[attr-defined]
                )
                .where(chunk_id_col.op("@@@")(paradedb_query))
                .order_by(text("bm25_score DESC"))
                .limit(candidate_limit)
            )

            if paper_ids:
                query_stmt = query_stmt.where(DBPaperChunk.paper_id.in_(paper_ids))

            raw_hits = await self.db.execute(query_stmt)
            hits = raw_hits.fetchall()

            if not hits:
                return []

            ordered_ids = [int(row.id) for row in hits if row.id is not None]
            score_map = {
                int(row.id): float(row.bm25_score)
                for row in hits
                if row.id is not None
            }

            chunk_query = select(DBPaperChunk).where(DBPaperChunk.id.in_(ordered_ids))
            if paper_ids:
                chunk_query = chunk_query.where(DBPaperChunk.paper_id.in_(paper_ids))

            result = await self.db.execute(chunk_query)
            filtered_chunks = result.scalars().all()
            chunk_map = {int(chunk.id): chunk for chunk in filtered_chunks}

            ranked: List[tuple[DBPaperChunk, float]] = []
            for row_id in ordered_ids:
                chunk = chunk_map.get(row_id)
                if not chunk:
                    continue
                ranked.append((chunk, score_map.get(row_id, 0.0)))
                if len(ranked) >= limit:
                    break

            return ranked
        except Exception as exc:
            logger.warning(
                f"ParadeDB BM25 chunk search failed, falling back to ts_rank BM25: {exc}"
            )
            try:
                await self.db.rollback()
            except Exception:
                logger.debug("Failed to rollback after ParadeDB BM25 chunk error", exc_info=True)
            return await self._bm25_search_tsrank(
                query=query,
                limit=limit,
                paper_ids=paper_ids,
            )
    
    async def delete_chunks_by_paper_id(self, paper_id: str) -> int:
        """
        Delete all chunks for a paper
        
        Args:
            paper_id: Paper ID to delete chunks for
            
        Returns:
            Number of chunks deleted
        """
        from sqlalchemy import delete as sql_delete
        
        result = await self.db.execute(
            sql_delete(DBPaperChunk).where(DBPaperChunk.paper_id == paper_id)
        )
        await self.db.commit()
        
        return result.rowcount # type: ignore
