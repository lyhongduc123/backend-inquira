import gc
from typing import AsyncGenerator, Dict, List, Optional, TYPE_CHECKING, Union, Tuple, Any
import asyncio
from app.core.dtos.paper import PaperEnrichedDTO
from app.core.singletons import (
    get_extractor_service,
    get_chunker_service,
    get_embedding_service,
    get_summarizer_service,
)
from .services.chunker import ChunkWithMetadata
from app.domain.papers import PaperRepository, PaperService
from app.domain.chunks.repository import ChunkRepository
from app.domain.authors import AuthorService
from app.domain.institutions import InstitutionService
from numpy import dot
from numpy.linalg import norm

if TYPE_CHECKING:
    from app.retriever.service import RetrievalService
    from .services.summarizer import SummarizerService
    from .services.extractor import ExtractorService
    from .services.chunker import ChunkingService
    from .services.embeddings import EmbeddingService

from app.extensions.logger import create_logger

logger = create_logger(__name__)


class PaperProcessor:
    def __init__(
        self,
        repository: PaperRepository,
        chunk_repository: Optional[ChunkRepository] = None,
        retrieval_service: Optional["RetrievalService"] = None,
        extractor_service: Optional["ExtractorService"] = None,
        chunker_service: Optional["ChunkingService"] = None,
        embedding_service: Optional["EmbeddingService"] = None,
        summarizer_service: Optional["SummarizerService"] = None,
    ):
        """Initialize PaperProcessor with dependency injection.

        Args:
            repository: Required paper repository
            chunk_repository: Optional chunk repository (created if not provided)
            retrieval_service: Optional RetrievalService (created if not provided)
            extractor_service: Optional ExtractorService (singleton if not provided)
            chunker_service: Optional ChunkingService (singleton if not provided)
            embedding_service: Optional EmbeddingService (singleton if not provided)
            summarizer_service: Optional SummarizerService (singleton if not provided)
        """
        self.repository = repository
        
        self.retrieval_service = retrieval_service
        self.paper_service = PaperService(repository, self.retrieval_service)
        
        self.author_service = AuthorService(self.repository.db)
        self.institution_service = InstitutionService(self.repository.db)
        self.chunk_repository = chunk_repository or ChunkRepository(self.repository.db)

        # Use singletons for stateless services
        self.extractor_service = extractor_service or get_extractor_service()
        self.chunker_service = chunker_service or get_chunker_service()
        self.embedding_service = embedding_service or get_embedding_service()
        self.summarizer_service = summarizer_service or get_summarizer_service()

    async def process_single_paper(self, paper: PaperEnrichedDTO) -> bool:
        """
        Process a paper: retrieve full-text from available sources, chunk, embed, summarize

        Tries multiple sources (arXiv, open access PDFs, etc.) and gracefully
        handles paywalled papers by using abstract only.

        Args:
            db_paper: Database paper object

        Returns:
            True if successful, False otherwise
        """
        paper_id_str = str(paper.paper_id)
        processed = await self.ensure_paper_record(paper)
        if processed:
            return True

        # Lazy-load retrieval service if not injected
        if self.retrieval_service is None:
            from app.retriever.service import RetrievalService

            self.retrieval_service = RetrievalService(self.repository.db)

        resolved = await self.retrieval_service.resolve_paper_content(paper)
        if not resolved:
            logger.warning(
                f"Could not resolve content for paper {paper_id_str}, skipping processing."
            )
            await self.repository.update_paper_processing_status(paper_id_str, "failed")
            return False

        try:
            chunks = []
            if resolved.kind == "tei_xml":
                structure = self.extractor_service.extract_tei_xml_structure(
                    resolved.content  # pyright: ignore[reportArgumentType]
                )
                chunks = self.chunker_service.chunk_from_tei_structure(
                    structure, paper_id_str
                )
                # Clear structure from memory
                del structure

            elif resolved.kind == "pdf_bytes":
                doc_structure = self.extractor_service.extract_pdf_structure(
                    resolved.content  # pyright: ignore[reportArgumentType]
                )
                chunks = self.chunker_service.chunk_from_docling_structure(
                    doc_structure, paper_id_str
                )
                # Clear PDF bytes and structure from memory immediately after chunking
                del doc_structure

            # if not chunks:
            #     logger.warning(
            #         f"Structure-based chunking failed for {paper_id_str}, falling back to text-based"
            #     )
            #     full_text = self.extractor_service.extract_pdf_text(pdfBytes)
            #     clean_text = self.extractor_service._fix_text_encoding(full_text)
            #     chunks = self.chunker_service.chunk_text(clean_text, paper_id_str)
            #     extraction_method = "pdf_text"

            if not chunks:
                logger.error(
                    f"[{paper_id_str}] No chunks generated from any extraction method"
                )
                await self.repository.update_paper_processing_status(
                    paper_id_str, "failed"
                )
                return False

            # Clear resolved content from memory
            del resolved

            logger.info(f"[{paper_id_str}] Generated {len(chunks)} chunks from content")
            
            # Prepare chunk texts with section titles for better embeddings
            chunk_texts = []
            for chunk in chunks:
                # Prepend section title if available for better semantic context
                if chunk.section_title:
                    chunk_text = f"Section: {chunk.section_title}\n\n{chunk.text}"
                else:
                    chunk_text = chunk.text
                chunk_texts.append(chunk_text)
            
            embeddings = await self.embedding_service.create_embeddings_batch(
                chunk_texts, batch_size=10, task="search_document"  # Use search_document task
            )
            
            del chunk_texts

            await self._store_chunks(paper_id_str, chunks, embeddings)

            del chunks
            del embeddings

            await self.repository.update_paper_processing_status(
                paper_id_str, "completed"
            )

            gc.collect()

            return True
        except Exception as e:
            logger.error(f"Error processing paper {paper_id_str}: {e}")
            await self.repository.update_paper_processing_status(paper_id_str, "failed")

            # Also collect on error to prevent memory leaks
            gc.collect()

            return False

    async def process_papers(self, papers: List[PaperEnrichedDTO]) -> Dict[str, bool]:
        """
        Process multiple papers, optionally streaming progress events.
        Optimized to batch-check existing papers first.

        Args:
            papers: List of PaperEnrichedDTO objects

        Returns:
            Dict mapping paper_id to processing success status
        """
        results = {}
        
        # Batch check which papers already exist
        paper_ids = [str(p.paper_id) for p in papers]
        existing_papers = await self.paper_service.batch_check_existing_papers(paper_ids)
        
        logger.info(
            f"Batch check: {sum(existing_papers.values())} already exist, "
            f"{len(papers) - sum(existing_papers.values())} need processing"
        )
        
        # Process only papers that don't exist or aren't processed
        for paper in papers:
            paper_id = str(paper.paper_id)
            
            # Skip if already exists and processed
            if existing_papers.get(paper_id, False):
                results[paper_id] = True
                logger.debug(f"Paper {paper_id} already processed, skipping")
                continue
            
            success = await self.process_single_paper(paper)
            results[paper_id] = success
        
        return results

    async def process_papers_with_progress(
        self, papers: List[PaperEnrichedDTO]
    ) -> AsyncGenerator[tuple, None]:
        """
        Process multiple papers with progress streaming

        Args:
            papers: List of PaperPreprocess objects

        Yields:
            tuple: (paper_id, success, current_index, total_papers)
        """
        total = len(papers)
        for idx, paper in enumerate(papers, 1):
            paper_id = str(paper.paper_id)
            success = await self.process_single_paper(paper)
            yield (paper_id, success, idx, total)

    async def ensure_paper_record(self, paper: PaperEnrichedDTO) -> bool:
        """
        Ensure that a paper record exists in the database.
        If not, create it.
        Args:
            paper: PaperEnrichedDTO object
        Returns:
            DBPaper: Database paper object
        """
        try:
            result = await self.paper_service.create_paper_from_schema(paper)
            if result and result.is_processed:
                return True
        except Exception as e:
            logger.error(f"Error ensuring paper record for {paper.paper_id}: {e}")

        return False

    async def _store_chunks(
        self,
        paper_id: str,
        chunks: List[ChunkWithMetadata],
        embeddings: List[List[float]],
    ):
        """
        Store chunks and their embeddings in the database.
        Batches all chunks into a single transaction to avoid session state issues.

        Args:
            paper_id: Paper ID
            chunks: List of ChunkWithMetadata objects
            embeddings: List of embeddings corresponding to chunks
        """
        try:
            # Batch create all chunks - pass defer_commit flag
            for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
                await self.chunk_repository.create_chunk(
                    chunk_id=f"{paper_id}::C{idx}",
                    paper_id=paper_id,
                    text=chunk.text,
                    token_count=chunk.token_count,
                    section_title=chunk.section_title,
                    chunk_index=idx,
                    page_number=chunk.page_number,
                    label=chunk.label,
                    level=chunk.level,
                    char_start=chunk.char_start,
                    char_end=chunk.char_end,
                    docling_metadata=chunk.docling_metadata,
                    embedding=emb if emb is not None else [],
                    defer_commit=True,  # Don't commit until all chunks are added
                )
            # Commit all chunks at once
            await self.repository.db.commit()
            logger.debug(f"Stored {len(chunks)} chunks for paper {paper_id}")
        except Exception as e:
            logger.error(f"Error storing chunks for {paper_id}: {e}")
            await self.repository.db.rollback()
            raise

    async def generate_paper_embeddings(
        self, papers: List[PaperEnrichedDTO]
    ) -> List[PaperEnrichedDTO]:
        """
        Generate title + abstract embeddings for papers that don't have them.
        Uses existing embeddings from cache if available.

        Args:
            papers: List of paper DTOs

        Returns:
            Same papers with embeddings populated
        """
        papers_needing_embedding = []
        papers_with_embedding = []

        for paper in papers:
            if paper.embedding is not None:
                papers_with_embedding.append(paper)
                logger.debug(f"Paper {paper.paper_id} already has embedding (cached)")
            else:
                papers_needing_embedding.append(paper)

        if not papers_needing_embedding:
            logger.info("All papers already have embeddings")
            return papers

        logger.info(
            f"Generating embeddings for {len(papers_needing_embedding)} papers "
            f"({len(papers_with_embedding)} cached)"
        )

        # Generate embeddings for papers that need them
        texts = [f"{p.title}\n\n{p.abstract or ''}" for p in papers_needing_embedding]

        embeddings = await self.embedding_service.create_embeddings_batch(
            texts, batch_size=20
        )

        for paper, embedding in zip(papers_needing_embedding, embeddings):
            paper.embedding = embedding

        all_papers = papers_with_embedding + papers_needing_embedding
        logger.info(f"Generated embeddings for {len(papers_needing_embedding)} papers")

        return all_papers

    async def filter_papers_by_similarity(
        self,
        papers: List[PaperEnrichedDTO],
        query: str,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
        prefer_open_access: bool = True,
    ) -> List[PaperEnrichedDTO]:
        """
        Filter papers by semantic similarity to query.
        Prioritizes papers with PDF availability (open access or PDF URL).

        Args:
            papers: Papers with embeddings already generated
            query: Query text to compare against
            top_k: Number of top papers to return (if None, use min_score)
            min_score: Minimum similarity score (0-1, if None, use top_k)
            prefer_open_access: Boost papers with PDFs in ranking

        Returns:
            Filtered papers sorted by similarity (highest first)
        """
        if not papers:
            return []

        if any(p.embedding is None for p in papers):
            logger.warning("Some papers missing embeddings, generating now...")
            papers = await self.generate_paper_embeddings(papers)

        query_embedding = await self.embedding_service.create_embedding(query, task="search_query")

        scored_papers = []
        for paper in papers:
            if paper.embedding is None:
                continue
            similarity = dot(query_embedding, paper.embedding) / (
                norm(query_embedding) * norm(paper.embedding)
            )
            boost = 1.0
            if prefer_open_access:
                if paper.is_open_access or paper.pdf_url:
                    boost = 1.1 
            final_score = similarity * boost
            scored_papers.append((paper, final_score))

        scored_papers.sort(key=lambda x: x[1], reverse=True)

        if top_k is not None:
            filtered = scored_papers[:top_k]
        elif min_score is not None:
            filtered = [(p, s) for p, s in scored_papers if s >= min_score]
        else:
            filtered = [
                (p, s)
                for p, s in scored_papers
                if s >= 0.5 or p.is_open_access or p.pdf_url
            ]

        result_papers = [p for p, s in filtered]

        logger.info(
            f"Filtered {len(papers)} papers to {len(result_papers)} "
            f"(top_k={top_k}, min_score={min_score})"
        )


        if filtered:
            scores = [s for _, s in filtered]
            logger.info(
                f"Score range: {min(scores):.3f} - {max(scores):.3f}, "
                f"avg: {sum(scores)/len(scores):.3f}"
            )

        return result_papers

    async def process_papers_concurrent(
        self,
        papers: List[PaperEnrichedDTO],
        filtered_papers: Optional[List[PaperEnrichedDTO]] = None,
        max_workers: int = 4,
    ) -> Dict[str, bool]:
        """
        Process multiple papers concurrently (PDF → chunk → embed).
        Uses batch operations to create all paper records with authors/journals efficiently,
        then processes content concurrently.

        This prevents database connection conflicts while maintaining performance benefits.

        Args:
            papers: Papers to process
            filtered_papers: Optional pre-filtered papers to process (if None, processes all)
            max_workers: Maximum concurrent workers (default: 4)

        Returns:
            Dict mapping paper_id to success status
        """
        logger.info(f"Batch creating {len(papers)} paper records with enrichment...")
        paper_ids = [str(p.paper_id) for p in papers]
        processed_papers = await self.paper_service.batch_check_processed_papers(paper_ids)

        papers_to_create = [p for p in papers if not processed_papers.get(str(p.paper_id), False)]
        already_processed = [pid for pid, is_proc in processed_papers.items() if is_proc]
        
        logger.info(
            f"Batch check: {len(already_processed)} already processed, "
            f"{len(papers_to_create)} need creation/processing"
        )
        
        papers_to_process: List[PaperEnrichedDTO] = []
        created_papers: List[Any] = []
        if papers_to_create:
            try:
                created_papers = await self.paper_service.batch_create_papers_from_schema(
                    papers=papers_to_create,  # type: ignore[arg-type]
                    enrich=True
                )
                
                if created_papers:
                    # Map created DBPapers back to PaperEnrichedDTOs for processing
                    created_paper_ids = {p.paper_id for p in created_papers}
                    papers_to_process = [
                        p for p in papers_to_create 
                        if p.paper_id in created_paper_ids
                    ]
                    
                    logger.info(f"Successfully created and enriched {len(created_papers)} papers")
                else:
                    logger.info("No new papers created (all already exist)")
                    
            except Exception as e:
                logger.error(f"Batch paper creation failed: {e}", exc_info=True)
                logger.info("Falling back to individual paper creation")
                for paper in papers_to_create:
                    try:
                        result = await self.paper_service.create_paper_from_schema(paper)
                        if result and not result.is_processed:
                            papers_to_process.append(paper)
                    except Exception as inner_e:
                        logger.error(f"Error creating paper {paper.paper_id}: {inner_e}")
        
        logger.info(
            f"Papers ready: {len(papers_to_process)} to process, "
            f"{len(already_processed)} already processed"
        )
        if not papers_to_process:
            return {pid: True for pid in already_processed}

        semaphore = asyncio.Semaphore(max_workers)
        results = {pid: True for pid in already_processed}
        if filtered_papers is not None and len(filtered_papers) > 0:
            filtered_ids = {str(p.paper_id) for p in filtered_papers}
            papers_to_process = [p for p in papers_to_process if str(p.paper_id) in filtered_ids]
            logger.info(f"{len(papers_to_process)} papers after filtering by similarity")

        if not papers_to_process:
            logger.info("No papers to process after filtering")
            return results

        logger.info(f"Starting concurrent processing of {len(papers_to_process)} papers with {max_workers} workers...")
        tasks = [self.process_content_only(paper, semaphore) for paper in papers_to_process]
        completed = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect results
        failed_count = 0
        exception_count = 0
        for result in completed:
            if isinstance(result, Exception):
                exception_count += 1
                logger.error(f"Paper processing exception: {result}")
            elif isinstance(result, tuple):
                paper_id, success = result
                results[paper_id] = success
                if not success:
                    failed_count += 1

        successful = sum(1 for s in results.values() if s)
        total_processed = len(results)
        logger.info(
            f"Concurrent processing complete: {successful}/{total_processed} successful, "
            f"{failed_count} failed, {exception_count} exceptions"
        )

        del tasks, completed, papers_to_process
        gc.collect()

        return results
    
    async def process_papers_v2(
        self,
        papers: List[PaperEnrichedDTO],
        filtered_papers: Optional[List[PaperEnrichedDTO]] = None,
        max_workers: int = 4,
    ) -> Dict[str, bool]:
        """Process a list of papers concurrently."""
        if filtered_papers is not None and len(filtered_papers) > 0:
            filtered_ids = {str(p.paper_id) for p in filtered_papers}
            papers = [p for p in papers if str(p.paper_id) in filtered_ids]
            logger.info(f"{len(papers)} papers after filtering by similarity")

        if not papers:
            logger.info("No papers to process after filtering")
            return {}

        semaphore = asyncio.Semaphore(max_workers)
        tasks = [self.process_content_only(paper, semaphore) for paper in papers]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        result_dict = {}
        for result in results:
            if isinstance(result, tuple) and len(result) == 2:
                paper_id, success = result
                result_dict[paper_id] = success
            elif isinstance(result, Exception):
                logger.error(f"Error processing paper: {result}")
        return result_dict
        
    
    async def process_content_only(
        self, paper: PaperEnrichedDTO, semaphore: asyncio.Semaphore
    ) -> Tuple[str, bool]:
        """Process paper content (PDF → chunks → embed) without DB record creation"""
        async with semaphore:
            paper_id = str(paper.paper_id)
            logger.info(f"[Worker] Processing content for {paper_id}")
            if self.retrieval_service is None:
                from app.retriever.service import RetrievalService
                self.retrieval_service = RetrievalService(self.repository.db)

            resolved = await self.retrieval_service.resolve_paper_content(paper)
            if not resolved:
                logger.warning(f"Could not resolve content for {paper_id}")
                await self.repository.update_paper_processing_status(paper_id, "failed")
                return (paper_id, False)

            try:
                chunks = []
                if resolved.kind == "tei_xml":
                    structure = self.extractor_service.extract_tei_xml_structure(
                        resolved.content  # pyright: ignore[reportArgumentType]
                    )
                    chunks = self.chunker_service.chunk_from_tei_structure(
                        structure, paper_id
                    )
                    del structure
                elif resolved.kind == "pdf_bytes":
                    doc_structure = self.extractor_service.extract_pdf_structure(
                        resolved.content  # pyright: ignore[reportArgumentType]
                    )
                    chunks = self.chunker_service.chunk_from_docling_structure(
                        doc_structure, paper_id
                    )
                    del doc_structure

                del resolved

                if not chunks:
                    logger.error(f"[{paper_id}] No chunks generated")
                    await self.repository.update_paper_processing_status(
                        paper_id, "failed"
                    )
                    return (paper_id, False)

                logger.info(f"[{paper_id}] Generated {len(chunks)} chunks")
                chunk_texts = [c.text for c in chunks]
                embeddings = await self.embedding_service.create_embeddings_batch(
                    chunk_texts,
                    batch_size=5,  
                )
                del chunk_texts

                await self._store_chunks(paper_id, chunks, embeddings)
                del chunks, embeddings
                await self.repository.update_paper_processing_status(
                    paper_id, "completed"
                )
                gc.collect()

                return (paper_id, True)

            except Exception as e:
                logger.error(f"Error processing {paper_id}: {e}")
                # Rollback any pending transaction
                try:
                    await self.repository.db.rollback()
                except Exception as rollback_error:
                    logger.error(f"Error during rollback for {paper_id}: {rollback_error}")
                
                await self.repository.update_paper_processing_status(paper_id, "failed")
                gc.collect()
                return (paper_id, False)
