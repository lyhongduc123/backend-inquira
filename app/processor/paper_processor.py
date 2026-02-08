from typing import AsyncGenerator, Dict, List, Optional, TYPE_CHECKING, Union
from app.models.papers import DBPaper
from app.retriever.paper_schemas import Paper, PaperPreprocess
from app.retriever.schemas import NormalizedResult
from .services.summarizer import SummarizerService
from .services.extractor import ExtractorService
from .services.embeddings import EmbeddingService, get_embedding_service
from .services.chunker import ChunkingService
from .services.transformer import TransformerService
from app.papers.repository import PaperRepository
from app.chunks.repository import ChunkRepository
from app.authors.service import AuthorService
from app.institutions.service import InstitutionService
from app.extensions.logger import create_logger

if TYPE_CHECKING:
    from app.retriever.paper_service import PaperRetrievalService

logger = create_logger(__name__)


class PaperProcessor:
    def __init__(
        self,
        repository: PaperRepository,
        chunk_repository: Optional[ChunkRepository] = None,
        paper_service: Optional["PaperRetrievalService"] = None,
        extractor_service: Optional[ExtractorService] = None,
        chunker_service: Optional[ChunkingService] = None,
        embedding_service: Optional[EmbeddingService] = None,
        summarizer_service: Optional[SummarizerService] = None,
    ):
        """Initialize PaperProcessor with dependency injection.

        Args:
            repository: Required paper repository
            chunk_repository: Optional chunk repository (created if not provided)
            paper_service: Optional PaperRetrievalService (created if not provided)
            extractor_service: Optional ExtractorService (created if not provided)
            chunker_service: Optional ChunkingService (created if not provided)
            embedding_service: Optional EmbeddingService (singleton if not provided)
            summarizer_service: Optional SummarizerService (created if not provided)
        """
        self.repository = repository
        self.transformer = TransformerService()

        # Services for author/institution enrichment
        self.author_service = AuthorService(self.repository.db)
        self.institution_service = InstitutionService(self.repository.db)

        # Use injected chunk repository or create default
        if chunk_repository:
            self.chunk_repository = chunk_repository
        else:
            self.chunk_repository = ChunkRepository(self.repository.db)

        # Use injected dependencies or create defaults
        # Lazy import to avoid circular dependency
        if paper_service is None:
            from app.retriever.paper_service import PaperRetrievalService

            paper_service = PaperRetrievalService(db=self.repository.db)

        self.paper_service = paper_service
        self.extractor_service = extractor_service or ExtractorService()
        self.chunker_service = chunker_service or ChunkingService()
        self.embedding_service = embedding_service or get_embedding_service()
        self.summarizer_service = summarizer_service or SummarizerService(
            self.chunker_service
        )

    async def process_single_paper(self, paper: PaperPreprocess) -> bool:
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

        exist = await self.paper_service.get_paper_if_exists(
            paper.external_ids or {}, paper.source
        )

        if exist is None:
            if not paper.abstract or paper.abstract.strip() == "":
                paper.abstract = "Abstract not available"
                logger.info(f"Paper {paper_id_str} has no abstract, using fallback")

            await self.repository.create_paper(paper)

        if exist is not None and exist.is_processed is True:
            logger.info(f"Paper {paper_id_str} already processed, skipping.")
            return True

        try:
            chunks = []
            extraction_method = None
            doc_structure = None
            if paper.has_content:
                if paper.has_content.get("grobid_xml"):

                    logger.info(f"[{paper_id_str}] Attempting TEI XML extraction...")
                    tei_xml = await self.paper_service.get_tei_xml(paper)
                    if tei_xml:
                        try:
                            tei_structure = self.extractor_service.extract_tei_xml_structure(
                                tei_xml
                            )
                            if tei_structure:
                                logger.info(
                                    f"[{paper_id_str}] Successfully extracted TEI structure"
                                )
                                tei_chunks = self.chunker_service.chunk_from_tei_structure(
                                    tei_structure, paper_id_str
                                )
                                # Convert ChunkWithMetadata to old tuple format
                                chunks = [
                                    (c.text, c.token_count, c.section_title) for c in tei_chunks
                                ]
                                extraction_method = "tei_xml"
                            else:
                                logger.warning(
                                    f"[{paper_id_str}] TEI structure extraction returned empty"
                                )
                        except Exception as e:
                            logger.error(f"[{paper_id_str}] TEI extraction failed: {e}")
                    else:
                        logger.info(f"[{paper_id_str}] No TEI XML available")

            # Fallback to PDF extraction
            if not chunks:
                logger.info(f"[{paper_id_str}] Falling back to PDF extraction...")
                if not paper.has_content.get("pdf"):
                    logger.info(f"[{paper_id_str}] No PDF available for extraction")
                pdfBytes = await self.paper_service.get_pdf_paper(paper)

                if not pdfBytes:
                    logger.warning(
                        f"Could not retrieve TEI XML or PDF for paper {paper_id_str}, skipping processing."
                    )
                    await self.repository.update_paper_processing_status(
                        paper_id_str, "failed"
                    )
                    return False

                doc_structure = self.extractor_service.extract_pdf_structure(pdfBytes)
                docling_chunks = self.chunker_service.chunk_from_docling_structure(
                    doc_structure, paper_id_str
                )
                # Convert ChunkWithMetadata to old tuple format
                chunks = [
                    (c.text, c.token_count, c.section_title) for c in docling_chunks
                ]
                extraction_method = "pdf_structure"

                if not chunks:
                    logger.warning(
                        f"Structure-based chunking failed for {paper_id_str}, falling back to text-based"
                    )
                    full_text = self.extractor_service.extract_pdf_text(pdfBytes)
                    clean_text = self.extractor_service._fix_text_encoding(full_text)
                    chunks = self.chunker_service.chunk_text(clean_text, paper_id_str)
                    extraction_method = "pdf_text"

            if not chunks:
                logger.error(
                    f"[{paper_id_str}] No chunks generated from any extraction method"
                )
                await self.repository.update_paper_processing_status(
                    paper_id_str, "failed"
                )
                return False

            logger.info(
                f"[{paper_id_str}] Generated {len(chunks)} chunks using {extraction_method}"
            )

            embeddings = await self.embedding_service.create_embeddings_batch(
                [c[0] for c in chunks]
            )

            for idx, (chunk_tuple, emb) in enumerate(zip(chunks, embeddings)):
                text, tokens, section = chunk_tuple
                await self.chunk_repository.create_chunk(
                    chunk_id=f"{paper_id_str}::C{idx}",
                    paper_id=paper_id_str,
                    text=text,
                    token_count=tokens,
                    section_title=section,
                    chunk_index=idx,
                    embedding=emb if emb is not None else [],
                )

            # Generate summary from chunks (now returns structured JSON)
            # combined_text = "\n\n".join([c[0] for c in chunks])
            # summary = await self.summarizer_service.generate_summary(
            #     paper, combined_text, doc_structure
            # )
            # summary_emb = await self.embedding_service.create_embedding(summary)
            # await self.repository.update_paper_summary(
            #     paper_id_str, summary, summary_emb if summary_emb else []
            # )

            await self.repository.update_paper_processing_status(
                paper_id_str, "completed"
            )
            return True
        except Exception as e:
            logger.error(f"Error processing paper {paper_id_str}: {e}")
            await self.repository.update_paper_processing_status(paper_id_str, "failed")
            return False

    async def process_papers(
        self, papers: List[PaperPreprocess], stream_progress: bool = False
    ) -> Union[
        AsyncGenerator[Dict[str, bool], None],
        AsyncGenerator[tuple[str, bool, int, int], None],
    ]:
        """
        Process multiple papers, optionally streaming progress events

        Args:
            papers: List of PaperPreprocess objects
            stream_progress: If True, yields progress tuples; if False, returns dict

        Yields (if stream_progress=True):
            tuple: (paper_id, success, current_index, total_papers)

        Returns (if stream_progress=False):
            Dict mapping paper_id to processing success status
        """
        total = len(papers)

        if stream_progress:
            for idx, paper in enumerate(papers, 1):
                paper_id = str(paper.paper_id)
                success = await self.process_single_paper(paper)
                yield (paper_id, success, idx, total)
        else:
            results = {}
            for paper in papers:
                success = await self.process_single_paper(paper)
                results[str(paper.paper_id)] = success
            yield results

    async def _process_from_tei_xml(self, paper: DBPaper, tei_xml: str) -> bool:
        """
        Process a paper given its TEI XML content.

        Args:
            paper: Database paper object
            tei_xml: TEI XML string

        Returns:
            True if successful, False otherwise
        """
        paper_id_str = str(paper.paper_id)

        try:
            tei_structure = self.extractor_service.extract_tei_xml_structure(tei_xml)
            if not tei_structure:
                logger.error(f"[{paper_id_str}] TEI structure extraction returned empty")
                return False

            tei_chunks = self.chunker_service.chunk_from_tei_structure(
                tei_structure, paper_id_str
            )

            if not tei_chunks:
                logger.error(f"[{paper_id_str}] No chunks generated from TEI XML")
                return False

            embeddings = await self.embedding_service.create_embeddings_batch(
                [c.text for c in tei_chunks]
            )

            for idx, (chunk, emb) in enumerate(zip(tei_chunks, embeddings)):
                await self.chunk_repository.create_chunk(
                    chunk_id=f"{paper_id_str}::C{idx}",
                    paper_id=paper_id_str,
                    text=chunk.text,
                    token_count=chunk.token_count,
                    section_title=chunk.section_title,
                    chunk_index=idx,
                    embedding=emb if emb is not None else [],
                )

            await self.repository.update_paper_processing_status(
                paper_id_str, "completed"
            )
            return True
        except Exception as e:
            logger.error(f"Error processing paper {paper_id_str} from TEI XML: {e}")
            await self.repository.update_paper_processing_status(paper_id_str, "failed")
            return False