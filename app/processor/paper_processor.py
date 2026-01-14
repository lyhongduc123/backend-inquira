from typing import Dict, List
from app.models.papers import DBPaper
from app.retriever.paper_schemas import Paper
from .services.summarizer import SummarizerService
from .services.extractor import ExtractorService
from .services.embeddings import EmbeddingService
from .services.chunker import TextChunker
from app.retriever.paper_repository import PaperRepository
from app.extensions.logger import create_logger

logger = create_logger(__name__)


class PaperProcessor:
    def __init__(self, repository: PaperRepository):
        self.repository = repository
        # Local import to avoid circular import
        from app.retriever.paper_service import PaperRetrievalService

        self.paper_service = PaperRetrievalService(self.repository.db)
        self.extractor_service = ExtractorService()
        self.chunker_service = TextChunker()
        self.embedding_service = EmbeddingService()
        self.summarizer_service = SummarizerService(self.chunker_service)

    async def process_single_paper(self, paper: Paper) -> bool:
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
            # Ensure we have a valid abstract (database requires NOT NULL)
            if not paper.abstract or paper.abstract.strip() == "":
                # Set a fallback abstract for papers without abstracts
                paper.abstract = "Abstract not available"
                logger.info(f"Paper {paper_id_str} has no abstract, using fallback")
            
            await self.repository.create_paper(paper)

        if exist is not None and exist.is_processed is True:
            logger.info(f"Paper {paper_id_str} already processed, skipping.")
            return True

        try:
            pdfBytes = await self.paper_service.get_pdf_paper(paper)
            if not pdfBytes:
                logger.warning(
                    f"Could not retrieve full-text PDF for paper {paper_id_str}, skipping processing."
                )
                await self.repository.update_paper_processing_status(
                    paper_id_str, "failed"
                )
                return False
            
            # Extract structured document using docling
            doc_structure = self.extractor_service.extract_pdf_structure(pdfBytes)
            
            # Use structure-aware chunking for better results
            chunks = self.chunker_service.chunk_from_structure(doc_structure, paper_id_str)
            
            # Fallback to text-based chunking if structure-based fails
            if not chunks:
                logger.warning(f"Structure-based chunking failed for {paper_id_str}, falling back to text-based")
                full_text = self.extractor_service.extract_pdf_text(pdfBytes)
                clean_text = self.extractor_service._fix_text_encoding(full_text)
                chunks = self.chunker_service.chunk_text(clean_text, paper_id_str)
            
            embeddings = await self.embedding_service.create_embeddings_batch(
                [c[0] for c in chunks]
            )

            for idx, ((text, tokens, section), emb) in enumerate(
                zip(chunks, embeddings)
            ):
                await self.repository.create_chunk(
                    chunk_id=f"{paper_id_str}::C{idx}",
                    paper_id=paper_id_str,
                    text=text,
                    token_count=tokens,
                    section_title=section,
                    chunk_index=idx,
                    embedding=emb if emb is not None else [],
                )

            # Generate summary from chunks
            combined_text = "\n\n".join([c[0] for c in chunks[:5]])  # Use first 5 chunks for summary
            summary = await self.summarizer_service.generate_summary(paper, combined_text)
            summary_emb = await self.embedding_service.create_embedding(summary)
            await self.repository.update_paper_summary(
                paper_id_str, summary, summary_emb if summary_emb else []
            )

            await self.repository.update_paper_processing_status(
                paper_id_str, "completed"
            )
            return True
        except Exception as e:
            logger.error(f"Error processing paper {paper_id_str}: {e}")
            await self.repository.update_paper_processing_status(paper_id_str, "failed")
            return False

    async def process_papers(self, papers: List[Paper]) -> Dict[str, bool]:
        """
        Process multiple papers concurrently

        Args:
            papers: List of Paper objects

        Returns:
            Dict mapping paper_id to processing success status
        """
        results = {}
        for paper in papers:
            success = await self.process_single_paper(paper)
            results[str(paper.paper_id)] = success
        return results
