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
            paper.external_ids, paper.source
        )

        if exist is not None and getattr(exist, "is_processed", False) is True:
            logger.info(f"Paper {paper_id_str} already processed, skipping.")
            return True
        else:
            logger.info(f"Processing paper {paper_id_str}")
            await self.repository.create_paper(paper)

        try:
            full_text = await self.paper_service.get_pdf_paper(paper)
            if not full_text:
                full_text = f"Title: {paper.title}\n\nAbstract: {paper.abstract}"
            clean_text = self.extractor_service._fix_text_encoding(full_text)
            sections = self.extractor_service.split_sections(clean_text)

            # TODO: Finish implementing section extraction and keyword extraction
            # results_text = self.extractor_service.extract_results_conclusion(sections)
            # keywords = self.extractor_service.extract_keywords(clean_text)

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

            summary = await self.summarizer_service.generate_summary(paper, clean_text)
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
