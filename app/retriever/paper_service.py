from typing import List, Optional, Dict, Any, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from app.retriever.paper_retriever import PaperRetriever
from app.retriever.chunker import TextChunker
from app.retriever.embeddings import EmbeddingService
from app.retriever.paper_repository import PaperRepository
from app.retriever.paper_schemas import Paper, PaperChunk, Author
from app.retriever.provider.base_schemas import NormalizedResult
from app.models.papers import DBPaper, DBPaperChunk
from app.extensions.logger import create_logger
from app.core.config import settings
from app.retriever.provider import SemanticScholarProvider, RetrievalConfig, RetrievalMode

logger = create_logger(__name__)


class PaperRetrievalService:
    """
    Service for retrieving and caching papers with full-text
    
    Implements the flow:
    1. Search Semantic Scholar for papers
    2. Check if papers exist in database
    3. If not, retrieve full-text from arXiv
    4. Chunk and embed the text
    5. Generate summary
    6. Store in database
    """
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.paper_retriever = PaperRetriever()
        
        # Use new provider architecture
        config = RetrievalConfig(
            mode=RetrievalMode.ENHANCED,
            enable_caching=True,
            enable_full_text=True,
            max_results=100
        )
        self.semantic_provider = SemanticScholarProvider(
            api_url=settings.SEMANTIC_API_URL,
            config=config,
            db_session=db
        )
        
        self.chunker = TextChunker(min_tokens=600, max_tokens=1200, overlap_tokens=100)
        self.embedding_service = EmbeddingService()
        self.repository = PaperRepository(db)
    
    def _convert_normalized_to_paper(self, normalized: NormalizedResult) -> Paper:
        """
        Convert normalized result from provider to Paper object
        
        Args:
            normalized: Normalized paper data from provider
            
        Returns:
            Paper object
        """
        # Convert authors
        authors_list = []
        authors_data = normalized.get('authors', [])
        if authors_data:
            authors_list = [
                Author(
                    name=a.get('name', '') if isinstance(a, dict) else str(a),
                    author_id=a.get('author_id') if isinstance(a, dict) else None
                )
                for a in authors_data
            ]
        
        # Extract DOI from external_ids or use paper_id as fallback
        external_ids = normalized.get('external_ids') or {}
        external_id = external_ids.get('DOI') or normalized.get('paper_id', '')
        
        print(normalized.get('paper_id', ''), normalized.get('influential_citation_count'))
        
        return Paper(
            paper_id=normalized.get('paper_id', ''),
            title=normalized.get('title', ''),
            abstract=normalized.get('abstract'),
            authors=authors_list,
            publication_date=None,  # Will be parsed from publication_date if needed
            venue=normalized.get('venue'),
            url=normalized.get('url'),
            citation_count=normalized.get('citation_count') or 0,
            influential_citation_count=normalized.get('influential_citation_count') or 0,
            reference_count=normalized.get('reference_count') or 0,
            external_id=external_id,
            source=normalized.get('source', 'semantic_scholar'),
            pdf_url=normalized.get('pdf_url'),
            is_open_access=normalized.get('is_open_access', False),
            open_access_pdf=normalized.get('open_access_pdf')
        )
    
    async def search_and_retrieve_papers(
        self,
        query: str,
        limit: int = 20,
        auto_process: bool = True
    ) -> List[DBPaper]:
        """
        Search for papers and optionally auto-process them
        
        Args:
            query: Search query
            limit: Number of papers to retrieve
            auto_process: Whether to automatically process papers (fetch full-text, chunk, embed)
            
        Returns:
            List of DBPaper objects
        """
        logger.info(f"Searching papers for query: {query[:100]}...")
        
        raw_papers = await self.semantic_provider.search_papers(query, limit=limit)
        
        if not raw_papers:
            logger.warning(f"No papers found for query: {query}")
            return []
        
        papers = []
        for raw_paper in raw_papers:
            try:
                normalized = self.semantic_provider.normalize_result(raw_paper)
                paper = self._convert_normalized_to_paper(normalized)
                papers.append(paper)
            except Exception as e:
                logger.error(f"Error converting paper: {e}")
                continue
        
        # Check which papers already exist in database
        db_papers = []
        for paper in papers:
            if not paper.external_id:
                continue
                
            print(f"Processing paper: {paper.title} ({paper.external_id}) source: {paper.source}")
            existing = await self.repository.get_paper_by_external_id(
                paper.external_id, paper.source
            )
            
            if existing:
                logger.info(f"Paper {paper.paper_id} already exists in database")
         
                await self.repository.update_last_accessed(str(existing.paper_id))
                db_papers.append(existing)
                
                # Process if not yet processed and auto_process is enabled
                # Access actual values from SQLAlchemy columns
                is_processed = bool(existing.is_processed)
                pdf_url = getattr(existing, 'pdf_url', None)
                pdf_url_str = str(pdf_url) if pdf_url is not None else None
                if auto_process and not is_processed and pdf_url_str:
                    logger.info(f"Auto-processing existing paper {str(existing.paper_id)}")
                    await self._process_paper(existing)
            else:
                db_paper = await self.repository.create_paper(paper)
                db_papers.append(db_paper)
                
                # Process if auto_process is enabled
                if auto_process:
                    logger.info(f"Auto-processing new paper {str(db_paper.paper_id)}")
                    await self._process_paper(db_paper)
        
        return db_papers
    
    async def _process_paper(self, db_paper: DBPaper) -> bool:
        """
        Process a paper: retrieve full-text from available sources, chunk, embed, summarize
        
        Tries multiple sources (arXiv, open access PDFs, etc.) and gracefully
        handles paywalled papers by using abstract only.
        
        Args:
            db_paper: Database paper object
            
        Returns:
            True if successful, False otherwise
        """
        # Get actual string values from SQLAlchemy columns
        paper_id_str = str(db_paper.paper_id)
        
        try:
            await self.repository.update_paper_processing_status(
                paper_id_str, "processing"
            )
            
            # Prepare paper metadata for multi-source retrieval
            # Extract open access PDF URL from JSONB field
            open_access_pdf_data = getattr(db_paper, 'open_access_pdf', None)
            open_access_pdf_url = None
            if isinstance(open_access_pdf_data, dict):
                open_access_pdf_url = open_access_pdf_data.get('url')
            
            paper_data = {
                'arxiv_id': getattr(db_paper, 'arxiv_id', None),
                'doi': getattr(db_paper, 'doi', None),
                'pdf_url': str(getattr(db_paper, 'pdf_url', '')) if getattr(db_paper, 'pdf_url', None) else None,
                'open_access_pdf': open_access_pdf_url,
                'is_open_access': getattr(db_paper, 'is_open_access', False),
            }
            
            # Check if paper is likely accessible
            access_info = self.paper_retriever.get_access_info(paper_data)
            
            if not access_info['is_open_access']:
                logger.warning(
                    f"Paper {paper_id_str} is not open access. "
                    f"Likely paywalled: {access_info['likely_paywalled']}"
                )
                
                # For paywalled papers, use abstract only
                abstract = str(getattr(db_paper, 'abstract', ''))
                if abstract and len(abstract) > 100:
                    logger.info(f"Using abstract for paywalled paper {paper_id_str}")
                    full_text = f"Title: {str(getattr(db_paper, 'title', ''))}\n\nAbstract: {abstract}"
                else:
                    logger.warning(f"No accessible content for paper {paper_id_str}")
                    await self.repository.update_paper_processing_status(
                        paper_id_str, "failed", "Paper is paywalled and no abstract available"
                    )
                    return False
            else:
                # 2. Try to retrieve full-text from multiple sources
                logger.info(
                    f"Retrieving full-text for paper {paper_id_str} "
                    f"from sources: {access_info['sources']}"
                )
                full_text = await self.paper_retriever.try_multiple_sources(paper_data)
                
                if not full_text:
                    # Fallback to abstract if full-text retrieval failed
                    logger.warning(f"Failed to retrieve full-text for paper {paper_id_str}, using abstract")
                    abstract = str(getattr(db_paper, 'abstract', ''))
                    if abstract and len(abstract) > 100:
                        full_text = f"Title: {str(getattr(db_paper, 'title', ''))}\n\nAbstract: {abstract}"
                    else:
                        logger.error(f"No content available for paper {paper_id_str}")
                        await self.repository.update_paper_processing_status(
                            paper_id_str, "failed", "Failed to retrieve any content"
                        )
                        return False
            
            # 3. Chunk text
            logger.info(f"Chunking text for paper {paper_id_str}")
            chunks = self.chunker.chunk_text(full_text, paper_id_str)
            
            if not chunks:
                logger.error(f"No chunks created for paper {paper_id_str}")
                await self.repository.update_paper_processing_status(
                    paper_id_str, "failed", "Failed to chunk text"
                )
                return False
            
            # 3. Generate embeddings for chunks
            logger.info(f"Generating embeddings for {len(chunks)} chunks")
            chunk_texts = [chunk[0] for chunk in chunks]
            embeddings = await self.embedding_service.create_embeddings_batch(chunk_texts)
            
            # 4. Store chunks in database
            for idx, ((chunk_text, token_count, section_title), embedding) in enumerate(zip(chunks, embeddings)):
                if embedding is None:
                    logger.warning(f"Skipping chunk {idx} due to embedding failure")
                    continue
                
                chunk_id = self.chunker.create_chunk_id(paper_id_str, idx)
                await self.repository.create_chunk(
                    chunk_id=chunk_id,
                    paper_id=paper_id_str,
                    text=chunk_text,
                    token_count=token_count,
                    chunk_index=idx,
                    embedding=embedding,
                    section_title=section_title
                )
            
            # 5. Generate summary
            logger.info(f"Generating summary for paper {paper_id_str}")
            summary = await self._generate_summary(db_paper, full_text)
            
            # 6. Generate summary embedding
            summary_embedding = await self.embedding_service.create_embedding(summary)
            
            if summary_embedding:
                await self.repository.update_paper_summary(
                    paper_id_str, summary, summary_embedding
                )
            
            # 7. Update status to completed
            await self.repository.update_paper_processing_status(
                paper_id_str, "completed"
            )
            
            logger.info(f"Successfully processed paper {paper_id_str}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing paper {paper_id_str}: {e}")
            await self.repository.update_paper_processing_status(
                paper_id_str, "failed", str(e)
            )
            return False
    
    async def _generate_summary(self, db_paper: DBPaper, full_text: str) -> str:
        """
        Generate a 300-800 token summary of the paper
        
        Args:
            db_paper: Database paper object
            full_text: Full text of the paper
            
        Returns:
            Summary text
        """
        # Truncate full text if too long (use first ~4000 tokens)
        tokens = self.chunker.count_tokens(full_text)
        if tokens > 4000:
            # Rough approximation: 1 token ≈ 4 characters
            truncated_text = full_text[:16000]
        else:
            truncated_text = full_text
        
        # Extract title and authors safely
        title_str = str(db_paper.title)
        abstract_val = getattr(db_paper, 'abstract', None)
        abstract_str = str(abstract_val) if abstract_val is not None else 'No abstract available'
        
        # Handle authors safely
        authors_str = 'Unknown'
        authors_val = getattr(db_paper, 'authors', None)
        if authors_val:
            try:
                if isinstance(authors_val, list):
                    author_names = [a.get('name', 'Unknown') if isinstance(a, dict) else str(a) for a in authors_val]
                    authors_str = ', '.join(author_names)
            except:
                authors_str = 'Unknown'
        
        prompt = f"""Generate a concise summary (300-800 tokens) of the following research paper.

Title: {title_str}
Authors: {authors_str}

Abstract:
{abstract_str}

Full Text (truncated if necessary):
{truncated_text}

Summary should include:
1. Main research question and objectives
2. Key methodology and approach
3. Primary findings and results
4. Significance and implications
5. Limitations (if mentioned)

Generate the summary:"""

        try:
            # Use synchronous completion for summary
            # Since stream_completion is a synchronous generator, we'll collect it
            from app.llm import llm_service
            response_text = ""
            
            # stream_completion is synchronous generator
            for chunk in llm_service.llm_provider.stream_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.3
            ):
                response_text += chunk
            
            summary = response_text.strip()
            
            # Verify summary length (should be 300-800 tokens)
            summary_tokens = self.chunker.count_tokens(summary)
            logger.info(f"Generated summary with {summary_tokens} tokens for paper {str(db_paper.paper_id)}")
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            # Fallback to abstract + title
            return f"Title: {title_str}\n\nAbstract: {abstract_str}"
    
    async def get_relevant_chunks(
        self,
        query: str,
        paper_ids: Optional[List[str]] = None,
        limit: int = 10
    ) -> List[DBPaperChunk]:
        """
        Get relevant chunks for a query
        
        Args:
            query: Query text
            paper_ids: Optional list of paper IDs to restrict search
            limit: Number of chunks to return
            
        Returns:
            List of relevant chunks
        """
        # Generate query embedding
        query_embedding = await self.embedding_service.create_embedding(query)
        
        if not query_embedding:
            logger.error("Failed to generate query embedding")
            return []
        
        # Search for similar chunks
        chunks = await self.repository.search_similar_chunks(
            query_embedding, limit=limit, paper_ids=paper_ids
        )
        
        return chunks
    
    async def get_paper_summaries(
        self,
        query: str,
        limit: int = 10
    ) -> List[DBPaper]:
        """
        Get relevant paper summaries for a query
        
        Args:
            query: Query text
            limit: Number of papers to return
            
        Returns:
            List of relevant papers with summaries
        """
        # Generate query embedding
        query_embedding = await self.embedding_service.create_embedding(query)
        
        if not query_embedding:
            logger.error("Failed to generate query embedding")
            return []
        
        # Search for similar papers
        papers = await self.repository.search_similar_papers(
            query_embedding, limit=limit
        )
        
        return papers
