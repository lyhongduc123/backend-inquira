"""
Chat service that orchestrates retrieval and LLM services for chatbot functionality
"""
import asyncio
import json
from typing import AsyncGenerator, Optional, Dict, Any, List
from sqlalchemy.ext.asyncio import AsyncSession
from app.llm import llm_service
from app.llm.schemas import CitationBasedResponse
from app.retriever import retriever, RetrievalServiceType
from app.chat.schemas import ChatMessageRequest
from app.models.papers import DBPaper
from app.extensions.logger import create_logger
from app.retriever.paper_service import PaperRetrievalService
from app.conversations.service import ConversationService

logger = create_logger(__name__)

# Default user ID until auth is implemented
DEFAULT_USER_ID = 1


def db_paper_to_dict(db_paper: DBPaper) -> Dict[str, Any]:
    """Convert SQLAlchemy DBPaper object to dictionary for JSON serialization"""
    pub_date = db_paper.publication_date
    return {
        "paper_id": db_paper.paper_id,
        "title": db_paper.title,
        "authors": db_paper.authors or [],
        "abstract": db_paper.abstract,
        "publication_date": pub_date.isoformat() if pub_date is not None else None,
        "venue": db_paper.venue,
        "url": db_paper.url,
        "pdf_url": db_paper.pdf_url,
        "is_open_access": getattr(db_paper, 'is_open_access', False),
        "open_access_pdf": getattr(db_paper, 'open_access_pdf', None),
        "source": db_paper.source,
        "external_id": db_paper.external_id,
        "citation_count": db_paper.citation_count or 0,
        "influential_citation_count": getattr(db_paper, 'influential_citation_count', None),
        "reference_count": getattr(db_paper, 'reference_count', None),
    }


class ChatService:
    """Service class for handling chat interactions"""
    
    def __init__(self):
        """Initialize chat service with LLM and retriever services"""
        self.llm_service = llm_service
        self.retriever = retriever
    
    async def process_message(
        self, 
        request: ChatMessageRequest,
        user_id: Optional[int] = None
    ) -> CitationBasedResponse:
        """
        Process a chat message and generate citation-based response with thought process
        
        Args:
            request: Chat message request containing user query
            user_id: Optional user ID for personalization
            
        Returns:
            CitationBasedResponse containing:
            - query: Original user question
            - thought_process: Step-by-step reasoning with citations
            - final_answer: Synthesized answer
            - all_citations: All citations with metadata (title, authors, year, quote)
            - sources: Full paper metadata for frontend display (title, authors, year, abstract, url, etc.)
            - sources_count: Number of unique sources cited
            - model_used: LLM model identifier
        """
        # Step 1: Break down the question into subtopics for better retrieval
        questions = await self.llm_service.breakdown_user_question(
            user_question=request.query
        )
        
        print(f"Question breakdown: {questions}")
        
        # Step 2: Search for relevant papers for each subtopic using Semantic Scholar
        search_results = []
        max_subtopics = 3  # Limit to first 3 subtopics to avoid over-fetching
        
        for idx, subtopic in enumerate(questions.subtopics[:max_subtopics], 1):
            try:
                # Use the new provider-based search method
                results = await self.retriever.search(
                    search_services=[RetrievalServiceType.SEMANTIC],
                    query=subtopic
                )
                search_results.extend(results)
                print(f"Subtopic {idx}/{max_subtopics}: Found {len(results)} papers")
                
                # Rate limiting to avoid overwhelming the API
                await asyncio.sleep(1)
            except Exception as e:
                print(f"Error searching for subtopic '{subtopic}': {e}")
                continue
        
        # Step 3: Deduplicate and limit results
        unique_results = []
        seen_titles = set()
        max_papers = 5
        
        for result in search_results:
            title = result.get('title', '').strip()
            # Skip if no title or already seen
            if not title or title.lower() in seen_titles:
                continue
                
            unique_results.append(result)
            seen_titles.add(title.lower())
            
            if len(unique_results) >= max_papers:
                break
        
        print(f"Total unique papers found: {len(unique_results)}")
        
        # Step 4: Generate citation-based response with thought process
        # The response will automatically include paper metadata in the sources field
        response = self.llm_service.generate_citation_based_response(
            query=request.query,
            context=unique_results,
            show_thought_process=True
        )
        
        # TODO: Save conversation and message to database
        
        return response
    
    async def stream_message(
        self, 
        request: ChatMessageRequest,
        user_id: Optional[int] = None,
        db_session=None
    ) -> AsyncGenerator[str, None]:
        """
        Stream chat message response with full paper processing and vector search
        
        This method performs the complete RAG pipeline:
        1. Get or create conversation
        2. Break down user question into subtopics
        3. Search and retrieve papers with database caching
        4. Auto-process papers (download PDF, chunk, embed) if not cached
        5. Use vector search on chunks to find most relevant content
        6. Stream LLM response with citations to specific paper sections
        7. Save message and update conversation
        
        Args:
            request: Chat message request containing user query
            user_id: Optional user ID for personalization
            db_session: Database session for caching and vector search
            
        Yields:
            Response chunks in Server-Sent Events (SSE) format:
            - event: conversation, data: JSON object with conversation_id
            - event: sources, data: JSON array of paper metadata
            - event: chunk, data: text chunks of the response
            - event: done, data: empty (signals completion)
        """
        if not db_session:
            logger.error("Database session required for stream_message")
            error_message = "Database connection error. Please try again."
            yield f"event: chunk\ndata: {json.dumps(error_message)}\n\n"
            yield "event: done\ndata: \n\n"
            return
        
        # Step 1: Get or create conversation
        user_id = user_id or DEFAULT_USER_ID
        conversation_service = ConversationService(db_session)
        
        conversation = await conversation_service.get_or_create_conversation(
            user_id=user_id,
            conversation_id=request.conversation_id
        )
        
        # Send conversation ID to frontend
        yield f"event: conversation\ndata: {json.dumps({'conversation_id': conversation.id})}\n\n"
        
        # Continue with existing logic
        questions = await self.llm_service.breakdown_user_question(
            user_question=request.query
        )
        logger.info(f"Question breakdown: {questions}")
        
        paper_service = PaperRetrievalService(db_session)
        
        db_papers = []
        max_subtopics = 3
        
        for idx, subtopic in enumerate(questions.subtopics[:max_subtopics], 1):
            try:
                papers = await self.retriever.search_with_caching(
                    query=subtopic,
                    db_session=db_session,
                    search_services=[RetrievalServiceType.SEMANTIC],
                    limit=3,
                    auto_process=True
                )
                db_papers.extend(papers)
                logger.info(f"Subtopic {idx}/{max_subtopics}: Retrieved {len(papers)} papers")
                
                await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Error retrieving papers for subtopic '{subtopic}': {e}")
                continue
        
        # Deduplicate papers
        unique_papers = []
        seen_ids = set()
        
        for paper in db_papers:
            paper_id = str(paper.paper_id)
            if paper_id not in seen_ids:
                unique_papers.append(paper)
                seen_ids.add(paper_id)
        
        logger.info(f"Total unique papers: {len(unique_papers)}")
        
        if len(unique_papers) == 0:
            error_message = "I couldn't find any relevant research papers for your question. This could be because:\n\n1. The topic might be too specific or recent\n2. There may be no academic papers published on this subject\n3. The papers may be behind paywalls\n\nPlease try asking a different question or rephrase your current one."
            yield f"event: chunk\ndata: {json.dumps(error_message)}\n\n"
            yield "event: done\ndata: \n\n"
            return
        
        # Use vector search to find most relevant chunks
        paper_ids = [str(p.paper_id) for p in unique_papers]
        relevant_chunks = await paper_service.get_relevant_chunks(
            query=request.query,
            paper_ids=paper_ids,
            limit=20
        )
        
        logger.info(f"Found {len(relevant_chunks)} relevant chunks via vector search")
        
        context = []
        chunk_papers = {}
        
        # Map chunks to their papers
        for chunk in relevant_chunks:
            chunk_paper_id = str(chunk.paper_id)
            if chunk_paper_id not in chunk_papers:
                # Find the paper for this chunk
                paper = next((p for p in unique_papers if str(p.paper_id) == chunk_paper_id), None)
                if paper:
                    chunk_papers[chunk_paper_id] = paper
        
        for chunk in relevant_chunks:
            chunk_paper_id = str(chunk.paper_id)
            paper = chunk_papers.get(chunk_paper_id)
            
            if paper:
                context.append({
                    'title': paper.title,
                    'authors': paper.authors or [],
                    'abstract': paper.abstract,
                    'year': paper.publication_date.year if paper.publication_date else None,
                    'url': paper.url,
                    'pdf_url': paper.pdf_url,
                    'citationCount': paper.citation_count or 0,
                    'paper_id': chunk_paper_id,
                    'chunk_text': chunk.text,  # Actual relevant content
                    'section': chunk.section_title or 'Unknown Section'
                })
        
        # Fallback: If no chunks found, use paper abstracts
        if len(context) == 0:
            logger.warning("No chunks found, falling back to paper abstracts")
            
            # Check if papers are still processing
            unprocessed_count = sum(1 for p in unique_papers if not p.is_processed)
            if unprocessed_count > 0:
                logger.info(f"{unprocessed_count}/{len(unique_papers)} papers are still processing")
            
            for paper in unique_papers[:5]:
                context.append({
                    'title': paper.title,
                    'authors': paper.authors or [],
                    'abstract': paper.abstract,
                    'year': paper.publication_date.year if paper.publication_date else None,
                    'url': paper.url,
                    'pdf_url': paper.pdf_url,
                    'citationCount': paper.citation_count or 0,
                    'paper_id': str(paper.paper_id),
                    'content': paper.abstract  # Use abstract as content when no chunks
                })
            
            # Warn the user that we're using limited information
            warning_msg = "\n\n> **Note:** Using paper abstracts only. Full paper content is being processed and will be available for future queries on this topic.\n\n"
            yield f"event: chunk\ndata: {json.dumps(warning_msg)}\n\n"
        
        sources = [db_paper_to_dict(p) for p in unique_papers[:20]]
        sources_event = f"event: sources\ndata: {json.dumps(sources)}\n\n"
        yield sources_event
        
        # Save user message
        try:
            await conversation_service.add_message_to_conversation(
                conversation_id=conversation.id,
                user_id=user_id,
                message_text=request.query,
                role="user",
                auto_title=True
            )
            logger.info(f"User message added to conversation {conversation.id}")
        except Exception as e:
            logger.error(f"Failed to save user message: {e}")
        
        # Stream assistant response and collect it
        assistant_response = []
        async for chunk_text in self.llm_service.stream_citation_based_response(
            query=request.query,
            context=context
        ):
            assistant_response.append(chunk_text)
            chunk_event = f"event: chunk\ndata: {json.dumps(chunk_text)}\n\n"
            yield chunk_event

        # Save assistant response
        try:
            full_response = "".join(assistant_response)
            await conversation_service.add_message_to_conversation(
                conversation_id=conversation.id,
                user_id=user_id,
                message_text=full_response,
                role="assistant",
                auto_title=False  # Don't update title from assistant message
            )
            logger.info(f"Assistant message added to conversation {conversation.id}")
        except Exception as e:
            logger.error(f"Failed to save assistant message: {e}")

        yield "event: done\ndata: \n\n"
    
    async def save_feedback(
        self,
        message_id: int,
        rating: int,
        comment: Optional[str] = None
    ) -> bool:
        """
        Save user feedback for a chat message
        
        Args:
            message_id: ID of the message to provide feedback for
            rating: Rating value (1-5 stars)
            comment: Optional text comment with additional feedback
            
        Returns:
            True if feedback was saved successfully, False otherwise
        """
        # TODO: Implement database persistence for feedback
        # Should save to a Feedback table with message_id, rating, comment, timestamp
        return True
    
    async def stream_message_with_papers(
        self,
        request: ChatMessageRequest,
        db_session: AsyncSession,
        user_id: Optional[int] = None
    ) -> AsyncGenerator[str, None]:
        """
        Stream chat response with full paper retrieval, caching, and vector search
        
        This method performs the complete pipeline:
        1. Break down user question into subtopics
        2. Search Semantic Scholar for relevant papers
        3. Check database cache, retrieve full-text if needed
        4. Process papers (chunk, embed, summarize) if not cached
        5. Use vector search to find most relevant chunks
        6. Stream LLM response with citations to specific chunks
        
        Args:
            request: Chat message request containing user query
            db_session: Database session for paper caching and retrieval
            user_id: Optional user ID for personalization
            
        Yields:
            Response chunks with citations to specific paper sections
        """
        # Step 1: Break down the question
        questions = await self.llm_service.breakdown_user_question(
            user_question=request.query
        )
        logger.info(f"Question breakdown: {questions}")
        
        # Step 2: Search and retrieve papers with caching
        # This uses the PaperRetrievalService which handles:
        # - Searching Semantic Scholar
        # - Checking DB cache
        # - Retrieving full-text if available
        # - Processing (chunking, embedding, summarizing)
        max_subtopics = 3
        db_papers = []
        
        for idx, subtopic in enumerate(questions.subtopics[:max_subtopics], 1):
            try:
                # Use search_with_caching for full pipeline
                papers = await self.retriever.search_with_caching(
                    query=subtopic,
                    db_session=db_session,
                    search_services=[RetrievalServiceType.SEMANTIC],
                    limit=3,  
                    auto_process=True
                )
                db_papers.extend(papers)
                print(f"Subtopic {idx}/{max_subtopics}: Retrieved {len(papers)} processed papers from DB")
                
                await asyncio.sleep(1)
            except Exception as e:
                print(f"Error retrieving papers for subtopic '{subtopic}': {e}")
                continue
        
        # Step 3: Deduplicate papers
        unique_papers = []
        seen_ids = set()
        
        for paper in db_papers:
            if paper.id not in seen_ids:
                unique_papers.append(paper)
                seen_ids.add(paper.id)
        
        print(f"Total unique processed papers: {len(unique_papers)}")
        
        # Step 4: Find most relevant chunks using vector search
        # TODO: Implement vector similarity search on chunks
        # For now, convert DB papers to context format
        context = []
        for paper in unique_papers[:5]:  # Top 5 papers
            context.append({
                'title': paper.title,
                'authors': paper.authors_list if hasattr(paper, 'authors_list') else paper.authors,
                'abstract': paper.abstract,
                'year': paper.publication_year if hasattr(paper, 'publication_year') else paper.publication_date,
                'url': paper.url,
                'citationCount': paper.citation_count or 0,
                'paper_id': paper.paper_id if hasattr(paper, 'paper_id') else str(paper.id)
            })
        
        sources_event = f"event: sources\ndata: {json.dumps(context)}\n\n"
        yield sources_event
        
        # Stream the response and collect it for conversation tracking
        full_response = ""
        async for chunk in self.llm_service.stream_citation_based_response(
            query=request.query,
            context=context
        ):
            full_response += chunk
            chunk_event = f"event: chunk\ndata: {json.dumps(chunk)}\n\n"
            yield chunk_event
        
        yield "event: done\ndata: \n\n"


chat_service = ChatService()
