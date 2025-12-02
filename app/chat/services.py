"""
Chat service that orchestrates retrieval and LLM services for chatbot functionality
"""

import asyncio
import json
from typing import AsyncGenerator, Optional, Dict, Any, List
from sqlalchemy.ext.asyncio import AsyncSession
from app.llm import llm_service
from app.chat.schemas import ChatMessageRequest
from app.extensions.logger import create_logger
from app.extensions.stream import stream_event, get_stream_response_content
from app.rag_pipeline.pipeline import RAGPipeline
from app.rag_pipeline.schemas import RAGResult
from app.conversations.service import ConversationService

from app.retriever.utils import batch_paper_to_dicts

logger = create_logger(__name__)

# Default user ID until auth is implemented
DEFAULT_USER_ID = 1


class ChatService:
    """Service class for handling chat interactions"""

    def __init__(self, db_session: AsyncSession):
        """Initialize chat service with LLM and retriever services"""
        self.llm_service = llm_service
        self.rag_pipeline = RAGPipeline(db_session=db_session)

    async def stream_message(
        self,
        request: ChatMessageRequest,
        user_id: Optional[int] = None,
        db_session=None,
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

        user_id = user_id or DEFAULT_USER_ID
        conversation_service = ConversationService(db_session)
        conversation = await conversation_service.get_or_create_conversation(
            user_id=user_id, conversation_id=request.conversation_id
        )
        yield f"event: conversation\ndata: {json.dumps({'conversation_id': conversation.id})}\n\n"

        results = None
        async for event in self.rag_pipeline.run(
            request.query, max_subtopics=3, per_subtopic_limit=3, top_chunks=20
        ):
            if event.type == "step":
                async for evt in stream_event(name="step", data=event.data):
                    yield evt
            elif event.type == "subtopics":
                async for evt in stream_event(name="subtopics", data=event.data):
                    yield evt
            elif event.type == "result":
                results = event.data

        if results is None:
            logger.error("RAG pipeline did not return any results")
            error_message = (
                "An error occurred while retrieving research papers. Please try again."
            )
            async for evt in stream_event(name="chunk", data=error_message):
                yield evt
            async for evt in stream_event(name="done", data=None):
                yield evt
            return

        results = RAGResult(papers=results.papers, chunks=results.chunks)  # type: ignore

        if len(results.papers) == 0:
            error_message = "I couldn't find any relevant research papers for your question. This could be because:\n\n1. The topic might be too specific or recent\n2. There may be no academic papers published on this subject\n3. The papers may be behind paywalls\n\nPlease try asking a different question or rephrase your current one."
            async for evt in stream_event(name="chunk", data=error_message):
                yield evt
            async for evt in stream_event(name="done", data=None):
                yield evt
            return

        logger.info(f"Found {len(results.chunks)} relevant chunks via vector search")

        context = []
        chunk_papers = {}
        for chunk in results.chunks:
            chunk_paper_id = str(chunk.paper_id)
            if chunk_paper_id not in chunk_papers:
                paper = next(
                    (p for p in results.papers if str(p.paper_id) == chunk_paper_id),
                    None,
                )
                if paper:
                    chunk_papers[chunk_paper_id] = paper
        for chunk in results.chunks:
            chunk_paper_id = str(chunk.paper_id)
            paper = chunk_papers.get(chunk_paper_id)
            if paper:
                context.append(
                    {
                        "title": paper.title,
                        "authors": paper.authors or [],
                        "abstract": paper.abstract,
                        "year": (
                            paper.publication_date.year
                            if paper.publication_date
                            else None
                        ),
                        "url": paper.url,
                        "pdf_url": paper.pdf_url,
                        "citationCount": paper.citation_count or 0,
                        "paper_id": chunk_paper_id,
                        "chunk_text": chunk.text,
                        "section": chunk.section_title or "Unknown Section",
                    }
                )

        async for evt in stream_event(
            name="sources", data=batch_paper_to_dicts(results.papers)
        ):
            yield evt

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

        # Thought: Generating assistant response
        async for evt in stream_event(
            name="thought", data="Reading relevant papers and generating response..."
        ):
            yield evt

        assistant_response = []
        async for chunk_text in self.llm_service.stream_citation_based_response(
            query=request.query, context=context
        ):
            text = get_stream_response_content(chunk_text)
            if text:  # Only yield and append non-empty chunks
                assistant_response.append(text)
                async for chunk_event in stream_event(name="chunk", data=text):
                    yield chunk_event

        try:
            # Filter out None values before joining
            full_response = "".join(
                [s for s in assistant_response if isinstance(s, str)]
            )
            await conversation_service.add_message_to_conversation(
                conversation_id=conversation.id,
                user_id=user_id,
                message_text=full_response,
                role="assistant",
                auto_title=False,
                paper_ids=[str(p.paper_id) for p in results.papers],
            )
            logger.info(f"Assistant message added to conversation {conversation.id}")
        except Exception as e:
            logger.error(f"Failed to save assistant message: {e}")

        async for done_event in stream_event(name="done", data=None):
            yield done_event

    async def save_feedback(
        self, message_id: int, rating: int, comment: Optional[str] = None
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

    # async def stream_message_with_papers(
    #     self,
    #     request: ChatMessageRequest,
    #     db_session: AsyncSession,
    #     user_id: Optional[int] = None
    # ) -> AsyncGenerator[str, None]:
    #     """
    #     Stream chat response with full paper retrieval, caching, and vector search

    #     This method performs the complete pipeline:
    #     1. Break down user question into subtopics
    #     2. Search Semantic Scholar for relevant papers
    #     3. Check database cache, retrieve full-text if needed
    #     4. Process papers (chunk, embed, summarize) if not cached
    #     5. Use vector search to find most relevant chunks
    #     6. Stream LLM response with citations to specific chunks

    #     Args:
    #         request: Chat message request containing user query
    #         db_session: Database session for paper caching and retrieval
    #         user_id: Optional user ID for personalization

    #     Yields:
    #         Response chunks with citations to specific paper sections
    #     """
    #     # Step 1: Break down the question
    #     questions = await self.llm_service.breakdown_user_question(
    #         user_question=request.query
    #     )
    #     logger.info(f"Question breakdown: {questions}")

    #     # Step 2: Search and retrieve papers with caching
    #     # This uses the PaperRetrievalService which handles:
    #     # - Searching Semantic Scholar
    #     # - Checking DB cache
    #     # - Retrieving full-text if available
    #     # - Processing (chunking, embedding, summarizing)
    #     max_subtopics = 3
    #     db_papers = []

    #     for idx, subtopic in enumerate(questions.subtopics[:max_subtopics], 1):
    #         try:
    #             # Use search_with_caching for full pipeline
    #             papers = await self.retriever.search_with_caching(
    #                 query=subtopic,
    #                 db_session=db_session,
    #                 search_services=[RetrievalServiceType.SEMANTIC],
    #                 limit=3,
    #                 auto_process=True
    #             )
    #             db_papers.extend(papers)
    #             print(f"Subtopic {idx}/{max_subtopics}: Retrieved {len(papers)} processed papers from DB")

    #             await asyncio.sleep(1)
    #         except Exception as e:
    #             print(f"Error retrieving papers for subtopic '{subtopic}': {e}")
    #             continue

    #     # Step 3: Deduplicate papers
    #     unique_papers = []
    #     seen_ids = set()

    #     for paper in db_papers:
    #         if paper.id not in seen_ids:
    #             unique_papers.append(paper)
    #             seen_ids.add(paper.id)

    #     print(f"Total unique processed papers: {len(unique_papers)}")

    #     # Step 4: Find most relevant chunks using vector search
    #     # TODO: Implement vector similarity search on chunks
    #     # For now, convert DB papers to context format
    #     context = []
    #     for paper in unique_papers[:5]:  # Top 5 papers
    #         context.append({
    #             'title': paper.title,
    #             'authors': paper.authors_list if hasattr(paper, 'authors_list') else paper.authors,
    #             'abstract': paper.abstract,
    #             'year': paper.publication_year if hasattr(paper, 'publication_year') else paper.publication_date,
    #             'url': paper.url,
    #             'citationCount': paper.citation_count or 0,
    #             'paper_id': paper.paper_id if hasattr(paper, 'paper_id') else str(paper.id)
    #         })

    #     sources_event = f"event: sources\ndata: {json.dumps(context)}\n\n"
    #     yield sources_event

    #     # Stream the response and collect it for conversation tracking
    #     full_response = ""
    #     async for chunk in self.llm_service.stream_citation_based_response(
    #         query=request.query,
    #         context=context
    #     ):
    #         full_response += chunk
    #         chunk_event = f"event: chunk\ndata: {json.dumps(chunk)}\n\n"
    #         yield chunk_event

    #     yield "event: done\ndata: \n\n"
