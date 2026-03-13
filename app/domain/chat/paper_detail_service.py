"""
Paper detail chat service for single-paper deep-dive conversations
"""

from typing import AsyncGenerator, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from app.extensions.logger import create_logger
from app.extensions.stream import stream_event, get_stream_response_content
from app.llm import get_llm_service

logger = create_logger(__name__)


class PaperDetailChatService:
    """Handles single-paper deep-dive conversations"""

    def __init__(self, db_session: AsyncSession):
        self.db_session = db_session
        self.llm_service = get_llm_service()

    async def stream_chat(
        self,
        paper_id: str,
        query: str,
        conversation_id: str,
        user_id: int,
        model: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """
        Stream chat response for single-paper detail conversation.
        
        This method:
        1. Retrieves relevant chunks from the specific paper
        2. Builds context from paper metadata + chunks + conversation history
        3. Streams LLM response with paper-specific context
        4. Saves messages to conversation
        
        Args:
            paper_id: Paper to discuss
            query: User's question
            conversation_id: Conversation ID for this paper
            user_id: User ID
            model: Optional model override
        """
        from app.domain.conversations.service import ConversationService
        from app.domain.papers.service import PaperService
        from app.domain.papers.repository import PaperRepository, LoadOptions
        from app.retriever.service import RetrievalService
        
        conversation_service = ConversationService(self.db_session)
        paper_repo = PaperRepository(self.db_session)
        retrieval_service = RetrievalService(db=self.db_session)
        paper_service = PaperService(paper_repo, retrieval_service)
        
        # Stream conversation metadata
        async for evt in stream_event(
            name="conversation",
            data={
                "conversation_id": conversation_id,
                "conversation_type": "single_paper_detail",
                "primary_paper_id": paper_id
            }
        ):
            yield evt
        
        # Save user message
        try:
            await conversation_service.add_message_to_conversation(
                conversation_id=conversation_id,
                user_id=user_id,
                message_text=query,
                role="user",
                auto_title=True
            )
        except Exception as e:
            logger.error(f"Failed to save user message: {e}")
        
        # Get paper with full details
        paper = await paper_service.get_paper(paper_id, load_options=LoadOptions.all())
        if not paper:
            error_msg = f"Paper {paper_id} not found"
            async for evt in stream_event(name="error", data={"message": error_msg}):
                yield evt
            return
        
        # Stream paper metadata
        async for evt in stream_event(name="paper", data=paper.model_dump(mode='json')):
            yield evt
        
        # Get relevant chunks from this paper only
        async for evt in stream_event(
            name="thought",
            data={
                "type": "retrieval",
                "content": f"Searching through {paper.title} for relevant sections..."
            }
        ):
            yield evt
        
        chunks = await retrieval_service.get_relevant_chunks(
            query=query,
            paper_ids=[paper_id],
            limit=15
        )
        
        if not chunks:
            async for evt in stream_event(
                name="thought",
                data={
                    "type": "no_chunks",
                    "content": "No processed content found for this paper. Answering based on abstract and metadata."
                }
            ):
                yield evt
        
        # Get conversation history for context
        from app.domain.messages.repository import MessageRepository
        
        message_repo = MessageRepository(self.db_session)
        messages = await message_repo.list_by_conversation(
            conversation_id=conversation_id,
            include_inactive=False
        )
        recent_history = messages[-5:] if len(messages) > 5 else messages
        
        # Build context
        history_text = ""
        for msg in recent_history[:-1]:  # Exclude current message
            history_text += f"{msg.role.upper()}: {msg.content}\n\n"
        
        chunk_context = "\n\n".join([
            f"[Section {i+1}] {chunk.text}"
            for i, chunk in enumerate(chunks)
        ])
        
        # Build prompt
        system_prompt = f"""You are an expert research assistant analyzing a specific paper.

Paper Title: {paper.title}
Authors: {', '.join([a['name'] if isinstance(a, dict) else a.name for a in (paper.authors or [])][:5])}
Published: {paper.publication_date or 'Unknown'}
Venue: {paper.venue or 'Unknown'}

Abstract:
{paper.abstract or 'No abstract available.'}

{'Available Full-Text Sections:' if chunks else 'Note: Full-text not available, answering from abstract only.'}
{chunk_context if chunks else ''}

Previous conversation:
{history_text if history_text else 'This is the first question about this paper.'}

Instructions:
- Answer the user's question based on the paper content
- Be specific and cite sections when possible
- If the answer isn't in the available content, acknowledge it
- Maintain conversational context from previous messages
- Be scholarly but accessible
"""
        
        # Stream LLM response
        full_response = ""
        async for evt in stream_event(
            name="thought",
            data={
                "type": "generation",
                "content": "Analyzing paper content and formulating response..."
            }
        ):
            yield evt
        
        async for chunk_text in self.llm_service.stream_citation_based_response(
            query=query,
            context=[{"text": system_prompt}]
        ):
            # Extract actual text content from the stream chunk
            text = get_stream_response_content(chunk_text)
            if text is None:
                continue
            
            full_response += text
            
            # Yield chunk as SSE event
            async for evt in stream_event(name="chunk", data={"content": text}):
                yield evt
        
        # Save assistant message
        try:
            await conversation_service.add_message_to_conversation(
                conversation_id=conversation_id,
                user_id=user_id,
                message_text=full_response,
                role="assistant",
                auto_title=False,
                paper_ids=[paper_id]
            )
        except Exception as e:
            logger.error(f"Failed to save assistant message: {e}")
        
        # Done
        async for evt in stream_event(
            name="done",
            data={
                "paper_id": paper_id,
                "chunks_used": len(chunks),
                "conversation_id": conversation_id
            }
        ):
            yield evt
