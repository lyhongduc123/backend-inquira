"""
Error and special case handlers for chat operations.
Handles gibberish input, no results, and other edge cases.
"""

from typing import AsyncGenerator
from app.domain.conversations.service import ConversationService
from app.extensions import stream_event
from app.extensions.logger import create_logger
from .event_emitter import EventEmitter

logger = create_logger(__name__)


class ChatErrorHandler:
    """Handles error cases and special responses for chat"""
    
    @staticmethod
    async def _stream_text_in_chunks(text: str, delay: float = 0.05) -> AsyncGenerator[str, None]:
        """Simulates LLM streaming by yielding paragraphs or sentences."""
        tokens = text.split(" ")
        for i, token in enumerate(tokens):
            content = token if i == len(tokens) - 1 else token + " "
            async for evt in EventEmitter.emit_chunk_event(content):
                yield evt

    @staticmethod
    async def handle_gibberish_input(
        conversation_service: ConversationService,
        conversation_id: str,
        user_id: int,
        query: str,
    ) -> AsyncGenerator[str, None]:
        """
        Handle gibberish input with helpful introduction message.
        
        Args:
            conversation_service: Service for conversation operations
            conversation_id: Conversation ID
            user_id: User ID
            query: The gibberish query
            
        Yields:
            SSE events with introduction message
        """
        logger.info(f"Gibberish detected: {query}")
        
        # Save user message
        try:
            await conversation_service.add_message_to_conversation(
                conversation_id=conversation_id,
                user_id=user_id,
                message_text=query,
                role="user",
                auto_title=False,
            )
        except Exception as e:
            logger.error(f"Failed to save user message: {e}")
        
        # Build introduction message
        intro_parts = [
            "Hello! I'm exegent, an academic research assistant.\n\n",
            "I'm here to help you explore and understand academic research papers! I can:\n\n",
            " **Search** through millions of research papers across all disciplines  \n",
            " **Analyze** and summarize complex scientific papers  \n",
            " **Find** relevant citations and evidence for your questions  \n",
            " **Compare** different research findings and methodologies  \n\n",
            "**How to get started:**\n\n",
            "Ask me clear research questions like:\n",
            '- "What are the latest findings on climate change?"\n',
            '- "How does machine learning improve medical diagnosis?"\n',
            '- "What are the ethical implications of AI?"\n\n',
            "**Tips for better results:**\n",
            "- Be specific about what you want to know\n",
            "- Use proper words and complete sentences\n",
            "- Ask about scientific topics, research areas, or academic questions\n\n",
            "Try asking me a research question, and I'll find and analyze relevant papers for you!",
        ]
        
        intro_message = "".join(intro_parts)
        
        # Stream the message
        async for evt in stream_event(name="chunk", data={"text": intro_message}):
            yield evt
        
        # Save assistant message
        try:
            await conversation_service.add_message_to_conversation(
                conversation_id=conversation_id,
                user_id=user_id,
                message_text=intro_message,
                role="assistant",
                auto_title=False,
            )
        except Exception as e:
            logger.error(f"Failed to save assistant message: {e}")
        
        # Signal completion
        async for evt in stream_event(name="done", data=None):
            yield evt
    
    @staticmethod
    async def handle_no_results(
        conversation_id: str,
        user_id: int,
        conversation_service: ConversationService,
    ) -> AsyncGenerator[str, None]:
        """
        Handle when no papers are found.
        
        Args:
            conversation_id: Conversation ID
            user_id: User ID
            conversation_service: Service for conversation operations
            
        Yields:
            SSE events with no results message
        """
        msg = """I couldn't find any relevant research papers for your question. This could be because:

1. The topic might be too specific or recent
2. There may be no academic papers published on this subject
3. The papers may be behind paywalls or not indexed in the databases I have access to.

Please try asking a different question or rephrase your current one."""
 
        try:
            await conversation_service.add_message_to_conversation(
                conversation_id=conversation_id,
                user_id=user_id,
                message_text=msg,
                role="assistant",
                auto_title=False,
            )
        except Exception as e:
            logger.error(f"Failed to save no results message: {e}")
        
        # Stream the message
        async for evt in EventEmitter.emit_chunk_event(msg):
            yield evt
        
        # Signal completion
        async for evt in EventEmitter.emit_done_event():
            yield evt
