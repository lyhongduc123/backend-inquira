"""
Chat service that orchestrates retrieval and LLM services for chatbot functionality
"""
import asyncio
from typing import AsyncGenerator, Optional, Dict, Any, List
from app.llm import llm_service
from app.llm.schemas import CitationBasedResponse
from app.retriever import retriever, RetrievalServiceType
from app.chat.schemas import ChatMessageRequest


class ChatService:
    """Service class for handling chat interactions"""
    
    def __init__(self):
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
            request: Chat message request
            user_id: Optional user ID for personalization
            
        Returns:
            Citation-based chat response with thought process and sources
        """
        # Step 1: Break down the question into subtopics
        questions = await self.llm_service.breakdown_user_question(
            user_question=request.query
        )
        
        print(questions)
        
        # Step 2: Search for relevant papers for each subtopic
        search_results = []
        count = 0
        for subtopic in questions.subtopics:
            count += 1
            if count > 3:
                break
            results = await self.retriever.search(
                query=subtopic,
                search_services=[RetrievalServiceType.SEMANTIC],
            )
            search_results.extend(results)
            await asyncio.sleep(1)
            
        
        # Remove duplicates and limit total results
        unique_results = []
        seen_titles = set()
        for result in search_results:
            title = result.get('title', '')
            if title and title not in seen_titles:
                unique_results.append(result)
                seen_titles.add(title)
                if len(unique_results) >= 5:  # Max 5 papers
                    break
        
        # Step 3: Generate citation-based response with thought process
        response = self.llm_service.generate_citation_based_response(
            query=request.query,
            context=unique_results,
            show_thought_process=True
        )
        
        # TODO: Save to database (conversation & message)
        
        return response
    
    async def stream_message(
        self, 
        request: ChatMessageRequest,
        user_id: Optional[int] = None
    ) -> AsyncGenerator[str, None]:
        """
        Stream chat message response with thought process and citations
        
        Args:
            request: Chat message request
            user_id: Optional user ID
            
        Yields:
            Response chunks with thought steps and citations
        """
        # Break down question first
        questions = await self.llm_service.breakdown_user_question(
            user_question=request.query
        )
        print(questions)
        
        # Search for relevant papers
        search_results = []
        count = 0
        for subtopic in questions.subtopics:
            count += 1
            if count > 3:
                break
            results = await self.retriever.search(
                query=subtopic,
                search_services=[RetrievalServiceType.SEMANTIC],
            )
            search_results.extend(results)
            
            await asyncio.sleep(1)
        
        # Remove duplicates
        unique_results = []
        seen_titles = set()
        for result in search_results:
            title = result.get('title', '')
            if title and title not in seen_titles:
                unique_results.append(result)
                seen_titles.add(title)
                if len(unique_results) >= 5:
                    break
        
        print(f"Unique results count: {len(unique_results)}")
        # Stream citation-based response
        async for chunk in self.llm_service.stream_citation_based_response(
            query=request.query,
            context=unique_results
        ):
            yield chunk
    
    async def save_feedback(
        self,
        message_id: int,
        rating: int,
        comment: Optional[str] = None
    ) -> bool:
        """
        Save user feedback for a message
        
        Args:
            message_id: ID of the message
            rating: Rating value (1-5)
            comment: Optional feedback comment
            
        Returns:
            Success status
        """
        # TODO: Implement database save
        return True


# Singleton instance
chat_service = ChatService()
