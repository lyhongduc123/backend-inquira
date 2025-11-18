"""
Content reading and comprehension module
"""
from typing import Optional, Dict, Any, List, Generator
from .base import BaseLLMClient
from .prompts import ReadingPrompts

class Reader:
    """Specialized class for reading and comprehension tasks"""
    
    def __init__(self, llm_client: BaseLLMClient):
        self.llm = llm_client
        self.prompts = ReadingPrompts
    
    def read_and_explain(
        self,
        content: str,
        explanation_level: str = "intermediate",
        target_audience: str = "general",
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Read content and provide explanations at appropriate level
        
        Args:
            content: Content to read and explain
            explanation_level: Level of explanation ('beginner', 'intermediate', 'advanced')
            target_audience: Target audience ('general', 'students', 'researchers', 'professionals')
            model: Model to use
        
        Returns:
            Explanation with different components
        """
        system_message = self.prompts.get_explanation_prompt(explanation_level, target_audience)
        
        explanation = self.llm.simple_prompt(
            prompt=content,
            system_message=system_message,
            model=model,
            temperature=0.4
        )
        
        return {
            "explanation": explanation,
            "explanation_level": explanation_level,
            "target_audience": target_audience,
            "model_used": model or self.llm.default_model
        }
    
    def generate_questions(
        self,
        content: str,
        question_types: Optional[List[str]] = None,
        num_questions: int = 5,
        difficulty: str = "mixed",
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate questions based on content for comprehension testing
        
        Args:
            content: Content to generate questions from
            question_types: Types of questions ('factual', 'conceptual', 'analytical', 'application')
            num_questions: Number of questions to generate
            difficulty: Difficulty level ('easy', 'medium', 'hard', 'mixed')
            model: Model to use
        
        Returns:
            Generated questions with answers
        """
        if not question_types:
            question_types = ["factual", "conceptual", "analytical"]
        
        system_message = self.prompts.get_question_generation_prompt(
            question_types=question_types,
            num_questions=num_questions,
            difficulty=difficulty
        )
        
        questions_response = self.llm.simple_prompt(
            prompt=content,
            system_message=system_message,
            model=model,
            temperature=0.5
        )
        
        return {
            "questions": questions_response,
            "question_types": question_types,
            "num_questions": num_questions,
            "difficulty": difficulty,
            "model_used": model or self.llm.default_model
        }
    
    def create_study_guide(
        self,
        content: str,
        include_sections: Optional[List[str]] = None,
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a comprehensive study guide from content
        
        Args:
            content: Content to create study guide from
            include_sections: Sections to include in study guide
            model: Model to use
        
        Returns:
            Structured study guide
        """
        if not include_sections:
            include_sections = ["summary", "key_concepts", "important_facts", "study_questions", "mnemonics"]
        
        system_message = self.prompts.get_study_guide_prompt(include_sections)
        
        study_guide = self.llm.simple_prompt(
            prompt=content,
            system_message=system_message,
            model=model,
            temperature=0.3
        )
        
        return {
            "study_guide": study_guide,
            "sections_included": include_sections,
            "model_used": model or self.llm.default_model
        }
    
    def interactive_reading(
        self,
        content: str,
        user_questions: List[str],
        context_mode: bool = True,
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Interactive reading session with user questions
        
        Args:
            content: Content being read
            user_questions: Questions from the user
            context_mode: Whether to maintain context across questions
            model: Model to use
        
        Returns:
            Responses to user questions with context
        """
        if context_mode:
            # Build conversation with context
            system_content = self.prompts.get_interactive_reading_context(content)
            messages = [
                {"role": "system", "content": system_content}
            ]
            
            responses = []
            for question in user_questions:
                messages.append({"role": "user", "content": question})
                
                response = self.llm.chat_completion(
                    messages=messages,
                    model=model,
                    temperature=0.4
                )
                
                if isinstance(response, dict):
                    answer = response["content"]
                    messages.append({"role": "assistant", "content": answer})
                    responses.append({"question": question, "answer": answer})
        else:
            # Answer each question independently
            responses = []
            for question in user_questions:
                system_message = f"Based on this content, answer the user's question: {content[:1000]}..."
                
                answer = self.llm.simple_prompt(
                    prompt=question,
                    system_message=system_message,
                    model=model,
                    temperature=0.4
                )
                
                responses.append({"question": question, "answer": answer})
        
        return {
            "responses": responses,
            "context_mode": context_mode,
            "num_questions": len(user_questions),
            "model_used": model or self.llm.default_model
        }
    
    def extract_main_ideas(
        self,
        content: str,
        num_ideas: int = 5,
        include_supporting_details: bool = True,
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract main ideas and supporting details from content
        
        Args:
            content: Content to analyze
            num_ideas: Number of main ideas to extract
            include_supporting_details: Whether to include supporting details
            model: Model to use
        
        Returns:
            Main ideas with optional supporting details
        """
        system_message = self.prompts.get_main_ideas_extraction_prompt(
            num_ideas=num_ideas,
            include_supporting_details=include_supporting_details
        )
        
        main_ideas = self.llm.simple_prompt(
            prompt=content,
            system_message=system_message,
            model=model,
            temperature=0.3
        )
        
        return {
            "main_ideas": main_ideas,
            "num_ideas": num_ideas,
            "include_supporting_details": include_supporting_details,
            "model_used": model or self.llm.default_model
        }
    
    def create_concept_map(
        self,
        content: str,
        format_type: str = "hierarchical",
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a concept map from content
        
        Args:
            content: Content to map
            format_type: Type of concept map ('hierarchical', 'network', 'sequential')
            model: Model to use
        
        Returns:
            Concept map representation
        """
        system_message = self.prompts.get_concept_map_prompt(format_type)
        
        concept_map = self.llm.simple_prompt(
            prompt=content,
            system_message=system_message,
            model=model,
            temperature=0.4
        )
        
        return {
            "concept_map": concept_map,
            "format_type": format_type,
            "model_used": model or self.llm.default_model
        }
    
    def reading_comprehension_test(
        self,
        content: str,
        num_questions: int = 10,
        include_answers: bool = True,
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a reading comprehension test
        
        Args:
            content: Content to test comprehension on
            num_questions: Number of questions
            include_answers: Whether to include answer key
            model: Model to use
        
        Returns:
            Comprehension test with optional answers
        """
        system_message = self.prompts.get_comprehension_test_prompt(
            num_questions=num_questions,
            include_answers=include_answers
        )
        
        test = self.llm.simple_prompt(
            prompt=content,
            system_message=system_message,
            model=model,
            temperature=0.4
        )
        
        return {
            "comprehension_test": test,
            "num_questions": num_questions,
            "include_answers": include_answers,
            "model_used": model or self.llm.default_model
        }