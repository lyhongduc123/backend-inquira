"""
Reading and comprehension prompts
"""
from typing import List, Optional
from .base_prompts import PromptTemplate


class ReadingPrompts:
    """Prompts for reading and comprehension tasks"""
    
    # Explanation Level Descriptions
    LEVEL_DESCRIPTIONS = {
        "beginner": "using simple language, avoiding jargon, and providing basic definitions",
        "intermediate": "using moderate complexity, explaining key terms, and making reasonable assumptions about background knowledge",
        "advanced": "using technical language appropriately, assuming strong background knowledge, and focusing on nuanced details"
    }
    
    # Audience Descriptions
    AUDIENCE_DESCRIPTIONS = {
        "general": "general public with no specialized knowledge",
        "students": "students learning this topic for the first time",
        "researchers": "researchers familiar with academic concepts",
        "professionals": "professionals working in this field"
    }
    
    @staticmethod
    def get_explanation_prompt(level: str, audience: str) -> str:
        """Get prompt for explaining content at specific level and audience"""
        level_desc = ReadingPrompts.LEVEL_DESCRIPTIONS.get(
            level, 
            ReadingPrompts.LEVEL_DESCRIPTIONS["intermediate"]
        )
        audience_desc = ReadingPrompts.AUDIENCE_DESCRIPTIONS.get(
            audience,
            ReadingPrompts.AUDIENCE_DESCRIPTIONS["general"]
        )
        
        return f"""Explain the following content for {audience_desc}, {level_desc}.

Structure your explanation with:
1. **Overview**: What this is about in simple terms
2. **Key Points**: Main concepts broken down clearly
3. **Examples**: Concrete examples or analogies where helpful
4. **Implications**: Why this matters or how it's used
5. **Common Misconceptions**: What people often get wrong (if applicable)

Make it engaging and easy to follow."""
    
    @staticmethod
    def get_question_generation_prompt(
        question_types: List[str],
        num_questions: int,
        difficulty: str
    ) -> str:
        """Get prompt for generating comprehension questions"""
        types_text = ", ".join(question_types)
        
        return f"""Based on the following content, generate {num_questions} questions 
of these types: {types_text}

Difficulty level: {difficulty}

For each question, provide:
1. The question
2. The correct answer
3. Brief explanation of why it's the correct answer
4. Question type (factual/conceptual/analytical/application)

Format as:
**Question 1** (Type): [Question text]
**Answer**: [Answer]
**Explanation**: [Why this is correct]

Make questions that test real understanding, not just memorization."""
    
    @staticmethod
    def get_study_guide_prompt(include_sections: List[str]) -> str:
        """Get prompt for creating study guides"""
        sections_text = ", ".join(include_sections)
        
        return f"""Create a comprehensive study guide that includes: {sections_text}

Structure your response as:

## Summary
[Brief overview of main topics]

## Key Concepts
[Important concepts with definitions]

## Important Facts & Figures
[Crucial facts, dates, numbers to remember]

## Study Questions
[Self-test questions with answers]

## Memory Aids
[Mnemonics, acronyms, or memory techniques]

## Further Reading
[Suggested additional resources or topics]

Make it practical and study-friendly."""
    
    @staticmethod
    def get_interactive_reading_context(content: str) -> str:
        """Get system message for interactive reading sessions"""
        content_preview = content[:1000] + "..." if len(content) > 1000 else content
        
        return f"""You are helping someone understand this content: 

{content_preview}

Answer their questions clearly and refer back to the content when relevant. 
If they ask about something not in the content, let them know."""
    
    @staticmethod
    def get_main_ideas_extraction_prompt(
        num_ideas: int,
        include_supporting_details: bool
    ) -> str:
        """Get prompt for extracting main ideas"""
        details_instruction = "with 2-3 supporting details for each" if include_supporting_details else "without supporting details"
        
        return f"""Extract the {num_ideas} most important main ideas from the following content {details_instruction}.

Format as:
**Main Idea 1**: [Idea statement]
- Supporting detail 1
- Supporting detail 2

**Main Idea 2**: [Idea statement]
- Supporting detail 1
- Supporting detail 2

Focus on the core concepts that someone must understand."""
    
    @staticmethod
    def get_concept_map_prompt(format_type: str) -> str:
        """Get prompt for creating concept maps"""
        if format_type == "hierarchical":
            return """Create a hierarchical concept map showing relationships between concepts.
Format as:

Main Topic
├── Subtopic 1
│   ├── Detail 1a
│   └── Detail 1b
├── Subtopic 2
│   ├── Detail 2a
│   └── Detail 2b
└── Subtopic 3
    ├── Detail 3a
    └── Detail 3b

Show clear hierarchical relationships."""
        
        elif format_type == "network":
            return """Create a network concept map showing interconnected relationships.
Format as:

**Core Concepts**: [List main concepts]

**Relationships**:
- Concept A → relates to → Concept B (because...)
- Concept B → influences → Concept C (through...)
- Concept C → supports → Concept A (by...)

Show how concepts connect and influence each other."""
        
        else:  # sequential
            return """Create a sequential concept map showing how concepts flow or build upon each other.
Format as:

Step 1: [Concept] → leads to → Step 2: [Concept] → results in → Step 3: [Concept]

Show the logical progression or causal sequence."""
    
    @staticmethod
    def get_comprehension_test_prompt(
        num_questions: int,
        include_answers: bool
    ) -> str:
        """Get prompt for creating comprehension tests"""
        answer_instruction = "Include an answer key at the end." if include_answers else "Do not include answers."
        
        return f"""Create a reading comprehension test with {num_questions} questions 
based on the following passage. 

Include a mix of:
- Literal comprehension (what was explicitly stated)
- Inferential questions (what can be inferred)
- Critical thinking questions (analysis and evaluation)

{answer_instruction}

Format questions clearly with multiple choice, short answer, or essay format as appropriate."""
