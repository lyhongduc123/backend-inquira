from typing import Dict, Any, Optional, List
import re
import json
from app.models.papers import DBPaper
from app.retriever.paper_schemas import Paper
from app.extensions.logger import create_logger
from app.llm import get_llm_service
from app.extensions.stream import get_simple_response_content
from app.llm.configs import PromptPresets

logger = create_logger(__name__)

def _get_llm():
    return get_llm_service()

class SummarizerService:
    def __init__(self, chunker):
        self.chunker = chunker
    
    def extract_key_sections(self, doc_dict: Dict[str, Any]) -> Dict[str, str]:
        """
        Extract key sections from docling structure.
        
        Args:
            doc_dict: Docling document dictionary
            
        Returns:
            Dictionary with {section_name: text_content}
        """
        sections = {
            "abstract": "",
            "introduction": "",
            "methodology": "",
            "results": "",
            "conclusion": "",
            "full_text": []
        }
        
        texts = doc_dict.get("texts", [])
        
        current_section = None
        current_content = []
        
        for text_item in texts:
            if text_item.get("content_layer") == "furniture":
                continue
            
            label = text_item.get("label", "")
            text_content = text_item.get("text", "")
            level = text_item.get("level", 0)
            
            if not text_content:
                continue
            
            if label == "section_header":
                if current_section and current_content:
                    if current_section in sections:
                        sections[current_section] = "\n\n".join(current_content)
                    current_content = []
                
                text_lower = text_content.lower()
                if "abstract" in text_lower and level <= 2:
                    current_section = "abstract"
                elif re.search(r"introduction|overview", text_lower) and level <= 2:
                    current_section = "introduction"
                elif re.search(r"method|approach|model|architecture", text_lower) and level <= 2:
                    current_section = "methodology"
                elif re.search(r"result|experiment|evaluation|finding", text_lower) and level <= 2:
                    current_section = "results"
                elif re.search(r"conclusion|discussion|future work", text_lower) and level <= 2:
                    current_section = "conclusion"
                else:
                    current_section = None
                    
            elif label in ["text", "paragraph"]:
                if current_section:
                    current_content.append(text_content)
                sections["full_text"].append(text_content)
        
        if current_section and current_content:
            if current_section in sections:
                sections[current_section] = "\n\n".join(current_content)
        
        sections["full_text"] = "\n\n".join(sections["full_text"])
        
        return sections
    
    async def generate_summary(self, paper: Paper, full_text: str, doc_dict: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate structured summary of the paper.
        Returns JSON string with: {input, output, methodology, solutions, key_findings}
        
        Args:
            paper: Paper Pydantic model
            full_text: Full text of the paper
            doc_dict: Optional docling structure for better section extraction
            
        Returns:
            JSON string with structured summary
        """
        # Extract sections from docling if available
        if doc_dict:
            sections = self.extract_key_sections(doc_dict)
            context_parts = []
            if sections.get("abstract"):
                context_parts.append(f"**Abstract:**\n{sections['abstract'][:1000]}")
            if sections.get("introduction"):
                context_parts.append(f"**Introduction:**\n{sections['introduction'][:1500]}")
            if sections.get("methodology"):
                context_parts.append(f"**Methodology:**\n{sections['methodology'][:1500]}")
            if sections.get("results"):
                context_parts.append(f"**Results:**\n{sections['results'][:1000]}")
            if sections.get("conclusion"):
                context_parts.append(f"**Conclusion:**\n{sections['conclusion'][:1000]}")
            
            context_text = "\n\n".join(context_parts)
            if not context_text:
                context_text = sections.get("full_text", "")[:5000]
        else:
            # Fallback to full text truncation
            tokens = self.chunker.count_tokens(full_text)
            if tokens > 4000:
                context_text = full_text[:16000]
            else:
                context_text = full_text
        
        title_str = str(paper.title)
        
        prompt = f"""Analyze this research paper and provide a structured summary.

Paper Title: {title_str}

{context_text}

Provide a structured summary with these exact fields (respond in JSON format):
1. "input": What problem or research question does this paper address? (1-2 sentences)
2. "output": What are the main contributions, results, or findings? (2-3 sentences)
3. "methodology": What approach, methods, or techniques did they use? (2-3 sentences)
4. "solutions": What specific solutions, algorithms, or frameworks did they propose? (2-3 sentences)
5. "key_findings": What are the most important takeaways? (1-2 sentences)

Respond with ONLY the JSON object, no additional text."""

        try:
            config = PromptPresets.SUMMARIZATION
            
            response = _get_llm().llm_provider.simple_prompt(
                messages=[{"role": "user", "content": prompt}],
                **config.model_dump()
            )
            
            
            response_text = get_simple_response_content(response).strip()
            
            # Extract JSON from response
            if "```json" in response_text:
                json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
                if json_match:
                    response_text = json_match.group(1)
            elif "```" in response_text:
                json_match = re.search(r'```\s*(\{.*?\})\s*```', response_text, re.DOTALL)
                if json_match:
                    response_text = json_match.group(1)
            
            # Validate JSON
            summary_dict = json.loads(response_text)
            summary_json = json.dumps(summary_dict)
            
            paper_id_str = str(getattr(paper, 'paper_id', 'Unknown'))
            logger.info(f"Generated structured summary for paper {paper_id_str}")
            return summary_json
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            # Fallback to simple summary
            abstract_val = getattr(paper, 'abstract', None)
            abstract_str = str(abstract_val) if abstract_val is not None else 'No abstract available'
            fallback = {
                "input": abstract_str[:200] if abstract_str != 'No abstract available' else "Not available",
                "output": "Summary generation failed",
                "methodology": "Summary generation failed",
                "solutions": "Summary generation failed",
                "key_findings": "Summary generation failed",
                "error": str(e)
            }
            return json.dumps(fallback)