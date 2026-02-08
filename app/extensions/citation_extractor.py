"""
Citation Extraction Utilities

Parses LLM responses to extract citation information including claims and confidence.
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from app.extensions.logger import create_logger

logger = create_logger(__name__)


class CitationExtractor:
    """Extract citation information from streaming LLM responses"""
    
    # Pattern to match (cite:paper_id) format
    CITATION_PATTERN = re.compile(r'\(cite:([^\)]+)\)')
    
    @staticmethod
    def extract_citations_from_text(text: str) -> List[str]:
        """
        Extract all paper IDs from citation markers in text.
        
        Args:
            text: Text containing citation markers like (cite:paper_id1)
        
        Returns:
            List of paper IDs
        """
        matches = CitationExtractor.CITATION_PATTERN.findall(text)
        return list(set(matches))  # Remove duplicates
    
    @staticmethod
    def extract_citation_with_context(
        text: str,
        window_size: int = 150
    ) -> List[Dict[str, Any]]:
        """
        Extract citations with their surrounding context (claim).
        
        Args:
            text: Full text containing citations
            window_size: Number of characters before citation to include as context
        
        Returns:
            List of dicts with paper_id, claim, position
        """
        citations = []
        
        for match in CitationExtractor.CITATION_PATTERN.finditer(text):
            paper_id = match.group(1)
            cite_position = match.start()
            
            # Extract claim (text before citation)
            claim_start = max(0, cite_position - window_size)
            claim_text = text[claim_start:cite_position].strip()
            
            # Try to get the full sentence
            # Look for sentence start (. ! ?) or beginning of text
            sentence_start = claim_text.rfind('.')
            if sentence_start == -1:
                sentence_start = claim_text.rfind('!')
            if sentence_start == -1:
                sentence_start = claim_text.rfind('?')
            
            if sentence_start != -1:
                claim_text = claim_text[sentence_start + 1:].strip()
            
            citations.append({
                "paper_id": paper_id,
                "claim": claim_text,
                "position": cite_position,
                "confidence": 1.0  # Default confidence
            })
        
        return citations
    
    @staticmethod
    def calculate_confidence(
        claim_length: int,
        citation_position: str,
        num_citations_nearby: int
    ) -> float:
        """
        Calculate confidence score for a citation based on heuristics.
        
        Args:
            claim_length: Length of the claim text
            citation_position: Position in text ('intro', 'body', 'conclusion')
            num_citations_nearby: Number of other citations in same paragraph
        
        Returns:
            Confidence score between 0.0 and 1.0
        """
        base_confidence = 0.8
        
        # Longer claims typically have higher confidence
        if claim_length > 100:
            base_confidence += 0.1
        elif claim_length < 30:
            base_confidence -= 0.1
        
        # Body citations are more confident than intro/conclusion
        if citation_position == 'body':
            base_confidence += 0.05
        
        # Multiple citations supporting same area increases confidence
        if num_citations_nearby > 2:
            base_confidence += 0.05
        
        return min(1.0, max(0.0, base_confidence))
    
    @staticmethod
    def group_citations_by_paper(
        text: str
    ) -> Dict[str, Dict[str, Any]]:
        """
        Group all citations by paper_id with aggregated claims.
        
        Args:
            text: Full response text
        
        Returns:
            Dict mapping paper_id to citation details
        """
        citations_with_context = CitationExtractor.extract_citation_with_context(text)
        
        grouped: Dict[str, Dict[str, Any]] = {}
        
        for citation in citations_with_context:
            paper_id = citation["paper_id"]
            
            if paper_id not in grouped:
                grouped[paper_id] = {
                    "paper_id": paper_id,
                    "claims": [],
                    "positions": [],
                    "confidence": 0.0
                }
            
            grouped[paper_id]["claims"].append(citation["claim"])
            grouped[paper_id]["positions"].append(citation["position"])
        
        # Calculate average confidence for each paper
        for paper_id, data in grouped.items():
            num_citations = len(data["claims"])
            avg_claim_length = sum(len(c) for c in data["claims"]) / num_citations if num_citations > 0 else 0
            
            # Simple confidence heuristic
            confidence = CitationExtractor.calculate_confidence(
                claim_length=int(avg_claim_length),
                citation_position='body',
                num_citations_nearby=num_citations
            )
            
            grouped[paper_id]["confidence"] = confidence
            # Take the longest claim as the primary claim
            grouped[paper_id]["claim"] = max(data["claims"], key=len) if data["claims"] else ""
        
        return grouped