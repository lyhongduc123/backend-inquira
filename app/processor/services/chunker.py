import re
import tiktoken
from typing import List, Optional, Tuple, Dict, Any
from app.extensions.logger import create_logger

logger = create_logger(__name__)


class TextChunker:
    """Chunk text into smaller pieces for embedding with intelligent structure-aware chunking"""
    
    def __init__(
        self,
        min_tokens: int = 600,
        max_tokens: int = 1200,
        overlap_tokens: int = 100,
        model: str = "gpt-4"
    ):
        """
        Initialize text chunker
        
        Args:
            min_tokens: Minimum tokens per chunk
            max_tokens: Maximum tokens per chunk
            overlap_tokens: Number of tokens to overlap between chunks
            model: Model name for tokenization
        """
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.encoding = tiktoken.encoding_for_model(model)
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.encoding.encode(text))
    
    def chunk_from_structure(
        self,
        doc_dict: Dict[str, Any],
        paper_id: str
    ) -> List[Tuple[str, int, Optional[str]]]:
        """
        Chunk document based on its structure from docling.
        This is the preferred method for new code.
        
        Args:
            doc_dict: Structured document dictionary from docling
            paper_id: Paper ID for logging
            
        Returns:
            List of (chunk_text, token_count, section_title) tuples
        """
        chunks = []
        
        # Extract main content - docling structure typically has these keys
        main_text = doc_dict.get("main-text", [])
        
        # Group content by sections intelligently
        current_section_title = None
        current_section_content = []
        current_tokens = 0
        
        for item in main_text:
            if not isinstance(item, dict):
                continue
                
            item_type = item.get("type", "")
            item_text = item.get("text", "")
            
            # Detect section headings
            if item_type in ["heading", "title", "section-header"]:
                # Save previous section if it exists and meets minimum size
                if current_section_content and current_tokens >= self.min_tokens:
                    chunk_text = "\n".join(current_section_content)
                    chunks.append((chunk_text, current_tokens, current_section_title))
                    current_section_content = []
                    current_tokens = 0
                
                # Start new section
                current_section_title = item_text
                continue
            
            # Add content to current section
            if item_text:
                item_tokens = self.count_tokens(item_text)
                
                # If adding this would exceed max, save current chunk
                if current_tokens + item_tokens > self.max_tokens and current_section_content:
                    chunk_text = "\n".join(current_section_content)
                    chunks.append((chunk_text, current_tokens, current_section_title))
                    
                    # Keep last paragraph for overlap
                    if current_section_content:
                        overlap_text = current_section_content[-1]
                        overlap_tokens = self.count_tokens(overlap_text)
                        current_section_content = [overlap_text] if overlap_tokens <= self.overlap_tokens else []
                        current_tokens = overlap_tokens if overlap_tokens <= self.overlap_tokens else 0
                    else:
                        current_section_content = []
                        current_tokens = 0
                
                current_section_content.append(item_text)
                current_tokens += item_tokens
        
        # Add final section
        if current_section_content and current_tokens >= self.min_tokens:
            chunk_text = "\n".join(current_section_content)
            chunks.append((chunk_text, current_tokens, current_section_title))
        elif current_section_content and chunks:
            # Append to last chunk if too small
            last_chunk_text, last_tokens, last_title = chunks[-1]
            additional_text = "\n".join(current_section_content)
            combined_text = last_chunk_text + "\n" + additional_text
            combined_tokens = self.count_tokens(combined_text)
            chunks[-1] = (combined_text, combined_tokens, last_title or current_section_title)
        elif current_section_content:
            # If only one small chunk, keep it anyway
            chunk_text = "\n".join(current_section_content)
            chunks.append((chunk_text, current_tokens, current_section_title))
        
        logger.info(f"Chunked paper {paper_id} into {len(chunks)} structure-aware chunks")
        return chunks
    
    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting (can be improved with spaCy or NLTK)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def chunk_text(
        self,
        text: str,
        paper_id: str,
        preserve_sections: bool = True
    ) -> List[Tuple[str, int, Optional[str]]]:
        """
        Chunk text into overlapping pieces
        
        Args:
            text: Full text to chunk
            paper_id: Paper ID for logging
            preserve_sections: Try to preserve section boundaries
            
        Returns:
            List of (chunk_text, token_count, section_title) tuples
        """
        chunks = []
        
        # Try to split by sections first
        if preserve_sections:
            sections = self._split_into_sections(text)
        else:
            sections = [(text, None)]
        
        chunk_index = 0
        for section_text, section_title in sections:
            section_chunks = self._chunk_section(section_text, section_title)
            chunks.extend(section_chunks)
        
        logger.info(f"Chunked paper {paper_id} into {len(chunks)} chunks")
        return chunks
    
    def _split_into_sections(self, text: str) -> List[Tuple[str, Optional[str]]]:
        """
        Split text into sections based on headings.
        Enhanced to work with markdown headings from docling.
        
        Returns:
            List of (section_text, section_title) tuples
        """
        sections: List[Tuple[str, Optional[str]]] = []
        
        # Pattern to match both markdown headings and traditional section headings
        markdown_heading = r'^#{1,3}\s+(.+)$'
        traditional_heading = r'^(?:\d+\.?\s+)?([A-Z][A-Za-z\s]+)$'
        
        lines = text.split('\n')
        current_section = []
        current_title: Optional[str] = None
        
        for line in lines:
            stripped = line.strip()
            
            # Check for markdown heading (from docling)
            md_match = re.match(markdown_heading, stripped)
            if md_match:
                # Save previous section
                if current_section:
                    section_text = '\n'.join(current_section)
                    sections.append((section_text, current_title))
                
                # Start new section
                current_title = md_match.group(1).strip()
                current_section = []
                continue
            
            # Check for traditional heading
            if len(stripped) < 100 and re.match(traditional_heading, stripped):
                # Save previous section
                if current_section:
                    section_text = '\n'.join(current_section)
                    sections.append((section_text, current_title))
                
                # Start new section
                current_title = stripped
                current_section = []
            else:
                current_section.append(line)
        
        # Add last section
        if current_section:
            section_text = '\n'.join(current_section)
            sections.append((section_text, current_title))
        
        # If no sections found, return entire text
        if not sections:
            sections = [(text, None)]
        
        return sections
    
    def _chunk_section(
        self,
        section_text: str,
        section_title: Optional[str]
    ) -> List[Tuple[str, int, Optional[str]]]:
        """
        Chunk a single section into overlapping pieces
        
        Returns:
            List of (chunk_text, token_count, section_title) tuples
        """
        chunks = []
        sentences = self.split_into_sentences(section_text)
        
        current_chunk = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)
            
            # If adding this sentence exceeds max_tokens, save current chunk
            if current_tokens + sentence_tokens > self.max_tokens and current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append((chunk_text, current_tokens, section_title))
                
                # Start new chunk with overlap
                overlap_chunk = []
                overlap_tokens = 0
                
                # Add sentences from end of previous chunk for overlap
                for prev_sentence in reversed(current_chunk):
                    prev_tokens = self.count_tokens(prev_sentence)
                    if overlap_tokens + prev_tokens <= self.overlap_tokens:
                        overlap_chunk.insert(0, prev_sentence)
                        overlap_tokens += prev_tokens
                    else:
                        break
                
                current_chunk = overlap_chunk
                current_tokens = overlap_tokens
            
            current_chunk.append(sentence)
            current_tokens += sentence_tokens
        
        # Add final chunk if it meets minimum size
        if current_chunk and current_tokens >= self.min_tokens:
            chunk_text = ' '.join(current_chunk)
            chunks.append((chunk_text, current_tokens, section_title))
        elif current_chunk and chunks:
            # If final chunk is too small, append to last chunk
            chunk_text = ' '.join(current_chunk)
            last_chunk, last_tokens, last_title = chunks[-1]
            combined_text = last_chunk + ' ' + chunk_text
            combined_tokens = self.count_tokens(combined_text)
            chunks[-1] = (combined_text, combined_tokens, last_title or section_title)
        elif current_chunk:
            # If only one small chunk, keep it anyway
            chunk_text = ' '.join(current_chunk)
            chunks.append((chunk_text, current_tokens, section_title))
        
        return chunks
    
    def create_chunk_id(self, paper_id: str, chunk_index: int) -> str:
        """
        Create chunk ID in format P12345::C7
        
        Args:
            paper_id: Paper ID (e.g., P12345)
            chunk_index: 0-based chunk index
            
        Returns:
            Chunk ID (e.g., P12345::C7)
        """
        return f"{paper_id}::C{chunk_index}"
