import re
import tiktoken
from typing import List, Optional, Tuple, Dict, Any, NamedTuple
from app.extensions.logger import create_logger

logger = create_logger(__name__)


class ChunkWithMetadata(NamedTuple):
    """Enhanced chunk with docling metadata"""
    text: str  # Clean text for embeddings
    token_count: int
    section_title: Optional[str]
    page_number: Optional[int]
    label: Optional[str]  # section_header, text, caption, etc.
    level: Optional[int]  # Hierarchy level
    char_start: Optional[int]  # Character offset start
    char_end: Optional[int]  # Character offset end
    docling_metadata: Dict[str, Any]  # bbox, prov, etc.


class ChunkingService:
    """Chunk text into smaller pieces for embedding with section-boundary-aware intelligent chunking"""
    
    def __init__(
        self,
        min_tokens: int = 600,
        max_tokens: int = 1200,
        overlap_ratio: float = 0.1,  # 10% overlap
        model: str = "gpt-4"
    ):
        """
        Initialize text chunker with section-aware strategy
        
        Args:
            min_tokens: Minimum tokens per chunk (not strictly enforced for small sections)
            max_tokens: Maximum tokens per chunk (sections may be split if they exceed this)
            overlap_ratio: Ratio of overlap when splitting long sections (default 10%)
            model: Model name for tokenization
        """
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.overlap_ratio = overlap_ratio
        self.encoding = tiktoken.encoding_for_model(model)
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.encoding.encode(text))
    
    def chunk_from_docling_structure(
        self,
        doc_dict: Dict[str, Any],
        paper_id: str
    ) -> List[ChunkWithMetadata]:
        """
        Chunk document from docling structure with SECTION-BOUNDARY-AWARE strategy.
        
        Strategy:
        - Never cross section boundaries
        - Keep entire abstract as one chunk (if <= max_tokens)
        - Keep entire section as one chunk (if <= max_tokens)
        - Split long sections with small overlap (10%) but maintain section boundaries
        - Chunks always end at section boundaries for clean semantic units
        
        Args:
            doc_dict: Docling document dictionary
            paper_id: Paper ID for logging
            
        Returns:
            List of ChunkWithMetadata tuples
        """
        chunks = []
        
        # Extract texts from docling
        texts = doc_dict.get("texts", [])
        
        # Step 1: Group text items by section
        sections = []  # List of {"header": ..., "items": [...]}
        current_section = {"header": None, "level": None, "items": []}
        
        for text_item in texts:
            # Skip furniture (headers, footers, page numbers)
            if text_item.get("content_layer") == "furniture":
                continue
            
            text_content = text_item.get("text", "")
            if not text_content:
                continue
            
            label = text_item.get("label", "")
            
            # If this is a section header, finalize previous section and start new one
            if label == "section_header":
                # Save previous section if it has content
                if current_section["items"]:
                    sections.append(current_section)
                
                # Start new section
                current_section = {
                    "header": text_content,
                    "level": text_item.get("level"),
                    "items": []
                }
            else:
                # Add item to current section
                current_section["items"].append(text_item)
        
        # Add final section
        if current_section["items"]:
            sections.append(current_section)
        
        # Step 2: Chunk each section independently
        char_offset = 0
        for section in sections:
            section_header = section["header"]
            section_level = section["level"]
            section_items = section["items"]
            
            # Extract section text and metadata
            section_parts = []
            section_metadata_items = []
            section_page = None
            
            for item in section_items:
                text_content = item.get("text", "")
                section_parts.append(text_content)
                
                # Track page number
                prov = item.get("prov", [])
                if prov and len(prov) > 0:
                    page_no = prov[0].get("page_no")
                    if page_no is not None:
                        section_page = page_no
                
                # Accumulate metadata
                section_metadata_items.append({
                    "text": text_content[:100],
                    "label": item.get("label", ""),
                    "level": item.get("level"),
                    "bbox": item.get("bbox"),
                    "prov": prov
                })
            
            # Join section text
            section_text = "\n\n".join(section_parts)
            section_tokens = self.count_tokens(section_text)
            
            # Determine primary label for this section
            # (use most common label, or "text" as default)
            labels = [item.get("label", "text") for item in section_items]
            primary_label = max(set(labels), key=labels.count) if labels else "text"
            
            # If section fits in one chunk, keep it whole
            if section_tokens <= self.max_tokens:
                char_start = char_offset
                char_end = char_offset + len(section_text)
                
                chunks.append(ChunkWithMetadata(
                    text=section_text,
                    token_count=section_tokens,
                    section_title=section_header,
                    page_number=section_page,
                    label=primary_label,
                    level=section_level,
                    char_start=char_start,
                    char_end=char_end,
                    docling_metadata={"items": section_metadata_items}
                ))
                
                char_offset = char_end + 2  # +2 for newlines between sections
            else:
                # Section too long - split with overlap but stay within section
                section_chunks = self._split_section_with_overlap(
                    section_text=section_text,
                    section_header=section_header,
                    section_level=section_level,
                    section_page=section_page,
                    primary_label=primary_label,
                    metadata_items=section_metadata_items,
                    char_offset=char_offset
                )
                chunks.extend(section_chunks)
                
                # Update char offset
                if section_chunks:
                    last_chunk_end = section_chunks[-1].char_end
                    char_offset = (last_chunk_end + 2) if last_chunk_end is not None else (char_offset + len(section_text) + 2)
        
        logger.info(f"[{paper_id}] Created {len(chunks)} section-aware chunks from docling structure")
        return chunks
    
    def _split_section_with_overlap(
        self,
        section_text: str,
        section_header: Optional[str],
        section_level: Optional[int],
        section_page: Optional[int],
        primary_label: str,
        metadata_items: List[Dict[str, Any]],
        char_offset: int
    ) -> List[ChunkWithMetadata]:
        """
        Split a long section into multiple chunks with overlap.
        Chunks stay within section boundaries.
        
        Args:
            section_text: Full section text
            section_header: Section title
            section_level: Section hierarchy level
            section_page: Page number
            primary_label: Primary label for chunks
            metadata_items: Docling metadata
            char_offset: Starting character offset
            
        Returns:
            List of ChunkWithMetadata for this section
        """
        chunks = []
        
        # Split into sentences for cleaner boundaries
        sentences = self.split_into_sentences(section_text)
        
        current_chunk_sentences = []
        current_tokens = 0
        chunk_start_offset = char_offset
        
        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)
            
            # If adding this sentence exceeds max, save current chunk
            if current_tokens + sentence_tokens > self.max_tokens and current_chunk_sentences:
                # Save current chunk
                chunk_text = " ".join(current_chunk_sentences)
                chunk_end_offset = chunk_start_offset + len(chunk_text)
                
                chunks.append(ChunkWithMetadata(
                    text=chunk_text,
                    token_count=current_tokens,
                    section_title=section_header,
                    page_number=section_page,
                    label=primary_label,
                    level=section_level,
                    char_start=chunk_start_offset,
                    char_end=chunk_end_offset,
                    docling_metadata={"items": metadata_items}
                ))
                
                # Calculate overlap (10% of max_tokens)
                overlap_tokens = int(self.max_tokens * self.overlap_ratio)
                
                # Keep last few sentences for overlap
                overlap_sentences = []
                overlap_token_count = 0
                for s in reversed(current_chunk_sentences):
                    s_tokens = self.count_tokens(s)
                    if overlap_token_count + s_tokens <= overlap_tokens:
                        overlap_sentences.insert(0, s)
                        overlap_token_count += s_tokens
                    else:
                        break
                
                # Start new chunk with overlap
                current_chunk_sentences = overlap_sentences
                current_tokens = overlap_token_count
                
                # Update start offset (skip overlap text)
                if overlap_sentences:
                    overlap_text = " ".join(overlap_sentences)
                    chunk_start_offset = chunk_end_offset - len(overlap_text)
                else:
                    chunk_start_offset = chunk_end_offset
            
            # Add sentence to current chunk
            current_chunk_sentences.append(sentence)
            current_tokens += sentence_tokens
        
        # Add final chunk
        if current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences)
            chunk_end_offset = chunk_start_offset + len(chunk_text)
            
            chunks.append(ChunkWithMetadata(
                text=chunk_text,
                token_count=current_tokens,
                section_title=section_header,
                page_number=section_page,
                label=primary_label,
                level=section_level,
                char_start=chunk_start_offset,
                char_end=chunk_end_offset,
                docling_metadata={"items": metadata_items}
            ))
        
        return chunks
    
    def chunk_from_tei_structure(
        self,
        tei_structure: Dict[str, Any],
        paper_id: str
    ) -> List[ChunkWithMetadata]:
        """
        Chunk document from TEI XML structure with SECTION-BOUNDARY-AWARE strategy.
        
        TEI structure format:
        {
            "title": str,
            "authors": List[{name, affiliation, email}],
            "abstract": str,
            "sections": List[{title, content}],
            "references": List[{raw_text}]
        }
        
        Strategy:
        - Keep entire abstract as one chunk (if <= max_tokens)
        - Keep entire section as one chunk (if <= max_tokens)
        - Split long sections with 10% overlap
        - Never cross section boundaries
        
        Args:
            tei_structure: TEI structure dictionary from extract_tei_xml_structure()
            paper_id: Paper ID for logging
            
        Returns:
            List of ChunkWithMetadata tuples
        """
        chunks = []
        char_offset = 0
        
        # Chunk abstract if present
        abstract = tei_structure.get("abstract", "").strip()
        if abstract:
            abstract_tokens = self.count_tokens(abstract)
            char_start = char_offset
            char_end = char_offset + len(abstract)
            
            if abstract_tokens <= self.max_tokens:
                # Keep entire abstract as one chunk
                chunks.append(ChunkWithMetadata(
                    text=abstract,
                    token_count=abstract_tokens,
                    section_title="Abstract",
                    page_number=None,
                    label="abstract",
                    level=1,
                    char_start=char_start,
                    char_end=char_end,
                    docling_metadata={}
                ))
            else:
                # Split long abstract with overlap
                abstract_chunks = self._split_section_with_overlap(
                    section_text=abstract,
                    section_header="Abstract",
                    section_level=1,
                    section_page=None,
                    primary_label="abstract",
                    metadata_items=[],
                    char_offset=char_start
                )
                chunks.extend(abstract_chunks)
            
            char_offset = char_end + 2
        
        # Chunk sections
        sections = tei_structure.get("sections", [])
        for section in sections:
            section_title = section.get("title", "").strip()
            section_content = section.get("content", "").strip()
            
            if not section_content:
                continue
            
            section_tokens = self.count_tokens(section_content)
            char_start = char_offset
            char_end = char_offset + len(section_content)
            
            if section_tokens <= self.max_tokens:
                # Keep entire section as one chunk
                chunks.append(ChunkWithMetadata(
                    text=section_content,
                    token_count=section_tokens,
                    section_title=section_title,
                    page_number=None,
                    label="section",
                    level=1,
                    char_start=char_start,
                    char_end=char_end,
                    docling_metadata={}
                ))
            else:
                # Split long section with overlap
                section_chunks = self._split_section_with_overlap(
                    section_text=section_content,
                    section_header=section_title,
                    section_level=1,
                    section_page=None,
                    primary_label="section",
                    metadata_items=[],
                    char_offset=char_start
                )
                chunks.extend(section_chunks)
            
            char_offset = char_end + 2
        
        logger.info(f"[{paper_id}] Created {len(chunks)} section-aware chunks from TEI structure")
        return chunks
    
    def _split_text_into_chunks(
        self,
        text: str,
        section_title: Optional[str] = None
    ) -> List[Tuple[str, int]]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Text to split
            section_title: Section title for context
            
        Returns:
            List of (chunk_text, token_count) tuples
        """
        chunks = []
        sentences = self.split_into_sentences(text)
        
        current_chunk = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)
            
            # If adding this sentence exceeds max_tokens, save current chunk
            if current_tokens + sentence_tokens > self.max_tokens and current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append((chunk_text, current_tokens))
                
                # Start new chunk with overlap
                # Calculate overlap tokens (10% of max_tokens)
                overlap_token_limit = int(self.max_tokens * self.overlap_ratio)
                
                # Keep last few sentences for context
                overlap_sentences = []
                overlap_tokens = 0
                for s in reversed(current_chunk):
                    s_tokens = self.count_tokens(s)
                    if overlap_tokens + s_tokens <= overlap_token_limit:
                        overlap_sentences.insert(0, s)
                        overlap_tokens += s_tokens
                    else:
                        break
                
                current_chunk = overlap_sentences
                current_tokens = overlap_tokens
            
            current_chunk.append(sentence)
            current_tokens += sentence_tokens
        
        # Add final chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append((chunk_text, current_tokens))
        
        return chunks
    
    def chunk_from_structure(
        self,
        doc_dict: Dict[str, Any],
        paper_id: str
    ) -> List[Tuple[str, int, Optional[str]]]:
        """
        DEPRECATED: Use chunk_from_docling_structure() instead.
        Kept for backward compatibility.
        """
        # Use new method and convert to old format
        new_chunks = self.chunk_from_docling_structure(doc_dict, paper_id)
        return [(c.text, c.token_count, c.section_title) for c in new_chunks]
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
                # Calculate overlap tokens (10% of max_tokens)
                overlap_token_limit = int(self.max_tokens * self.overlap_ratio)
                
                overlap_chunk = []
                overlap_tokens = 0
                
                # Add sentences from end of previous chunk for overlap
                for prev_sentence in reversed(current_chunk):
                    prev_tokens = self.count_tokens(prev_sentence)
                    if overlap_tokens + prev_tokens <= overlap_token_limit:
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
