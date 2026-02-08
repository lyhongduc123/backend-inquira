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
    docling_metadata: Dict[str, Any]  # bbox, prov, etc.


class ChunkingService:
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
    
    def chunk_from_docling_structure(
        self,
        doc_dict: Dict[str, Any],
        paper_id: str
    ) -> List[ChunkWithMetadata]:
        """
        Chunk document from docling structure WITH RICH METADATA.
        
        Args:
            doc_dict: Docling document dictionary
            paper_id: Paper ID for logging
            
        Returns:
            List of ChunkWithMetadata tuples
        """
        chunks = []
        
        # Extract texts from docling
        texts = doc_dict.get("texts", [])
        
        current_section_title = None
        current_section_level = None
        current_parts = []  # Clean text parts
        current_tokens = 0
        current_page = None
        current_label = None
        current_metadata_items = []  # Accumulate metadata from text_items
        
        for text_item in texts:
            # Skip furniture (headers, footers)
            if text_item.get("content_layer") == "furniture":
                continue
            
            text_content = text_item.get("text", "")
            if not text_content:
                continue
            
            label = text_item.get("label", "")
            level = text_item.get("level")
            page_no = None
            
            # Extract page number from prov
            prov = text_item.get("prov", [])
            if prov and len(prov) > 0:
                page_no = prov[0].get("page_no")
                if page_no is not None:
                    current_page = page_no
            
            item_tokens = self.count_tokens(text_content)
            
            # Handle section headers
            if label == "section_header":
                # Save previous chunk if exists
                if current_parts and current_tokens >= self.min_tokens:
                    chunks.append(ChunkWithMetadata(
                        text="\n\n".join(current_parts),
                        token_count=current_tokens,
                        section_title=current_section_title,
                        page_number=current_page,
                        label=current_label,
                        level=current_section_level,
                        docling_metadata={"items": current_metadata_items}
                    ))
                    current_parts = []
                    current_tokens = 0
                    current_metadata_items = []
                
                # Update section context
                current_section_title = text_content
                current_section_level = level
                current_label = label
                continue
            
            # Check if adding would exceed max
            if current_tokens + item_tokens > self.max_tokens and current_parts:
                # Save current chunk
                chunks.append(ChunkWithMetadata(
                    text="\n\n".join(current_parts),
                    token_count=current_tokens,
                    section_title=current_section_title,
                    page_number=current_page,
                    label=current_label,
                    level=current_section_level,
                    docling_metadata={"items": current_metadata_items}
                ))
                
                # Keep last part for overlap
                if current_parts:
                    overlap_text = current_parts[-1]
                    overlap_tokens = self.count_tokens(overlap_text)
                    
                    if overlap_tokens <= self.overlap_tokens:
                        current_parts = [overlap_text]
                        current_tokens = overlap_tokens
                        # Keep last metadata item for overlap
                        current_metadata_items = [current_metadata_items[-1]] if current_metadata_items else []
                    else:
                        current_parts = []
                        current_tokens = 0
                        current_metadata_items = []
                else:
                    current_parts = []
                    current_tokens = 0
                    current_metadata_items = []
            
            # Add content and metadata
            current_parts.append(text_content)
            current_tokens += item_tokens
            current_label = label
            
            # Accumulate docling metadata
            item_metadata = {
                "text": text_content[:100],  # Preview
                "label": label,
                "level": level,
                "bbox": text_item.get("bbox"),
                "prov": prov
            }
            current_metadata_items.append(item_metadata)
        
        # Add final chunk
        if current_parts:
            if current_tokens >= self.min_tokens:
                chunks.append(ChunkWithMetadata(
                    text="\n\n".join(current_parts),
                    token_count=current_tokens,
                    section_title=current_section_title,
                    page_number=current_page,
                    label=current_label,
                    level=current_section_level,
                    docling_metadata={"items": current_metadata_items}
                ))
            elif chunks:
                # Append to last chunk
                last = chunks[-1]
                combined_text = last.text + "\n\n" + "\n\n".join(current_parts)
                combined_tokens = self.count_tokens(combined_text)
                
                # Merge metadata
                combined_metadata_items = last.docling_metadata.get("items", []) + current_metadata_items
                
                chunks[-1] = ChunkWithMetadata(
                    text=combined_text,
                    token_count=combined_tokens,
                    section_title=last.section_title or current_section_title,
                    page_number=current_page or last.page_number,
                    label=current_label or last.label,
                    level=last.level,
                    docling_metadata={"items": combined_metadata_items}
                )
            else:
                # First chunk, keep even if small
                chunks.append(ChunkWithMetadata(
                    text="\n\n".join(current_parts),
                    token_count=current_tokens,
                    section_title=current_section_title,
                    page_number=current_page,
                    label=current_label,
                    level=current_section_level,
                    docling_metadata={"items": current_metadata_items}
                ))
        
        logger.info(f"[{paper_id}] Created {len(chunks)} chunks from docling structure")
        return chunks
    
    def chunk_from_tei_structure(
        self,
        tei_structure: Dict[str, Any],
        paper_id: str
    ) -> List[ChunkWithMetadata]:
        """
        Chunk document from TEI XML structure extracted by GROBID.
        
        TEI structure format:
        {
            "title": str,
            "authors": List[{name, affiliation, email}],
            "abstract": str,
            "sections": List[{title, content}],
            "references": List[{raw_text}]
        }
        
        Args:
            tei_structure: TEI structure dictionary from extract_tei_xml_structure()
            paper_id: Paper ID for logging
            
        Returns:
            List of ChunkWithMetadata tuples
        """
        chunks = []
        
        # Chunk abstract if present
        abstract = tei_structure.get("abstract", "").strip()
        if abstract:
            abstract_tokens = self.count_tokens(abstract)
            if abstract_tokens > self.max_tokens:
                # Split long abstract
                abstract_chunks = self._split_text_into_chunks(abstract, "Abstract")
                for chunk_text, token_count in abstract_chunks:
                    chunks.append(ChunkWithMetadata(
                        text=chunk_text,
                        token_count=token_count,
                        section_title="Abstract",
                        page_number=None,
                        label="abstract",
                        level=1,
                        docling_metadata={}
                    ))
            else:
                chunks.append(ChunkWithMetadata(
                    text=abstract,
                    token_count=abstract_tokens,
                    section_title="Abstract",
                    page_number=None,
                    label="abstract",
                    level=1,
                    docling_metadata={}
                ))
        
        # Chunk sections
        sections = tei_structure.get("sections", [])
        for section in sections:
            section_title = section.get("title", "").strip()
            section_content = section.get("content", "").strip()
            
            if not section_content:
                continue
            
            section_tokens = self.count_tokens(section_content)
            
            if section_tokens > self.max_tokens:
                # Split long section
                section_chunks = self._split_text_into_chunks(section_content, section_title)
                for chunk_text, token_count in section_chunks:
                    chunks.append(ChunkWithMetadata(
                        text=chunk_text,
                        token_count=token_count,
                        section_title=section_title,
                        page_number=None,
                        label="section",
                        level=1,
                        docling_metadata={}
                    ))
            else:
                chunks.append(ChunkWithMetadata(
                    text=section_content,
                    token_count=section_tokens,
                    section_title=section_title,
                    page_number=None,
                    label="section",
                    level=1,
                    docling_metadata={}
                ))
        
        logger.info(f"[{paper_id}] Created {len(chunks)} chunks from TEI structure")
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
                # Keep last few sentences for context
                overlap_sentences = []
                overlap_tokens = 0
                for s in reversed(current_chunk):
                    s_tokens = self.count_tokens(s)
                    if overlap_tokens + s_tokens <= self.overlap_tokens:
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
