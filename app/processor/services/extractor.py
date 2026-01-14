import io
import re
import unicodedata
from typing import Dict, Any, List
from docling.document_converter import DocumentConverter
from docling_core.types.io import DocumentStream
from app.extensions.logger import create_logger

logger = create_logger(__name__)


class ExtractorService:
    def __init__(self):
        """Initialize the extractor service with docling converter"""
        self.converter = DocumentConverter()

    def extract_pdf_structure(self, pdf_bytes: bytes) -> Dict[str, Any]:
        """
        Extract structured data from PDF bytes using docling.
        Returns a dictionary with document structure preserved.
        
        Args:
            pdf_bytes (bytes): The PDF file content in bytes.
        Returns:
            Dictionary containing structured document data with sections, paragraphs, tables, etc.
        """
        try:
            # Convert PDF bytes to file-like object
            pdf_file = DocumentStream(name="input.pdf", stream=io.BytesIO(pdf_bytes))

            # Use docling to convert the PDF
            result = self.converter.convert(source=pdf_file)

            # Export to dict for structured access
            doc_dict = result.document.export_to_dict()

            logger.info(
                f"Successfully extracted structured document using docling"
            )
            return doc_dict

        except Exception as e:
            logger.error(f"Error extracting PDF structure with docling: {e}")
            raise Exception(f"Failed to extract PDF structure: {e}")
    
    def extract_pdf_text(self, pdf_bytes: bytes) -> str:
        """
        Extract text from PDF bytes using docling (backward compatibility).
        For new code, prefer extract_pdf_structure() for better results.
        
        Args:
            pdf_bytes (bytes): The PDF file content in bytes.
        Returns:
            Extracted text as a string.
        """
        try:
            # Get structured document
            doc_dict = self.extract_pdf_structure(pdf_bytes)
            
            # Convert to markdown for text-based processing
            text = self._dict_to_markdown(doc_dict)
            
            # Clean up the text
            text = self._fix_text_encoding(text)

            logger.info(
                f"Successfully extracted text using docling: {len(text)} characters"
            )
            return text

        except Exception as e:
            logger.error(f"Error extracting PDF text with docling: {e}")
            raise Exception(f"Failed to extract PDF text: {e}")

    def _dict_to_markdown(self, doc_dict: Dict[str, Any]) -> str:
        """
        Convert document dict to markdown text.
        This is a simplified converter - docling's export_to_markdown is better.
        
        Args:
            doc_dict: Document dictionary from docling
        Returns:
            Markdown formatted text
        """
        # Try to get main text content from the document structure
        # The exact structure depends on docling's output format
        text_parts = []
        
        # Extract text from document structure
        if "main-text" in doc_dict:
            for item in doc_dict.get("main-text", []):
                if isinstance(item, dict):
                    if "text" in item:
                        text_parts.append(item["text"])
                elif isinstance(item, str):
                    text_parts.append(item)
        
        return "\n\n".join(text_parts) if text_parts else str(doc_dict)

    def split_sections(self, text: str) -> dict:
        """
        Split text into sections based on common academic paper headings.
        Docling preserves markdown structure, making this easier.
        """
        pattern = r"^#{1,3}\s*(Abstract|Introduction|Methods?|Results?|Discussion|Conclusion|Findings|Background|Related Work)"
        sections = {}
        current_heading = "Intro"
        sections[current_heading] = ""

        for line in text.splitlines():
            line_strip = line.strip()
            match = re.match(pattern, line_strip, re.I)
            if match:
                current_heading = match.group(1)
                sections[current_heading] = ""
            else:
                sections[current_heading] += line_strip + " "
        return sections

    def extract_results_conclusion(self, sections: dict) -> str:
        result_text = ""
        for key in sections:
            if key.lower() in ["results", "discussion", "conclusion", "findings"]:
                result_text += sections[key] + "\n"
        return result_text.strip()

    def extract_keywords(self, text: str) -> list:
        """
        Extract keywords from the text using a simple regex approach.
        Args:
            text (str): The input text from which to extract keywords.
        Returns:
            List of extracted keywords.
        """
        # Dummy implementation: extract words longer than 5 characters
        keywords = re.findall(r"\b\w{6,}\b", text)
        return list(set(keywords))

    def _fix_text_encoding(self, text: str) -> str:
        """
        Fix common PDF text encoding issues
        Args:
            text: Raw extracted text
        Returns:
            Cleaned text
        """
        # Normalize unicode characters
        text = unicodedata.normalize("NFKD", text)

        # Remove non-printable characters except newlines/tabs
        text = "".join(char for char in text if char.isprintable() or char in "\n\t")

        # Fix common ligatures that get mangled
        ligature_map = {
            "ﬁ": "fi",
            "ﬂ": "fl",
            "ﬀ": "ff",
            "ﬃ": "ffi",
            "ﬄ": "ffl",
            "ﬅ": "ft",
            "ﬆ": "st",
            "€": "e",
            "�": "",
        }

        for bad, good in ligature_map.items():
            text = text.replace(bad, good)

        # Fix multiple spaces
        text = re.sub(r" +", " ", text)

        # Fix multiple newlines (keep max 2)
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()
