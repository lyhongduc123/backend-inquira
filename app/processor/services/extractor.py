import io
import re
import unicodedata
from PyPDF2 import PdfReader
from app.extensions.logger import create_logger

logger = create_logger(__name__)

class ExtractorService:
    def extract_pdf_text(self, pdf_bytes: bytes) -> str:
        """
        Extract text from PDF bytes.
        Args:
            pdf_bytes (bytes): The PDF file content in bytes.
            
        Returns:
            Extracted text as a string.
        """
        pdf_file = io.BytesIO(pdf_bytes)
        reader = PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    
    import re

    def split_sections(self, text: str) -> dict:
        pattern = r"(Results|Discussion|Conclusion|Findings)"
        sections = {}
        current_heading = "Intro"
        sections[current_heading] = ""

        for line in text.splitlines():
            line_strip = line.strip()
            if re.match(pattern, line_strip, re.I):
                current_heading = line_strip
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
    
    # TODO: Implement a more advanced keyword extraction method
    def extract_keywords(self, text: str) -> list:
        """
        Extract keywords from the text using a simple regex approach.
        
        Args:
            text (str): The input text from which to extract keywords.
        Returns:
            List of extracted keywords.
        """
        # Dummy implementation: extract words longer than 5 characters
        keywords = re.findall(r'\b\w{6,}\b', text)
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
        text = unicodedata.normalize('NFKD', text)
        
        # Remove non-printable characters except newlines/tabs
        text = ''.join(char for char in text if char.isprintable() or char in '\n\t')
        
        # Fix common ligatures that get mangled
        ligature_map = {
            'ﬁ': 'fi',
            'ﬂ': 'fl',
            'ﬀ': 'ff',
            'ﬃ': 'ffi',
            'ﬄ': 'ffl',
            'ﬅ': 'ft',
            'ﬆ': 'st',
            '€': 'e',  # Common encoding error
            '�': '',   # Replacement character - remove it
        }
        
        for bad, good in ligature_map.items():
            text = text.replace(bad, good)
        
        # Fix multiple spaces
        text = re.sub(r' +', ' ', text)
        
        # Fix multiple newlines (keep max 2)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()