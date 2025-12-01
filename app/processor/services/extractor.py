import io
import re
import unicodedata
from pypdf import PdfReader
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
        for page_number, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            page_text = self._remove_header_footer(page_text, page_number)
            text += page_text + "\n"
        text = self._fix_text_encoding(text)
        return text

    def _remove_header_footer(self, text: str, page_number: int) -> str:
        """
        Remove common headers/footers and page numbers.
        This is heuristic-based.

        Args:
            text: The page text
            page_number: Current page number
        Returns:
            Cleaned text
        """
        lines = text.splitlines()
        cleaned_lines = []

        # Simple heuristics:
        # - Remove first/last line if it matches page number or repeated header/footer
        # - Remove lines that are too short (1-2 chars) if they appear at top/bottom

        if lines:
            # Check top lines
            for i, line in enumerate(lines[:3]):  # first 3 lines
                if self._is_header_footer(line, page_number):
                    lines[i] = ""  # blank out header

            # Check bottom lines
            for i, line in enumerate(lines[-3:]):  # last 3 lines
                if self._is_header_footer(line, page_number):
                    lines[-(i + 1)] = ""  # blank out footer

            # Remove empty lines after header/footer removal
            cleaned_lines = [l.strip() for l in lines if l.strip()]
        return "\n".join(cleaned_lines)

    def _is_header_footer(self, line: str, page_number: int) -> bool:
        """
        Heuristic to detect headers/footers:
        - Page numbers
        - Repeated short lines (author names, journal titles)
        """
        line_strip = line.strip()
        # Remove lines that are just the page number
        if line_strip.isdigit() and int(line_strip) == page_number:
            return True
        # Remove very short lines
        if len(line_strip) < 5:
            return True
        # Remove common repeated patterns (e.g., 'Journal of ...')
        # This can be extended with regex for your specific corpus
        if re.match(r'^(©|All rights reserved|Journal|Proceedings)', line_strip, re.I):
            return True
        return False

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
            '€': 'e',
            '�': '',
        }
        
        for bad, good in ligature_map.items():
            text = text.replace(bad, good)
        
        # Fix multiple spaces
        text = re.sub(r' +', ' ', text)
        
        # Fix multiple newlines (keep max 2)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()
