import io
import re
import unicodedata
from typing import Dict, Any, List, Optional
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.pipeline_options import (
    ThreadedPdfPipelineOptions,
)
from docling.pipeline.threaded_standard_pdf_pipeline import ThreadedStandardPdfPipeline
from docling.datamodel.base_models import InputFormat
from docling_core.types.io import DocumentStream
from app.extensions.logger import create_logger
import xml.etree.ElementTree as ET

logger = create_logger(__name__)

class ExtractorService:
    _pipeline_options = ThreadedPdfPipelineOptions(
        accelerator_options=AcceleratorOptions(
            device=AcceleratorDevice.CUDA,  # or AcceleratorDevice.AUTO
        ),
        do_ocr=False
    )

    def __init__(self):
        """Initialize the extractor service with docling converter"""
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_cls=ThreadedStandardPdfPipeline,
                    pipeline_options=self._pipeline_options
                )
            }
        )
        self.converter.initialize_pipeline(InputFormat.PDF)

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

            result = self.converter.convert(source=pdf_file)
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
        Convert docling document dict to clean text (no markdown).
        Extracts text from docling's structured format.
        
        Args:
            doc_dict: Document dictionary from docling
        Returns:
            Clean text without markdown formatting
        """
        text_parts = []
        
        # Extract texts from docling structure
        texts = doc_dict.get("texts", [])
        
        for text_item in texts:
            # Skip furniture (headers, footers, page numbers)
            if text_item.get("content_layer") == "furniture":
                continue
            
            # Get the text content
            text_content = text_item.get("text", "")
            if not text_content:
                continue
            
            label = text_item.get("label", "")
            
            # Add section headers as plain text (no markdown)
            if label == "section_header":
                text_parts.append(f"\n{text_content}\n")
            else:
                text_parts.append(text_content)
        
        return "\n\n".join(text_parts)

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

    def extract_tei_xml_structure(self, tei_xml: str) -> Dict[str, Any]:
        """
        Extract structured data from GROBID TEI XML.
        
        TEI (Text Encoding Initiative) XML from GROBID provides rich structured data:
        - Title, authors, affiliations
        - Abstract
        - Full text with section headers
        - References
        - Figures and tables metadata
        
        Args:
            tei_xml (str): TEI XML string from GROBID
            
        Returns:
            Dictionary with structured document data
        """
        try:
            root = ET.fromstring(tei_xml)
            
            # TEI namespace
            ns = {'tei': 'http://www.tei-c.org/ns/1.0'}
            
            # Extract metadata
            title_elem = root.find('.//tei:titleStmt/tei:title[@type="main"]', ns)
            title = title_elem.text if title_elem is not None else ""
            
            # Extract authors
            authors = []
            for author in root.findall('.//tei:sourceDesc//tei:author', ns):
                persName = author.find('.//tei:persName', ns)
                if persName is not None:
                    forename = persName.find('.//tei:forename[@type="first"]', ns)
                    surname = persName.find('.//tei:surname', ns)
                    
                    author_name = ""
                    if forename is not None and forename.text:
                        author_name = forename.text + " "
                    if surname is not None and surname.text:
                        author_name += surname.text
                    
                    # Extract affiliation
                    affiliation_elem = author.find('.//tei:affiliation/tei:orgName[@type="institution"]', ns)
                    affiliation = affiliation_elem.text if affiliation_elem is not None else None
                    
                    if author_name.strip():
                        authors.append({
                            'name': author_name.strip(),
                            'affiliation': affiliation
                        })
            
            # Extract abstract
            abstract_elem = root.find('.//tei:profileDesc/tei:abstract', ns)
            abstract = self._extract_text_from_element(abstract_elem, ns) if abstract_elem is not None else ""
            
            # Extract body sections
            sections = []
            body = root.find('.//tei:text/tei:body', ns)
            if body is not None:
                for div in body.findall('.//tei:div', ns):
                    head_elem = div.find('.//tei:head', ns)
                    section_title = head_elem.text if head_elem is not None else "Unknown Section"
                    
                    # Extract paragraphs in this section
                    paragraphs = []
                    for p in div.findall('.//tei:p', ns):
                        p_text = self._extract_text_from_element(p, ns)
                        if p_text.strip():
                            paragraphs.append(p_text.strip())
                    
                    if paragraphs:
                        sections.append({
                            'title': section_title,
                            'content': paragraphs
                        })
            
            # Extract references
            references = []
            for biblStruct in root.findall('.//tei:listBibl/tei:biblStruct', ns):
                ref_title_elem = biblStruct.find('.//tei:analytic/tei:title[@type="main"]', ns)
                if ref_title_elem is None:
                    ref_title_elem = biblStruct.find('.//tei:monogr/tei:title', ns)
                
                ref_title = ref_title_elem.text if ref_title_elem is not None else ""
                
                # Extract reference authors
                ref_authors = []
                for ref_author in biblStruct.findall('.//tei:author/tei:persName', ns):
                    ref_author_text = self._extract_text_from_element(ref_author, ns)
                    if ref_author_text.strip():
                        ref_authors.append(ref_author_text.strip())
                
                if ref_title and ref_title.strip():
                    references.append({
                        'title': ref_title.strip(),
                        'authors': ref_authors
                    })
            
            result = {
                'title': title,
                'authors': authors,
                'abstract': abstract,
                'sections': sections,
                'references': references
            }
            
            logger.info(f"Successfully extracted TEI XML structure: {len(sections)} sections, {len(references)} references")
            return result
            
        except Exception as e:
            logger.error(f"Error extracting TEI XML structure: {e}")
            raise Exception(f"Failed to extract TEI XML: {e}")
    
    def _extract_text_from_element(self, element: Optional[ET.Element], ns: Dict[str, str]) -> str:
        """
        Recursively extract text from an XML element and its children.
        
        Args:
            element: XML element
            ns: Namespace dictionary
            
        Returns:
            Extracted text
        """
        if element is None:
            return ""
        
        # Get text content
        text_parts = []
        
        # Add element's direct text
        if element.text:
            text_parts.append(element.text)
        
        # Add text from children
        for child in element:
            child_text = self._extract_text_from_element(child, ns)
            if child_text:
                text_parts.append(child_text)
            
            # Add tail text after child element
            if child.tail:
                text_parts.append(child.tail)
        
        return " ".join(text_parts)
    
    def extract_tei_xml_text(self, tei_xml: str) -> str:
        """
        Extract plain text from GROBID TEI XML (backward compatibility).
        For new code, prefer extract_tei_xml_structure() for better results.
        
        Args:
            tei_xml (str): TEI XML string from GROBID
            
        Returns:
            Extracted text as a string
        """
        try:
            structure = self.extract_tei_xml_structure(tei_xml)
            
            # Combine title, abstract, and sections into text
            text_parts = []
            
            if structure.get('title'):
                text_parts.append(structure['title'])
            
            if structure.get('abstract'):
                text_parts.append(structure['abstract'])
            
            for section in structure.get('sections', []):
                text_parts.append(section['title'])
                text_parts.extend(section['content'])
            
            full_text = "\n\n".join(text_parts)
            
            # Clean up the text
            full_text = self._fix_text_encoding(full_text)
            
            logger.info(f"Successfully extracted text from TEI XML: {len(full_text)} characters")
            return full_text
            
        except Exception as e:
            logger.error(f"Error extracting text from TEI XML: {e}")
            raise Exception(f"Failed to extract text from TEI XML: {e}")
