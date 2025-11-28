import httpx
import io
import unicodedata
import re
from typing import Optional, Dict, Any
from PyPDF2 import PdfReader
from app.extensions.logger import create_logger

logger = create_logger(__name__)


class PaperRetriever:
    """
    Universal paper retriever supporting multiple sources (arXiv, PubMed, etc.)
    
    Handles:
    - Open access papers (arXiv, PubMed Central, bioRxiv)
    - Paywalled papers (graceful fallback to abstract)
    - Multiple PDF URL formats
    """
    
    def __init__(self):
        self.timeout = 30.0  # PDF downloads can take longer
        
        # Known open access sources
        self.open_access_domains = {
            'arxiv.org',
            'biorxiv.org', 
            'medrxiv.org',
            'ncbi.nlm.nih.gov/pmc',  # PubMed Central
            'europepmc.org',
            'plos.org',
            'frontiersin.org',
            'mdpi.com',
            'nature.com/articles',  # Some Nature articles are OA
        }
    
    def is_likely_open_access(self, url: str) -> bool:
        """
        Check if URL is likely from an open access source
        
        Args:
            url: URL to check
            
        Returns:
            True if likely open access, False otherwise
        """
        if not url:
            return False
        
        url_lower = url.lower()
        return any(domain in url_lower for domain in self.open_access_domains)
    
    async def download_pdf(self, pdf_url: str, check_open_access: bool = True) -> Optional[bytes]:
        """
        Download PDF from URL with open access checking
        
        Args:
            pdf_url: URL to PDF file (or webpage with full-text)
            check_open_access: If True, only download from known OA sources
            
        Returns:
            PDF content as bytes, or None if failed/paywalled/not a PDF
        """
        if check_open_access:
            if self.get_access_info({'pdf_url': pdf_url})['likely_paywalled']:
                logger.info(f"Skipping potential paywalled URL: {pdf_url}")
                return None
        
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            async with httpx.AsyncClient(
                timeout=self.timeout, 
                follow_redirects=True,
                headers=headers
            ) as client:
                response = await client.get(pdf_url)
                
                # Handle paywalls (403, 401, 402)
                if response.status_code in [401, 402, 403]:
                    logger.info(f"Access denied (likely paywalled): {pdf_url} - Status {response.status_code}")
                    return None
                
                response.raise_for_status()
                
                # Verify content type
                content_type = response.headers.get("content-type", "")
                
                # Check if it's actually a PDF
                if "pdf" in content_type.lower() or "application/octet-stream" in content_type.lower():
                    # Check file size (avoid downloading huge files)
                    content_length = response.headers.get("content-length")
                    if content_length and int(content_length) > 50 * 1024 * 1024:  # 50MB limit
                        logger.warning(f"PDF too large ({content_length} bytes): {pdf_url}")
                        return None
                    
                    logger.info(f"Downloaded PDF from {pdf_url} ({len(response.content)} bytes)")
                    return response.content
                
                # If it's HTML/text, it's probably a webpage, not a direct PDF
                elif "html" in content_type.lower() or "text" in content_type.lower():
                    logger.warning(f"URL is a webpage, not a direct PDF link: {pdf_url} (content-type: {content_type})")
                    return None
                
                else:
                    logger.warning(f"Unknown content type, not a PDF: {pdf_url} (content-type: {content_type})")
                    return None
                
        except httpx.HTTPStatusError as e:
            if e.response.status_code in [401, 402, 403, 404]:
                logger.info(f"Paper not accessible: {pdf_url} - {e.response.status_code}")
            else:
                logger.error(f"HTTP error downloading PDF from {pdf_url}: {e}")
            return None
        except httpx.TimeoutException:
            logger.warning(f"Timeout downloading PDF from {pdf_url}")
            return None
        except Exception as e:
            logger.error(f"Error downloading PDF from {pdf_url}: {e}")
            return None
    
    # def extract_text_from_pdf(self, pdf_content: bytes) -> Optional[str]:
    #     """
    #     Extract text from PDF content with encoding fixes
        
    #     Args:
    #         pdf_content: PDF file as bytes
            
    #     Returns:
    #         Extracted text, or None if failed
    #     """
    #     try:
    #         pdf_file = io.BytesIO(pdf_content)
    #         reader = PdfReader(pdf_file)
            
    #         text_parts = []
    #         for page_num, page in enumerate(reader.pages):
    #             try:
    #                 text = page.extract_text()
    #                 if text:
    #                     # Fix common PDF encoding issues
    #                     text = self._fix_text_encoding(text)
    #                     text_parts.append(text)
    #             except Exception as e:
    #                 logger.warning(f"Error extracting text from page {page_num}: {e}")
            
    #         full_text = "\n\n".join(text_parts)
    #         logger.info(f"Extracted {len(full_text)} characters from {len(reader.pages)} pages")
            
    #         return full_text if full_text.strip() else None
            
    #     except Exception as e:
    #         logger.error(f"Error extracting text from PDF: {e}")
    #         return None
    
    # def _fix_text_encoding(self, text: str) -> str:
    #     """
    #     Fix common PDF text encoding issues
        
    #     Args:
    #         text: Raw extracted text
            
    #     Returns:
    #         Cleaned text
    #     """

        
    #     # Normalize unicode characters
    #     text = unicodedata.normalize('NFKD', text)
        
    #     # Remove non-printable characters except newlines/tabs
    #     text = ''.join(char for char in text if char.isprintable() or char in '\n\t')
        
    #     # Fix common ligatures that get mangled
    #     ligature_map = {
    #         'ﬁ': 'fi',
    #         'ﬂ': 'fl',
    #         'ﬀ': 'ff',
    #         'ﬃ': 'ffi',
    #         'ﬄ': 'ffl',
    #         'ﬅ': 'ft',
    #         'ﬆ': 'st',
    #         '€': 'e',  # Common encoding error
    #         '�': '',   # Replacement character - remove it
    #     }
        
    #     for bad, good in ligature_map.items():
    #         text = text.replace(bad, good)
        
    #     # Fix multiple spaces
    #     text = re.sub(r' +', ' ', text)
        
    #     # Fix multiple newlines (keep max 2)
    #     text = re.sub(r'\n{3,}', '\n\n', text)
        
    #     return text.strip()
    
    # async def get_paper_text(self, pdf_url: str, check_open_access: bool = True) -> Optional[bytes]:
    #     """
    #     Download PDF and extract text
        
    #     Args:
    #         pdf_url: URL to PDF file
    #         check_open_access: If True, only download from known OA sources
            
    #     Returns:
    #         Extracted text, or None if failed/paywalled
    #     """
    #     pdf_content = await self.download_pdf(pdf_url, check_open_access)
    #     if not pdf_content:
    #         return None
        
    #     # text = self.extract_text_from_pdf(pdf_content)
    #     return pdf_content
    
    async def try_multiple_sources(self, paper_data: Dict[str, Any]) -> Optional[str]:
        """
        Try to retrieve paper text from multiple possible sources
        
        Args:
            paper_data: Dictionary containing paper metadata
                       Keys: arxiv_id, pmid, doi, pdf_url, open_access_pdf (URL string)
                       
        Returns:
            Extracted text, or None if all sources failed
        """
        # Priority order for retrieving papers
        sources = []
        
        # 1. arXiv (most reliable for open access)
        if 'arxiv_id' in paper_data and paper_data['arxiv_id']:
            arxiv_url = self.get_pdf_url_from_arxiv_id(paper_data['arxiv_id'])
            sources.append(('arXiv', arxiv_url))
        
        # 2. Direct open access PDF URL (from Semantic Scholar, extracted from dict)
        open_access_pdf_url = paper_data.get('open_access_pdf')
        if open_access_pdf_url and isinstance(open_access_pdf_url, str):
            sources.append(('OpenAccess', open_access_pdf_url))
        
        # 3. Generic PDF URL (if from OA source)
        if 'pdf_url' in paper_data and paper_data['pdf_url']:
            if self.is_likely_open_access(paper_data['pdf_url']):
                sources.append(('PDF', paper_data['pdf_url']))
        
        # 4. bioRxiv/medRxiv
        if 'doi' in paper_data and paper_data['doi']:
            doi = paper_data['doi']
            if 'biorxiv' in doi.lower() or 'medrxiv' in doi.lower():
                # Construct bioRxiv/medRxiv PDF URL
                biorxiv_url = f"https://www.biorxiv.org/content/{doi}.full.pdf"
                sources.append(('bioRxiv', biorxiv_url))
        
        # Try each source in priority order
        # for source_name, url in sources:
        #     logger.info(f"Trying to retrieve paper from {source_name}: {url}")
        #     text = await self.get_paper_text(url, check_open_access=True)
        #     if text:
        #         logger.info(f"Successfully retrieved paper from {source_name}")
        #         return text
        #     else:
        #         logger.debug(f"Failed to retrieve from {source_name}")
        
        logger.warning(f"Could not retrieve full-text from any source. Paper may be paywalled.")
        return None
    
    def extract_arxiv_id(self, url_or_id: str) -> Optional[str]:
        """
        Extract arXiv ID from URL or ID string
        
        Args:
            url_or_id: arXiv URL or ID (e.g., https://arxiv.org/abs/2301.12345 or 2301.12345)
            
        Returns:
            arXiv ID (e.g., 2301.12345) or None
        """
        if not url_or_id:
            return None
        
        patterns = [
            r'arxiv\.org/abs/(\d+\.\d+)',
            r'arxiv\.org/pdf/(\d+\.\d+)',
            r'ar[Xx]iv:(\d+\.\d+)',
            r'^(\d+\.\d+)$',  # Just the ID itself
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url_or_id)
            if match:
                return match.group(1)
        
        return None
    
    def extract_doi(self, url_or_doi: str) -> Optional[str]:
        """
        Extract DOI from URL or DOI string
        
        Args:
            url_or_doi: DOI URL or DOI string
            
        Returns:
            DOI or None
        """
        if not url_or_doi:
            return None
        
        patterns = [
            r'doi\.org/(.+)',
            r'dx\.doi\.org/(.+)',
            r'^(10\.\d+/.+)$',  # DOI format: 10.xxxx/...
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url_or_doi)
            if match:
                return match.group(1)
        
        return None
    
    def get_pdf_url_from_arxiv_id(self, arxiv_id: str) -> str:
        """
        Construct PDF URL from arXiv ID
        
        Args:
            arxiv_id: arXiv ID (e.g., 2301.12345)
            
        Returns:
            PDF URL
        """
        return f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    
    def get_access_info(self, paper_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Determine access availability for a paper
        
        Args:
            paper_data: Dictionary containing paper metadata
                       Keys: arxiv_id, doi, pdf_url, open_access_pdf (URL string), is_open_access (bool)
            
        Returns:
            Dictionary with access information:
            {
                'is_open_access': bool,
                'has_pdf_url': bool,
                'sources': List[str],  # Available sources
                'likely_paywalled': bool
            }
        """
        sources = []
        has_pdf = False
        
        # Use is_open_access from database if available (most reliable)
        is_open_access_db = paper_data.get('is_open_access', False)
        
        # Check arXiv
        if paper_data.get('arxiv_id'):
            sources.append('arXiv')
            has_pdf = True
        
        # Check Semantic Scholar open access PDF (stored as URL string from dict)
        open_access_pdf_url = paper_data.get('open_access_pdf')
        if open_access_pdf_url and isinstance(open_access_pdf_url, str):
            sources.append('OpenAccess')
            has_pdf = True
        
        # Check other sources
        pdf_url = paper_data.get('pdf_url', '')
        if pdf_url and self.is_likely_open_access(pdf_url):
            sources.append('PDF')
            has_pdf = True
        
        # Check bioRxiv/medRxiv
        doi = paper_data.get('doi', '')
        if doi and ('biorxiv' in doi.lower() or 'medrxiv' in doi.lower()):
            sources.append('bioRxiv/medRxiv')
            has_pdf = True
        
        # Trust database is_open_access if set, otherwise fallback to source detection
        is_open_access = is_open_access_db if is_open_access_db else (len(sources) > 0)
        likely_paywalled = not is_open_access and bool(paper_data.get('doi'))
        
        return {
            'is_open_access': is_open_access,
            'has_pdf_url': has_pdf,
            'sources': sources,
            'likely_paywalled': likely_paywalled
        }


# Backward compatibility alias
ArxivRetriever = PaperRetriever
