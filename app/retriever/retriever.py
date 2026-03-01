import httpx
import io
import unicodedata
import re
from typing import Optional, Dict, Any
from PyPDF2 import PdfReader
from app.extensions.logger import create_logger
from app.core.config import settings
from app.core.dtos.paper import PaperEnrichedDTO

logger = create_logger(__name__)


class PaperRetriever:
    """
    Universal paper retriever supporting multiple sources.

    Supports:
    - arXiv, PubMed, bioRxiv (open access)
    - OpenAlex PDF proxy and GROBID TEI XML
    - Paywalled papers (graceful fallback to abstract)
    - Multiple PDF URL formats
    """

    def __init__(self):
        self.timeout = 30.0
        self.openalex_content_url = "https://content.openalex.org/works"
        self.openalex_api_key = settings.OPENALEX_API_KEY

        # Known open access sources
        self.open_access_domains = {
            "arxiv.org",
            "biorxiv.org",
            "medrxiv.org",
            "ncbi.nlm.nih.gov/pmc",  # PubMed Central
            "europepmc.org",
            "plos.org",
            "frontiersin.org",
            "mdpi.com",
            "nature.com/articles",
        }

    async def download_pdf(
        self, pdf_url: str, check_open_access: bool = True
    ) -> Optional[bytes]:
        """
        Download PDF from URL with open access checking

        Args:
            pdf_url: URL to PDF file (or webpage with full-text)
            check_open_access: If True, only download from known OA sources

        Returns:
            PDF content as bytes, or None if failed/paywalled/not a PDF
        """
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }

            async with httpx.AsyncClient(
                timeout=self.timeout, follow_redirects=True, headers=headers
            ) as client:
                response = await client.get(pdf_url)

                # Handle paywalls (403, 401, 402)
                if response.status_code in [401, 402, 403]:
                    logger.info(
                        f"Access denied (likely paywalled): {pdf_url} - Status {response.status_code}"
                    )
                    return None

                response.raise_for_status()
                content_type = response.headers.get("content-type", "")

                # Check if it's actually a PDF
                if (
                    "pdf" in content_type.lower()
                    or "application/octet-stream" in content_type.lower()
                ):
                    # Check file size (avoid downloading huge files)
                    content_length = response.headers.get("content-length")
                    if (
                        content_length and int(content_length) > 50 * 1024 * 1024
                    ):  # 50MB limit
                        logger.warning(
                            f"PDF too large ({content_length} bytes): {pdf_url}"
                        )
                        return None

                    logger.info(
                        f"Downloaded PDF from {pdf_url} ({len(response.content)} bytes)"
                    )
                    return response.content

                # If it's HTML/text, it's probably a webpage, not a direct PDF
                elif "html" in content_type.lower() or "text" in content_type.lower():
                    logger.warning(
                        f"URL is a webpage, not a direct PDF link: {pdf_url} (content-type: {content_type})"
                    )
                    return None

                else:
                    logger.warning(
                        f"Unknown content type, not a PDF: {pdf_url} (content-type: {content_type})"
                    )
                    return None

        except httpx.HTTPStatusError as e:
            if e.response.status_code in [401, 402, 403, 404]:
                logger.info(
                    f"Paper not accessible: {pdf_url} - {e.response.status_code}"
                )
            else:
                logger.error(f"HTTP error downloading PDF from {pdf_url}: {e}")
            return None
        except httpx.TimeoutException:
            logger.warning(f"Timeout downloading PDF from {pdf_url}")
            return None
        except Exception as e:
            logger.error(f"Error downloading PDF from {pdf_url}: {e}")
            return None

    async def download_pdf_from_openalex(self, openalex_id: str) -> Optional[bytes]:
        """
        Download PDF via OpenAlex content delivery network.

        Args:
            openalex_id: OpenAlex work ID (e.g., 'W2741809807' or full URL)

        Returns:
            PDF content as bytes, or None if not available
        """
        if openalex_id.startswith("http"):
            openalex_id = openalex_id.split("/")[-1]

        if not openalex_id.startswith("W"):
            openalex_id = f"W{openalex_id}"

        pdf_url = f"{self.openalex_content_url}/{openalex_id}.pdf?api_key={self.openalex_api_key}"

        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (compatible; ExegentBot/1.0; +https://exegent.ai)"
            }

            async with httpx.AsyncClient(
                timeout=60.0,  # PDFs can be large
                follow_redirects=True,
                headers=headers,
            ) as client:
                response = await client.get(pdf_url)

                if response.status_code == 404:
                    logger.info(f"PDF not available via OpenAlex for {openalex_id}")
                    return None

                response.raise_for_status()

                content_type = response.headers.get("content-type", "")
                if (
                    "pdf" in content_type.lower()
                    or "application/octet-stream" in content_type.lower()
                ):
                    logger.info(
                        f"Downloaded PDF via OpenAlex for {openalex_id} ({len(response.content)} bytes)"
                    )
                    return response.content
                else:
                    logger.warning(
                        f"OpenAlex returned non-PDF content for {openalex_id}: {content_type}"
                    )
                    return None

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.info(f"PDF not found via OpenAlex for {openalex_id}")
            else:
                logger.error(f"HTTP error downloading from OpenAlex {openalex_id}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error downloading from OpenAlex {openalex_id}: {e}")
            return None

    async def download_tei_from_openalex(self, openalex_id: str) -> Optional[str]:
        """
        Download GROBID TEI XML via OpenAlex.

        Args:
            openalex_id: OpenAlex work ID

        Returns:
            TEI XML string, or None if not available
        """
        if openalex_id.startswith("http"):
            openalex_id = openalex_id.split("/")[-1]

        if not openalex_id.startswith("W"):
            openalex_id = f"W{openalex_id}"

        tei_url = f"{self.openalex_content_url}/{openalex_id}.grobid-xml?api_key={self.openalex_api_key}"

        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (compatible; ExegentBot/1.0; +https://exegent.ai)"
            }

            async with httpx.AsyncClient(
                timeout=60.0, follow_redirects=True, headers=headers
            ) as client:
                response = await client.get(tei_url)

                if response.status_code == 404:
                    logger.info(f"GROBID TEI not available for {openalex_id}")
                    return None

                response.raise_for_status()

                content_type = response.headers.get("content-type", "")
                if "xml" in content_type.lower() or "text" in content_type.lower():
                    logger.info(
                        f"Downloaded GROBID TEI for {openalex_id} ({len(response.text)} chars)"
                    )
                    return response.text
                else:
                    logger.warning(
                        f"OpenAlex returned non-XML content for {openalex_id}: {content_type}"
                    )
                    return None

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.info(f"GROBID TEI not found for {openalex_id}")
            else:
                logger.error(
                    f"HTTP error downloading TEI from OpenAlex {openalex_id}: {e}"
                )
            return None
        except Exception as e:
            logger.error(f"Error downloading TEI from OpenAlex {openalex_id}: {e}")
            return None

    async def try_multiple_sources(self, paper_data: Dict[str, Any]) -> Optional[bytes]:
        """
        DEPRECATED: Use 'paper_service.resolve_paper_content' instead.
        
        Try to retrieve paper PDF from multiple sources.

        Args:
            paper_data: Dictionary containing paper metadata
                       Keys: openalex_id, arxiv_id, pmid, doi, pdf_url, open_access_pdf

        Returns:
            PDF content as bytes, or None if all sources failed
        """
        sources = []

        if "openalex_id" in paper_data and paper_data["openalex_id"]:
            sources.append(("OpenAlex", "openalex", paper_data["openalex_id"]))

        if "arxiv_id" in paper_data and paper_data["arxiv_id"]:
            arxiv_url = self.get_pdf_url_from_arxiv_id(paper_data["arxiv_id"])
            sources.append(("arXiv", "url", arxiv_url))

        open_access_pdf_url = paper_data.get("open_access_pdf")
        if open_access_pdf_url and isinstance(open_access_pdf_url, str):
            sources.append(("OpenAccess", "url", open_access_pdf_url))

        if "doi" in paper_data and paper_data["doi"]:
            doi = paper_data["doi"]
            if "biorxiv" in doi.lower() or "medrxiv" in doi.lower():
                biorxiv_url = f"https://www.biorxiv.org/content/{doi}.full.pdf"
                sources.append(("bioRxiv", "url", biorxiv_url))

        # Try each source in priority order
        for source_name, source_type, source_value in sources:
            logger.info(f"Trying to retrieve paper from {source_name}")

            try:
                if source_type == "openalex":
                    pdf_content = await self.download_pdf_from_openalex(source_value)
                elif source_type == "url":
                    pdf_content = await self.download_pdf(
                        source_value, check_open_access=True
                    )
                else:
                    continue

                if pdf_content:
                    logger.info(f"Successfully retrieved paper from {source_name}")
                    return pdf_content
                else:
                    logger.debug(f"Failed to retrieve from {source_name}")

            except Exception as e:
                logger.error(f"Error retrieving from {source_name}: {e}")

        logger.warning(
            f"Could not retrieve full-text from any source. Paper may be paywalled."
        )
        return None

    async def get_structured_content(
        self, paper_data: Dict[str, Any], prefer_tei: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        DEPRECATED: Use 'download_pdf' and 'download_pdf_from_openalex', 'download_tei_from_openalex' instead.

        Get structured content from paper (TEI XML or PDF).

        TEI XML (from GROBID via OpenAlex) provides:
        - Structured sections and paragraphs
        - Parsed references
        - Formulas, tables, figures
        - Author affiliations

        Args:
            paper_data: Paper metadata with openalex_id
            prefer_tei: If True, try TEI XML first (recommended)

        Returns:
            Dict with 'format' ('tei' or 'pdf') and 'content' (str or bytes)
        """
        openalex_id = paper_data.get("openalex_id")

        if not openalex_id:
            # No OpenAlex ID, fallback to PDF only
            pdf_content = await self.try_multiple_sources(paper_data)
            if pdf_content:
                return {"format": "pdf", "content": pdf_content}
            return None

        if prefer_tei:
            # Try GROBID TEI XML first (best for structured extraction)
            tei_xml = await self.download_tei_from_openalex(openalex_id)
            if tei_xml:
                return {"format": "tei", "content": tei_xml}

            # Fallback to PDF
            logger.info("TEI not available, falling back to PDF")

        # Try PDF (includes OpenAlex proxy)
        pdf_content = await self.try_multiple_sources(paper_data)
        if pdf_content:
            return {"format": "pdf", "content": pdf_content}

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
            r"arxiv\.org/abs/(\d+\.\d+)",
            r"arxiv\.org/pdf/(\d+\.\d+)",
            r"ar[Xx]iv:(\d+\.\d+)",
            r"^(\d+\.\d+)$",  # Just the ID itself
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
            r"doi\.org/(.+)",
            r"dx\.doi\.org/(.+)",
            r"^(10\.\d+/.+)$",  # DOI format: 10.xxxx/...
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

    def get_access_info(self, paper_data: PaperEnrichedDTO) -> Dict[str, Any]:
        """
        Determine access availability for a paper.

        Args:
            paper_data: PaperPreprocess object containing paper metadata

        Returns:
            Dictionary with access information
        """
        sources = []
        has_pdf = False
        has_xml = False

        is_open_access_db = paper_data.is_open_access
        pdf_url = paper_data.pdf_url
        if pdf_url:
            sources.append("pdf_url")
            has_pdf = True
        if paper_data.external_ids:
            if paper_data.external_ids.get("openalex_id"):
                if paper_data.has_content:
                    if paper_data.has_content.get("grobid_tei"):
                        sources.append("openalex_grobid_xml")
                        has_xml = True
                    if paper_data.has_content.get("pdf"):
                        sources.append("openalex_pdf")
                        has_pdf = True

        is_open_access = is_open_access_db if is_open_access_db else (len(sources) > 0)

        return {
            "is_open_access": is_open_access,
            "has_pdf_url": has_pdf,
            "has_tei_xml": has_xml,
            "sources": sources,
        }


# Backward compatibility alias
ArxivRetriever = PaperRetriever
