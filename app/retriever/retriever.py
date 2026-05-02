import asyncio
import re
import sys
from typing import Optional, Dict, Any, Awaitable, Callable, Mapping, cast
from urllib.parse import urlparse

import httpx
from playwright.async_api import (
    BrowserContext,
    TimeoutError as PlaywrightTimeoutError,
    async_playwright,
)
import playwright_stealth

from app.extensions.logger import create_logger
from app.core.config import settings
from app.core.dtos.paper import PaperEnrichedDTO
from app.retriever.html_pdf_resolver import HtmlPdfResolver

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
        self.html_pdf_resolver = HtmlPdfResolver()

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

    @staticmethod
    def _supports_playwright_subprocess() -> bool:
        """Best-effort check for Playwright subprocess compatibility on current loop."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return False

        if sys.platform.startswith("win"):
            loop_name = loop.__class__.__name__.lower()
            if "proactor" not in loop_name:
                return False

        return True

    async def download_pdf(
        self,
        pdf_url: str,
        check_open_access: bool = True,
        allow_manual_verification: bool = False,
    ) -> Optional[bytes]:
        """
        Download PDF from URL with open access checking

        Args:
            pdf_url: URL to PDF file (or webpage with full-text)
            check_open_access: If True, only download from known OA sources
            allow_manual_verification: If True, open a headed browser on challenge pages
                so a human can complete anti-bot verification locally

        Returns:
            PDF content as bytes, or None if failed/paywalled/not a PDF
        """
        normalized_url = self._normalize_download_target(pdf_url)

        # Dedicated arXiv fast-path: no Playwright fallback needed.
        arxiv_pdf_url = await self._resolve_arxiv_pdf_url(normalized_url)
        if arxiv_pdf_url:
            arxiv_pdf = await self._download_arxiv_pdf(arxiv_pdf_url)
            if arxiv_pdf:
                return arxiv_pdf

        try:
            default_headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }

            async with httpx.AsyncClient(
                timeout=self.timeout,
                follow_redirects=True,
                headers=default_headers,
            ) as client:
                response = await client.get(normalized_url)
                if response.status_code in [401, 402, 403]:
                    self._log_httpx_access_denied(
                        response=response,
                        attempted_url=normalized_url,
                        source="direct-client",
                    )
                    browser_pdf = await self._try_browser_agent_download(
                        normalized_url,
                        allow_manual_verification=allow_manual_verification,
                    )
                    if browser_pdf:
                        return browser_pdf
                    return None
                response.raise_for_status()
                content_type = response.headers.get("content-type", "")

                if (
                    "pdf" in content_type.lower()
                    or "application/octet-stream" in content_type.lower()
                    or "binary/octet-stream" in content_type.lower()
                ):
                    return self._validate_pdf_size_and_return(
                        response.content,
                        response.headers.get("content-length"),
                        str(response.url),
                    )

                if "html" in content_type.lower() or "text" in content_type.lower():
                    logger.warning(
                        f"URL is a webpage, not a direct PDF link: {normalized_url} (content-type: {content_type})"
                    )
                    browser_pdf = await self._try_browser_agent_download(
                        str(response.url),
                        html_hint=response.text,
                        allow_manual_verification=allow_manual_verification,
                    )
                    if browser_pdf:
                        return browser_pdf
                    return None
                raise Exception(
                    f"Unexpected content type: {content_type} for URL: {normalized_url}"
                )
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error downloading PDF from {normalized_url}: {e}")
            return None
        except httpx.TimeoutException:
            logger.warning(f"Timeout downloading PDF from {normalized_url}")
            return None
        except Exception as e:
            logger.error(f"Error downloading PDF from {pdf_url}: {e}")
            return None

    async def _resolve_arxiv_pdf_url(self, target: str) -> Optional[str]:
        """Resolve an input URL/DOI/arXiv ID to a direct arXiv PDF URL, when possible."""
        candidate = str(target or "").strip()
        if not candidate:
            return None

        # Direct arXiv references (URL/identifier/DOI suffix)
        arxiv_id = self._extract_arxiv_id(candidate)
        if arxiv_id:
            return self.get_pdf_url_from_arxiv_id(arxiv_id)

        parsed = urlparse(candidate)
        host = (parsed.netloc or "").lower()
        if host.endswith("doi.org"):
            try:
                async with httpx.AsyncClient(
                    timeout=min(self.timeout, 20.0),
                    follow_redirects=True,
                    headers={
                        "User-Agent": "Mozilla/5.0 (compatible; ExegentBot/1.0; +https://exegent.ai)",
                    },
                ) as client:
                    response = await client.head(candidate)
                    if response.status_code >= 400 or response.status_code == 405:
                        response = await client.get(candidate)
                    final_url = str(response.url)
                    resolved_arxiv_id = self._extract_arxiv_id(final_url)
                    if resolved_arxiv_id:
                        return self.get_pdf_url_from_arxiv_id(resolved_arxiv_id)
            except Exception as exc:
                logger.debug("DOI resolution for arXiv fast-path failed (%s): %s", candidate, exc)

        return None

    async def _download_arxiv_pdf(self, pdf_url: str) -> Optional[bytes]:
        """Download PDF from arXiv directly without browser automation."""
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (compatible; ExegentBot/1.0; +https://exegent.ai)",
                "Accept": "application/pdf,*/*;q=0.8",
                "Referer": "https://arxiv.org/",
            }
            async with httpx.AsyncClient(
                timeout=min(self.timeout, 45.0),
                follow_redirects=True,
                headers=headers,
            ) as client:
                response = await client.get(pdf_url)
                response.raise_for_status()

                content_type = response.headers.get("content-type", "")
                if "pdf" not in content_type.lower() and "octet-stream" not in content_type.lower():
                    logger.warning(
                        "arXiv response is not a PDF (content-type=%s): %s",
                        content_type,
                        pdf_url,
                    )
                    return None

                return self._validate_pdf_size_and_return(
                    response.content,
                    response.headers.get("content-length"),
                    str(response.url),
                )
        except Exception as exc:
            logger.warning("Failed to download arXiv PDF from %s: %s", pdf_url, exc)
            return None

    @staticmethod
    def _extract_arxiv_id(value: str) -> Optional[str]:
        """Extract arXiv identifier from URL/DOI/raw ID forms."""
        text = str(value or "").strip()
        if not text:
            return None

        # DOI form: 10.48550/arXiv.1706.03762
        doi_match = re.search(r"10\.48550/arxiv\.([^/?#\s]+)", text, flags=re.IGNORECASE)
        if doi_match:
            doi_arxiv_id = doi_match.group(1).strip()
            if doi_arxiv_id.lower().endswith(".pdf"):
                doi_arxiv_id = doi_arxiv_id[:-4]
            return doi_arxiv_id

        # URL forms: /abs/<id>, /pdf/<id>(.pdf)
        url_match = re.search(
            r"arxiv\.org/(?:abs|pdf)/([^?#\s]+)",
            text,
            flags=re.IGNORECASE,
        )
        if url_match:
            raw = url_match.group(1).strip().rstrip("/")
            if raw.lower().endswith(".pdf"):
                raw = raw[:-4]
            return raw

        # Raw prefixed ID: arXiv:1706.03762
        prefixed_match = re.search(r"\barxiv:([^\s]+)", text, flags=re.IGNORECASE)
        if prefixed_match:
            prefixed_id = prefixed_match.group(1).strip()
            if prefixed_id.lower().endswith(".pdf"):
                prefixed_id = prefixed_id[:-4]
            return prefixed_id

        # Raw ID fallback (new and old style)
        raw = text.removeprefix("https://").removeprefix("http://").strip()
        raw = raw.removeprefix("arXiv:").removeprefix("arxiv:").strip()
        id_pattern = re.compile(r"^(?:\d{4}\.\d{4,5}(?:v\d+)?|[a-z\-]+(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?)$", re.IGNORECASE)
        if id_pattern.match(raw):
            return raw

        return None

    @staticmethod
    def _truncate_for_log(value: str, max_len: int = 2000) -> str:
        """Truncate long text for safe, readable logs."""
        text = (value or "").strip()
        if len(text) <= max_len:
            return text
        return f"{text[:max_len]}... [truncated, total={len(text)} chars]"

    def _guess_access_denied_reason(
        self,
        body_text: str,
        headers: Mapping[str, str],
    ) -> str:
        """Best-effort classification for common access denied causes."""
        lowered = (body_text or "").lower()
        server = headers.get("server", "").lower()
        cf_ray = headers.get("cf-ray", "")

        if "cloudflare" in lowered or "cf-challenge" in lowered or cf_ray:
            return "Cloudflare bot/challenge protection"
        if "captcha" in lowered or "recaptcha" in lowered:
            return "CAPTCHA or human verification required"
        if "paywall" in lowered or "purchase" in lowered or "subscribe" in lowered:
            return "Publisher paywall / subscription required"
        if "access denied" in lowered or "forbidden" in lowered:
            return "Generic access-control policy denied the request"
        if "akamai" in lowered or "akamai" in server:
            return "Akamai edge security policy denied the request"

        return "Unknown (inspect headers/body snippet)"

    def _log_httpx_access_denied(
        self,
        response: httpx.Response,
        attempted_url: str,
        source: str,
    ) -> None:
        """Log detailed diagnostics for access-denied HTTP responses."""
        header_keys = [
            "server",
            "cf-ray",
            "x-cache",
            "via",
            "set-cookie",
            "location",
            "retry-after",
            "www-authenticate",
            "content-type",
        ]
        headers_subset = {
            key: response.headers.get(key)
            for key in header_keys
            if response.headers.get(key) is not None
        }
        body_snippet = self._truncate_for_log(response.text)
        reason = self._guess_access_denied_reason(body_snippet, response.headers)

        logger.info(
            "Access denied (%s): attempted_url=%s final_url=%s status=%s reason=%s headers=%s body_snippet=%s",
            source,
            attempted_url,
            str(response.url),
            response.status_code,
            reason,
            headers_subset,
            body_snippet,
        )

    def _log_playwright_access_denied(
        self,
        attempted_url: str,
        final_url: str,
        status_code: int,
        headers: Mapping[str, str],
        body_text: str,
        source: str,
    ) -> None:
        """Log detailed diagnostics for Playwright access-denied responses."""
        header_keys = [
            "server",
            "cf-ray",
            "x-cache",
            "via",
            "set-cookie",
            "location",
            "retry-after",
            "www-authenticate",
            "content-type",
        ]
        headers_subset = {
            key: headers.get(key)
            for key in header_keys
            if headers.get(key) is not None
        }
        body_snippet = self._truncate_for_log(body_text)
        reason = self._guess_access_denied_reason(body_snippet, headers)

        logger.info(
            "Access denied (%s): attempted_url=%s final_url=%s status=%s reason=%s headers=%s body_snippet=%s",
            source,
            attempted_url,
            final_url,
            status_code,
            reason,
            headers_subset,
            body_snippet,
        )

    def _normalize_download_target(self, target: str) -> str:
        """Normalize raw DOI/URL-like targets into a valid fetch URL."""
        raw = str(target or "").strip()
        if not raw:
            return raw

        if raw.lower().startswith("doi:"):
            raw = raw[4:].strip()

        if raw.startswith("10.") and "/" in raw:
            return f"https://doi.org/{raw}"

        if raw.startswith("www."):
            return f"https://{raw}"

        parsed = urlparse(raw)
        if not parsed.scheme:
            return f"https://{raw}"
        return raw

    async def _try_browser_agent_download(
        self,
        url: str,
        html_hint: Optional[str] = None,
        allow_manual_verification: bool = False,
    ) -> Optional[bytes]:
        """
        Browser-agent fallback for sites that deny simple HTTP clients.

        Strategy:
        1) Retry URL with browser-like headers and referer
        2) If HTML is returned, parse likely PDF link and fetch that link
        """
        browser_headers = {
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,application/pdf,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://scholar.google.com/",
        }

        if not self._supports_playwright_subprocess():
            logger.debug(
                "Skipping browser-agent fallback for %s because current event loop does not support subprocess execution",
                url,
            )
            return None

        browser = None
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(
                    headless=True,
                    args=["--disable-blink-features=AutomationControlled"],
                )
                context = await browser.new_context(
                    user_agent=(
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/120.0.0.0 Safari/537.36"
                    ),
                    locale="en-US",
                    extra_http_headers=browser_headers,
                )
                page = await context.new_page()
                stealth_async_fn = cast(
                    Optional[Callable[[Any], Awaitable[Any]]],
                    getattr(playwright_stealth, "stealth_async", None),
                )
                if stealth_async_fn:
                    await stealth_async_fn(page)

                if html_hint:
                    candidate_pdf_url = self.html_pdf_resolver.extract_pdf_url(
                        html_hint,
                        base_url=url,
                    )
                    if candidate_pdf_url:
                        pdf = await self._fetch_pdf_with_playwright(
                            context, candidate_pdf_url
                        )
                        if pdf:
                            logger.info(
                                f"Downloaded PDF via browser-agent discovered link: {candidate_pdf_url}"
                            )
                            return pdf

                response = await page.goto(
                    url,
                    wait_until="domcontentloaded",
                    timeout=int(self.timeout * 1000),
                )

                if response and response.status in [401, 402, 403]:
                    denied_html = await page.content()
                    self._log_playwright_access_denied(
                        attempted_url=url,
                        final_url=page.url or url,
                        status_code=response.status,
                        headers=response.headers,
                        body_text=denied_html,
                        source="playwright-page-goto",
                    )

                    if (
                        allow_manual_verification
                        and self._looks_like_cloudflare_challenge(denied_html)
                    ):
                        logger.info(
                            "Attempting interactive manual verification flow for %s",
                            url,
                        )
                        manual_pdf = await self._try_manual_browser_verification(url)
                        if manual_pdf:
                            return manual_pdf

                    return None

                current_url = page.url or url
                content_type = ""
                if response:
                    content_type = response.headers.get("content-type", "")
                    if (
                        "pdf" in content_type.lower()
                        or "application/octet-stream" in content_type.lower()
                        or "binary/octet-stream" in content_type.lower()
                    ):
                        pdf = await self._fetch_pdf_with_playwright(context, current_url)
                        if pdf:
                            logger.info(
                                f"Downloaded PDF via browser-agent URL: {current_url}"
                            )
                        return pdf

                html = await page.content()
                if self._looks_like_cloudflare_challenge(html):
                    if allow_manual_verification:
                        logger.info(
                            "Detected challenge page; attempting interactive manual verification for %s",
                            current_url,
                        )
                        manual_pdf = await self._try_manual_browser_verification(
                            current_url
                        )
                        if manual_pdf:
                            return manual_pdf

                    for alt_url in self._derive_landing_page_urls(current_url):
                        pdf = await self._try_resolve_pdf_from_landing(context, alt_url)
                        if pdf:
                            return pdf

                candidate_pdf_url = self.html_pdf_resolver.extract_pdf_url(
                    html,
                    base_url=current_url,
                )
                if candidate_pdf_url:
                    pdf = await self._fetch_pdf_with_playwright(context, candidate_pdf_url)
                    if pdf:
                        logger.info(
                            f"Downloaded PDF via browser-agent extracted link: {candidate_pdf_url}"
                        )
                        return pdf

        except PlaywrightTimeoutError:
            logger.debug(f"Playwright fallback timed out for {url}")
        except Exception as e:
            logger.debug(f"Browser-agent fallback failed for {url}: {e}")
        finally:
            if browser:
                await browser.close()

        return None

    async def _try_manual_browser_verification(self, url: str) -> Optional[bytes]:
        """
        Open a headed browser so a human can solve challenge pages locally.

        This flow is intended for local/dev usage and may block up to 2 minutes.
        """
        browser = None
        try:
            if not self._supports_playwright_subprocess():
                logger.debug(
                    "Skipping interactive browser verification for %s because current event loop does not support subprocess execution",
                    url,
                )
                return None

            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=False)
                context = await browser.new_context(
                    user_agent=(
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/120.0.0.0 Safari/537.36"
                    ),
                    locale="en-US",
                )
                page = await context.new_page()

                logger.info(
                    "Interactive verification: complete any Cloudflare/captcha checks in the opened browser window. "
                    "Waiting up to 120 seconds..."
                )

                await page.goto(
                    url,
                    wait_until="domcontentloaded",
                    timeout=int(self.timeout * 1000),
                )

                max_wait_seconds = 120
                for _ in range(max_wait_seconds // 2):
                    html = await page.content()
                    if not self._looks_like_cloudflare_challenge(html):
                        current_url = page.url or url

                        candidate_pdf_url = self.html_pdf_resolver.extract_pdf_url(
                            html,
                            base_url=current_url,
                        )

                        if candidate_pdf_url:
                            pdf = await self._fetch_pdf_with_playwright(
                                context,
                                candidate_pdf_url,
                            )
                            if pdf:
                                logger.info(
                                    "Manual verification succeeded; downloaded via extracted PDF URL: %s",
                                    candidate_pdf_url,
                                )
                                return pdf

                        # If current URL itself appears PDF-like, try downloading directly.
                        pdf = await self._fetch_pdf_with_playwright(context, current_url)
                        if pdf:
                            logger.info(
                                "Manual verification succeeded; downloaded from current URL: %s",
                                current_url,
                            )
                            return pdf

                        break

                    await asyncio.sleep(2)

                logger.info(
                    "Interactive verification did not produce a downloadable PDF within timeout for %s",
                    url,
                )
                return None
        except Exception as e:
            logger.debug(f"Manual verification flow failed for {url}: {e}")
            return None
        finally:
            if browser:
                await browser.close()

    async def _try_resolve_pdf_from_landing(
        self,
        context: BrowserContext,
        landing_url: str,
    ) -> Optional[bytes]:
        """Fetch landing page, extract PDF URL, then download PDF with same session."""
        try:
            response = await context.request.get(
                landing_url,
                fail_on_status_code=False,
                timeout=int(self.timeout * 1000),
            )
            if response.status >= 400:
                return None

            content_type = response.headers.get("content-type", "")
            if "html" not in content_type.lower() and "text" not in content_type.lower():
                return None

            html = await response.text()
            if self._looks_like_cloudflare_challenge(html):
                return None

            candidate_pdf_url = self.html_pdf_resolver.extract_pdf_url(
                html,
                base_url=landing_url,
            )
            if not candidate_pdf_url:
                return None

            pdf = await self._fetch_pdf_with_playwright(context, candidate_pdf_url)
            if pdf:
                logger.info(
                    f"Downloaded PDF via landing-page resolver: {candidate_pdf_url}"
                )
            return pdf
        except Exception:
            return None

    @staticmethod
    def _looks_like_cloudflare_challenge(html: str) -> bool:
        text = (html or "").lower()
        return (
            "cloudflare" in text
            and (
                "attention required" in text
                or "checking your browser" in text
                or "cf-challenge" in text
            )
        )

    def _derive_landing_page_urls(self, url: str) -> list[str]:
        """Generate likely HTML landing page URLs for resolver fallback."""
        urls: list[str] = []
        normalized = self._normalize_download_target(url)

        if "/article/" in normalized and normalized.rstrip("/").endswith("/pdf"):
            urls.append(normalized.rsplit("/pdf", 1)[0])

        pmc_match = re.search(r"/pmc/articles/(PMC\d+)", normalized, flags=re.IGNORECASE)
        if pmc_match:
            pmc_id = pmc_match.group(1)
            urls.extend(
                [
                    f"https://pmc.ncbi.nlm.nih.gov/articles/{pmc_id}/",
                    f"https://pmc.ncbi.nlm.nih.gov/articles/{pmc_id}/pdf/",
                    f"https://pmc.ncbi.nlm.nih.gov/articles/{pmc_id}/pdf/?download=1",
                    f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmc_id}/",
                    f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmc_id}/pdf/",
                    f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmc_id}/pdf/?download=1",
                ]
            )

        science_doi = self._extract_doi_from_url(normalized)
        if science_doi:
            urls.extend(
                [
                    f"https://www.science.org/doi/{science_doi}",
                    f"https://www.science.org/doi/full/{science_doi}",
                    f"https://doi.org/{science_doi}",
                ]
            )

        deduped: list[str] = []
        seen = set()
        for candidate in urls:
            if candidate not in seen:
                seen.add(candidate)
                deduped.append(candidate)
        return deduped


    @staticmethod
    def _extract_doi_from_url(url: str) -> Optional[str]:
        """Extract DOI from DOI-like URLs (publisher/doi.org forms)."""
        # Matches /doi/<doi> or /doi/pdf/<doi>
        match = re.search(r"/doi/(?:pdf/|full/)?(10\.\d{4,9}/[^?#]+)", url, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip().rstrip("/")

        match = re.search(r"doi\.org/(10\.\d{4,9}/[^?#]+)", url, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip().rstrip("/")

        return None

    async def _fetch_pdf_with_playwright(
        self,
        context: BrowserContext,
        url: str,
    ) -> Optional[bytes]:
        """Fetch a URL via Playwright request context and return bytes for valid PDFs."""
        try:
            response = await context.request.get(
                url,
                fail_on_status_code=False,
                timeout=int(self.timeout * 1000),
            )
            if response.status in [401, 402, 403]:
                try:
                    denied_text = await response.text()
                except Exception:
                    denied_text = ""
                self._log_playwright_access_denied(
                    attempted_url=url,
                    final_url=response.url,
                    status_code=response.status,
                    headers=response.headers,
                    body_text=denied_text,
                    source="playwright-request-context",
                )
                return None

            if response.status >= 400:
                raise Exception(f"Failed to fetch PDF URL {url} - Status {response.status}: {response.status_text} - {response}")

            content_type = response.headers.get("content-type", "")
            if "pdf" not in content_type.lower() and "octet-stream" not in content_type.lower():
                return None

            body = await response.body()
            return self._validate_pdf_size_and_return(
                body,
                response.headers.get("content-length"),
                url,
            )
        except Exception as e:
            logger.error(f"Error fetching PDF URL {url}: {e}")
            raise

    def _validate_pdf_size_and_return(
        self,
        content: bytes,
        content_length_header: Optional[str],
        source_url: str,
    ) -> Optional[bytes]:
        """Validate PDF size limits before returning bytes."""
        content_length = content_length_header
        if content_length and int(content_length) > 50 * 1024 * 1024:
            logger.warning(f"PDF too large ({content_length} bytes): {source_url}")
            return None

        logger.info(f"Downloaded PDF from {source_url} ({len(content)} bytes)")
        return content

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
                    or "binary/octet-stream" in content_type.lower()
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

    def get_pdf_url_from_arxiv_id(self, arxiv_id: str) -> str:
        """
        Construct PDF URL from arXiv ID

        Args:
            arxiv_id: arXiv ID (e.g., 2301.12345)

        Returns:
            PDF URL
        """
        return f"https://arxiv.org/pdf/{arxiv_id}.pdf"
