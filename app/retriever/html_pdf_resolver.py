"""HTML PDF link resolver utilities for browser-agent fallback flows."""

from __future__ import annotations

import re
from typing import Optional
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from app.extensions.logger import create_logger

logger = create_logger(__name__)


class HtmlPdfResolver:
    """Resolve likely PDF URLs from publisher landing pages."""

    # Common publisher meta/link hints for full-text PDF URLs.
    META_KEYS = (
        "citation_pdf_url",
        "dc.identifier",
        "dc.identifier.uri",
        "og:url",
        "og:see_also",
    )

    PDF_PATH_HINT_PATTERN = re.compile(r"(\.pdf(?:$|\?)|/pdf/?(?:$|\?))", flags=re.IGNORECASE)

    @staticmethod
    def _attr_text(value: object) -> str:
        """Normalize BeautifulSoup attribute values to plain string."""
        if value is None:
            return ""
        if isinstance(value, list):
            return " ".join(str(v) for v in value)
        return str(value)

    def extract_pdf_url(self, html: str, base_url: str) -> Optional[str]:
        """
        Extract a likely PDF URL from HTML using meta tags and element hints.

        Resolution order:
        1) meta tags (`citation_pdf_url`, `og:url`, `dc.identifier*`)
        2) `<a href="...pdf">` anchors (including label hints)
        3) `<link href="...pdf">` elements
        4) script text fallback regex for embedded PDF URLs
        """
        if not html:
            return None

        soup = BeautifulSoup(html, "html.parser")

        for url in self._from_meta_tags(soup, base_url):
            return url

        for url in self._from_anchor_hints(soup, base_url):
            return url

        for url in self._from_link_tags(soup, base_url):
            return url

        for url in self._from_script_regex(html, base_url):
            return url

        return None

    def _from_meta_tags(self, soup: BeautifulSoup, base_url: str):
        for meta in soup.find_all("meta"):
            key = (
                self._attr_text(meta.get("name"))
                or self._attr_text(meta.get("property"))
                or self._attr_text(meta.get("itemprop"))
            ).strip().lower()
            if key not in self.META_KEYS:
                continue

            content = self._attr_text(meta.get("content")).strip()
            if not content:
                continue

            # Prefer explicit PDF-looking values.
            if self.PDF_PATH_HINT_PATTERN.search(content):
                yield urljoin(base_url, content)
                continue

            # Some keys may hold landing URLs; still accept if they appear PDF-related.
            if "/pdf" in content.lower():
                yield urljoin(base_url, content)

    def _from_anchor_hints(self, soup: BeautifulSoup, base_url: str):
        for anchor in soup.find_all("a"):
            href = self._attr_text(anchor.get("href")).strip()
            if not href:
                continue

            text = (anchor.get_text(" ", strip=True) or "").lower()
            title = self._attr_text(anchor.get("title")).lower()
            rel = self._attr_text(anchor.get("rel")).lower()
            aria = self._attr_text(anchor.get("aria-label")).lower()
            joined_hint = " ".join([text, title, rel, aria])

            if self.PDF_PATH_HINT_PATTERN.search(href):
                yield urljoin(base_url, href)
                continue

            if "pdf" in joined_hint and href:
                yield urljoin(base_url, href)

    def _from_link_tags(self, soup: BeautifulSoup, base_url: str):
        for link in soup.find_all("link"):
            href = self._attr_text(link.get("href")).strip()
            if not href:
                continue

            type_attr = self._attr_text(link.get("type")).lower()
            title = self._attr_text(link.get("title")).lower()
            rel = self._attr_text(link.get("rel")).lower()

            if "pdf" in type_attr or "pdf" in title or "pdf" in rel:
                yield urljoin(base_url, href)
                continue

            if self.PDF_PATH_HINT_PATTERN.search(href):
                yield urljoin(base_url, href)

    def _from_script_regex(self, html: str, base_url: str):
        # Last-resort extraction for JS-embedded absolute or relative PDF URLs.
        pattern = re.compile(r"([\"'])([^\"']+\.pdf(?:\?[^\"']*)?)\1", re.IGNORECASE)
        for _quote, raw_url in pattern.findall(html):
            if raw_url:
                yield urljoin(base_url, raw_url)
