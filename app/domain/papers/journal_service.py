"""
Journal Service - Moved from app.trust.journal_lookup

Provides functions to look up journal metadata from the SJR database.
Used for venue prestige scoring, academic legitimacy validation, and ranking.
Part of the papers module as it's directly related to paper enrichment.
"""

from typing import Optional, List, Dict, Any, TYPE_CHECKING
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
import re

from app.models.journals import DBJournal
from app.extensions.logger import create_logger

if TYPE_CHECKING:
    from app.models.papers import DBPaper

logger = create_logger(__name__)


class JournalService:
    """Service for looking up and enriching journal metadata from SJR database."""

    def __init__(self, db: AsyncSession):
        self.db = db

    @staticmethod
    def normalize_title(title: str) -> str:
        """Normalize title for fuzzy matching."""
        if not title:
            return ""

        normalized = title.lower()
        normalized = re.sub(r"[^\w\s-]", "", normalized)
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized.strip()

    @staticmethod
    def _build_issn_variants(issn: str) -> List[str]:
        """Build normalized ISSN variants for matching."""
        if not issn:
            return []

        value = str(issn).strip()
        if not value:
            return []

        clean = value.replace("-", "")
        variants: List[str] = [value, clean]

        if "-" not in value and len(clean) == 8:
            variants.append(f"{clean[:4]}-{clean[4:]}")

        # Preserve order while removing duplicates
        deduped: List[str] = []
        seen: set[str] = set()
        for item in variants:
            if item and item not in seen:
                seen.add(item)
                deduped.append(item)
        return deduped

    async def lookup_by_issn(
        self, issn: str, year: Optional[int] = None
    ) -> Optional[DBJournal]:
        """
        Look up journal by ISSN using array containment.

        Args:
            issn: ISSN to search for (with or without hyphen)
            year: Specific year to search, or latest if None

        Returns:
            Journal record or None if not found
        """
        if not issn:
            return None

        issn_variants = self._build_issn_variants(issn)

        # Use array containment operator for efficient GIN index lookup
        query = select(DBJournal).where(
            DBJournal.issn.overlap(issn_variants)  # PostgreSQL array overlap operator
        )

        if year:
            query = query.where(DBJournal.data_year == year)
        else:
            # Get latest year
            query = query.order_by(DBJournal.data_year.desc())

        result = await self.db.execute(query)
        return result.scalars().first()

    async def lookup_by_title(
        self, title: str, year: Optional[int] = None, fuzzy: bool = True
    ) -> Optional[DBJournal]:
        """
        Look up journal by title.

        Args:
            title: Journal title to search for
            year: Specific year to search, or latest if None
            fuzzy: Use fuzzy matching (normalized titles)

        Returns:
            Journal record or None if not found
        """
        if not title:
            return None

        if fuzzy:
            normalized = self.normalize_title(title)
            query = select(DBJournal).where(DBJournal.title_normalized == normalized)
        else:
            query = select(DBJournal).where(DBJournal.title.ilike(title))

        if year:
            query = query.where(DBJournal.data_year == year)
        else:
            query = query.order_by(DBJournal.data_year.desc())

        result = await self.db.execute(query)
        return result.scalars().first()

    async def lookup_by_venue(
        self, venue: str, issn: Optional[str] = None, issn_l: Optional[str] = None, year: Optional[int] = None
    ) -> Optional[DBJournal]:
        """
        Look up journal by venue name (from paper metadata).
        Tries ISSN_L first (most reliable), then ISSN, then title matching.

        Args:
            venue: Venue name from paper
            issn: Optional ISSN from paper
            issn_l: Optional linking ISSN (OpenAlex standard)
            year: Publication year for time-appropriate data

        Returns:
            Journal record or None if not found
        """
        # Try ISSN_L first (most reliable identifier)
        if issn_l:
            journal = await self.lookup_by_issn(issn_l, year)
            if journal:
                logger.debug(f"Matched journal by ISSN-L: {issn_l} -> {journal.title}")
                return journal

        # Try regular ISSN if available
        if issn:
            journal = await self.lookup_by_issn(issn, year)
            if journal:
                logger.debug(f"Matched journal by ISSN: {issn} -> {journal.title}")
                return journal

        # Try exact title match
        if venue:
            journal = await self.lookup_by_title(venue, year, fuzzy=True)
            if journal:
                logger.debug(f"Matched journal by venue: {venue} -> {journal.title}")
                return journal

        logger.debug(f"No journal match found for venue: {venue}, ISSN: {issn}, ISSN-L: {issn_l}")
        return None

    async def link_journal_to_paper(
        self, 
        paper: "DBPaper", 
        venue: Optional[str] = None,
        issn: Optional[str] = None, 
        issn_l: Optional[str] = None
    ) -> Optional[DBJournal]:
        """
        Enrich a paper with journal data by looking up and linking to journal table.
        
        This method:
        1. Looks up journal by ISSN-L, ISSN, or venue
        2. Links the journal to the paper (sets journal_id)
        3. Returns the matched journal
        
        Args:
            paper: DBPaper instance to enrich
            venue: Venue name (uses paper.venue if not provided)
            issn: Primary ISSN (uses paper.issn if not provided)
            issn_l: Linking ISSN (uses paper.issn_l if not provided)
            
        Returns:
            Matched DBJournal or None if not found
        """
        # Use provided values or fall back to paper attributes
        venue = venue or paper.venue
        issn_l = issn_l or paper.issn_l
        issn = issn or (paper.issn[0] if paper.issn and len(paper.issn) > 0 else None)
        
        # Get publication year for time-appropriate journal data
        year = paper.publication_date.year if paper.publication_date else None
        
        # Look up journal
        journal = await self.lookup_by_venue(venue, issn, issn_l, year)
        
        # Link journal if found
        if journal:
            paper.journal_id = journal.id
            logger.info(f"Enriched paper {paper.paper_id} with journal: {journal.title} (SJR: {journal.sjr_score})")
        else:
            logger.debug(f"No journal match for paper {paper.paper_id} (venue: {venue})")
            
        return journal
    
    async def batch_lookup_journals(
        self,
        papers_data: List[Dict[str, Any]]
    ) -> Dict[str, Optional[DBJournal]]:
        """
        Batch lookup journals for multiple papers efficiently.
        Uses venue, ISSN, and ISSN-L to find journal matches.
        
        Args:
            papers_data: List of dicts with keys: paper_id, venue, issn, issn_l, year
            Example: [
                {"paper_id": "123", "venue": "Nature", "issn": "0028-0836", "issn_l": "0028-0836", "year": 2023},
                ...
            ]
        
        Returns:
            Dict mapping paper_id -> DBJournal (or None if not found)
        """
        if not papers_data:
            return {}
        
        results = {}
        
        # Group papers by lookup strategy for efficient queries
        issn_l_lookups = []
        issn_lookups = []
        venue_lookups = []
        
        for paper_data in papers_data:
            paper_id = paper_data.get("paper_id")
            if not paper_id:
                continue
            
            issn_l = paper_data.get("issn_l")
            issn = paper_data.get("issn")
            venue = paper_data.get("venue")
            year = paper_data.get("year")
            
            # Priority: ISSN-L > ISSN > Venue
            if issn_l:
                issn_l_lookups.append((paper_id, issn_l, year))
            elif issn:
                issn_lookups.append((paper_id, issn, year))
            elif venue:
                venue_lookups.append((paper_id, venue, year))
            else:
                results[paper_id] = None
        
        # Batch lookup by ISSN-L (most reliable)
        if issn_l_lookups:
            issn_l_values: List[str] = []
            for _, issn_l, _ in issn_l_lookups:
                issn_l_values.extend(self._build_issn_variants(issn_l))

            # IMPORTANT: journals.issn is VARCHAR[] so use overlap, not IN (= scalar)
            query = (
                select(DBJournal)
                .where(DBJournal.issn.overlap(issn_l_values))
                .order_by(DBJournal.data_year.desc())
            )
            result = await self.db.execute(query)
            journals = result.scalars().all()
            
            # Map each ISSN variant to best (newest) journal.
            journal_map: Dict[str, DBJournal] = {}
            for journal in journals:
                for journal_issn in (journal.issn or []):
                    for variant in self._build_issn_variants(journal_issn):
                        if variant not in journal_map:
                            journal_map[variant] = journal
            
            for paper_id, issn_l, year in issn_l_lookups:
                journal = None
                for variant in self._build_issn_variants(issn_l):
                    journal = journal_map.get(variant)
                    if journal:
                        break
                results[paper_id] = journal
                if journal:
                    logger.debug(f"Batch matched {paper_id} by ISSN-L: {journal.title}")
        
        # Batch lookup by ISSN (array containment - need individual queries)
        for paper_id, issn, year in issn_lookups:
            journal = await self.lookup_by_issn(issn, year)
            results[paper_id] = journal
            if journal:
                logger.debug(f"Batch matched {paper_id} by ISSN: {journal.title}")
        
        # Batch lookup by venue (fuzzy matching - need individual queries)
        for paper_id, venue, year in venue_lookups:
            journal = await self.lookup_by_title(venue, year, fuzzy=True)
            results[paper_id] = journal
            if journal:
                logger.debug(f"Batch matched {paper_id} by venue: {journal.title}")
        
        found_count = sum(1 for j in results.values() if j is not None)
        logger.info(f"Batch journal lookup: {found_count}/{len(papers_data)} matched")
        
        return results
