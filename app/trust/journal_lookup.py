"""
Journal Lookup Service

Provides functions to look up journal metadata from the SJR database.
Used for venue prestige scoring, academic legitimacy validation, and ranking.
"""

from typing import Optional, List, Dict, Any, TYPE_CHECKING
from sqlalchemy import select, or_, func
from sqlalchemy.ext.asyncio import AsyncSession
import re

from app.models.journals import DBJournal

if TYPE_CHECKING:
    from app.models.papers import DBPaper


class JournalLookupService:
    """Service for looking up journal metadata from SJR database."""

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

        # Remove hyphens from ISSN for matching
        issn_clean = issn.replace("-", "")
        
        # Try both formats (with and without hyphen)
        issn_variants = [issn, issn_clean]
        if "-" not in issn and len(issn) == 8:
            # Add hyphenated version if input was without hyphen
            issn_variants.append(f"{issn[:4]}-{issn[4:]}")

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

    async def search_by_title(
        self, title: str, year: Optional[int] = None, limit: int = 10
    ) -> List[DBJournal]:
        """
        Search journals by partial title match.

        Args:
            title: Partial title to search for
            year: Specific year to search, or latest if None
            limit: Maximum results to return

        Returns:
            List of matching journals
        """
        if not title:
            return []

        normalized = self.normalize_title(title)

        query = select(DBJournal).where(
            DBJournal.title_normalized.ilike(f"%{normalized}%")
        )

        if year:
            query = query.where(DBJournal.data_year == year)
        else:
            # Prefer latest year
            query = query.order_by(DBJournal.data_year.desc())

        query = query.limit(limit)

        result = await self.db.execute(query)
        return list(result.scalars().all())

    async def lookup_by_venue(
        self, venue: str, issn: Optional[str] = None, year: Optional[int] = None
    ) -> Optional[DBJournal]:
        """
        Look up journal by venue name (from paper metadata).
        Tries ISSN first, then title matching.

        Args:
            venue: Venue name from paper
            issn: Optional ISSN from paper
            year: Publication year for time-appropriate data

        Returns:
            Journal record or None if not found
        """
        # Try ISSN first if available
        if issn:
            journal = await self.lookup_by_issn(issn, year)
            if journal:
                return journal

        # Try exact title match
        if venue:
            journal = await self.lookup_by_title(venue, year, fuzzy=True)
            if journal:
                return journal

        # Try partial match if exact fails
        if venue:
            results = await self.search_by_title(venue, year, limit=1)
            if results:
                return results[0]

        return None

    async def get_venue_prestige_score(
        self, venue: str, issn: Optional[str] = None, year: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get venue prestige scoring information.

        Args:
            venue: Venue name
            issn: Optional ISSN
            year: Publication year

        Returns:
            Dictionary with prestige metrics:
            - sjr_score: SJR indicator value
            - quartile: Q1-Q4
            - h_index: Journal h-index
            - impact_factor: Citations per doc (2 years)
            - is_open_access: Open access status
            - rank: Global rank
            - percentile: Percentile rank (0-100)
        """
        journal = await self.lookup_by_venue(venue, issn, year)

        if not journal:
            return {
                "found": False,
                "sjr_score": None,
                "quartile": None,
                "h_index": None,
                "impact_factor": None,
                "is_open_access": False,
                "rank": None,
                "percentile": None,
            }

        # Calculate percentile from rank
        # Get total journals in that year
        total_count_query = select(func.count(DBJournal.id)).where(
            DBJournal.data_year == journal.data_year
        )
        total_result = await self.db.execute(total_count_query)
        total_count = total_result.scalar() or 1

        percentile = None
        if journal.rank and total_count:
            percentile = ((total_count - journal.rank + 1) / total_count) * 100

        return {
            "found": True,
            "source_id": journal.source_id,
            "title": journal.title,
            "sjr_score": journal.sjr_score,
            "quartile": journal.sjr_best_quartile,
            "h_index": journal.h_index,
            "impact_factor": journal.cites_per_doc_2years,
            "is_open_access": journal.is_open_access,
            "rank": journal.rank,
            "percentile": percentile,
            "publisher": journal.publisher,
            "country": journal.country,
            "categories": journal.categories,
            "areas": journal.areas,
        }

    async def is_prestigious_venue(
        self,
        venue: str,
        issn: Optional[str] = None,
        year: Optional[int] = None,
        threshold: str = "Q1",
    ) -> bool:
        """
        Check if venue meets prestige threshold.

        Args:
            venue: Venue name
            issn: Optional ISSN
            year: Publication year
            threshold: Minimum quartile ('Q1', 'Q2', 'Q3', 'Q4')

        Returns:
            True if venue meets threshold
        """
        journal = await self.lookup_by_venue(venue, issn, year)

        if not journal or not journal.sjr_best_quartile:
            return False

        quartile_order = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}
        journal_quartile = quartile_order.get(journal.sjr_best_quartile, 5)
        threshold_quartile = quartile_order.get(threshold, 1)

        return journal_quartile <= threshold_quartile

    async def get_top_journals(
        self,
        year: Optional[int] = None,
        category: Optional[str] = None,
        limit: int = 100,
    ) -> List[DBJournal]:
        """
        Get top journals by SJR score.

        Args:
            year: Specific year, or latest if None
            category: Filter by research category
            limit: Maximum results

        Returns:
            List of top journals
        """
        query = select(DBJournal)

        if year:
            query = query.where(DBJournal.data_year == year)
        else:
            # Get latest year available
            latest_year_query = select(func.max(DBJournal.data_year))
            result = await self.db.execute(latest_year_query)
            latest_year = result.scalar()
            if latest_year:
                query = query.where(DBJournal.data_year == latest_year)

        if category:
            query = query.where(
                DBJournal.categories.any(
                    DBJournal.categories.property.mapper.class_.name == category
                )
            )

        query = query.order_by(DBJournal.sjr_score.desc()).limit(limit)

        result = await self.db.execute(query)
        return list(result.scalars().all())

    async def get_journal_stats(self, year: Optional[int] = None) -> Dict[str, Any]:
        """
        Get aggregate statistics about journals in the database.

        Args:
            year: Specific year, or all years if None

        Returns:
            Dictionary with statistics
        """
        base_query = select(DBJournal)
        if year:
            base_query = base_query.where(DBJournal.data_year == year)

        # Total journals
        count_query = select(func.count(DBJournal.id))
        if year:
            count_query = count_query.where(DBJournal.data_year == year)
        result = await self.db.execute(count_query)
        total = result.scalar() or 0

        # Quartile distribution
        quartile_counts = {}
        for q in ["Q1", "Q2", "Q3", "Q4"]:
            q_query = select(func.count(DBJournal.id)).where(
                DBJournal.sjr_best_quartile == q
            )
            if year:
                q_query = q_query.where(DBJournal.data_year == year)
            result = await self.db.execute(q_query)
            quartile_counts[q] = result.scalar() or 0

        # Open access count
        oa_query = select(func.count(DBJournal.id)).where(
            DBJournal.is_open_access == True
        )
        if year:
            oa_query = oa_query.where(DBJournal.data_year == year)
        result = await self.db.execute(oa_query)
        open_access_count = result.scalar() or 0

        return {
            "total_journals": total,
            "quartile_distribution": quartile_counts,
            "open_access_count": open_access_count,
            "open_access_percentage": (
                (open_access_count / total * 100) if total > 0 else 0
            ),
            "year": year or "all",
        }

    async def enrich_paper_with_sjr(self, paper_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enrich paper metadata with SJR journal data by matching ISSN or venue.

        Args:
            paper_data: Paper metadata dict (from DBPaper or Paper schema)

        Returns:
            Paper data enriched with 'sjr_data' field containing journal metrics
        """
        # Extract ISSN and venue
        issn = None
        venue = paper_data.get("venue")
        year = paper_data.get("year")
        
        # Try to get ISSN from primary_location (OpenAlex format) if available
        primary_location = paper_data.get("primary_location")
        if primary_location and isinstance(primary_location, dict):
            source = primary_location.get("source", {})
            if isinstance(source, dict):
                # Try issn_l first (linking ISSN)
                issn = source.get("issn_l")
                # Fallback to first ISSN in list
                if not issn:
                    issn_list = source.get("issn")
                    if issn_list and isinstance(issn_list, list) and len(issn_list) > 0:
                        issn = issn_list[0]
        
        # If no ISSN or venue, return original data
        if not issn and not venue:
            return paper_data
        
        # Ensure venue is a string
        if venue and not isinstance(venue, str):
            venue = str(venue)
        
        # Look up journal by ISSN or venue
        journal = await self.lookup_by_venue(venue or "", issn, year)
        
        if journal:
            # Calculate percentile
            total_count_query = select(func.count(DBJournal.id)).where(
                DBJournal.data_year == journal.data_year
            )
            total_result = await self.db.execute(total_count_query)
            total_count = total_result.scalar() or 1
            
            percentile = None
            if journal.rank and total_count:
                percentile = ((total_count - journal.rank + 1) / total_count) * 100
            
            # Add SJR data to paper
            paper_data["sjr_data"] = {
                "journal_title": journal.title,
                "issn": journal.issn,
                "sjr_score": journal.sjr_score,
                "quartile": journal.sjr_best_quartile,
                "h_index": journal.h_index,
                "impact_factor": journal.cites_per_doc_2years,
                "rank": journal.rank,
                "percentile": round(percentile, 2) if percentile else None,
                "is_open_access": journal.is_open_access,
                "publisher": journal.publisher,
                "country": journal.country,
                "data_year": journal.data_year,
            }
        
        return paper_data

    async def update_paper_with_journal(
        self, paper: "DBPaper", issn: Optional[str] = None, 
        issn_l: Optional[str] = None
    ) -> Optional[DBJournal]:
        """
        Update a paper's ISSN fields and journal relationship.
        
        Args:
            paper: DBPaper instance to update
            issn: Primary ISSN (if available)
            issn_l: Linking ISSN (if available from OpenAlex)
            
        Returns:
            Matched journal or None if not found
        """
        # Update ISSN fields if provided
        if issn:
            paper.issn = issn
        if issn_l:
            paper.issn_l = issn_l
            
        # Try to match journal using available ISSNs
        search_issn = issn_l or issn
        if not search_issn and paper.venue:
            # Fallback to venue-only matching
            journal = await self.lookup_by_venue(
                paper.venue, 
                None, 
                paper.publication_date.year if paper.publication_date else None
            )
        elif search_issn:
            journal = await self.lookup_by_issn(
                search_issn,
                paper.publication_date.year if paper.publication_date else None
            )
        else:
            return None
            
        # Set journal relationship if found
        if journal:
            paper.journal_id = journal.id
            
        return journal
