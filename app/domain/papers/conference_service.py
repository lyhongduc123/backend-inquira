"""
Conference Service - Business logic for conference operations.
Provides conference lookup and paper enrichment with conference data.
"""

from typing import Optional, List, Dict, Any, TYPE_CHECKING
from sqlalchemy import select, or_
from sqlalchemy.ext.asyncio import AsyncSession

from app.extensions.logger import create_logger
from app.models.conferences import DBConference

if TYPE_CHECKING:
    from app.models.papers import DBPaper

logger = create_logger(__name__)


class ConferenceService:
    """
    Service for conference operations.
    
    Provides:
    - Conference lookup by acronym and title
    - Paper enrichment with conference data
    """
    
    def __init__(self, db_session: AsyncSession):
        self.db_session = db_session
    
    async def lookup_by_venue(
        self,
        venue: Optional[str] = None,
        year: Optional[int] = None
    ) -> Optional[DBConference]:
        """
        Look up conference by venue name or acronym.
        
        Matching strategy:
        1. Exact acronym match (case-insensitive)
        2. Partial title match (contains, case-insensitive)
        3. Prioritize primary conferences and higher ranks
        
        Args:
            venue: Conference venue name or acronym
            year: Publication year (currently not used for filtering)
            
        Returns:
            Matched DBConference or None if not found
        """
        if not venue:
            return None
        
        venue_normalized = venue.strip().upper()
        
        # Build query with multiple matching strategies
        stmt = select(DBConference).where(
            or_(
                # Exact acronym match
                DBConference.acronym.ilike(venue_normalized),
                # Partial title match
                DBConference.title.ilike(f"%{venue}%")
            )
        ).order_by(
            # Prioritize primary conferences
            DBConference.is_primary.desc(),
            # Prioritize higher ranks (A* > A > B > C)
            DBConference.rank.asc()
        ).limit(1)
        
        result = await self.db_session.execute(stmt)
        conference = result.scalar_one_or_none()
        
        if conference:
            logger.debug(
                f"Matched venue '{venue}' to conference: {conference.title} "
                f"({conference.acronym}, rank: {conference.rank})"
            )
        
        return conference
    
    async def enrich_paper_with_conference(
        self, 
        paper: "DBPaper", 
        venue: Optional[str] = None
    ) -> Optional[DBConference]:
        """
        Enrich a paper with conference data by looking up and linking to conference table.
        
        This method:
        1. Looks up conference by venue (acronym or title)
        2. Links the conference to the paper (sets conference_id)
        3. Returns the matched conference
        
        Args:
            paper: DBPaper instance to enrich
            venue: Venue name (uses paper.venue if not provided)
            
        Returns:
            Matched DBConference or None if not found
        """
        # Use provided value or fall back to paper attribute
        venue = venue or paper.venue
        
        # Get publication year for potential time-appropriate filtering
        year = paper.publication_date.year if paper.publication_date else None
        
        # Look up conference
        conference = await self.lookup_by_venue(venue, year)
        
        # Link conference if found
        if conference:
            paper.conference_id = conference.id
            logger.info(
                f"Enriched paper {paper.paper_id} with conference: "
                f"{conference.title} ({conference.acronym}, rank: {conference.rank})"
            )
        else:
            logger.debug(f"No conference match for paper {paper.paper_id} (venue: {venue})")
            
        return conference
    
    async def batch_lookup_conferences(
        self,
        venues: List[str]
    ) -> Dict[str, Optional[DBConference]]:
        """
        Batch lookup conferences for multiple venues efficiently.
        
        Args:
            venues: List of venue names/acronyms
            
        Returns:
            Dict mapping venue to matched DBConference (or None)
        """
        if not venues:
            return {}
        
        # Normalize venues for matching
        venue_map: Dict[str, str] = {}  # normalized -> original
        for venue in venues:
            if venue:
                normalized = venue.strip().upper()
                venue_map[normalized] = venue
        
        if not venue_map:
            return {}
        
        # Query all potential matches
        stmt = select(DBConference).where(
            or_(
                DBConference.acronym.in_(list(venue_map.keys())),
                *[DBConference.title.ilike(f"%{v}%") for v in venues if v]
            )
        ).order_by(
            DBConference.is_primary.desc(),
            DBConference.rank.asc()
        )
        
        result = await self.db_session.execute(stmt)
        conferences = result.scalars().all()
        
        # Match conferences to original venues
        results: Dict[str, Optional[DBConference]] = {v: None for v in venues}
        
        for conference in conferences:
            for venue in venues:
                if not venue:
                    continue
                venue_upper = venue.strip().upper()
                
                # Get actual values (conferences are ORM objects from query)
                conf_acronym = getattr(conference, 'acronym', None)
                conf_title = getattr(conference, 'title', None)
                
                # Check acronym match
                if conf_acronym and conf_acronym.upper() == venue_upper:
                    if results[venue] is None:  # First match wins
                        results[venue] = conference
                # Check title contains
                elif conf_title and venue.lower() in conf_title.lower():
                    if results[venue] is None:
                        results[venue] = conference
        
        logger.info(
            f"Batch lookup: matched {sum(1 for v in results.values() if v)} "
            f"conferences out of {len(venues)} venues"
        )
        
        return results
