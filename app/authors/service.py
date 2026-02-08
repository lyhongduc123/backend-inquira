"""
Service for author data enrichment from OpenAlex API.
Handles extraction, transformation, and persistence of author data.
"""
from typing import Dict, List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from app.authors.repository import AuthorRepository
from app.models.authors import DBAuthor
from app.extensions.logger import create_logger

logger = create_logger(__name__)


class AuthorService:
    """Service for managing author data from OpenAlex"""
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.repository = AuthorRepository(db)
    
    def extract_author_id_from_url(self, url: str) -> str:
        """Extract OpenAlex author ID from URL (e.g., https://openalex.org/A5114007683 -> A5114007683)"""
        if not url:
            return ""
        return url.split("/")[-1] if "/" in url else url
    
    async def upsert_from_openalex(self, authorship: Dict, s2_stats: Optional[Dict] = None) -> Optional[DBAuthor]:
        """
        Extract and persist author data from OpenAlex authorship object.
        Enriches with Semantic Scholar stats if available.
        
        IMPORTANT: Uses Semantic Scholar ID as primary author_id if available,
        falls back to OpenAlex ID. OpenAlex ID stored separately.
        
        Args:
            authorship: OpenAlex authorship object containing author info
            s2_stats: Optional Semantic Scholar stats (h_index, citation_count, paper_count)
            Example:
            {
                "author_position": "first",
                "author": {
                    "id": "https://openalex.org/A5114007683",
                    "display_name": "Frederick Sanger",
                    "orcid": "https://orcid.org/0000-0002-5926-4032"
                },
                "institutions": [...],
                "is_corresponding": false
            }
            
        Returns:
            DBAuthor object (created or updated) or None if invalid data
        """
        author_info = authorship.get("author", {})
        if not author_info:
            logger.warning("No author info in authorship object")
            return None
        
        author_id_url = author_info.get("id")
        if not author_id_url:
            logger.warning("No author ID in authorship object")
            return None
        
        openalex_id = self.extract_author_id_from_url(author_id_url)
        display_name = author_info.get("display_name", "Unknown Author")
        
        # IMPORTANT: Use S2 ID as primary if available, else OpenAlex ID
        s2_author_id = s2_stats.get('author_id') if s2_stats else None
        primary_author_id = s2_author_id if s2_author_id else openalex_id
        
        # Extract ORCID if available
        orcid_url = author_info.get("orcid")
        orcid = None
        if orcid_url:
            # Extract ORCID from URL (https://orcid.org/0000-0002-5926-4032 -> 0000-0002-5926-4032)
            orcid = orcid_url.split("/")[-1] if "/" in orcid_url else orcid_url
        
        # Build external_ids dictionary with both OpenAlex and Semantic Scholar IDs
        external_ids = {}
        if author_id_url:
            external_ids["openalex"] = author_id_url
        if orcid_url:
            external_ids["orcid"] = orcid_url
        if s2_author_id:
            external_ids["semantic_scholar"] = s2_author_id
        
        # Prepare author data with S2 stats
        author_data = {
            "author_id": primary_author_id,  # S2 ID preferred, else OpenAlex
            "openalex_id": openalex_id,  # Always store OpenAlex ID separately
            "name": display_name,
            "display_name": display_name,
            "orcid": orcid,
            "external_ids": external_ids,
            "verified": bool(orcid)  # Authors with ORCID are considered verified
        }
        
        # Add Semantic Scholar stats if available
        if s2_stats:
            if s2_stats.get('h_index') is not None:
                author_data['h_index'] = s2_stats.get('h_index')
            if s2_stats.get('citation_count') is not None:
                author_data['total_citations'] = s2_stats.get('citation_count')
            if s2_stats.get('paper_count') is not None:
                author_data['total_papers'] = s2_stats.get('paper_count')
        
        # Upsert author
        try:
            db_author = await self.repository.upsert_author(author_data)
            return db_author
        except Exception as e:
            logger.error(f"Failed to upsert author {primary_author_id}: {e}")
            return None
    
    async def link_author_to_paper(
        self,
        author: DBAuthor,
        paper_id: int,
        authorship: Dict,
        institution_id: Optional[int] = None
    ):
        """
        Create author-paper relationship with metadata.
        
        Args:
            author: DBAuthor object
            paper_id: Database ID of paper
            authorship: OpenAlex authorship object with position and affiliation info
            institution_id: Database ID of institution (if available)
        """
        # Extract author position
        author_position_str = authorship.get("author_position", "")
        author_position = None
        if author_position_str == "first":
            author_position = 1
        elif author_position_str == "last":
            author_position = 999  # Use high number for last author
        # Middle authors would need counting, skip for now
        
        # Extract is_corresponding
        is_corresponding = authorship.get("is_corresponding", False)
        
        # Extract raw affiliation strings
        affiliations = authorship.get("affiliations", [])
        institution_raw = None
        if affiliations:
            # Take first affiliation string
            institution_raw = affiliations[0].get("raw_affiliation_string")
        
        # Extract author name as appeared in paper
        author_string = authorship.get("raw_author_name")
        
        try:
            await self.repository.create_author_paper_link(
                author_id=author.id,
                paper_id=paper_id,
                author_position=author_position,
                is_corresponding=is_corresponding,
                institution_id=institution_id,
                institution_raw=institution_raw,
                author_string=author_string
            )
        except Exception as e:
            logger.error(f"Failed to link author {author.id} to paper {paper_id}: {e}")
    
    async def link_author_to_institution(
        self,
        author: DBAuthor,
        institution_id: int,
        year: Optional[int] = None,
        is_current: bool = False
    ):
        """
        Create author-institution relationship.
        
        Args:
            author: DBAuthor object
            institution_id: Database ID of institution
            year: Year of publication (used to infer affiliation period)
            is_current: Whether this is current affiliation
        """
        try:
            await self.repository.create_author_institution_link(
                author_id=author.id,
                institution_id=institution_id,
                year=year,
                is_current=is_current
            )
        except Exception as e:
            logger.error(f"Failed to link author {author.id} to institution {institution_id}: {e}")
