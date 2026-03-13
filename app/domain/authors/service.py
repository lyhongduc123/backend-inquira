"""
Service for author data enrichment from OpenAlex API.
Handles extraction, transformation, and persistence of author data.
"""
from typing import Dict, List, Optional, Any
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.authors import DBAuthor
from app.extensions.logger import create_logger

from .repository import AuthorRepository

logger = create_logger(__name__)


class AuthorService:
    """Service for managing author data from OpenAlex"""
    
    def __init__(self, db: AsyncSession, repository: Optional["AuthorRepository"] = None):
        self.db = db
        self.repository = repository or AuthorRepository(db)
    
    def extract_author_id_from_url(self, url: str) -> str:
        """Extract OpenAlex author ID from URL (e.g., https://openalex.org/A5114007683 -> A5114007683)"""
        if not url:
            return ""
        return url.split("/")[-1] if "/" in url else url

    
    async def upsert_from_merged_author(self, author_data: Dict) -> Optional[DBAuthor]:
        """
        Extract and persist author data from merged author dict.
        Merged authors contain both Semantic Scholar stats and OpenAlex institutions.
        
        IMPORTANT: Uses Semantic Scholar ID as primary author_id if available,
        falls back to OpenAlex ID. OpenAlex ID stored separately.
        
        Args:
            author_data: Merged author dict from retrieval service
            Example:
            {
                "name": "Frederick Sanger",
                "author_id": "123456789",  # S2 ID or OA ID
                "h_index": 45,
                "citation_count": 12000,
                "paper_count": 150,
                "url": "https://www.semanticscholar.org/author/123456789",
                "homepage_url": "https://www.semanticscholar.org/author/123456789",
                "orcid": "0000-0002-5926-4032",
                "institutions": [{...}],
                "affiliations": ["MRC Laboratory"]
            }
            
        Returns:
            DBAuthor object (created or updated) or None if invalid data
        """
        name = author_data.get("name", "").strip()
        if not name:
            logger.warning("No author name in merged author data")
            return None
        
        author_id = author_data.get("author_id")
        if not author_id:
            logger.warning(f"No author ID for {name}")
            return None
        
        orcid = author_data.get("orcid")
        if orcid and "/" in orcid:
            orcid = orcid.split("/")[-1]
        
        openalex_id = author_data.get("openalex_id")
        if not openalex_id and author_data.get("openalex_id"):
            oa_id = author_data["openalex_id"]
            if isinstance(oa_id, str):
                openalex_id = oa_id.removeprefix("https://openalex.org/")
        
        external_ids = {}
        if author_data.get("url"):
            external_ids["semantic_scholar"] = author_data.get("url")
        if orcid:
            external_ids["orcid"] = f"https://orcid.org/{orcid}"
        if openalex_id:
            external_ids["openalex"] = f"https://openalex.org/{openalex_id}"
        
        # Prepare author data for upsert
        db_author_data = {
            "author_id": author_id,
            "openalex_id": openalex_id,
            "name": name,
            "display_name": name,
            "orcid": orcid,
            "external_ids": external_ids,
            "verified": bool(orcid),
            "url": author_data.get("url"),
            "homepage_url": author_data.get("homepage_url"),
        }
        
        # Add Semantic Scholar stats if available
        if author_data.get('h_index') is not None:
            db_author_data['h_index'] = author_data.get('h_index')
        if author_data.get('citation_count') is not None:
            db_author_data['total_citations'] = author_data.get('citation_count')
        if author_data.get('paper_count') is not None:
            db_author_data['total_papers'] = author_data.get('paper_count')
        
        try:
            db_author = await self.repository.upsert_author(db_author_data)
            return db_author
        except Exception as e:
            logger.error(f"Failed to upsert author {author_id}: {e}")
            return None
    
    async def link_author_to_paper(
        self,
        author: DBAuthor,
        paper_id: int,
        author_data: Dict,
        institution_id: Optional[int] = None,
        author_position: Optional[int] = None
    ):
        """
        Create author-paper relationship with metadata.
        
        Args:
            author: DBAuthor object
            paper_id: Database ID of paper
            author_data: Merged author dict with affiliation info
            institution_id: Database ID of institution (if available)
            author_position: Position in author list (1-indexed, 1=first author)
        """
        # Extract raw affiliation strings from merged data
        affiliations = author_data.get("affiliations", [])
        institution_raw = None
        if affiliations and isinstance(affiliations, list) and len(affiliations) > 0:
            # Take first affiliation
            institution_raw = affiliations[0] if isinstance(affiliations[0], str) else None
        
        # Use author name as it appears in the paper
        author_string = author_data.get("name")
        
        try:
            await self.repository.create_author_paper_link(
                author_id=author.id,
                paper_id=paper_id,
                author_position=author_position,
                is_corresponding=False,  # Not available in merged data
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
    
    async def compute_career_metrics(self, author_id: str):
        """
        Compute career trajectory and reputation metrics from author's papers.
        
        Computes:
        - first_publication_year
        - last_known_institution_id
        - field_weighted_citation_impact (avg FWCI)
        - is_corresponding_author_frequently
        - average_author_position
        - g_index (largest number g where top g papers have at least g² citations)
        
        Args:
            author_id: Author identifier
        """
        # Get all author's papers with relationships
        papers = await self.repository.get_author_papers_with_metadata(author_id)
        author_paper_links = await self.repository.get_author_paper_links(author_id)
        
        if not papers:
            logger.warning(f"No papers found for author {author_id}")
            return
        
        logger.info(f"Computing career metrics for {author_id} from {len(papers)} papers")
        
        # Compute first publication year
        pub_years = [p.publication_date.year for p in papers if p.publication_date]
        first_year = min(pub_years) if pub_years else None
        
        # Find last known institution (most recent paper with institution)
        papers_sorted = sorted(
            [p for p in papers if p.publication_date],
            key=lambda x: x.publication_date,
            reverse=True
        )
        last_institution_id = None
        for paper in papers_sorted:
            # Find author-paper link for this paper
            link = next((ap for ap in author_paper_links if ap.paper_id == paper.id), None)
            if link and link.institution_id:
                last_institution_id = link.institution_id
                break
        
        # Compute average FWCI
        fwci_values = [p.fwci for p in papers if p.fwci is not None]
        avg_fwci = sum(fwci_values) / len(fwci_values) if fwci_values else None
        
        # Compute corresponding author frequency
        total_links = len(author_paper_links)
        corresponding_count = sum(1 for ap in author_paper_links if ap.is_corresponding)
        is_corresponding_freq = corresponding_count / total_links > 0.5 if total_links > 0 else False
        
        # Compute average author position
        positions = [ap.author_position for ap in author_paper_links if ap.author_position]
        avg_position = sum(positions) / len(positions) if positions else None
        
        # Compute collaboration diversity (unique institutions)
        unique_institutions = set(
            ap.institution_id for ap in author_paper_links
            if ap.institution_id is not None
        )
        collaboration_diversity = len(unique_institutions)
        
        # Update author record
        await self.repository.update_author(author_id, {
            "first_publication_year": first_year,
            "last_known_institution_id": last_institution_id,
            "field_weighted_citation_impact": avg_fwci,
            "is_corresponding_author_frequently": is_corresponding_freq,
            "average_author_position": avg_position,
        })
        
        logger.info(f"Updated career metrics for author {author_id}")
    
    async def batch_upsert_authors(
        self, 
        authors_data: List[Dict[str, Any]]
    ) -> Dict[str, DBAuthor]:
        """
        Batch upsert multiple authors in a single database operation.
        More efficient than individual upserts for bulk operations.
        
        Args:
            authors_data: List of merged author dicts
            
        Returns:
            Dict mapping author_id -> DBAuthor object
        """
        if not authors_data:
            return {}
        
        from sqlalchemy.dialects.postgresql import insert as pg_insert
        from app.models.authors import DBAuthor
        
        # Prepare all author records
        author_records: List[Dict[str, Any]] = []
        author_id_map: Dict[str, Dict[str, Any]] = {}
        
        for author_data in authors_data:
            name = author_data.get("name", "").strip()
            if not name:
                continue
            
            author_id = author_data.get("author_id")
            if not author_id:
                continue
            
            # Extract ORCID (clean format)
            orcid = author_data.get("orcid")
            if orcid and "/" in orcid:
                orcid = orcid.split("/")[-1]
            
            # Extract OpenAlex ID from institutions data or author_id
            # In merged data, OpenAlex ID may be in institutions or as author_id
            openalex_id = None
            if author_id.startswith("A") and len(author_id) > 1 and author_id[1:].isdigit():
                # This is an OpenAlex ID (format: A1234567890)
                openalex_id = author_id
            
            # Also check for openalex_id field directly in author_data
            if not openalex_id and author_data.get("openalex_id"):
                oa_id = author_data["openalex_id"]
                if isinstance(oa_id, str):
                    openalex_id = oa_id.removeprefix("https://openalex.org/")
            
            # Build external_ids dictionary
            external_ids = {}
            if author_data.get("url"):
                external_ids["semantic_scholar"] = author_data.get("url")
            if orcid:
                external_ids["orcid"] = f"https://orcid.org/{orcid}"
            if openalex_id:
                external_ids["openalex"] = f"https://openalex.org/{openalex_id}"
            
            # Prepare record
            record = {
                "author_id": author_id,
                "openalex_id": openalex_id,
                "name": name,
                "display_name": name,
                "orcid": orcid,
                "external_ids": external_ids,
                "verified": bool(orcid),
                "url": author_data.get("url"),
                "homepage_url": author_data.get("homepage_url"),
            }
            
            # Add Semantic Scholar stats if available
            if author_data.get('h_index') is not None:
                record['h_index'] = author_data.get('h_index')
            if author_data.get('citation_count') is not None:
                record['total_citations'] = author_data.get('citation_count')
            if author_data.get('paper_count') is not None:
                record['total_papers'] = author_data.get('paper_count')
            
            author_records.append(record)
            author_id_map[author_id] = author_data
        
        if not author_records:
            return {}
        
        try:
            # Batch insert with ON CONFLICT UPDATE
            stmt = (
                pg_insert(DBAuthor)
                .values(author_records)
                .on_conflict_do_update(
                    index_elements=['author_id'],
                    set_={
                        'name': pg_insert(DBAuthor).excluded.name,
                        'display_name': pg_insert(DBAuthor).excluded.display_name,
                        'openalex_id': pg_insert(DBAuthor).excluded.openalex_id,
                        'h_index': pg_insert(DBAuthor).excluded.h_index,
                        'total_citations': pg_insert(DBAuthor).excluded.total_citations,
                        'total_papers': pg_insert(DBAuthor).excluded.total_papers,
                        'orcid': pg_insert(DBAuthor).excluded.orcid,
                        'external_ids': pg_insert(DBAuthor).excluded.external_ids,
                        'verified': pg_insert(DBAuthor).excluded.verified,
                        'url': pg_insert(DBAuthor).excluded.url,
                        'homepage_url': pg_insert(DBAuthor).excluded.homepage_url,
                        'external_ids': pg_insert(DBAuthor).excluded.external_ids,
                        'verified': pg_insert(DBAuthor).excluded.verified,
                        'url': pg_insert(DBAuthor).excluded.url,
                        'homepage_url': pg_insert(DBAuthor).excluded.homepage_url,
                        'updated_at': datetime.now()
                    }
                )
                .returning(DBAuthor)
            )
            
            result = await self.db.execute(stmt)
            await self.db.commit()
            
            # Build result dict
            db_authors = result.scalars().all()
            result_map = {author.author_id: author for author in db_authors}
            
            logger.info(f"Batch upserted {len(result_map)} authors")
            return result_map
            
        except Exception as e:
            logger.error(f"Batch upsert authors failed: {e}", exc_info=True)
            await self.db.rollback()
            return {}
    
    async def get_author_statistics(self) -> dict:
        """
        Get comprehensive author statistics.
        
        Returns:
            Dict with statistics about all authors
        """
        return await self.repository.get_author_statistics()
    
    async def list_authors(
        self,
        page: int,
        page_size: int,
        search: Optional[str] = None,
        verified_only: bool = False
    ) -> tuple[list[DBAuthor], int]:
        """
        List authors with pagination and filters.
        
        Args:
            page: Page number (1-indexed)
            page_size: Items per page
            search: Optional search term
            verified_only: Filter for verified authors only
            
        Returns:
            Tuple of (authors list, total count)
        """
        return await self.repository.list_authors(
            page=page,
            page_size=page_size,
            search=search,
            verified_only=verified_only
        )
    
    async def get_author_by_id(self, author_id: str) -> DBAuthor:
        """
        Get author by ID.
        
        Args:
            author_id: Author identifier
            
        Returns:
            DBAuthor object
            
        Raises:
            NotFoundException if not found
        """
        from app.core.exceptions import NotFoundException
        author = await self.repository.get_author(author_id)
        if not author:
            raise NotFoundException(f"Author {author_id} not found")
        return author
    
    async def get_author_papers_with_metadata(self, author_id: str) -> list:
        """Get author's papers with metadata"""
        return await self.repository.get_author_papers_with_metadata(author_id)
    
    async def get_quartile_breakdown(self, author_id: str) -> dict:
        """Get quartile breakdown for author's publications"""
        return await self.repository.get_quartile_breakdown(author_id)
    
    async def get_co_authors(
        self,
        author_id: str,
        limit: int = 10,
        offset: int = 0,
        min_collaborations: int = 1
    ) -> list[dict]:
        """Get co-authors with collaboration counts"""
        return await self.repository.get_co_authors(
            author_id=author_id,
            limit=limit,
            offset=offset,
        )
    
    async def get_citing_authors(
        self,
        author_id: str,
        page: int,
        page_size: int,
        min_citations: int = 1
    ) -> tuple[list[dict], int]:
        """Get authors who cite this author's work"""
        return await self.repository.get_citing_authors(
            author_id=author_id,
            limit=page_size,
            offset=(page - 1) * page_size,
        )
    
    async def get_referenced_authors(
        self,
        author_id: str,
        page: int,
        page_size: int,
    ) -> tuple[list[dict], int]:
        """Get authors this author references"""
        return await self.repository.get_referenced_authors(
            author_id=author_id,
            limit=page_size,
            offset=(page - 1) * page_size,
        )
    
    async def get_author_details_with_papers(
        self,
        author_id: str,
        auto_enrich: bool = True
    ) -> Dict[str, Any]:
        """
        Get comprehensive author profile with papers, quartile breakdown, and co-authors.
        
        Args:
            author_id: Author identifier
            auto_enrich: Whether to trigger background enrichment if needed
            
        Returns:
            Dictionary with author details, papers, quartile breakdown, co-authors, etc.
        """
        from app.core.exceptions import NotFoundException
        from datetime import timezone
        
        # Get author
        author = await self.repository.get_author(author_id)
        if not author:
            raise NotFoundException(f"Author {author_id} not found")
        
        # Check if enrichment is needed
        now = datetime.now(timezone.utc)
        last_indexed = author.last_paper_indexed_at
        needs_enrichment = (
            last_indexed is None or
            (now - last_indexed).total_seconds() > 30 * 24 * 3600  # 30 days in seconds
        )
        
        enrichment_status = None
        if needs_enrichment and auto_enrich:
            # Note: Enrichment triggering should be done at router level
            # to avoid circular dependencies with task queue
            enrichment_status = {
                "status": "needs_enrichment",
                "message": "Author needs enrichment"
            }
        
        # Get papers and related data
        papers = await self.repository.get_author_papers_with_metadata(author_id)
        
        # Convert papers to metadata format
        from app.domain.papers.schemas import PaperMetadata
        paper_metadata_list = [
            PaperMetadata.from_db_model(paper)
            for paper in papers
        ]
        
        # Get quartile breakdown
        quartile_dict = await self.repository.get_quartile_breakdown(author_id)
        
        # Get co-authors
        co_author_data = await self.repository.get_co_authors(author_id, limit=10)
        
        # Calculate papers by year
        papers_by_year = {}
        for paper in papers:
            if paper.publication_date:
                year = paper.publication_date.year
                papers_by_year[year] = papers_by_year.get(year, 0) + 1
        
        return {
            "author": author,
            "papers": paper_metadata_list,
            "quartile_breakdown": quartile_dict,
            "co_authors": co_author_data,
            "papers_by_year": papers_by_year,
            "enrichment_status": enrichment_status,
            "is_enriched": author.last_paper_indexed_at is not None
        }
    
    def detect_conflict_in_citations(
        self, 
        semantic_citations: Optional[int], 
        openalex_citations: Optional[int],
        threshold_percent: float = 60.0
    ) -> bool:
        """
        Detect if there's a significant conflict between Semantic Scholar and OpenAlex citation counts.
        
        Considers it a conflict if:
        - One source has data but the other doesn't (missing data conflict)
        - Citation counts differ by >threshold_percent (default 60%)
        
        Args:
            semantic_citations: Citation count from Semantic Scholar
            openalex_citations: Citation count from OpenAlex
            threshold_percent: Percentage difference threshold (default 60%)
            
        Returns:
            True if conflict detected, False otherwise
        """
        # If both are None, no conflict
        if semantic_citations is None and openalex_citations is None:
            return False
        
        # If only one source has data, it's a conflict
        if (semantic_citations is None) != (openalex_citations is None):
            return True
        
        # Both have data - check percentage difference
        if semantic_citations and openalex_citations:
            # Calculate percentage difference from the higher value
            max_citations = max(semantic_citations, openalex_citations)
            if max_citations == 0:
                return False
            
            percent_diff = abs(semantic_citations - openalex_citations) / max_citations * 100
            return percent_diff >= threshold_percent
        
        return False
    
    async def process_author_for_conflicts(
        self, 
        author_id: str, 
        semantic_data: Dict[str, Any], 
        openalex_data: Dict[str, Any],
        threshold_percent: float = 60.0
    ) -> bool:
        """
        Process author by comparing data between Semantic Scholar and OpenAlex.
        Detects conflicts and updates the author record.
        
        Args:
            author_id: Author identifier
            semantic_data: Author data from Semantic Scholar API
            openalex_data: Author data from OpenAlex API
            threshold_percent: Citation difference threshold for flagging conflict
            
        Returns:
            True if conflict detected, False otherwise
        """
        try:
            # Extract citation counts from both sources
            semantic_citations = semantic_data.get("citation_count")
            openalex_citations = openalex_data.get("cited_by_count")
            
            # Detect conflict
            has_conflict = self.detect_conflict_in_citations(
                semantic_citations, 
                openalex_citations, 
                threshold_percent
            )
            
            # Update author record
            await self.repository.update_author(author_id, {
                "is_processed": True,
                "is_conflict": has_conflict
            })
            
            if has_conflict:
                logger.warning(
                    f"Author {author_id} flagged as conflict: "
                    f"Semantic={semantic_citations}, OpenAlex={openalex_citations} "
                    f"(diff >= {threshold_percent}%)"
                )
            else:
                logger.info(f"Author {author_id} processed successfully, no conflicts detected")
            
            return has_conflict
            
        except Exception as e:
            logger.error(f"Error processing author {author_id} for conflicts: {e}")
            # Still mark as processed even if there's an error
            try:
                await self.repository.update_author(author_id, {"is_processed": True})
            except Exception as update_error:
                logger.error(f"Failed to mark author {author_id} as processed: {update_error}")
            raise e
