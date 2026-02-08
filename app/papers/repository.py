from typing import List, Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, and_
from datetime import datetime, date
from app.models.papers import DBPaper, DBPaperChunk
from app.authors.service import AuthorService
from app.institutions.service import InstitutionService
from app.retriever.paper_schemas import Paper
from app.extensions.logger import create_logger

logger = create_logger(__name__)


class PaperRepository:
    """Repository for paper database operations"""
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.author_service = AuthorService(db)
        self.institution_service = InstitutionService(db)
    
    async def create_paper(self, paper: Paper) -> DBPaper:
        """
        Create a new paper in database with all available attributes.
        If paper already exists, return the existing record.
        
        Args:
            paper: Paper schema object
            
        Returns:
            Created or existing DBPaper object
        """
        # Check if paper already exists
        existing = await self.get_paper_by_id(paper.paper_id)
        if existing:
            logger.info(f"Paper {paper.paper_id} already exists in database")
            return existing
        
        # Convert authors to JSONB format (handle both dict and Author objects)
        authors_json = []
        if paper.authors:
            for author in paper.authors:
                if isinstance(author, dict):
                    authors_json.append(author)
                else:
                    # Author is a Pydantic model
                    authors_json.append({
                        "name": author.name,
                        "author_id": author.author_id
                    })
        
        db_paper = DBPaper(
            # Core identifiers and metadata
            paper_id=paper.paper_id,
            title=paper.title or "Untitled",
            authors=authors_json,
            abstract=paper.abstract or "Abstract not available",
            publication_date=paper.publication_date,
            venue=paper.venue,
            issn=paper.issn,
            issn_l=paper.issn_l,
            
            # URLs
            url=paper.url or None,
            pdf_url=paper.pdf_url or None,
            is_open_access=paper.is_open_access or False,
            open_access_pdf=paper.open_access_pdf,
            
            # Source and identifiers
            source=paper.source or "SemanticScholar",
            external_ids=paper.external_ids or {},
            
            # Summary and embeddings (populated during processing)
            summary=paper.summary,
            summary_embedding=paper.summary_embedding,
            
            # Relevance and citation metrics
            relevance_score=paper.relevance_score,
            citation_count=paper.citation_count or 0,
            influential_citation_count=paper.influential_citation_count or 0,
            reference_count=paper.reference_count or 0,
            
            # Rich metadata (OpenAlex specific) - only set if not None
            topics=paper.topics,
            keywords=paper.keywords,
            concepts=paper.concepts,
            mesh_terms=paper.mesh_terms,
            
            # Citation quality metrics
            citation_percentile=paper.citation_percentile,
            fwci=paper.fwci,
            
            # Trust scores (computed later)
            author_trust_score=paper.author_trust_score,
            institutional_trust_score=paper.institutional_trust_score,
            network_diversity_score=paper.network_diversity_score,
            
            # Journal relationship
            journal_id=paper.journal_id,
            
            # Paper quality indicators
            is_retracted=paper.is_retracted or False,
            language=paper.language,
            
            # Author collaboration metadata
            corresponding_author_ids=paper.corresponding_author_ids,
            institutions_distinct_count=paper.institutions_distinct_count,
            countries_distinct_count=paper.countries_distinct_count,
            
            # Processing status
            is_processed=paper.is_processed or False,
            processing_status=paper.processing_status or "pending",
            processing_error=None
        )
        
        self.db.add(db_paper)
        await self.db.commit()
        await self.db.refresh(db_paper)
        
        logger.info(f"Created paper {paper.paper_id} in database")
        
        # Enrich with authors and institutions if authorships data is available
        if paper.authorships:
            try:
                # Pass semantic_authors for stats enrichment if available
                semantic_authors = getattr(paper, 'semantic_authors', None)
                await self.enrich_paper_with_authors_institutions(
                    db_paper, 
                    paper.authorships,
                    semantic_authors=semantic_authors
                )
            except Exception as e:
                logger.error(f"Failed to enrich paper {paper.paper_id} with authors/institutions: {e}")
        
        return db_paper
    
    async def enrich_paper_with_authors_institutions(
        self, 
        db_paper: DBPaper, 
        authorships: List[Dict],
        semantic_authors: Optional[List[Dict]] = None
    ):
        """
        Enrich paper with author and institution data from OpenAlex authorships.
        Creates DBAuthor, DBInstitution entities and links them to the paper.
        
        Args:
            db_paper: Database paper entity
            authorships: List of OpenAlex authorship objects
            semantic_authors: Optional list of Semantic Scholar authors with stats (h_index, citationCount, etc.)
            
        Example authorship:
        {
            "author_position": "first",
            "author": {
                "id": "https://openalex.org/A5114007683",
                "display_name": "Frederick Sanger",
                "orcid": "https://orcid.org/0000-0002-5926-4032"
            },
            "institutions": [
                {
                    "id": "https://openalex.org/I170203145",
                    "display_name": "MRC Laboratory of Molecular Biology",
                    "ror": "https://ror.org/00tw3jy02",
                    "country_code": "GB",
                    "type": "facility",
                    "city": "Cambridge",
                    "country": "United Kingdom"
                }
            ],
            "is_corresponding": false,
            "raw_author_name": "F. Sanger"
        }
        
        Example semantic_author:
        {
            "name": "Frederick Sanger",
            "author_id": "1234567",
            "h_index": 45,
            "citation_count": 12000,
            "paper_count": 150
        }
        """
        if not authorships:
            logger.debug(f"No authorships data for paper {db_paper.paper_id}")
            return
        
        # Build lookup map of Semantic Scholar author stats by name (fuzzy match)
        s2_stats_map = {}
        
        # Build lookup map of Semantic Scholar author stats by name (fuzzy match)
        s2_stats_map = {}
        if semantic_authors:
            for s2_author in semantic_authors:
                name = s2_author.get('name', '').lower().strip()
                if name:
                    s2_stats_map[name] = {
                        'author_id': s2_author.get('author_id'),
                        'h_index': s2_author.get('h_index'),
                        'citation_count': s2_author.get('citation_count'),
                        'paper_count': s2_author.get('paper_count')
                    }
        
        # Extract publication year for author-institution tracking
        # publication_date is a datetime.date object from SQLAlchemy Date column
        pub_year = None
        if db_paper.publication_date:
            if isinstance(db_paper.publication_date, (datetime, date)):
                pub_year = db_paper.publication_date.year
            elif isinstance(db_paper.publication_date, int):
                pub_year = db_paper.publication_date
        
        for authorship in authorships:
            # Process author with enriched stats
            author_name = authorship.get("author", {}).get("display_name", "").lower().strip()
            s2_stats = s2_stats_map.get(author_name, {})
            
            db_author = await self.author_service.upsert_from_openalex(authorship, s2_stats)
            if not db_author:
                continue
            
            # Process institutions for this author
            institutions = authorship.get("institutions", [])
            institution_id = None
            
            if institutions:
                # Process first institution (primary affiliation)
                primary_institution = institutions[0]
                db_institution = await self.institution_service.upsert_from_openalex(primary_institution)
                
                if db_institution:
                    institution_id = db_institution.id
                    
                    # Link author to institution
                    await self.author_service.link_author_to_institution(
                        author=db_author,
                        institution_id=db_institution.id,
                        year=pub_year,
                        is_current=False  # We can't determine if it's current from paper data
                    )
            
            # Link author to paper
            await self.author_service.link_author_to_paper(
                author=db_author,
                paper_id=db_paper.id,
                authorship=authorship,
                institution_id=institution_id
            )
        
        logger.info(f"Enriched paper {db_paper.paper_id} with {len(authorships)} authors and their institutions")
        
    async def get_papers(
        self,
        skip: int = 0,
        limit: int = 20,
        processed_only: bool = False,
        source: Optional[str] = None
    ) -> tuple[List[DBPaper], int]:
        """
        Get papers with pagination and optional filtering
        
        Args:
            skip: Number of records to skip
            limit: Maximum number of records to return
            processed_only: If True, only return processed papers
            source: Optional source filter (e.g., 'SemanticScholar', 'OpenAlex')
            
        Returns:
            Tuple of (papers list, total count)
        """
        # Build base query
        query = select(DBPaper)
        count_query = select(DBPaper)
        
        # Apply filters
        if processed_only:
            query = query.where(DBPaper.is_processed == True)
            count_query = count_query.where(DBPaper.is_processed == True)
        
        if source:
            query = query.where(DBPaper.source == source)
            count_query = count_query.where(DBPaper.source == source)
        
        # Get total count
        from sqlalchemy import func
        count_result = await self.db.execute(
            select(func.count()).select_from(count_query.subquery())
        )
        total = count_result.scalar_one()
        
        # Apply pagination and ordering
        query = query.order_by(DBPaper.created_at.desc()).offset(skip).limit(limit)
        
        # Execute query
        result = await self.db.execute(query)
        papers = list(result.scalars().all())
        
        return papers, total
    
    async def get_paper_by_id(self, paper_id: str) -> Optional[DBPaper]:
        """Get paper by internal paper_id"""
        result = await self.db.execute(
            select(DBPaper).where(DBPaper.paper_id == paper_id)
        )
        return result.scalar_one_or_none()
    
    async def get_paper_by_db_id(self, id: int) -> Optional[DBPaper]:
        """Get paper by database ID"""
        result = await self.db.execute(
            select(DBPaper).where(DBPaper.id == id)
        )
        return result.scalar_one_or_none()
    
    async def get_paper_by_external_ids(self, external_ids: Dict[str, str], source: str) -> Optional[DBPaper]:
        """Get paper by external IDs and source"""
        # Try to find by DOI first, then other IDs
        doi = external_ids.get('DOI')
        if doi:
            result = await self.db.execute(
                select(DBPaper).where(
                    and_(
                        DBPaper.external_ids['DOI'].astext == doi,
                        DBPaper.source == source
                    )
                )
            )
            paper = result.scalar_one_or_none()
            if paper:
                return paper
        
        # Fallback to other IDs
        for key, value in external_ids.items():
            if value:
                result = await self.db.execute(
                    select(DBPaper).where(
                        and_(
                            DBPaper.external_ids[key].astext == value,
                            DBPaper.source == source
                        )
                    )
                )
                paper = result.scalar_one_or_none()
                if paper:
                    return paper
        
        return None
    
    async def update_paper_processing_status(
        self,
        paper_id: str,
        status: str,
        error: Optional[str] = None
    ):
        """Update paper processing status"""
        update_data = {
            "processing_status": status,
            "updated_at": datetime.utcnow()
        }
        
        if status == "completed":
            update_data["is_processed"] = True
        
        if error:
            update_data["processing_error"] = error
        
        await self.db.execute(
            update(DBPaper)
            .where(DBPaper.paper_id == paper_id)
            .values(**update_data)
        )
        await self.db.commit()
    
    async def update_paper(self, paper_id: str, update_data: Dict[str, Any]) -> Optional[DBPaper]:
        """Update paper with provided data"""
        await self.db.execute(
            update(DBPaper)
            .where(DBPaper.paper_id == paper_id)
            .values(**update_data, updated_at=datetime.utcnow())
        )
        await self.db.commit()
        return await self.get_paper_by_id(paper_id)
    
    async def delete_paper(self, paper_id: str) -> bool:
        """
        Delete paper from database.
        Note: Chunks should be deleted separately via ChunkRepository before calling this.
        """
        from sqlalchemy import delete as sql_delete
        
        # Delete paper
        result = await self.db.execute(
            sql_delete(DBPaper).where(DBPaper.paper_id == paper_id)
        )
        await self.db.commit()
        
        return result.rowcount > 0 # type: ignore
    
    async def update_paper_summary(
        self,
        paper_id: str,
        summary: str,
        summary_embedding: List[float]
    ):
        """Update paper summary and embedding"""
        await self.db.execute(
            update(DBPaper)
            .where(DBPaper.paper_id == paper_id)
            .values(
                summary=summary,
                summary_embedding=summary_embedding,
                updated_at=datetime.now()
            )
        )
        await self.db.commit()
    
    async def search_similar_papers(
        self,
        query_embedding: List[float],
        limit: int = 10
    ) -> List[DBPaper]:
        """
        Search for similar papers using summary embeddings
        
        Args:
            query_embedding: Query embedding vector
            limit: Number of results to return
            
        Returns:
            List of similar papers ordered by similarity
        """
        result = await self.db.execute(
            select(DBPaper)
            .where(DBPaper.summary_embedding.isnot(None))
            .order_by(DBPaper.summary_embedding.cosine_distance(query_embedding))
            .limit(limit)
        )
        return list(result.scalars().all())
    
    async def update_last_accessed(self, paper_id: str):
        """Update last accessed timestamp"""
        await self.db.execute(
            update(DBPaper)
            .where(DBPaper.paper_id == paper_id)
            .values(last_accessed_at=datetime.utcnow())
        )
        await self.db.commit()
