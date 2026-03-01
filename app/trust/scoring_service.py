"""Trust Scoring Service

Orchestration layer that handles:
- Database access and queries
- Fetching enriched objects (authors, institutions, papers with relationships)
- Calling compute_scores functions
- Updating database with computed scores
- Batch operations (update_all, offline scoring)

For pure computation functions, see compute_scores.py
"""

from typing import Optional, Union
import logging
from sqlalchemy import func
from sqlalchemy.orm import Session, joinedload
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import DBAuthor, DBInstitution, DBPaper, DBAuthorPaper
from app.trust import compute_scores

logger = logging.getLogger(__name__)


class ScoringService:
    """Service for trust score computation with database orchestration."""
    
    def __init__(self, db: Union[Session, AsyncSession]):
        self.db = db
        self.is_async = isinstance(db, AsyncSession)
    
    async def compute_author_metrics_and_reputation(
        self, 
        author_id: Optional[int] = None,
        author: Optional[DBAuthor] = None
    ) -> float:
        """
        Compute author metrics (h-index, citations, etc.) and reputation score.
        
        Args:
            author_id: Database ID to fetch author, OR
            author: Already loaded DBAuthor object
            
        Returns:
            Computed reputation score (0-100)
        """
        if author is None:
            if author_id is None:
                raise ValueError("Must provide either author_id or author object")
            
            if self.is_async:
                author = await self.db.get(DBAuthor, author_id)
            else:
                author = self.db.query(DBAuthor).get(author_id)
        
        if not author:
            return 0.0
        
        # Compute metrics from papers
        if self.is_async:
            from sqlalchemy import select
            result = await self.db.execute(
                select(DBPaper)
                .join(DBAuthorPaper)
                .filter(DBAuthorPaper.author_id == author.id)
            )
            papers = result.scalars().all()
        else:
            papers = self.db.query(DBPaper).join(DBAuthorPaper).filter(
                DBAuthorPaper.author_id == author.id
            ).all()
        
        # Total papers
        author.total_papers = len(papers)
        
        # Total citations
        author.total_citations = sum(p.citation_count or 0 for p in papers)
        
        # H-index calculation
        citation_counts = sorted([p.citation_count or 0 for p in papers], reverse=True)
        h_index = 0
        for i, citations in enumerate(citation_counts, 1):
            if citations >= i:
                h_index = i
            else:
                break
        author.h_index = h_index
        
        # i10-index (papers with >=10 citations)
        author.i10_index = sum(1 for p in papers if (p.citation_count or 0) >= 10)
        
        # Average FWCI
        fwci_values = [p.fwci for p in papers if p.fwci is not None]
        if fwci_values:
            author.field_weighted_citation_impact = sum(fwci_values) / len(fwci_values)
        
        # Check retractions
        retracted_papers = [p for p in papers if p.is_retracted]
        author.has_retracted_papers = len(retracted_papers) > 0
        author.retracted_papers_count = len(retracted_papers)
        
        # Collaboration diversity (unique institutions / total papers)
        if self.is_async:
            result = await self.db.execute(
                select(func.count(func.distinct(DBAuthorPaper.institution_id)))
                .filter(
                    DBAuthorPaper.author_id == author.id,
                    DBAuthorPaper.institution_id.isnot(None)
                )
            )
            unique_institutions = result.scalar() or 0
        else:
            unique_institutions = self.db.query(
                func.count(func.distinct(DBAuthorPaper.institution_id))
            ).filter(
                DBAuthorPaper.author_id == author.id,
                DBAuthorPaper.institution_id.isnot(None)
            ).scalar() or 0
        
        if author.total_papers > 0:
            author.collaboration_diversity_score = min(1.0, unique_institutions / author.total_papers)
        
        # Corresponding author frequency
        if self.is_async:
            result = await self.db.execute(
                select(func.count())
                .select_from(DBAuthorPaper)
                .filter(
                    DBAuthorPaper.author_id == author.id,
                    DBAuthorPaper.is_corresponding == True
                )
            )
            corresponding_count = result.scalar() or 0
        else:
            corresponding_count = self.db.query(DBAuthorPaper).filter(
                DBAuthorPaper.author_id == author.id,
                DBAuthorPaper.is_corresponding == True
            ).count()
        
        author.is_corresponding_author_frequently = (
            corresponding_count / author.total_papers > 0.5 if author.total_papers > 0 else False
        )
        
        # Average author position
        if self.is_async:
            result = await self.db.execute(
                select(DBAuthorPaper.author_position)
                .filter(
                    DBAuthorPaper.author_id == author.id,
                    DBAuthorPaper.author_position.isnot(None)
                )
            )
            positions = result.scalars().all()
        else:
            positions = self.db.query(DBAuthorPaper.author_position).filter(
                DBAuthorPaper.author_id == author.id,
                DBAuthorPaper.author_position.isnot(None)
            ).all()
        
        if positions:
            author.average_author_position = sum(positions) / len(positions)
        
        # Extract data for pure computation function
        author_data = {
            'h_index': author.h_index,
            'total_citations': author.total_citations,
            'field_weighted_citation_impact': author.field_weighted_citation_impact,
            'verified': author.verified,
            'retracted_papers_count': author.retracted_papers_count,
            'collaboration_diversity_score': author.collaboration_diversity_score,
        }
        
        # Compute reputation score using pure function
        reputation_score = compute_scores.compute_author_reputation(author_data)
        author.reputation_score = reputation_score
        
        if self.is_async:
            await self.db.commit()
        else:
            self.db.commit()
        
        return reputation_score
    
    async def compute_institution_metrics_and_reputation(
        self,
        institution_id: Optional[int] = None,
        institution: Optional[DBInstitution] = None
    ) -> float:
        """
        Compute institution metrics and reputation score.
        
        Args:
            institution_id: Database ID to fetch institution, OR
            institution: Already loaded DBInstitution object
            
        Returns:
            Computed reputation score (0-100)
        """
        if institution is None:
            if institution_id is None:
                raise ValueError("Must provide either institution_id or institution object")
            
            if self.is_async:
                institution = await self.db.get(DBInstitution, institution_id)
            else:
                institution = self.db.query(DBInstitution).get(institution_id)
        
        if not institution:
            return 0.0
        
        # Compute metrics from papers
        if self.is_async:
            from sqlalchemy import select
            result = await self.db.execute(
                select(DBPaper)
                .join(DBAuthorPaper)
                .filter(DBAuthorPaper.institution_id == institution.id)
            )
            papers = result.scalars().all()
        else:
            papers = self.db.query(DBPaper).join(DBAuthorPaper).filter(
                DBAuthorPaper.institution_id == institution.id
            ).all()
        
        institution.total_papers = len(papers)
        institution.total_citations = sum(p.citation_count or 0 for p in papers)
        
        # Average paper quality
        fwci_values = [p.fwci for p in papers if p.fwci]
        if fwci_values:
            institution.avg_paper_quality = sum(fwci_values) / len(fwci_values)
        
        # Retraction rate
        retracted = sum(1 for p in papers if p.is_retracted)
        institution.retraction_rate = retracted / institution.total_papers if institution.total_papers > 0 else 0
        
        # H-index calculation
        citation_counts = sorted([p.citation_count or 0 for p in papers], reverse=True)
        h_index = 0
        for i, citations in enumerate(citation_counts, 1):
            if citations >= i:
                h_index = i
            else:
                break
        institution.h_index = h_index
        
        # Extract data for pure computation function
        institution_data = {
            'total_citations': institution.total_citations,
            'h_index': institution.h_index,
            'avg_paper_quality': institution.avg_paper_quality,
            'retraction_rate': institution.retraction_rate,
            'total_papers': institution.total_papers,
        }
        
        # Compute reputation using pure function
        reputation_score = compute_scores.compute_institution_reputation(institution_data)
        institution.reputation_score = reputation_score
        
        if self.is_async:
            await self.db.commit()
        else:
            self.db.commit()
        
        return reputation_score
    
    async def compute_paper_trust_scores(
        self,
        paper_id: Optional[int] = None,
        paper: Optional[DBPaper] = None
    ) -> dict:
        """
        Compute trust scores for a paper.
        
        Args:
            paper_id: Database ID to fetch paper with relationships, OR
            paper: Already loaded DBPaper object with author_papers relationship
            
        Returns:
            Dict with computed trust scores
        """
        if paper is None:
            if paper_id is None:
                raise ValueError("Must provide either paper_id or paper object")
            
            # Fetch paper with loaded relationships
            if self.is_async:
                from sqlalchemy import select
                result = await self.db.execute(
                    select(DBPaper)
                    .options(
                        joinedload(DBPaper.author_papers)
                        .joinedload(DBAuthorPaper.author),
                        joinedload(DBPaper.author_papers)
                        .joinedload(DBAuthorPaper.institution)
                    )
                    .filter(DBPaper.id == paper_id)
                )
                paper = result.scalar_one_or_none()
            else:
                paper = self.db.query(DBPaper).options(
                    joinedload(DBPaper.author_papers)
                    .joinedload(DBAuthorPaper.author),
                    joinedload(DBPaper.author_papers)
                    .joinedload(DBAuthorPaper.institution)
                ).filter(DBPaper.id == paper_id).first()
        
        if not paper:
            return {}
        
        # Get author_papers relationship
        if self.is_async:
            from sqlalchemy import select
            result = await self.db.execute(
                select(DBAuthorPaper)
                .options(
                    joinedload(DBAuthorPaper.author),
                    joinedload(DBAuthorPaper.institution)
                )
                .filter(DBAuthorPaper.paper_id == paper.id)
            )
            author_papers = result.scalars().all()
        else:
            author_papers = self.db.query(DBAuthorPaper).options(
                joinedload(DBAuthorPaper.author),
                joinedload(DBAuthorPaper.institution)
            ).filter(DBAuthorPaper.paper_id == paper.id).all()
        
        # Extract data for pure computation function
        author_papers_data = [
            {
                'author_reputation_score': ap.author.reputation_score if ap.author else None,
                'institution_reputation_score': ap.institution.reputation_score if ap.institution else None,
                'institution_id': ap.institution_id,
                'country_code': ap.institution.country_code if ap.institution else None,
            }
            for ap in author_papers
        ]
        
        # Compute scores using pure function
        scores = compute_scores.compute_paper_trust_scores(author_papers_data)
        
        # Update paper fields
        paper.author_trust_score = scores["author_trust_score"]
        paper.institutional_trust_score = scores["institutional_trust_score"]
        paper.network_diversity_score = scores["network_diversity_score"]
        paper.author_count = scores["author_count"]
        paper.is_single_author = scores["is_single_author"]
        paper.institutions_distinct_count = scores["institutions_distinct_count"]
        paper.countries_distinct_count = scores["countries_distinct_count"]
        
        if self.is_async:
            await self.db.commit()
        else:
            self.db.commit()
        
        return scores
    
    async def update_all_authors(self) -> None:
        """Batch update: Compute metrics and reputation for all authors."""
        logger.info("Computing metrics for all authors...")
        
        if self.is_async:
            from sqlalchemy import select
            result = await self.db.execute(select(DBAuthor))
            authors = result.scalars().all()
        else:
            authors = self.db.query(DBAuthor).all()
        
        total = len(authors)
        
        for i, author in enumerate(authors, 1):
            if i % 100 == 0:
                logger.info(f"Processing author {i}/{total}")
            
            await self.compute_author_metrics_and_reputation(author=author)
        
        logger.info(f"Completed author reputation computation for {total} authors")
    
    async def update_all_institutions(self) -> None:
        """Batch update: Compute metrics and reputation for all institutions."""
        logger.info("Computing metrics for all institutions...")
        
        if self.is_async:
            from sqlalchemy import select
            result = await self.db.execute(select(DBInstitution))
            institutions = result.scalars().all()
        else:
            institutions = self.db.query(DBInstitution).all()
        
        total = len(institutions)
        
        for i, institution in enumerate(institutions, 1):
            if i % 50 == 0:
                logger.info(f"Processing institution {i}/{total}")
            
            await self.compute_institution_metrics_and_reputation(institution=institution)
        
        logger.info(f"Completed institution reputation computation for {total} institutions")
    
    async def update_all_papers(self) -> None:
        """Batch update: Compute trust scores for all papers."""
        logger.info("Computing trust scores for all papers...")
        
        if self.is_async:
            from sqlalchemy import select
            result = await self.db.execute(select(DBPaper))
            papers = result.scalars().all()
        else:
            papers = self.db.query(DBPaper).all()
        
        total = len(papers)
        
        for i, paper in enumerate(papers, 1):
            if i % 500 == 0:
                logger.info(f"Processing paper {i}/{total}")
            
            await self.compute_paper_trust_scores(paper=paper)
        
        logger.info(f"Completed paper trust score computation for {total} papers")
    
    async def compute_all_trust_scores(self) -> None:
        """Run complete trust score computation pipeline."""
        logger.info("Starting full trust score computation...")
        
        # Order matters: authors -> institutions -> papers
        await self.update_all_authors()
        await self.update_all_institutions()
        await self.update_all_papers()
        
        logger.info("Trust score computation complete!")


# Sync wrapper for backward compatibility
def compute_all_trust_scores(db: Session) -> None:
    """Run complete trust score computation pipeline (sync version)."""
    import asyncio
    service = ScoringService(db)
    
    # Run async methods in sync context
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(service.compute_all_trust_scores())
    finally:
        loop.close()


if __name__ == "__main__":
    import argparse
    import sys
    import os
    
    # Add parent directory to path for imports
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    
    try:
        from app.db.database import SessionLocal
    except ImportError:
        print("Error: Could not import SessionLocal. Make sure you're running from the backend-exegent directory.")
        print("Usage: python -m app.trust.scoring_service --all")
        sys.exit(1)
    
    parser = argparse.ArgumentParser(description="Compute trust scores")
    parser.add_argument("--all", action="store_true", help="Compute all scores")
    parser.add_argument("--authors-only", action="store_true", help="Compute author scores only")
    parser.add_argument("--institutions-only", action="store_true", help="Compute institution scores only")
    parser.add_argument("--papers-only", action="store_true", help="Compute paper scores only")
    parser.add_argument("--paper-id", type=int, help="Compute scores for specific paper")
    parser.add_argument("--author-id", type=int, help="Compute scores for specific author")
    
    args = parser.parse_args()
    
    db = SessionLocal()
    service = ScoringService(db)
    
    import asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        if args.all:
            loop.run_until_complete(service.compute_all_trust_scores())
        elif args.authors_only:
            loop.run_until_complete(service.update_all_authors())
        elif args.institutions_only:
            loop.run_until_complete(service.update_all_institutions())
        elif args.papers_only:
            loop.run_until_complete(service.update_all_papers())
        elif args.paper_id:
            scores = loop.run_until_complete(service.compute_paper_trust_scores(paper_id=args.paper_id))
            logger.info(f"Computed trust scores for paper {args.paper_id}: {scores}")
        elif args.author_id:
            score = loop.run_until_complete(service.compute_author_metrics_and_reputation(author_id=args.author_id))
            logger.info(f"Author {args.author_id} reputation score: {score:.2f}")
        else:
            parser.print_help()
    finally:
        loop.close()
        db.close()
