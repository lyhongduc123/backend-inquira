"""
Trust Score Computation Utilities

Provides functions to compute and update trust/reputation scores for:
- Authors (individual researcher reputation)
- Institutions (organizational trust)
- Papers (combined author + institutional trust)
- Citations (network analysis metrics)

Usage:
    python -m app.trust.compute_scores --all
    python -m app.trust.compute_scores --authors-only
    python -m app.trust.compute_scores --paper-id <paper_id>
"""

from typing import List, Optional
import logging
from sqlalchemy import func, select
from sqlalchemy.orm import Session
from app.models import DBAuthor, DBInstitution, DBPaper, DBAuthorPaper, DBCitation

logger = logging.getLogger(__name__)


class TrustScoreComputer:
    """Computes trust and reputation scores across the database."""
    
    def __init__(self, db: Session):
        self.db = db
    
    def compute_author_reputation(self, author_id: int) -> float:
        """
        Compute composite reputation score for an author (0-100).
        
        Factors:
        - H-index (weight: 30%)
        - Total citations (weight: 20%)
        - Field-weighted citation impact (weight: 20%)
        - Verification status (weight: 10%)
        - Retraction penalty (weight: -20% per retraction)
        - Self-citation rate (weight: -10% if > 30%)
        - Collaboration diversity (weight: 10%)
        """
        author = self.db.query(DBAuthor).get(author_id)
        if not author:
            return 0.0
        
        score = 0.0
        h_score = min(30, (author.h_index or 0) * 0.6)
        score += h_score
        
        if author.total_citations and author.total_citations > 0:
            import math
            citation_score = min(20, math.log10(author.total_citations) * 4)
            score += citation_score
        
        if author.field_weighted_citation_impact:
            fwci_score = min(20, author.field_weighted_citation_impact * 10)
            score += fwci_score
        
        if author.verified:
            score += 10
        
        retraction_penalty = (author.retracted_papers_count or 0) * 20
        score -= retraction_penalty
        
        if author.self_citation_rate and author.self_citation_rate > 0.3:
            score -= 10
        
        if author.collaboration_diversity_score:
            collab_score = author.collaboration_diversity_score * 10
            score += collab_score
        
        return max(0.0, min(100.0, score))
    
    def compute_institution_reputation(self, institution_id: int) -> float:
        """
        Compute composite reputation score for an institution (0-100).
        
        Factors:
        - Total citations (weight: 30%)
        - H-index (weight: 25%)
        - Average paper quality (FWCI) (weight: 25%)
        - Retraction rate penalty (weight: -20%)
        - Paper volume (weight: 10%)
        """
        institution = self.db.query(DBInstitution).get(institution_id)
        if not institution:
            return 0.0
        
        score = 0.0
        
        # 1. Citation count (0-30 points, logarithmic)
        if institution.total_citations and institution.total_citations > 0:
            import math
            citation_score = min(30, math.log10(institution.total_citations) * 6)
            score += citation_score
        
        # 2. H-index (0-25 points, capped at 100)
        h_score = min(25, (institution.h_index or 0) * 0.25)
        score += h_score
        
        # 3. Average paper quality (0-25 points)
        if institution.avg_paper_quality:
            quality_score = min(25, institution.avg_paper_quality * 12.5)
            score += quality_score
        
        # 4. Retraction rate penalty
        retraction_penalty = (institution.retraction_rate or 0) * 100  # 1% = -1 point
        score -= retraction_penalty
        
        # 5. Paper volume (0-10 points, logarithmic)
        if institution.total_papers and institution.total_papers > 0:
            import math
            volume_score = min(10, math.log10(institution.total_papers) * 3)
            score += volume_score
        
        return max(0.0, min(100.0, score))
    
    def compute_author_metrics(self, author_id: int) -> None:
        """Compute h-index, total citations, FWCI, etc. for an author."""
        author = self.db.query(DBAuthor).get(author_id)
        if not author:
            return
        
        # Get all papers by this author
        papers = self.db.query(DBPaper).join(DBAuthorPaper).filter(
            DBAuthorPaper.author_id == author_id
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
        unique_institutions = self.db.query(
            func.count(func.distinct(DBAuthorPaper.institution_id))
        ).filter(
            DBAuthorPaper.author_id == author_id,
            DBAuthorPaper.institution_id.isnot(None)
        ).scalar() or 0
        
        if author.total_papers > 0:
            author.collaboration_diversity_score = min(1.0, unique_institutions / author.total_papers)
        
        # Corresponding author frequency
        corresponding_count = self.db.query(DBAuthorPaper).filter(
            DBAuthorPaper.author_id == author_id,
            DBAuthorPaper.is_corresponding == True
        ).count()
        author.is_corresponding_author_frequently = (
            corresponding_count / author.total_papers > 0.5 if author.total_papers > 0 else False
        )
        
        # Average author position
        positions = self.db.query(DBAuthorPaper.author_position).filter(
            DBAuthorPaper.author_id == author_id,
            DBAuthorPaper.author_position.isnot(None)
        ).all()
        if positions:
            author.average_author_position = sum(p[0] for p in positions) / len(positions)
        
        # Compute self-citation rate (requires citations table)
        self_citations = self.db.query(DBCitation).filter(
            DBCitation.is_self_citation == True
        ).join(
            DBAuthorPaper,
            DBCitation.cited_paper_id == DBAuthorPaper.paper_id
        ).filter(
            DBAuthorPaper.author_id == author_id
        ).count()
        
        total_citations_received = self.db.query(DBCitation).join(
            DBAuthorPaper,
            DBCitation.cited_paper_id == DBAuthorPaper.paper_id
        ).filter(
            DBAuthorPaper.author_id == author_id
        ).count()
        
        if total_citations_received > 0:
            author.self_citation_rate = self_citations / total_citations_received
        
        self.db.commit()
    
    def compute_paper_trust_scores(self, paper_id: int) -> None:
        """Compute trust scores for a paper based on its authors and institutions."""
        paper = self.db.query(DBPaper).get(paper_id)
        if not paper:
            return
        
        # Get all authors for this paper
        author_papers = self.db.query(DBAuthorPaper).filter_by(paper_id=paper_id).all()
        
        if not author_papers:
            return
        
        # Author trust score (average)
        author_scores = []
        for ap in author_papers:
            if ap.author.reputation_score:
                author_scores.append(ap.author.reputation_score)
        
        if author_scores:
            paper.author_trust_score = sum(author_scores) / len(author_scores)
        
        # Institutional trust score (average)
        institution_scores = []
        for ap in author_papers:
            if ap.institution and ap.institution.reputation_score:
                institution_scores.append(ap.institution.reputation_score)
        
        if institution_scores:
            paper.institutional_trust_score = sum(institution_scores) / len(institution_scores)
        
        # Network diversity score
        unique_institutions = len(set(ap.institution_id for ap in author_papers if ap.institution_id))
        paper.network_diversity_score = min(1.0, unique_institutions / len(author_papers))
        
        # Author count
        paper.author_count = len(author_papers)
        paper.is_single_author = len(author_papers) == 1
        
        # Update legacy fields from relationships
        paper.institutions_distinct_count = unique_institutions
        paper.countries_distinct_count = len(set(
            ap.institution.country_code for ap in author_papers 
            if ap.institution and ap.institution.country_code
        ))
        
        self.db.commit()
    
    def update_all_authors(self) -> None:
        """Compute metrics and reputation for all authors."""
        logger.info("Computing metrics for all authors...")
        
        authors = self.db.query(DBAuthor).all()
        total = len(authors)
        
        for i, author in enumerate(authors, 1):
            if i % 100 == 0:
                logger.info(f"Processing author {i}/{total}")
            
            self.compute_author_metrics(author.id)
            author.reputation_score = self.compute_author_reputation(author.id)
        
        self.db.commit()
        logger.info(f"Completed author reputation computation for {total} authors")
    
    def update_all_institutions(self) -> None:
        """Compute metrics and reputation for all institutions."""
        logger.info("Computing metrics for all institutions...")
        
        institutions = self.db.query(DBInstitution).all()
        total = len(institutions)
        
        for i, institution in enumerate(institutions, 1):
            if i % 50 == 0:
                logger.info(f"Processing institution {i}/{total}")
            
            # Compute metrics from papers
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
            
            # Compute reputation
            institution.reputation_score = self.compute_institution_reputation(institution.id)
        
        self.db.commit()
        logger.info(f"Completed institution reputation computation for {total} institutions")
    
    def update_all_papers(self) -> None:
        """Compute trust scores for all papers."""
        logger.info("Computing trust scores for all papers...")
        
        papers = self.db.query(DBPaper).all()
        total = len(papers)
        
        for i, paper in enumerate(papers, 1):
            if i % 500 == 0:
                logger.info(f"Processing paper {i}/{total}")
            
            self.compute_paper_trust_scores(paper.id)
        
        self.db.commit()
        logger.info(f"Completed paper trust score computation for {total} papers")


def compute_all_trust_scores(db: Session) -> None:
    """Run complete trust score computation pipeline."""
    computer = TrustScoreComputer(db)
    
    # Order matters: authors -> institutions -> papers
    logger.info("Starting full trust score computation...")
    
    computer.update_all_authors()
    computer.update_all_institutions()
    computer.update_all_papers()
    
    logger.info("Trust score computation complete!")


if __name__ == "__main__":
    import argparse
    import sys
    import os
    
    # Add parent directory to path for imports
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    
    try:
        from app.db.database import AsyncSession
    except ImportError:
        print("Error: Could not import SessionLocal. Make sure you're running from the backend-exegent directory.")
        print("Usage: python -m app.trust.compute_scores --all")
        sys.exit(1)
    
    parser = argparse.ArgumentParser(description="Compute trust scores")
    parser.add_argument("--all", action="store_true", help="Compute all scores")
    parser.add_argument("--authors-only", action="store_true", help="Compute author scores only")
    parser.add_argument("--institutions-only", action="store_true", help="Compute institution scores only")
    parser.add_argument("--papers-only", action="store_true", help="Compute paper scores only")
    parser.add_argument("--paper-id", type=int, help="Compute scores for specific paper")
    parser.add_argument("--author-id", type=int, help="Compute scores for specific author")
    
    args = parser.parse_args()
    
    db = Session()
    computer = TrustScoreComputer(db)
    
    try:
        if args.all:
            compute_all_trust_scores(db)
        elif args.authors_only:
            computer.update_all_authors()
        elif args.institutions_only:
            computer.update_all_institutions()
        elif args.papers_only:
            computer.update_all_papers()
        elif args.paper_id:
            computer.compute_paper_trust_scores(args.paper_id)
            logger.info(f"Computed trust scores for paper {args.paper_id}")
        elif args.author_id:
            computer.compute_author_metrics(args.author_id)
            score = computer.compute_author_reputation(args.author_id)
            logger.info(f"Author {args.author_id} reputation score: {score:.2f}")
        else:
            parser.print_help()
    finally:
        db.close()
