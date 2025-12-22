from typing import Dict, Any, Optional, List
from app.retriever.paper_schemas import Paper, Author
from app.retriever.provider.base_schemas import NormalizedResult

class TransformerService:
    def convert_normalized_to_paper(self, normalized: NormalizedResult) -> Paper:
        """
        Convert normalized result from provider to Paper object
        
        Args:
            normalized: Normalized paper data from provider
            
        Returns:
            Paper object
        """
        # Convert authors
        authors_list = []
        authors_data = normalized.get('authors', [])
        if authors_data:
            authors_list = [
                Author(
                    name=a.get('name', '') if isinstance(a, dict) else str(a),
                    author_id=a.get('author_id') if isinstance(a, dict) else None
                )
                for a in authors_data
            ]
        
        # Extract external_ids dict from normalized result
        external_ids = normalized.get('external_ids') or {}
        
        print(normalized.get('paper_id', ''), normalized.get('influential_citation_count'))
        
        return Paper(
            paper_id=normalized.get('paper_id', ''),
            title=normalized.get('title', ''),
            abstract=normalized.get('abstract'),
            authors=authors_list,
            publication_date=None,  # Will be parsed from publication_date if needed
            venue=normalized.get('venue'),
            url=normalized.get('url'),
            citation_count=normalized.get('citation_count') or 0,
            influential_citation_count=normalized.get('influential_citation_count') or 0,
            reference_count=normalized.get('reference_count') or 0,
            external_ids=external_ids,
            source=normalized.get('source', 'semantic_scholar'),
            pdf_url=normalized.get('pdf_url'),
            is_open_access=normalized.get('is_open_access', False),
            open_access_pdf=normalized.get('open_access_pdf')
        )