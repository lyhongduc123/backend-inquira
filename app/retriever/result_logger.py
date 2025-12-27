"""
Utility module for logging and saving retrieved paper results to JSON files.
Useful for debugging and analyzing retrieval results.
"""
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from app.retriever.provider.base_schemas import NormalizedResult
from app.extensions.logger import create_logger

logger = create_logger(__name__)


def save_retrieval_results(
    results: List[NormalizedResult],
    output_dir: str = "retrieval_logs",
    query: Optional[str] = None,
    provider: Optional[str] = None
) -> str:
    """
    Save retrieved paper results to a JSON file for debugging and analysis.
    
    Args:
        results: List of NormalizedResult dictionaries from retrieval
        output_dir: Directory to save logs (default: retrieval_logs)
        query: Optional search query that was used
        provider: Optional provider name (e.g., "semantic_scholar")
        
    Returns:
        Path to the saved JSON file
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    provider_str = f"_{provider}" if provider else ""
    query_str = f"_{query.replace(' ', '_')[:50]}" if query else ""
    filename = f"retrieval_{timestamp}{provider_str}{query_str}.json"
    
    filepath = output_path / filename
    
    # Convert results to serializable format
    serializable_results = []
    for result in results:
        # Create a clean copy of the result
        clean_result = {}
        for key, value in result.items():
            # Handle non-serializable types
            if value is None or isinstance(value, (str, int, float, bool, list, dict)):
                clean_result[key] = value
            else:
                clean_result[key] = str(value)
        serializable_results.append(clean_result)
    
    # Prepare output data with metadata
    output_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "provider": provider,
            "total_results": len(results),
            "results_with_pdf_url": sum(1 for r in results if r.get("pdf_url")),
            "results_open_access": sum(1 for r in results if r.get("is_open_access"))
        },
        "results": serializable_results
    }
    
    # Save to JSON file
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved {len(results)} retrieval results to {filepath}")
    
    # Print summary
    print(f"\n✓ Retrieval Results Summary:")
    print(f"  Total results: {len(results)}")
    print(f"  Results with PDF URL: {output_data['metadata']['results_with_pdf_url']}")
    print(f"  Open access papers: {output_data['metadata']['results_open_access']}")
    print(f"  Saved to: {filepath}")
    
    return str(filepath)


def save_paper_analysis(
    results: List[NormalizedResult],
    output_dir: str = "retrieval_logs"
) -> str:
    """
    Save detailed analysis of retrieval results focusing on PDF availability and metadata.
    
    Args:
        results: List of NormalizedResult dictionaries
        output_dir: Directory to save logs
        
    Returns:
        Path to the saved analysis file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = output_path / f"analysis_{timestamp}.json"
    
    # Analyze results
    analysis = {
        "timestamp": datetime.now().isoformat(),
        "total_papers": len(results),
        "papers_by_status": {
            "has_pdf_url": [],
            "no_pdf_url": [],
            "open_access_no_pdf": [],
            "missing_abstract": [],
            "missing_publication_date": []
        }
    }
    
    for result in results:
        paper_summary = {
            "paper_id": result.get("paper_id"),
            "title": result.get("title"),
            "venue": result.get("venue"),
            "publication_date": result.get("publication_date"),
            "is_open_access": result.get("is_open_access"),
            "pdf_url": result.get("pdf_url"),
            "citation_count": result.get("citation_count")
        }
        
        if result.get("pdf_url"):
            analysis["papers_by_status"]["has_pdf_url"].append(paper_summary)
        else:
            analysis["papers_by_status"]["no_pdf_url"].append(paper_summary)
            
        if result.get("is_open_access") and not result.get("pdf_url"):
            analysis["papers_by_status"]["open_access_no_pdf"].append(paper_summary)
            
        if not result.get("abstract"):
            analysis["papers_by_status"]["missing_abstract"].append(paper_summary)
            
        if not result.get("publication_date"):
            analysis["papers_by_status"]["missing_publication_date"].append(paper_summary)
    
    # Add summary statistics
    analysis["statistics"] = {
        "papers_with_pdf": len(analysis["papers_by_status"]["has_pdf_url"]),
        "papers_without_pdf": len(analysis["papers_by_status"]["no_pdf_url"]),
        "open_access_without_pdf": len(analysis["papers_by_status"]["open_access_no_pdf"]),
        "missing_abstracts": len(analysis["papers_by_status"]["missing_abstract"]),
        "missing_dates": len(analysis["papers_by_status"]["missing_publication_date"]),
        "pdf_coverage_percent": round(
            (len(analysis["papers_by_status"]["has_pdf_url"]) / len(results) * 100) if results else 0,
            2
        ) if results else 0
    }
    
    # Save analysis
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved analysis to {filepath}")
    
    # Print summary
    print(f"\n✓ Detailed Analysis:")
    print(f"  Papers with PDF URLs: {analysis['statistics']['papers_with_pdf']}")
    print(f"  Papers without PDF URLs: {analysis['statistics']['papers_without_pdf']}")
    print(f"  Open Access papers without PDFs: {analysis['statistics']['open_access_without_pdf']}")
    print(f"  PDF Coverage: {analysis['statistics']['pdf_coverage_percent']}%")
    print(f"  Saved to: {filepath}")
    
    return str(filepath)
