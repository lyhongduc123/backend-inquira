"""
Utility module for logging and saving retrieved paper results to JSON files.
Useful for debugging and analyzing retrieval results.
"""
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from app.retriever.schemas import NormalizedPaperResult
from app.extensions.logger import create_logger

logger = create_logger(__name__)


def save_retrieval_results(
    results: List[NormalizedPaperResult],
    output_dir: str = "retrieval_logs",
    query: Optional[str] = None,
    provider: Optional[str] = None
) -> str:
    """
    Save retrieved paper results to a JSON file for debugging and analysis.
    
    Args:
        results: List of NormalizedPaperResult dictionaries from retrieval
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
    
    serializable_results = []
    for result in results:
        clean_result = {}
        for key, value in result.model_dump().items():
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
            "results_with_pdf_url": sum(1 for r in serializable_results if r.get("pdf_url")),
            "results_open_access": sum(1 for r in serializable_results if r.get("is_open_access"))
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