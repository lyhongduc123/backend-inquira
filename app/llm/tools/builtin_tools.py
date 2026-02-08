"""
Built-in tools for the LLM agent
"""
from app.llm.tools.registry import tool_registry
from app.llm.tools.schemas import Tool, ToolParameter, ParameterType
from typing import List, Dict, Any, Optional
from app.extensions.logger import create_logger
from sqlalchemy.ext.asyncio import AsyncSession

logger = create_logger(__name__)


# ============================================================================
# COMPARISON TOOL
# ============================================================================

async def compare_papers_tool(
    paper_ids: List[str],
    comparison_aspects: Optional[List[str]] = None,
    db_session: Optional[AsyncSession] = None
) -> Dict[str, Any]:
    """
    Compare multiple research papers across different aspects
    
    Args:
        paper_ids: List of paper IDs to compare
        comparison_aspects: Aspects to compare (methodology, results, datasets, etc.)
        db_session: Database session (injected from context)
    
    Returns:
        Comparison results
    """
    from app.retriever.paper_repository import PaperRepository
    from app.llm import get_llm_service
    
    logger.info(f"Comparing papers: {paper_ids} on aspects: {comparison_aspects}")
    
    if not db_session:
        return {
            "error": "Database session not available",
            "papers_compared": 0
        }
    
    # Fetch papers from database
    repo = PaperRepository(db_session)
    papers = []
    
    for paper_id in paper_ids:
        paper = await repo.get_paper_by_id(paper_id)
        if paper:
            papers.append({
                "id": str(paper.id),
                "title": paper.title,
                "abstract": paper.abstract,
                "year": paper.publication_date.year if paper.publication_date else None,
                "citation_count": paper.citation_count or 0
            })
    
    if not papers:
        return {
            "error": "No valid papers found",
            "papers_compared": 0
        }
    
    # Use LLM service to compare papers
    comparison = await get_llm_service().compare_papers(
        papers_data=papers,
        comparison_aspects=comparison_aspects or ["methodology", "findings", "limitations"]
    )
    
    return {
        "comparison": comparison.comparison,
        "papers_compared": len(papers),
        "aspects": comparison.comparison_aspects,
        "papers": papers
    }


# Register comparison tool
comparison_tool = Tool(
    name="compare_papers",
    description="Compare multiple research papers across different aspects like methodology, results, datasets, or limitations. Use this when user asks to compare papers or wants to see differences/similarities between papers.",
    parameters=[
        ToolParameter(
            name="paper_ids",
            type=ParameterType.ARRAY,
            description="List of paper IDs to compare (minimum 2 papers)",
            required=True,
            items={"type": "string"}
        ),
        ToolParameter(
            name="comparison_aspects",
            type=ParameterType.ARRAY,
            description="Aspects to compare: methodology, results, datasets, limitations, novelty, etc.",
            required=False,
            items={"type": "string"}
        )
    ]
)
# Tool will be registered on first use
_comparison_tool = (comparison_tool, compare_papers_tool)


# ============================================================================
# OPINION METER TOOL
# ============================================================================

async def opinion_meter_tool(
    topic: str,
    paper_ids: Optional[List[str]] = None,
    db_session: Optional[AsyncSession] = None
) -> Dict[str, Any]:
    """
    Measure the distribution of opinions/stances on a topic across papers
    
    Args:
        topic: The topic or question to analyze
        paper_ids: Optional list of specific papers to analyze
        db_session: Database session (injected from context)
    
    Returns:
        Opinion distribution analysis
    """
    from app.retriever.paper_repository import PaperRepository
    from app.llm import get_llm_service
    
    logger.info(f"Analyzing opinions on topic: {topic}")
    
    if not db_session:
        return {
            "error": "Database session not available",
            "topic": topic
        }
    
    repo = PaperRepository(db_session)
    papers = []
    
    # If specific papers provided, fetch them
    if paper_ids:
        for paper_id in paper_ids:
            paper = await repo.get_paper_by_id(paper_id)
            if paper:
                papers.append({
                    "id": str(paper.id),
                    "title": paper.title,
                    "abstract": paper.abstract,
                    "year": paper.publication_date.year if paper.publication_date else None
                })
    else:
        # Otherwise, search for papers on the topic
        from app.retriever.paper_retriever import PaperRetriever
        retriever = PaperRetriever(db_session)
        results = await retriever.search_papers(query=topic, limit=10)
        
        for paper in results.papers:
            papers.append({
                "id": str(paper.id),
                "title": paper.title,
                "abstract": paper.abstract,
                "year": paper.publication_date.year if paper.publication_date else None
            })
    
    if not papers:
        return {
            "error": "No papers found for the topic",
            "topic": topic
        }
    
    # Analyze opinions using LLM
    opinions = {
        "topic": topic,
        "total_papers": len(papers),
        "stances": {
            "positive": [],
            "negative": [],
            "neutral": [],
            "mixed": []
        },
        "summary": ""
    }
    
    # Use LLM to classify each paper's stance
    for paper in papers:
        prompt = f"""
        Analyze the stance of this paper on the topic: "{topic}"
        
        Paper Title: {paper['title']}
        Abstract: {paper['abstract']}
        
        Classify the stance as: POSITIVE, NEGATIVE, NEUTRAL, or MIXED
        Provide a brief justification.
        
        Format:
        STANCE: [POSITIVE/NEGATIVE/NEUTRAL/MIXED]
        JUSTIFICATION: [brief explanation]
        """
        
        from app.llm.provider import LLMProvider
        provider = get_llm_service().llm_provider
        response = provider.simple_prompt(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        
        # Parse response
        stance = "neutral"
        justification = ""
        
        from app.extensions.stream import get_simple_response_content
        content = get_simple_response_content(response)
        
        if "POSITIVE" in content.upper():
            stance = "positive"
        elif "NEGATIVE" in content.upper():
            stance = "negative"
        elif "MIXED" in content.upper():
            stance = "mixed"
        
        # Extract justification
        lines = content.split('\n')
        for line in lines:
            if "JUSTIFICATION:" in line.upper():
                justification = line.split(':', 1)[1].strip()
                break
        
        opinions["stances"][stance].append({
            "paper_id": paper["id"],
            "title": paper["title"],
            "year": paper["year"],
            "justification": justification
        })
    
    # Generate summary
    opinions["summary"] = f"""
    Opinion Distribution for "{topic}":
    - Positive: {len(opinions['stances']['positive'])} papers ({len(opinions['stances']['positive'])/len(papers)*100:.1f}%)
    - Negative: {len(opinions['stances']['negative'])} papers ({len(opinions['stances']['negative'])/len(papers)*100:.1f}%)
    - Neutral: {len(opinions['stances']['neutral'])} papers ({len(opinions['stances']['neutral'])/len(papers)*100:.1f}%)
    - Mixed: {len(opinions['stances']['mixed'])} papers ({len(opinions['stances']['mixed'])/len(papers)*100:.1f}%)
    """
    
    return opinions


# Register opinion meter tool
opinion_meter = Tool(
    name="opinion_meter",
    description="Analyze the distribution of opinions, stances, or perspectives on a specific topic across research papers. Use this when user asks about consensus, controversy, or different viewpoints on a topic.",
    parameters=[
        ToolParameter(
            name="topic",
            type=ParameterType.STRING,
            description="The topic, question, or claim to analyze opinions about",
            required=True
        ),
        ToolParameter(
            name="paper_ids",
            type=ParameterType.ARRAY,
            description="Optional list of specific paper IDs to analyze. If not provided, will search for relevant papers.",
            required=False,
            items={"type": "string"}
        )
    ]
)
# Tool will be registered on first use
_opinion_meter = (opinion_meter, opinion_meter_tool)


# ============================================================================
# CITATION ANALYSIS TOOL
# ============================================================================

async def citation_analysis_tool(
    paper_id: str,
    analysis_type: str = "impact",
    db_session: Optional[AsyncSession] = None
) -> Dict[str, Any]:
    """
    Analyze citation patterns and impact of a paper
    
    Args:
        paper_id: The paper ID to analyze
        analysis_type: Type of analysis (impact, trends, network)
        db_session: Database session
    
    Returns:
        Citation analysis results
    """
    from app.retriever.paper_repository import PaperRepository
    
    logger.info(f"Analyzing citations for paper: {paper_id}, type: {analysis_type}")
    
    if not db_session:
        return {"error": "Database session not available"}
    
    repo = PaperRepository(db_session)
    paper = await repo.get_paper_by_id(paper_id)
    
    if not paper:
        return {"error": f"Paper {paper_id} not found"}
    
    result = {
        "paper_id": paper_id,
        "title": paper.title,
        "citation_count": paper.citation_count or 0,
        "publication_year": paper.publication_date.year if paper.publication_date else None,
        "analysis_type": analysis_type
    }
    
    # Calculate citations per year if publication year is available
    if result["publication_year"]:
        from datetime import datetime
        years_since_pub = datetime.now().year - result["publication_year"]
        if years_since_pub > 0:
            result["citations_per_year"] = result["citation_count"] / years_since_pub
            result["impact_level"] = "high" if result["citations_per_year"] > 10 else "medium" if result["citations_per_year"] > 2 else "low"
    
    return result


# Register citation analysis tool
citation_analysis = Tool(
    name="citation_analysis",
    description="Analyze the citation impact and patterns of a research paper. Use when user asks about paper impact, influence, or citation metrics.",
    parameters=[
        ToolParameter(
            name="paper_id",
            type=ParameterType.STRING,
            description="The ID of the paper to analyze",
            required=True
        ),
        ToolParameter(
            name="analysis_type",
            type=ParameterType.STRING,
            description="Type of analysis: 'impact' (default), 'trends', or 'network'",
            required=False,
            enum=["impact", "trends", "network"]
        )
    ]
)
# Tool will be registered on first use
_citation_analysis = (citation_analysis, citation_analysis_tool)


# ============================================================================
# RESEARCH TRENDS TOOL
# ============================================================================

async def research_trends_tool(
    topic: str,
    time_range: str = "recent",
    db_session: Optional[AsyncSession] = None
) -> Dict[str, Any]:
    """
    Analyze research trends on a topic over time
    
    Args:
        topic: The research topic to analyze
        time_range: Time range (recent, 5years, 10years)
        db_session: Database session
    
    Returns:
        Trend analysis results
    """
    from app.retriever.paper_retriever import PaperRetriever
    from datetime import datetime, timedelta
    
    logger.info(f"Analyzing research trends for: {topic}, range: {time_range}")
    
    if not db_session:
        return {"error": "Database session not available"}
    
    retriever = PaperRetriever(db_session)
    
    # Determine date range
    now = datetime.now()
    if time_range == "recent":
        start_year = now.year - 2
    elif time_range == "5years":
        start_year = now.year - 5
    else:  # 10years
        start_year = now.year - 10
    
    # Search for papers
    results = await retriever.search_papers(query=topic, limit=50)
    
    # Organize by year
    papers_by_year = {}
    for paper in results.papers:
        if paper.publication_date:
            year = paper.publication_date.year
            if year >= start_year:
                if year not in papers_by_year:
                    papers_by_year[year] = []
                papers_by_year[year].append({
                    "title": paper.title,
                    "citations": paper.citation_count or 0
                })
    
    # Calculate trends
    trend_data = {
        "topic": topic,
        "time_range": time_range,
        "total_papers": len(results.papers),
        "papers_by_year": {str(year): len(papers) for year, papers in papers_by_year.items()},
        "trend": "increasing" if len(papers_by_year.get(now.year, [])) > len(papers_by_year.get(start_year, [])) else "decreasing"
    }
    
    return trend_data


# Register research trends tool
research_trends = Tool(
    name="research_trends",
    description="Analyze research trends and publication patterns on a topic over time. Use when user asks about trends, growth, or temporal patterns in research.",
    parameters=[
        ToolParameter(
            name="topic",
            type=ParameterType.STRING,
            description="The research topic to analyze trends for",
            required=True
        ),
        ToolParameter(
            name="time_range",
            type=ParameterType.STRING,
            description="Time range for analysis: 'recent' (2 years), '5years', or '10years'",
            required=False,
            enum=["recent", "5years", "10years"]
        )
    ]
)
# Tool will be registered on first use
_research_trends = (research_trends, research_trends_tool)


# Lazy registration to avoid slow startup
_tools_registered = False

def register_builtin_tools():
    """Register all built-in tools (lazy initialization)"""
    global _tools_registered
    if _tools_registered:
        return
    
    tool_registry.register(*_comparison_tool)
    tool_registry.register(*_opinion_meter)
    tool_registry.register(*_citation_analysis)
    tool_registry.register(*_research_trends)
    
    _tools_registered = True
    logger.info("Built-in tools registered successfully")
