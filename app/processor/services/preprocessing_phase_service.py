"""
Phase-based preprocessing services for queue execution.

This service splits heavy preprocessing work into independently triggerable phases:
- Embedding backfill
- Citation linking
- Author trust metrics
- Paper tag computation
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.extensions.logger import create_logger
from app.processor.preprocessing_repository import PreprocessingRepository
from app.processor.preprocessing_service import PreprocessingService
from app.retriever.service import RetrievalService
from app.domain.papers.linking_service import PaperLinkingService
from app.processor.services.zeroshot_tagger import ZeroShotTaggerService

logger = create_logger(__name__)


DEFAULT_TOPIC_LABELS: List[str] = [
    "Artificial Intelligence",
    "Machine Learning",
    "Natural Language Processing",
    "Computer Vision",
    "Data Mining",
    "Healthcare",
    "Economics",
    "Social Science",
    "Physics",
    "Biology",
]


class PreprocessingPhaseService:
    """Service that runs preprocessing phases independently."""

    def __init__(
        self,
        preprocessing_service: PreprocessingService,
        preprocessing_repository: PreprocessingRepository,
        retriever: RetrievalService,
        linking_service: PaperLinkingService,
        zeroshot_tagger_service: ZeroShotTaggerService,
    ):
        self.preprocessing_service = preprocessing_service
        self.preprocessing_repository = preprocessing_repository
        self.retriever = retriever
        self.linking_service = linking_service
        self.zeroshot_tagger_service = zeroshot_tagger_service

    async def run_embedding_backfill(self) -> Dict[str, int]:
        """Run embedding backfill for papers missing embeddings."""
        papers = await self.preprocessing_repository.get_papers_missing_embeddings(limit=1000)
        before_count = len(papers)

        await self.preprocessing_service._generate_missing_embeddings(state=None)

        after_papers = await self.preprocessing_repository.get_papers_missing_embeddings(limit=1000)
        after_count = len(after_papers)

        return {
            "considered": before_count,
            "remaining_missing": after_count,
            "updated": max(0, before_count - after_count),
        }

    async def run_citation_linking(
        self,
        limit: int = 200,
        references_limit: int = 50,
        citations_limit: int = 50,
    ) -> Dict[str, int]:
        """
        Build citation links for existing papers by fetching references from S2.

        Notes:
        - Re-running is safe because inserts use ON CONFLICT DO NOTHING.
        - Only references where cited paper already exists will be linked.
        """
        papers = await self.preprocessing_repository.get_papers_for_citation_linking(limit=limit)
        citation_map: Dict[str, set[str]] = {}
        related_paper_ids: set[str] = set()
        source_papers = 0

        relation_fields = (
            "paperId,corpusId,title,abstract,authors,year,publicationDate,venue,"
            "citationCount,influentialCitationCount,url,openAccessPdf,isOpenAccess,"
            "externalIds,fieldsOfStudy,publicationTypes,isInfluential,contexts,intents"
        )

        for paper in papers:
            try:
                source_paper_id = str(paper.paper_id)

                refs_response = await self.retriever.get_paper_references(
                    paper_id=source_paper_id,
                    limit=references_limit,
                    offset=0,
                    fields=relation_fields,
                )
                ref_ids = self._extract_reference_ids(refs_response)

                cits_response = await self.retriever.get_paper_citations(
                    paper_id=source_paper_id,
                    limit=citations_limit,
                    offset=0,
                    fields=relation_fields,
                )
                citing_ids = self._extract_citation_ids(cits_response)

                has_links = False
                if ref_ids:
                    citation_map.setdefault(source_paper_id, set()).update(ref_ids)
                    related_paper_ids.update(ref_ids)
                    has_links = True

                if citing_ids:
                    for citing_id in citing_ids:
                        citation_map.setdefault(citing_id, set()).add(source_paper_id)
                    related_paper_ids.update(citing_ids)
                    has_links = True

                if has_links:
                    source_papers += 1
            except Exception as exc:
                logger.warning(
                    "[PreprocessingPhase] Failed fetching references for %s: %s",
                    paper.paper_id,
                    exc,
                )

        # Enrich related papers with OpenAlex and index them before citation linking.
        indexed_related_papers = 0
        if related_paper_ids:
            try:
                related_ids_list = sorted(related_paper_ids)
                enriched_related_papers = []
                chunk_size = 100

                for i in range(0, len(related_ids_list), chunk_size):
                    chunk_ids = related_ids_list[i:i + chunk_size]
                    chunk_papers = await self.retriever.get_multiple_papers(chunk_ids)
                    if chunk_papers:
                        enriched_related_papers.extend(chunk_papers)

                if enriched_related_papers:
                    created = await self.preprocessing_service.paper_service.batch_create_papers_from_schema(
                        papers=enriched_related_papers,
                        enrich=True,
                    )
                    indexed_related_papers = len(created)
            except Exception as exc:
                logger.warning(
                    "[PreprocessingPhase] Failed enriching/indexing related papers: %s",
                    exc,
                )

        citation_data: List[tuple[str, List[str]]] = [
            (citing_id, sorted(cited_ids))
            for citing_id, cited_ids in citation_map.items()
            if cited_ids
        ]

        if not citation_data:
            return {
                "candidate_papers": len(papers),
                "source_papers": 0,
                "indexed_related_papers": indexed_related_papers,
                "citation_links_attempted": 0,
                "citation_links_created": 0,
            }

        citation_links_attempted = sum(len(ref_ids) for _, ref_ids in citation_data)
        citation_links_created = await self.linking_service.batch_link_citations_references(
            citation_data=citation_data
        )

        return {
            "candidate_papers": len(papers),
            "source_papers": source_papers,
            "indexed_related_papers": indexed_related_papers,
            "citation_links_attempted": citation_links_attempted,
            "citation_links_created": int(citation_links_created),
        }

    async def run_author_metrics(
        self,
        only_unprocessed: bool = False,
        conflict_threshold_percent: float = 50.0,
        batch_size: int = 200,
    ) -> Dict[str, int]:
        """Run author trust metric computation phase."""
        return await self.preprocessing_service.compute_all_author_metrics(
            only_unprocessed=only_unprocessed,
            conflict_threshold_percent=conflict_threshold_percent,
            batch_size=batch_size,
        )

    async def run_paper_tagging(
        self,
        limit: int = 200,
        only_missing_tags: bool = True,
        candidate_labels: Optional[List[str]] = None,
        category: str = "topic",
        min_confidence: float = 50.0,
        max_tags_per_paper: int = 3,
    ) -> Dict[str, int]:
        """
        Compute and persist zero-shot tags for papers.

        Returns aggregate stats for queue status/monitoring.
        """
        labels = candidate_labels or DEFAULT_TOPIC_LABELS
        papers = await self.preprocessing_repository.get_papers_for_tagging(
            limit=limit,
            only_missing_tags=only_missing_tags,
        )

        updated = 0
        skipped = 0
        failed = 0

        for paper in papers:
            try:
                content = self._build_tagging_content(paper.title, paper.abstract)
                if not content:
                    skipped += 1
                    continue

                raw_tags = self.zeroshot_tagger_service.compute_tags(
                    content,
                    labels,
                    category=category,
                )

                filtered_tags = [t for t in raw_tags if float(t.get("confidence", 0.0)) >= min_confidence]
                filtered_tags = filtered_tags[: max(1, max_tags_per_paper)]

                await self.preprocessing_repository.update_paper_tags(
                    paper_id=str(paper.paper_id),
                    tags=filtered_tags,
                )
                updated += 1
            except Exception as exc:
                failed += 1
                logger.error(
                    "[PreprocessingPhase] Failed paper tagging for %s: %s",
                    paper.paper_id,
                    exc,
                    exc_info=True,
                )

        return {
            "considered": len(papers),
            "updated": updated,
            "skipped": skipped,
            "failed": failed,
        }

    @staticmethod
    def _extract_reference_ids(refs_response: Dict[str, Any]) -> List[str]:
        """Extract referenced paper IDs from multiple possible S2 response shapes."""
        data = refs_response.get("data", []) if isinstance(refs_response, dict) else []
        ref_ids: List[str] = []

        for item in data:
            if not isinstance(item, dict):
                continue

            # Raw S2 shape
            cited_paper = item.get("citedPaper")
            if isinstance(cited_paper, dict) and cited_paper.get("paperId"):
                ref_ids.append(str(cited_paper["paperId"]))
                continue

            # Snake_case mapped shape
            cited_paper = item.get("cited_paper")
            if isinstance(cited_paper, dict) and cited_paper.get("paper_id"):
                ref_ids.append(str(cited_paper["paper_id"]))
                continue

            # Fallback flat shape
            if item.get("paperId"):
                ref_ids.append(str(item["paperId"]))

        return ref_ids

    @staticmethod
    def _extract_citation_ids(citations_response: Dict[str, Any]) -> List[str]:
        """Extract citing paper IDs from multiple possible S2 response shapes."""
        data = citations_response.get("data", []) if isinstance(citations_response, dict) else []
        citing_ids: List[str] = []

        for item in data:
            if not isinstance(item, dict):
                continue

            # Raw S2 shape
            citing_paper = item.get("citingPaper")
            if isinstance(citing_paper, dict) and citing_paper.get("paperId"):
                citing_ids.append(str(citing_paper["paperId"]))
                continue

            # Snake_case mapped shape
            citing_paper = item.get("citing_paper")
            if isinstance(citing_paper, dict) and citing_paper.get("paper_id"):
                citing_ids.append(str(citing_paper["paper_id"]))
                continue

            # Fallback flat shape
            if item.get("paperId"):
                citing_ids.append(str(item["paperId"]))

        return citing_ids

    @staticmethod
    def _build_tagging_content(title: Optional[str], abstract: Optional[str]) -> str:
        """Build robust tagging text from available title/abstract."""
        title_text = (title or "").strip()
        abstract_text = (abstract or "").strip()
        if title_text and abstract_text:
            return f"Title: {title_text}\n\nAbstract: {abstract_text}"
        return title_text or abstract_text
