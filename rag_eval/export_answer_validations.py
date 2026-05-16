from __future__ import annotations

import argparse
import asyncio
import csv
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import select
from sqlalchemy.orm import joinedload

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.db.database import async_session, init_db
from app.extensions.logger import create_logger
from app.models.answer_vaidations import DBAnswerValidation
from app.models.papers import DBPaper

logger = create_logger(__name__)


DEFAULT_DATASET_PATH = (
    Path(__file__).resolve().parent
    / "evals"
    / "datasets"
    / "litqa_answer_validations.csv"
)


def _normalize_question(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def _iter_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = line.strip()
            if not payload:
                continue
            item = json.loads(payload)
            if isinstance(item, dict):
                rows.append(item)
    return rows


def _serialize_cell(value: Any) -> str:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    if value is None:
        return ""
    return str(value)


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "validation_id",
        "dataset",
        "paper_ids",
        "ground_truth_paper_ids",
        "doi",
        "question",
        "answer",
        "contexts",
        "key_passage",
        "ground_truth",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _serialize_cell(row.get(key)) for key in fieldnames})


DOI_PATTERN = re.compile(r"^10\.\d{4,9}/\S+$", flags=re.IGNORECASE)
DOI_URL_PREFIX = re.compile(r"^https?://(dx\.)?doi\.org/", flags=re.IGNORECASE)
DOI_PREFIX = re.compile(r"^doi:\s*", flags=re.IGNORECASE)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _normalize_doi(raw: str) -> Optional[str]:
    value = (raw or "").strip()
    if not value:
        return None
    value = DOI_URL_PREFIX.sub("", value)
    value = DOI_PREFIX.sub("", value).strip().rstrip("/")
    if DOI_PATTERN.fullmatch(value):
        return value.lower()
    return None


def _extract_doi_from_external_ids(external_ids: Any) -> Optional[str]:
    if isinstance(external_ids, str):
        return _normalize_doi(external_ids)
    if not isinstance(external_ids, dict):
        return None
    for key in ("DOI", "doi"):
        if key in external_ids:
            doi = _normalize_doi(str(external_ids.get(key) or ""))
            if doi:
                return doi
    for value in external_ids.values():
        doi = _normalize_doi(str(value or ""))
        if doi:
            return doi
    return None


def _extract_contexts_from_research_content(context_used: str) -> List[str]:
    if not context_used or "--- RESEARCH CONTENT ---" not in context_used:
        return []
        
    content_section = context_used.split("--- RESEARCH CONTENT ---")[1].strip()
    raw_blocks = content_section.split("SOURCE_ID:")
    
    contexts: List[str] = []
    for block in raw_blocks:
        block = block.strip()
        if not block:
            continue
            
        chunk_text = f"SOURCE_ID: {block}"
        contexts.append(chunk_text)
        
    return contexts


def _parse_reference_legend(context_used: str) -> Dict[int, Dict[str, str]]:
    """Parse REFERENCE LEGEND section from context_used field."""
    legend: Dict[int, Dict[str, str]] = {}
    
    if "--- REFERENCE LEGEND ---" not in context_used:
        return legend
    
    legend_section = context_used.split("--- REFERENCE LEGEND ---")[1]
    if "--- RESEARCH CONTENT ---" in legend_section:
        legend_section = legend_section.split("--- RESEARCH CONTENT ---")[0]
    
    lines = legend_section.strip().split("\n")
    for line in lines:
        if line.startswith("PAPER ID:"):
            try:
                parts = line.split(" | ")
                paper_idx = int(parts[0].replace("PAPER ID:", "").strip())
                entry: Dict[str, str] = {}
                
                for part in parts[1:]:
                    if part.startswith("Title:"):
                        entry["title"] = part.replace("Title:", "").strip()
                    elif part.startswith("DOI:"):
                        entry["doi"] = part.replace("DOI:", "").strip()
                    elif part.startswith("APA:"):
                        entry["apa"] = part.replace("APA:", "").strip()
                
                if entry:
                    legend[paper_idx] = entry
            except (ValueError, IndexError):
                continue
    
    return legend


def _extract_source_ids_from_content(context_used: str) -> List[int]:
    """Extract SOURCE_ID references from RESEARCH CONTENT section."""
    source_ids: List[int] = []
    
    if "--- RESEARCH CONTENT ---" not in context_used:
        return source_ids
    
    content_section = context_used.split("--- RESEARCH CONTENT ---")[1]
    
    for line in content_section.split("\n"):
        if line.startswith("SOURCE_ID:"):
            try:
                source_id = int(line.replace("SOURCE_ID:", "").strip())
                if source_id not in source_ids:
                    source_ids.append(source_id)
            except ValueError:
                continue
    
    return source_ids

def _extract_paper_info_from_context(
    context_used: str, 
    doi_to_paper_ids: Dict[str, List[str]]
) -> Tuple[List[int], List[int], Dict[str, Any]]:
    legend = _parse_reference_legend(context_used)
    used_source_ids = _extract_source_ids_from_content(context_used)
    
    if not legend:
        return [], [], {}

    avail_index_ids = list(legend.keys())
    chunked_index_ids = list(set(used_source_ids))
    
    mapping_info = {}

    for source_idx, entry in legend.items():
        raw_doi = entry.get("doi", "").strip()
        if not raw_doi:
            continue
            
        normalized_doi = _normalize_doi(raw_doi) 
        if not normalized_doi:
            continue
        
        mapped_paper_ids = doi_to_paper_ids.get(normalized_doi, [])
        mapping_info[str(source_idx)] = {
            "legend_doi": raw_doi,
            "database_paper_ids": mapped_paper_ids
        }
            
    return avail_index_ids, chunked_index_ids, mapping_info


def _extract_paper_ids_from_context_fallback(validation: DBAnswerValidation) -> List[str]:
    """Fallback: old format using context_chunks for exists_paper_ids."""
    paper_ids: List[str] = []
    chunks = (
        validation.context_chunks if isinstance(validation.context_chunks, list) else []
    )
    exists_paper_ids = set()
    for item in chunks:
        if not isinstance(item, dict):
            continue
        for key in ("paper_id", "paperId", "source_id", "sourceId", "id"):
            value = str(item.get(key) or "").strip()
            if value:
                if value not in exists_paper_ids:
                    exists_paper_ids.add(value)
                    paper_ids.append(value)
                break

    return list(dict.fromkeys(paper_ids))


def _load_litqa_map(litqa_path: Path) -> Dict[str, Dict[str, Any]]:
    if not litqa_path.exists():
        return {}
    litqa_map: Dict[str, Dict[str, Any]] = {}
    for row in _iter_jsonl(litqa_path):
        question = str(row.get("question") or "").strip()
        if not question:
            continue
        doi_list: List[str] = []
        for source in row.get("sources") or []:
            doi = _normalize_doi(str(source))
            if doi:
                doi_list.append(doi)
        litqa_map[_normalize_question(question)] = {
            "question": question,
            "ideal": str(row.get("ideal") or "").strip(),
            "key_passage": str(row.get("key-passage") or "").strip(),
            "doi": list(dict.fromkeys(doi_list)),
        }
    return litqa_map


def _load_scifact_map(
    scifact_queries_path: Path,
    scifact_qrels_path: Path,
) -> Dict[str, Dict[str, Any]]:
    if not scifact_queries_path.exists() or not scifact_qrels_path.exists():
        return {}

    qrels_by_query: Dict[str, List[str]] = {}
    with scifact_qrels_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            qid = str(row.get("query-id") or "").strip()
            cid = str(row.get("corpus-id") or "").strip()
            score = str(row.get("score") or "0").strip()
            if not qid or not cid or score == "0":
                continue
            qrels_by_query.setdefault(qid, []).append(cid)

    scifact_map: Dict[str, Dict[str, Any]] = {}
    for row in _iter_jsonl(scifact_queries_path):
        query_id = str(row.get("_id") or "").strip()
        text = str(row.get("text") or "").strip()
        if not query_id or not text:
            continue
        scifact_map[_normalize_question(text)] = {
            "query_id": query_id,
            "relevant_corpus_ids": list(
                dict.fromkeys(qrels_by_query.get(query_id, []))
            ),
        }
    return scifact_map


def _to_dataset_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    dataset_rows: List[Dict[str, Any]] = []
    for item in rows:
        query = str(item.get("question") or "").strip()
        answer = str(item.get("answer") or "").strip()
        contexts = item.get("contexts") or []
        if not query or not answer or not isinstance(contexts, list) or not contexts:
            logger.warning(f"Skipping validation_id {item.get('validation_id')} due to missing query, answer, or contexts.")
            continue

        paper_ids = (
            item.get("paper_ids") if isinstance(item.get("paper_ids"), list) else []
        )
        doi = item.get("doi") if isinstance(item.get("doi"), list) else []

        dataset_rows.append(
            {
                "validation_id": item.get("validation_id"),
                "dataset": item.get("dataset"),
                "paper_ids": paper_ids,
                "ground_truth_paper_ids": (
                    item.get("ground_truth_paper_ids")
                    if isinstance(item.get("ground_truth_paper_ids"), list)
                    else []
                ),
                "doi": doi,
                "question": query,
                "answer": answer,
                "contexts": contexts,
                "key_passage": str(item.get("key_passage") or "").strip(),
                "ground_truth": str(item.get("ground_truth") or "").strip(),
            }
        )
    return dataset_rows


async def _fetch_export_rows(
    *,
    dataset: str,
    max_samples: int | None,
    litqa_map: Dict[str, Dict[str, Any]],
    scifact_map: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    await init_db()
    session = async_session()
    try:
        doi_to_paper_ids: Dict[str, List[str]] = {}
        papers = (await session.execute(select(DBPaper))).scalars().all()
        for paper in papers:
            paper_doi = _extract_doi_from_external_ids(paper.external_ids)
            if paper_doi:
                doi_to_paper_ids.setdefault(paper_doi, []).append(paper.paper_id)

        validations = (
            (
                await session.execute(
                    select(DBAnswerValidation)
                    .options(joinedload(DBAnswerValidation.message))
                    .where(DBAnswerValidation.status == "completed")
                    .order_by(DBAnswerValidation.created_at.desc())
                )
            )
            .scalars()
            .all()
        )

        rows: List[Dict[str, Any]] = []
        for validation in validations:
            question = str(validation.query_text or "").strip()
            if not question:
                continue
            norm_question = _normalize_question(question)

            matched_dataset: str | None = None
            litqa_entry = litqa_map.get(norm_question)
            scifact_entry = scifact_map.get(norm_question)
            if dataset in ("litqa", "both") and litqa_entry:
                matched_dataset = "litqa"
            elif dataset in ("scifact", "both") and scifact_entry:
                matched_dataset = "scifact"

            if not matched_dataset:
                continue

            answer = str(
                validation.message.content if validation.message else ""
            ).strip()
            contexts = _extract_contexts_from_research_content(str(validation.context_used or ""))
            
            exists_paper_ids = _extract_paper_ids_from_context_fallback(validation)
            context_used_str = str(validation.context_used or "")
            avail_index_ids, chunked_index_ids, mapping_info = _extract_paper_info_from_context(
                context_used_str, doi_to_paper_ids
            )

            if not answer or not contexts:
                continue

            key_passage = ""
            source_dois: List[str] = []
            ground_truth = ""
            indexed_paper_ids: List[str] = []

            if matched_dataset == "litqa" and litqa_entry:
                key_passage = str(litqa_entry.get("key_passage") or "").strip()
                ground_truth = str(
                    litqa_entry.get("ideal") or key_passage or ""
                ).strip()
                source_dois = [
                    str(x).strip()
                    for x in (litqa_entry.get("doi") or [])
                    if str(x).strip()
                ]

                for source_doi in source_dois:
                    indexed_paper_ids.extend(doi_to_paper_ids.get(source_doi, []))
                indexed_paper_ids = list(dict.fromkeys(indexed_paper_ids))

            hash_to_index: Dict[str, int] = {}
            doi_to_index: Dict[str, int] = {}
            
            legend = _parse_reference_legend(context_used_str)
            for idx, entry in legend.items():
                norm_doi = _normalize_doi(entry.get("doi", ""))
                if norm_doi:
                    doi_to_index[norm_doi] = idx
                    for h in doi_to_paper_ids.get(norm_doi, []):
                        hash_to_index[h] = idx

            final_paper_ids = []
            if chunked_index_ids:
                final_paper_ids = list(chunked_index_ids)
            else:
                for pid in exists_paper_ids:
                    if pid in hash_to_index:
                        final_paper_ids.append(hash_to_index[pid])
                    else:
                        final_paper_ids.append(pid)
                final_paper_ids = list(dict.fromkeys(final_paper_ids))

            final_ground_truth_ids = []
            for s_doi in source_dois:
                norm_s_doi = _normalize_doi(s_doi)
                if norm_s_doi in doi_to_index:
                    final_ground_truth_ids.append(doi_to_index[norm_s_doi])
            
            if not final_ground_truth_ids:
                for pid in indexed_paper_ids:
                    if pid in hash_to_index:
                        final_ground_truth_ids.append(hash_to_index[pid])
                    else:
                        final_ground_truth_ids.append(pid)
            final_ground_truth_ids = list(dict.fromkeys(final_ground_truth_ids))

            rows.append(
                {
                    "validation_id": validation.id,
                    "dataset": matched_dataset,
                    "question": question,
                    "enhanced_query": validation.enhanced_query,
                    "answer": answer,
                    "contexts": contexts,
                    "paper_ids": final_paper_ids,
                    "ground_truth_paper_ids": final_ground_truth_ids,
                    "doi": source_dois,
                    "key_passage": key_passage,
                    "ground_truth": ground_truth,
                    "avail_paper_ids": avail_index_ids,
                    "legend_doi_mapping": mapping_info,
                }
            )

            if max_samples is not None and len(rows) >= max_samples:
                break

        return rows
    finally:
        await session.close()


async def _main_async() -> None:
    parser = argparse.ArgumentParser(
        description="Export answer_validations to a CSV dataset for rag_eval"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="litqa",
        choices=["litqa", "scifact", "both"],
        help="Dataset scope used for query matching",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap for exported samples",
    )
    parser.add_argument(
        "--dataset-output",
        type=str,
        default=str(DEFAULT_DATASET_PATH),
        help="Output CSV dataset file",
    )
    parser.add_argument(
        "--litqa-path",
        type=str,
        default=str(
            Path(__file__).resolve().parent.parent
            / "data"
            / "litqa"
            / "litqa-v2-public.jsonl"
        ),
    )
    parser.add_argument(
        "--scifact-queries-path",
        type=str,
        default=str(
            Path(__file__).resolve().parent.parent
            / "data"
            / "beir_datasets"
            / "scifact"
            / "queries.jsonl"
        ),
    )
    parser.add_argument(
        "--scifact-qrels-path",
        type=str,
        default=str(
            Path(__file__).resolve().parent.parent
            / "data"
            / "beir_datasets"
            / "scifact"
            / "qrels"
            / "test.tsv"
        ),
    )

    args = parser.parse_args()

    litqa_map = _load_litqa_map(Path(args.litqa_path))
    scifact_map = _load_scifact_map(
        scifact_queries_path=Path(args.scifact_queries_path),
        scifact_qrels_path=Path(args.scifact_qrels_path),
    )

    export_rows = await _fetch_export_rows(
        dataset=args.dataset,
        max_samples=args.max_samples,
        litqa_map=litqa_map,
        scifact_map=scifact_map,
    )
    dataset_rows = _to_dataset_rows(export_rows)

    dataset_output = Path(args.dataset_output)
    if dataset_output.suffix.lower() == ".jsonl":
        dataset_output.parent.mkdir(parents=True, exist_ok=True)
        with dataset_output.open("w", encoding="utf-8") as handle:
            for row in dataset_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    else:
        _write_csv(dataset_output, dataset_rows)

    print(f"Exported {len(dataset_rows)} rows to {dataset_output}")


if __name__ == "__main__":
    asyncio.run(_main_async())
