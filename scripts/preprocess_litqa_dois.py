"""
Preprocess LitQA papers by DOI only (skip bulk search).

Workflow:
1) Read LitQA JSONL entries from data folder
2) Extract and validate DOI strings from `sources`
3) Fetch papers using Semantic Scholar multiple-papers route via RetrievalService
4) Run selected-paper preprocessing phases (metadata, linking, embeddings, content)

This script is strict/fail-fast by design:
- Any malformed source DOI stops the run
- Any unresolved DOI stops the run
- Any selected open-access paper that fails content processing stops the run

Usage:
    python scripts/preprocess_litqa_dois.py
    python scripts/preprocess_litqa_dois.py --job-id litqa-sync-20260407
    python scripts/preprocess_litqa_dois.py --limit 50
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, final

from app.core import container
from app.core.container import ServiceContainer
from app.core.db.database import db_session_context, init_db
from app.extensions.logger import create_logger

logger = create_logger(__name__)

DOI_PATTERN = re.compile(r"^10\.\d{4,9}/\S+$", flags=re.IGNORECASE)
DOI_URL_PREFIX = re.compile(r"^https?://(dx\.)?doi\.org/", flags=re.IGNORECASE)
DOI_PREFIX = re.compile(r"^doi:\s*", flags=re.IGNORECASE)


@dataclass
class ExtractionResult:
    dois: List[str]
    source_records: int


class JsonlReportWriter:
    """Simple JSONL report writer for sync auditability."""

    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: str, payload: Dict[str, Any]) -> None:
        record = {
            "timestamp": _utc_iso_z(),
            "event": event,
            "payload": payload,
        }
        with self.output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_iso_z() -> str:
    return _utc_now().isoformat().replace("+00:00", "Z")


def _normalize_doi(source: str) -> str:
    """Normalize source string to strict DOI string (no URL prefix, no trailing slash)."""
    raw = source.strip()
    cleaned = DOI_URL_PREFIX.sub("", raw)
    cleaned = DOI_PREFIX.sub("", cleaned).strip()
    cleaned = cleaned.rstrip("/")

    if not DOI_PATTERN.fullmatch(cleaned):
        raise ValueError(f"Invalid DOI source value: {source}")

    return cleaned.lower()


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                yield json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_no}: {exc}") from exc


def extract_litqa_dois(dataset_path: Path) -> ExtractionResult:
    """
    Extract strict DOI strings from LitQA `sources` fields.

    Fail-fast behavior:
    - Missing/non-list `sources` field raises immediately
    - Non-string source values raise immediately
    - Malformed DOI values raise immediately
    - Skip not OA
    """
    if not dataset_path.exists():
        raise FileNotFoundError(f"LitQA dataset not found: {dataset_path}")

    dois: List[str] = []
    records = 0

    for record in _iter_jsonl(dataset_path):
        records += 1
        is_oa = record.get("is_opensource", False)
        if not is_oa:
            continue

        sources = record.get("sources")
        if not isinstance(sources, list):
            raise ValueError(
                f"Invalid record (id={record.get('id')}): `sources` must be a list"
            )

        for source in sources:
            if not isinstance(source, str):
                raise ValueError(
                    f"Invalid source type in record (id={record.get('id')}): {type(source)}"
                )
            doi = _normalize_doi(source)
            dois.append(doi)

    deduped = list(dict.fromkeys(dois))
    return ExtractionResult(dois=deduped, source_records=records)


def _load_exception_index(exception_path: Path) -> Dict[str, Dict[str, Optional[str]]]:
    if not exception_path.exists():
        return {}

    payload = json.loads(exception_path.read_text(encoding="utf-8"))
    rows = payload.get("except", []) if isinstance(payload, dict) else []
    index: Dict[str, Dict[str, Optional[str]]] = {}

    for row in rows:
        if not isinstance(row, dict):
            continue
        doi = _normalize_possible_doi(str(row.get("doi") or ""))
        if not doi:
            continue
        index[doi] = {
            "doi": doi,
            "ver_doi": _normalize_possible_doi(str(row.get("ver_doi") or "")),
            "title": str(row.get("title") or "").strip() or None,
        }

    return index


def _normalize_possible_doi(identifier: str) -> Optional[str]:
    value = str(identifier).strip()
    if not value:
        return None

    lower_value = value.lower()
    if lower_value.startswith("doi:"):
        value = value[4:].strip()
    elif lower_value.startswith("https://doi.org/"):
        value = value[len("https://doi.org/") :].strip()
    elif lower_value.startswith("http://doi.org/"):
        value = value[len("http://doi.org/") :].strip()
    elif lower_value.startswith("https://dx.doi.org/"):
        value = value[len("https://dx.doi.org/") :].strip()
    elif lower_value.startswith("http://dx.doi.org/"):
        value = value[len("http://dx.doi.org/") :].strip()

    value = value.rstrip("/").strip()
    if not value:
        return None

    if value.startswith("10.") and "/" in value:
        return value.lower()

    return None


def _apply_exception_replacements(
    *,
    doi_list: List[str],
    report_writer: JsonlReportWriter,
    exception_path: Path,
) -> List[str]:
    """
    Apply exception strategy:
    - if DOI exists in exception mapping and has `ver_doi` -> replace with `ver_doi`
    - if DOI exists in exception mapping and no `ver_doi` -> filter it out
    - otherwise keep DOI as-is
    """
    exception_index = _load_exception_index(exception_path)
    if not exception_index:
        return doi_list

    output_identifiers: List[str] = []
    replaced_count = 0
    filtered_count = 0

    for doi in doi_list:
        mapping = exception_index.get(doi)
        if not mapping:
            output_identifiers.append(doi)
            continue

        ver_doi = mapping.get("ver_doi")
        if ver_doi:
            output_identifiers.append(ver_doi)
            replaced_count += 1
            report_writer.write(
                event="exception_doi_replaced",
                payload={
                    "litqa_doi": doi,
                    "replacement_ver_doi": ver_doi,
                    "title": mapping.get("title"),
                },
            )
        else:
            filtered_count += 1
            report_writer.write(
                event="exception_doi_filtered",
                payload={
                    "litqa_doi": doi,
                    "reason": "missing_ver_doi",
                    "title": mapping.get("title"),
                },
            )

    deduped = list(dict.fromkeys(output_identifiers))

    logger.info(
        "Exception strategy applied: replaced=%s, filtered=%s, output=%s",
        replaced_count,
        filtered_count,
        len(deduped),
    )
    report_writer.write(
        event="exception_strategy_summary",
        payload={
            "exception_path": str(exception_path),
            "input_count": len(doi_list),
            "replaced_count": replaced_count,
            "filtered_count": filtered_count,
            "output_count": len(deduped),
        },
    )

    return deduped


async def run_sync(
    dataset_path: Path,
    job_id: str,
    limit: int | None,
    report_writer: JsonlReportWriter,
    exception_path: Path,
) -> Dict[str, Any]:
    """Run strict LitQA DOI sync and preprocessing."""
    extraction = extract_litqa_dois(dataset_path)
    doi_list = extraction.dois[:limit] if limit is not None else extraction.dois

    if not doi_list:
        raise ValueError("No DOI values found in LitQA dataset")

    logger.info(
        "Extracted %s unique DOIs from %s records",
        len(doi_list),
        extraction.source_records,
    )
    report_writer.write(
        event="doi_extraction_completed",
        payload={
            "dataset_path": str(dataset_path),
            "records": extraction.source_records,
            "doi_count": len(doi_list),
            "limit": limit,
            "job_id": job_id,
        },
    )

    await init_db()

    async with db_session_context() as session:
        try:
            identifiers_to_process = _apply_exception_replacements(
                doi_list=doi_list,
                report_writer=report_writer,
                exception_path=exception_path,
            )

            if not identifiers_to_process:
                raise ValueError(
                    "No identifiers remain after exception strategy filtering (all exception DOIs missing ver_doi)"
                )

            container = ServiceContainer(session)

            stats = await container.preprocessing_service.process_selected_papers(
                job_id=job_id,
                paper_ids=identifiers_to_process,
                resume=False,
                strict=False,
            )

            report_writer.write(
                event="sync_completed",
                payload={
                    "job_id": job_id,
                    "stats": stats,
                    "doi_count": len(doi_list),
                },
            )
            return stats
        except Exception as exc:
            report_writer.write(
                event="sync_failed",
                payload={
                    "job_id": job_id,
                    "error": str(exc),
                    "doi_count": len(doi_list),
                },
            )
            raise


async def run_check_sync(
    dataset_path: Path,
    report_writer: JsonlReportWriter,
    exception_path: Path,
) -> Dict[str, Any]:
    """Run checking on preprocessed DOIs."""
    extraction = extract_litqa_dois(dataset_path)
    doi_list = extraction.dois

    if not doi_list:
        raise ValueError("No DOI values found in LitQA dataset")

    logger.info(
        "Extracted %s unique DOIs from %s records for checking",
        len(doi_list),
        extraction.source_records,
    )
    report_writer.write(
        event="doi_extraction_completed_for_check",
        payload={
            "dataset_path": str(dataset_path),
            "records": extraction.source_records,
            "doi_count": len(doi_list),
        },
    )

    identifiers_to_check = _apply_exception_replacements(
        doi_list=doi_list,
        report_writer=report_writer,
        exception_path=exception_path,
    )

    async with db_session_context() as session:
        try:
            container = ServiceContainer(session)

            idx = 0
            exist_dois = []
            fulltext_dois = set()
            for doi in identifiers_to_check:
                idx += 1
                logger.info(
                    "Checking DOI %d/%d: %s", idx, len(identifiers_to_check), doi
                )
                paper_exist = (
                    await container.paper_repository.get_paper_by_external_ids(
                        external_ids={"doi": doi},
                    )
                )
                if paper_exist:
                    if paper_exist.is_processed:
                        fulltext_dois.add(doi)
                    exist_dois.append(doi)

            report_writer.write(
                event="check_sync_completed",
                payload={
                    "total_identifiers": len(doi_list),
                    "existing_dois_count": len(exist_dois),
                    "fulltext_dois_count": len(fulltext_dois),
                    "identifiers_after_exception": len(identifiers_to_check),
                    "identifiers_to_check": identifiers_to_check,
                    "existing_dois": exist_dois,
                    "fulltext_dois": list(fulltext_dois),
                },
            )
            return {
                "total_identifiers": len(doi_list),
                "existing_dois_count": len(exist_dois),
                "fulltext_dois_count": len(fulltext_dois),
                "identifiers_after_exception": len(identifiers_to_check),
            }
        except Exception as exc:
            report_writer.write(
                event="check_sync_failed",
                payload={
                    "error": str(exc),
                    "total_identifiers": len(doi_list),
                    "identifiers_after_exception": len(identifiers_to_check),
                },
            )
            raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preprocess LitQA sources via DOI-only selected-paper pipeline"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Run checking on preprocessed dois",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(
            Path(__file__).parent.parent / "data" / "litqa" / "litqa-v2-public.jsonl"
        ),
        help="Path to LitQA JSONL file",
    )
    parser.add_argument(
        "--job-id",
        type=str,
        default=f"litqa-doi-sync-{_utc_now().strftime('%Y%m%d-%H%M%S')}",
        help="Unique preprocessing job id",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of unique DOIs to sync",
    )
    parser.add_argument(
        "--report-file",
        type=str,
        default=str(
            Path("preprocessing_logs")
            / "litqa"
            / f"litqa_doi_sync_{_utc_now().strftime('%Y%m%d-%H%M%S')}.jsonl"
        ),
        help="JSONL report output path",
    )
    parser.add_argument(
        "--exception-file",
        type=str,
        default=str(Path(__file__).parent.parent / "data" / "litqa" / "exception.json"),
        help="Path to LitQA exception mapping JSON file",
    )
    return parser


async def _main_async() -> None:
    parser = build_parser()
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    report_path = Path(args.report_file)
    exception_path = Path(args.exception_file)
    report_writer = JsonlReportWriter(report_path)

    logger.info("Starting LitQA DOI preprocessing sync")
    logger.info("Dataset: %s", dataset_path)
    logger.info("Exception file: %s", exception_path)
    logger.info("Job ID: %s", args.job_id)
    logger.info("Report: %s", report_path)

    if args.check:
        report_path = Path(args.report_file).with_name(
            Path(args.report_file).stem + "_check" + Path(args.report_file).suffix
        )
        report_writer = JsonlReportWriter(report_path)
        stats = await run_check_sync(
            dataset_path=dataset_path,
            report_writer=report_writer,
            exception_path=exception_path,
        )
        logger.info("Check sync completed: %s", stats)
        return

    stats = await run_sync(
        dataset_path=dataset_path,
        job_id=args.job_id,
        limit=args.limit,
        report_writer=report_writer,
        exception_path=exception_path,
    )

    logger.info("Sync completed: %s", stats)


if __name__ == "__main__":
    asyncio.run(_main_async())
