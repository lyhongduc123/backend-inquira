"""
Batch-run database pipeline stream API with resume checkpointing.

Features
--------
- Sends LitQA/SciFact questions to chat stream endpoint (`/api/v1/chat/stream`)
- Stores per-question result in JSONL
- Maintains checkpoint so reruns continue from last processed index
- Supports LitQA DOI filtering using `data/litqa/info.json`
- Supports DOI exception mapping using `data/litqa/exception.json`

Usage
-----
python scripts/run_database_pipeline_stream_batch.py --auth-token <TOKEN>
python scripts/run_database_pipeline_stream_batch.py --dataset litqa --resume
python scripts/run_database_pipeline_stream_batch.py --dataset both --max-questions 200 --resume
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import requests

DOI_PATTERN = re.compile(r"^10\.\d{4,9}/\S+$", flags=re.IGNORECASE)
DOI_URL_PREFIX = re.compile(r"^https?://(dx\.)?doi\.org/", flags=re.IGNORECASE)
DOI_PREFIX = re.compile(r"^doi:\s*", flags=re.IGNORECASE)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _normalize_question(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def _normalize_doi(raw: Any) -> Optional[str]:
    value = str(raw or "").strip()
    if not value:
        return None
    value = DOI_URL_PREFIX.sub("", value)
    value = DOI_PREFIX.sub("", value).strip().rstrip("/")
    if DOI_PATTERN.fullmatch(value):
        return value.lower()
    return None


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                yield json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_no} in {path}: {exc}") from exc


def _load_exception_index(path: Path) -> Dict[str, Optional[str]]:
    if not path.exists():
        return {}

    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("except", []) if isinstance(payload, dict) else []

    mapping: Dict[str, Optional[str]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        src = _normalize_doi(row.get("doi"))
        if not src:
            continue
        mapping[src] = _normalize_doi(row.get("ver_doi"))
    return mapping


@dataclass
class LitqaInfo:
    indexed_dois: set[str]
    fulltext_dois: set[str]


def _load_litqa_info(path: Path) -> LitqaInfo:
    if not path.exists():
        return LitqaInfo(indexed_dois=set(), fulltext_dois=set())

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return LitqaInfo(indexed_dois=set(), fulltext_dois=set())

    indexed_raw = payload.get("indexed_dois")
    if not isinstance(indexed_raw, list):
        indexed_raw = payload.get("existing_dois")
    if not isinstance(indexed_raw, list):
        indexed_raw = []

    fulltext_raw = payload.get("fulltext_dois")
    if not isinstance(fulltext_raw, list):
        fulltext_raw = []

    indexed_dois = {_normalize_doi(x) for x in indexed_raw}
    fulltext_dois = {_normalize_doi(x) for x in fulltext_raw}

    return LitqaInfo(
        indexed_dois={x for x in indexed_dois if x},
        fulltext_dois={x for x in fulltext_dois if x},
    )


def _load_litqa_questions(
    litqa_path: Path,
    litqa_info_path: Path,
    litqa_exception_path: Path,
) -> List[str]:
    exception_index = _load_exception_index(litqa_exception_path)
    litqa_info = _load_litqa_info(litqa_info_path)

    questions: List[str] = []
    seen: set[str] = set()

    for row in _iter_jsonl(litqa_path):
        question = str(row.get("question") or "").strip()
        if not question:
            continue

        if not bool(row.get("is_opensource", False)):
            continue

        mapped_dois: List[str] = []
        for src in row.get("sources") or []:
            doi = _normalize_doi(src)
            if not doi:
                continue
            mapped = exception_index.get(doi, doi)
            if mapped:
                mapped_dois.append(mapped)

        if not mapped_dois:
            continue

        # If info.json has constraints, enforce them.
        if litqa_info.indexed_dois:
            mapped_dois = [d for d in mapped_dois if d in litqa_info.indexed_dois]
        if litqa_info.fulltext_dois:
            mapped_dois = [d for d in mapped_dois if d in litqa_info.fulltext_dois]

        if not mapped_dois:
            continue

        key = _normalize_question(question)
        if key in seen:
            continue
        seen.add(key)
        questions.append(question)

    return questions


def _load_scifact_questions(path: Path) -> List[str]:
    questions: List[str] = []
    seen: set[str] = set()
    for row in _iter_jsonl(path):
        text = str(row.get("text") or "").strip()
        if not text:
            continue
        key = _normalize_question(text)
        if key in seen:
            continue
        seen.add(key)
        questions.append(text)
    return questions


def _load_checkpoint(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"next_index": 0}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass
    return {"next_index": 0}


def _save_checkpoint(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _extract_event_content(event_type: str, event_data: str) -> tuple[Optional[str], Optional[str], Dict[str, Any]]:
    payload: Dict[str, Any] = {}
    text_piece: Optional[str] = None
    error_message: Optional[str] = None

    try:
        parsed = json.loads(event_data)
        if isinstance(parsed, dict):
            payload = parsed
        else:
            payload = {"value": parsed}
    except json.JSONDecodeError:
        payload = {"raw": event_data}

    if event_type in ("chunk", "token"):
        text_piece = str(
            payload.get("content")
            or payload.get("text")
            or payload.get("token")
            or ""
        )
    elif event_type == "error":
        error_message = str(payload.get("message") or payload.get("error") or event_data)

    return text_piece, error_message, payload


def _call_stream_endpoint(
    *,
    base_url: str,
    auth_token: str,
    query: str,
    timeout_seconds: int,
    model: Optional[str],
    conversation_id: Optional[str],
) -> Dict[str, Any]:
    url = f"{base_url.rstrip('/')}/api/v1/chat/stream"
    headers = {
        "Accept": "text/event-stream",
        "Content-Type": "application/json",
    }
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"

    body: Dict[str, Any] = {
        "query": query,
        "pipeline": "database",
        "stream": True,
    }
    if model:
        body["model"] = model
    if conversation_id:
        body["conversationId"] = conversation_id

    started = time.time()
    answer_parts: List[str] = []
    event_counts: Dict[str, int] = {}
    error_message: Optional[str] = None
    latest_conversation_id: Optional[str] = None

    with requests.post(url, headers=headers, json=body, stream=True, timeout=timeout_seconds) as response:
        status_code = response.status_code
        if status_code >= 400:
            text = response.text[:2000]
            return {
                "ok": False,
                "status_code": status_code,
                "error": f"HTTP {status_code}: {text}",
                "answer": "",
                "elapsed_seconds": round(time.time() - started, 3),
                "event_counts": event_counts,
                "conversation_id": None,
            }

        current_event = "message"
        data_lines: List[str] = []

        for raw_line in response.iter_lines(decode_unicode=True):
            if raw_line is None:
                continue
            line = raw_line.rstrip("\r")

            if line == "":
                if data_lines:
                    event_data = "\n".join(data_lines)
                    event_counts[current_event] = event_counts.get(current_event, 0) + 1
                    text_piece, event_error, payload = _extract_event_content(current_event, event_data)

                    if current_event == "conversation" and isinstance(payload, dict):
                        conv_id = payload.get("conversation_id") or payload.get("conversationId")
                        if conv_id:
                            latest_conversation_id = str(conv_id)

                    if text_piece:
                        answer_parts.append(text_piece)
                    if event_error:
                        error_message = event_error

                current_event = "message"
                data_lines = []
                continue

            if line.startswith(":"):
                continue
            if line.startswith("event:"):
                current_event = line[6:].strip() or "message"
                continue
            if line.startswith("data:"):
                data_lines.append(line[5:].strip())

    return {
        "ok": error_message is None,
        "status_code": 200,
        "error": error_message,
        "answer": "".join(answer_parts).strip(),
        "elapsed_seconds": round(time.time() - started, 3),
        "event_counts": event_counts,
        "conversation_id": latest_conversation_id,
    }


def run_batch(args: argparse.Namespace) -> Dict[str, Any]:
    auth_token = args.auth_token or os.getenv("EXEGENT_API_TOKEN", "")

    all_questions: List[str] = []
    if args.dataset in ("litqa", "both"):
        all_questions.extend(
            _load_litqa_questions(
                litqa_path=Path(args.litqa_path),
                litqa_info_path=Path(args.litqa_info_path),
                litqa_exception_path=Path(args.litqa_exception_path),
            )
        )
    if args.dataset in ("scifact", "both"):
        all_questions.extend(_load_scifact_questions(Path(args.scifact_queries_path)))

    if args.max_questions is not None:
        all_questions = all_questions[: args.max_questions]

    checkpoint_path = Path(args.checkpoint_file)
    output_path = Path(args.output_file)

    checkpoint = _load_checkpoint(checkpoint_path) if args.resume else {"next_index": 0}
    next_index = int(checkpoint.get("next_index", 0))
    if args.start_index is not None:
        next_index = max(0, args.start_index)

    processed = 0
    failed = 0
    conversation_id: Optional[str] = args.conversation_id

    for idx in range(next_index, len(all_questions)):
        question = all_questions[idx]
        total = len(all_questions)
        print(f"[{idx + 1}/{total}] Running query...", flush=True)

        result: Optional[Dict[str, Any]] = None
        last_error: Optional[str] = None

        for attempt in range(1, args.max_retries + 1):
            try:
                result = _call_stream_endpoint(
                    base_url=args.base_url,
                    auth_token=auth_token,
                    query=question,
                    timeout_seconds=args.timeout_seconds,
                    model=args.model,
                    conversation_id=conversation_id,
                )
                if result["ok"]:
                    break
                last_error = result.get("error")
            except requests.RequestException as exc:
                last_error = str(exc)

            if attempt < args.max_retries:
                time.sleep(args.retry_delay_seconds)

        if result is None:
            result = {
                "ok": False,
                "status_code": None,
                "error": last_error or "unknown_error",
                "answer": "",
                "elapsed_seconds": 0.0,
                "event_counts": {},
                "conversation_id": None,
            }

        if result.get("conversation_id"):
            conversation_id = str(result["conversation_id"])

        row = {
            "timestamp": _utc_now_iso(),
            "index": idx,
            "dataset": args.dataset,
            "question": question,
            "ok": bool(result.get("ok")),
            "status_code": result.get("status_code"),
            "error": result.get("error"),
            "answer": result.get("answer"),
            "answer_length": len(str(result.get("answer") or "")),
            "elapsed_seconds": result.get("elapsed_seconds"),
            "event_counts": result.get("event_counts") or {},
            "conversation_id": result.get("conversation_id") or conversation_id,
        }
        _append_jsonl(output_path, row)

        status_text = "OK" if row["ok"] else "FAILED"
        print(
            f"[{idx + 1}/{total}] {status_text} | elapsed={row['elapsed_seconds']}s | answer_len={row['answer_length']}",
            flush=True,
        )

        processed += 1
        if not row["ok"]:
            failed += 1
            if not args.continue_on_error:
                _save_checkpoint(
                    checkpoint_path,
                    {
                        "next_index": idx,
                        "updated_at": _utc_now_iso(),
                        "processed": processed,
                        "failed": failed,
                        "dataset": args.dataset,
                        "total_questions": len(all_questions),
                    },
                )
                raise RuntimeError(f"Stopped at index {idx}: {row['error']}")

        _save_checkpoint(
            checkpoint_path,
            {
                "next_index": idx + 1,
                "updated_at": _utc_now_iso(),
                "processed": processed,
                "failed": failed,
                "dataset": args.dataset,
                "total_questions": len(all_questions),
                "conversation_id": conversation_id,
            },
        )

        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    return {
        "total_questions": len(all_questions),
        "processed": processed,
        "failed": failed,
        "checkpoint_file": str(checkpoint_path),
        "output_file": str(output_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch-run /api/v1/chat/stream with checkpoint resume")
    parser.add_argument("--base-url", type=str, default="http://localhost:8000", help="Backend base URL")
    parser.add_argument(
        "--auth-token",
        type=str,
        default="",
        help="Optional Bearer token (or EXEGENT_API_TOKEN env). Leave empty to use anonymous chat.",
    )
    parser.add_argument("--dataset", type=str, default="litqa", choices=["litqa", "scifact", "both"])

    parser.add_argument(
        "--litqa-path",
        type=str,
        default=str(Path(__file__).parent.parent / "data" / "litqa" / "litqa-v2-public.jsonl"),
    )
    parser.add_argument(
        "--litqa-info-path",
        type=str,
        default=str(Path(__file__).parent.parent / "data" / "litqa" / "info.json"),
    )
    parser.add_argument(
        "--litqa-exception-path",
        type=str,
        default=str(Path(__file__).parent.parent / "data" / "litqa" / "exception.json"),
    )
    parser.add_argument(
        "--scifact-queries-path",
        type=str,
        default=str(Path(__file__).parent.parent / "data" / "beir_datasets" / "scifact" / "queries.jsonl"),
    )

    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint file")
    parser.add_argument("--start-index", type=int, default=None, help="Override start index")
    parser.add_argument("--max-questions", type=int, default=None, help="Optional cap on number of questions")
    parser.add_argument("--conversation-id", type=str, default=None, help="Optional fixed conversation id")
    parser.add_argument("--model", type=str, default=None, help="Optional model override")

    parser.add_argument("--timeout-seconds", type=int, default=240)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--retry-delay-seconds", type=float, default=2.0)
    parser.add_argument("--sleep-seconds", type=float, default=0.2)
    parser.add_argument("--continue-on-error", action="store_true")

    parser.add_argument(
        "--checkpoint-file",
        type=str,
        default=str(Path("evaluation_results") / "database_pipeline" / "stream_checkpoint.json"),
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=str(
            Path("evaluation_results")
            / "database_pipeline"
            / f"stream_batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run_batch(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
