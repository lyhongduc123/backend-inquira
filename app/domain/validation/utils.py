"""Small shared utilities for validation module."""
from typing import Set
import re

def extract_paper_ids_from_context(context: str) -> Set[str]:
    ids = set()
    patterns = [
        r'(?:^|\n)SOURCE_ID:\s*([^\n\r]+)',
        r'(?:^|\n)PAPER\s+ID:\s*([^\n\r]+)',
        r'(?:^|\n)paper_id:\s*([^\n\r]+)',
        r'(?:^|\n)paperId:\s*([^\n\r]+)',
        r'"paperId":\s*"([^"]+)"',
        r'"paper_id":\s*"([^"]+)"',
        r'"id":\s*"([^"]+)"',
    ]
    for pattern in patterns:
        matches = re.findall(pattern, context, re.MULTILINE)
        ids.update(match.strip().strip(',') for match in matches)
    return ids


def extract_chunk_ids_from_context(context: str) -> Set[str]:
    ids: Set[str] = set()
    patterns = [
        r'(?:^|\n)CHUNK_ID:\s*([^\n\r]+)',
        r'(?:^|\n)chunk_id:\s*([^\n\r]+)',
        r'"chunkId":\s*"([^"]+)"',
        r'"chunk_id":\s*"([^"]+)"',
    ]
    for pattern in patterns:
        matches = re.findall(pattern, context, re.MULTILINE)
        ids.update(match.strip().strip(',') for match in matches)
    return ids


def extract_context_evidence(context: str):
    from app.domain.validation.schemas import ContextEvidence

    paper_ids = sorted(list(extract_paper_ids_from_context(context)))
    chunk_ids = sorted(list(extract_chunk_ids_from_context(context)))
    return ContextEvidence(
        paper_ids=paper_ids,
        chunk_ids=chunk_ids,
        total_papers=len(paper_ids),
        total_chunks=len(chunk_ids),
    )
