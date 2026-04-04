import re
import json
from pathlib import Path
import tiktoken
from typing import List, Optional, Tuple, Dict, Any, NamedTuple
from app.extensions.logger import create_logger
from app.modules.r2_storage import R2StorageService

logger = create_logger(__name__)


class ChunkWithMetadata(NamedTuple):
    """Enhanced chunk with docling metadata"""

    text: str  # Clean text for embeddings
    token_count: int
    section_title: Optional[str]
    page_number: Optional[int]
    label: Optional[str]  # section_header, text, caption, etc.
    level: Optional[int]  # Hierarchy level
    char_start: Optional[int]  # Character offset start
    char_end: Optional[int]  # Character offset end
    docling_metadata: Dict[str, Any]  # bbox, prov, etc.


class ChunkingService:
    """Chunk text into smaller pieces for embedding with section-boundary-aware intelligent chunking"""

    def __init__(
        self,
        min_tokens: int = 300,
        max_tokens: int = 600,
        overlap_ratio: float = 0.1,  # 10% overlap
        model: str = "gpt-4",
    ):
        """
        Initialize text chunker with section-aware strategy

        Args:
            min_tokens: Minimum tokens per chunk (not strictly enforced for small sections)
            max_tokens: Maximum tokens per chunk (sections may be split if they exceed this)
            overlap_ratio: Ratio of overlap when splitting long sections (default 10%)
            model: Model name for tokenization
        """
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.overlap_ratio = overlap_ratio
        self.encoding = tiktoken.encoding_for_model(model)
        self.r2_storage = R2StorageService()

    @staticmethod
    def _normalize_docling_label(label: Optional[str]) -> str:
        if not label:
            return ""
        return label.strip().lower().replace("-", "_").replace(" ", "_")

    @staticmethod
    def _resolve_ref_index(ref_obj: Dict[str, Any], expected: str) -> Optional[int]:
        ref = ref_obj.get("$ref")
        if not isinstance(ref, str):
            return None
        prefix = f"#/{expected}/"
        if not ref.startswith(prefix):
            return None
        try:
            return int(ref.split("/")[-1])
        except Exception:
            return None

    @staticmethod
    def _clean_noise_text(text: str) -> str:
        cleaned = text.replace("\u0000", "").strip()
        cleaned = re.sub(r"\s+", " ", cleaned)
        cleaned = cleaned.replace("Please cite this article as:", "")
        return cleaned.strip()

    @staticmethod
    def _is_boilerplate_text(text: str) -> bool:
        """Detect repeated PDF boilerplate/noise rows from hierarchical export."""
        normalized = re.sub(r"\s+", " ", (text or "")).strip().lower()
        if not normalized:
            return True

        # Symbol-only / OCR artifacts
        if not re.search(r"[a-z0-9]", normalized):
            return True

        boilerplate_patterns = (
            r"^please cite this article as:?$",
            r"^corresponding author",
            r"^tel\s*\+?\d",
        )
        if any(re.search(pattern, normalized) for pattern in boilerplate_patterns):
            return True

        return False

    def _extract_text_item_content(self, item: Dict[str, Any]) -> str:
        text = (item.get("text") or "").strip()
        if text:
            return self._clean_noise_text(text)
        orig = (item.get("orig") or "").strip()
        return self._clean_noise_text(orig)

    @staticmethod
    def _is_formula_like_text(text: str) -> bool:
        if len(text) < 40:
            return False
        symbols = sum(1 for c in text if c in "∑∏√∞≈≤≥±∫πλμσΩ{}[]<>|=+*/^_")
        ratio = symbols / max(len(text), 1)
        return ratio > 0.08

    @staticmethod
    def _extract_page_number_from_item(item: Dict[str, Any]) -> Optional[int]:
        prov = item.get("prov", [])
        if prov and isinstance(prov, list) and len(prov) > 0:
            page_no = prov[0].get("page_no")
            if page_no is not None:
                return page_no
        return None

    def _extract_section_anchor_by_page(
        self, texts: List[Dict[str, Any]]
    ) -> Dict[int, str]:
        """Build a page->section anchor map from section headers in doc texts."""
        anchors: Dict[int, str] = {}
        current_header = "Document"

        for item in texts:
            if item.get("content_layer") == "furniture":
                continue

            label = self._normalize_docling_label(item.get("label", ""))
            text = self._extract_text_item_content(item)
            if label == "section_header" and text:
                current_header = text

            page = self._extract_page_number_from_item(item)
            if page is not None and page not in anchors:
                anchors[page] = current_header

        return anchors

    def _collect_formula_items(self, texts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Collect formula/equation-like text items for dedicated evidence chunks."""
        formula_labels = {
            "formula",
            "equation",
            "math",
            "equation_block",
            "formula_block",
        }
        formula_items: List[Dict[str, Any]] = []

        for item in texts:
            if item.get("content_layer") == "furniture":
                continue

            label = self._normalize_docling_label(item.get("label", ""))
            text = self._extract_text_item_content(item)
            if not text:
                continue

            if label in formula_labels or self._is_formula_like_text(text):
                formula_items.append(item)

        return formula_items

    @staticmethod
    def _normalize_section_key(section_title: Optional[str]) -> str:
        if not section_title:
            return ""

        cleaned = re.sub(
            r"\s*/\s*(Figure|Table|Formula)\s+Evidence\s*$",
            "",
            section_title,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(r"\s+", " ", cleaned).strip().lower()
        return cleaned

    def _link_evidence_to_nearby_text_chunks(
        self,
        text_chunks: List[ChunkWithMetadata],
        evidence_chunks: List[ChunkWithMetadata],
    ) -> List[ChunkWithMetadata]:
        """Attach nearest text chunk indices/sections to evidence chunks (tables/figures/formulas)."""
        if not text_chunks or not evidence_chunks:
            return evidence_chunks

        text_index_rows: List[Dict[str, Any]] = []
        for idx, chunk in enumerate(text_chunks):
            text_index_rows.append(
                {
                    "index": idx,
                    "page": chunk.page_number,
                    "section_key": self._normalize_section_key(chunk.section_title),
                    "section_title": chunk.section_title,
                }
            )

        linked_chunks: List[ChunkWithMetadata] = []

        for chunk in evidence_chunks:
            target_page = chunk.page_number
            target_section_key = self._normalize_section_key(chunk.section_title)

            scored: List[Tuple[float, int, Dict[str, Any]]] = []
            for row in text_index_rows:
                score = 0.0

                if target_page is not None and row["page"] == target_page:
                    score += 3.0

                row_section_key = row["section_key"]
                if target_section_key and row_section_key:
                    if row_section_key == target_section_key:
                        score += 4.0
                    elif (
                        row_section_key in target_section_key
                        or target_section_key in row_section_key
                    ):
                        score += 2.0

                if (
                    target_page is not None
                    and row["page"] is not None
                    and row["page"] != target_page
                ):
                    distance = abs(int(row["page"]) - int(target_page))
                    if distance == 1:
                        score += 1.0
                    elif distance == 2:
                        score += 0.5

                if score <= 0:
                    continue

                page_distance = (
                    abs(int(row["page"]) - int(target_page))
                    if target_page is not None and row["page"] is not None
                    else 99
                )
                scored.append((score, page_distance, row))

            scored.sort(key=lambda x: (-x[0], x[1], x[2]["index"]))
            top_rows = [row for _, _, row in scored[:3]]

            linked_indices = [int(row["index"]) for row in top_rows]
            linked_sections: List[str] = []
            for row in top_rows:
                section_title = row.get("section_title")
                if isinstance(section_title, str) and section_title and section_title not in linked_sections:
                    linked_sections.append(section_title)

            metadata = dict(chunk.docling_metadata or {})
            metadata["linked_text_chunk_indices"] = linked_indices
            metadata["linked_sections"] = linked_sections[:2]
            metadata["linkage"] = {
                "strategy": "section_page_proximity",
                "target_page": target_page,
                "target_section_key": target_section_key,
            }

            linked_chunks.append(
                ChunkWithMetadata(
                    text=chunk.text,
                    token_count=chunk.token_count,
                    section_title=chunk.section_title,
                    page_number=chunk.page_number,
                    label=chunk.label,
                    level=chunk.level,
                    char_start=chunk.char_start,
                    char_end=chunk.char_end,
                    docling_metadata=metadata,
                )
            )

        return linked_chunks

    def _read_hierarchical_chunks_from_assets(
        self, doc_dict: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Read pre-exported hierarchical chunks from extractor assets if available."""
        asset_paths = doc_dict.get("asset_paths") or {}
        chunk_path = asset_paths.get("hierarchical_chunks_path")
        if not chunk_path:
            return []

        try:
            path = Path(chunk_path)
            if not path.exists() or not path.is_file():
                return []
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                return payload
            return []
        except Exception as error:
            logger.warning(f"Failed to read hierarchical chunks from assets: {error}")
            return []

    def _read_assets_manifest(self, doc_dict: Dict[str, Any]) -> Dict[str, Any]:
        inline_manifest = doc_dict.get("asset_manifest")
        if isinstance(inline_manifest, dict):
            return inline_manifest

        asset_paths = doc_dict.get("asset_paths") or {}
        inline_manifest = asset_paths.get("manifest_inline")
        if isinstance(inline_manifest, dict):
            return inline_manifest

        manifest_path = asset_paths.get("manifest_path")
        if manifest_path:
            try:
                path = Path(manifest_path)
                if path.exists() and path.is_file():
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    if isinstance(payload, dict):
                        return payload
            except Exception as error:
                logger.warning(f"Failed to read assets manifest: {error}")

        manifest_r2_key = asset_paths.get("manifest_r2_key")
        if isinstance(manifest_r2_key, str) and manifest_r2_key.strip():
            try:
                manifest_bytes = self.r2_storage.get_object_bytes(manifest_r2_key)
                if manifest_bytes:
                    payload = json.loads(manifest_bytes.decode("utf-8"))
                    if isinstance(payload, dict):
                        doc_dict["asset_manifest"] = payload
                        return payload
            except Exception as error:
                logger.warning(f"Failed to read R2 assets manifest: {error}")

        return {}

    @staticmethod
    def _resolve_manifest_link(entry: Dict[str, Any], field: str) -> Optional[str]:
        if not isinstance(entry, dict):
            return None

        # Prefer uploaded R2 references first to avoid leaking local temp paths.
        r2_block = entry.get("r2")
        if isinstance(r2_block, dict):
            r2_value = r2_block.get(field)
            if isinstance(r2_value, dict):
                url_value = r2_value.get("url")
                if isinstance(url_value, str) and url_value.strip():
                    return url_value
                key_value = r2_value.get("key")
                if isinstance(key_value, str) and key_value.strip():
                    return key_value

        direct_value = entry.get(field)
        if isinstance(direct_value, str) and direct_value.strip():
            return direct_value
        if isinstance(direct_value, dict):
            url_value = direct_value.get("url")
            if isinstance(url_value, str) and url_value.strip():
                return url_value
            key_value = direct_value.get("key")
            if isinstance(key_value, str) and key_value.strip():
                return key_value

        return None

    @staticmethod
    def _manifest_entry_by_index(entries: Any, index: int) -> Dict[str, Any]:
        if not isinstance(entries, list):
            return {}
        for row in entries:
            if isinstance(row, dict) and int(row.get("index", -1)) == index:
                return row
        return {}

    def _build_section_metadata_summary(
        self,
        section_items: List[Dict[str, Any]],
        asset_paths: Dict[str, Any],
    ) -> Dict[str, Any]:
        labels = [self._normalize_docling_label(i.get("label", "")) for i in section_items]
        labels = [x for x in labels if x]
        pages = [self._extract_page_number_from_item(i) for i in section_items]
        page_numbers = sorted({int(p) for p in pages if p is not None})
        return {
            "source": "docling_section",
            "item_count": len(section_items),
            "labels": sorted(set(labels))[:10],
            "pages": page_numbers,
            "assets_manifest_path": asset_paths.get("manifest_path"),
        }

    def _build_table_blocks_by_page(
        self,
        doc_dict: Dict[str, Any],
    ) -> Dict[int, List[str]]:
        """Build compact table text blocks grouped by page so tables stay in normal chunks."""
        result: Dict[int, List[str]] = {}
        tables = doc_dict.get("tables", []) or []
        texts = doc_dict.get("texts", []) or []

        for table in tables:
            page_no = self._extract_page_number_from_item(table)
            if page_no is None:
                continue

            caption_parts: List[str] = []
            for caption_ref in table.get("captions", []) or []:
                text_idx = self._resolve_ref_index(caption_ref, "texts")
                if text_idx is not None and 0 <= text_idx < len(texts):
                    caption = self._extract_text_item_content(texts[text_idx])
                    if caption:
                        caption_parts.append(caption)

            data = table.get("data", {}) or {}
            cells = data.get("table_cells", []) or []
            rows: Dict[int, List[Dict[str, Any]]] = {}
            for cell in cells:
                row_idx = int(cell.get("start_row_offset_idx", 0) or 0)
                rows.setdefault(row_idx, []).append(cell)

            row_lines: List[str] = []
            for row_idx in sorted(rows.keys()):
                ordered = sorted(rows[row_idx], key=lambda c: int(c.get("start_col_offset_idx", 0) or 0))
                vals = [str(c.get("text") or "").strip() for c in ordered]
                vals = [v for v in vals if v]
                if vals:
                    row_lines.append(" | ".join(vals))

            if not caption_parts and not row_lines:
                continue

            table_block = "\n".join([
                "[TABLE]",
                *caption_parts,
                *row_lines,
            ]).strip()
            result.setdefault(page_no, []).append(table_block)

        return result

    def _build_figure_assets_by_page(
        self,
        doc_dict: Dict[str, Any],
    ) -> Dict[int, List[Dict[str, Any]]]:
        """Map page -> figure asset entries for retrieval-time figure lookup."""
        mapping: Dict[int, List[Dict[str, Any]]] = {}
        manifest = self._read_assets_manifest(doc_dict)
        figure_entries = manifest.get("figures", []) if isinstance(manifest, dict) else []
        pictures = doc_dict.get("pictures", []) or []

        for idx, picture in enumerate(pictures):
            page_no = self._extract_page_number_from_item(picture)
            if page_no is None:
                continue
            asset = self._manifest_entry_by_index(figure_entries, idx)
            if not asset:
                continue
            normalized_asset = dict(asset)
            normalized_asset["metadata_path"] = self._resolve_manifest_link(
                asset,
                "metadata_path",
            )
            normalized_asset["image_path"] = self._resolve_manifest_link(asset, "image_path")
            normalized_asset["crop_path"] = self._resolve_manifest_link(asset, "crop_path")
            mapping.setdefault(page_no, []).append(normalized_asset)

        return mapping

    def _build_table_assets_by_page(
        self,
        doc_dict: Dict[str, Any],
    ) -> Dict[int, List[Dict[str, Any]]]:
        """Map page -> table asset entries for linking text chunks to table chunks."""
        mapping: Dict[int, List[Dict[str, Any]]] = {}
        manifest = self._read_assets_manifest(doc_dict)
        table_entries = manifest.get("tables", []) if isinstance(manifest, dict) else []
        tables = doc_dict.get("tables", []) or []

        for idx, table in enumerate(tables):
            page_no = self._extract_page_number_from_item(table)
            if page_no is None:
                continue

            asset = self._manifest_entry_by_index(table_entries, idx)
            entry = {
                "table_index": idx,
                "table_chunk_ref": f"table_{idx:03d}",
                "path": self._resolve_manifest_link(asset, "path"),
                "crop_path": self._resolve_manifest_link(asset, "crop_path"),
            }
            mapping.setdefault(page_no, []).append(entry)

        return mapping

    @staticmethod
    def _extract_hierarchical_section_title(metadata: Any) -> Optional[str]:
        if not isinstance(metadata, dict):
            return None
        for key in ("section_title", "heading", "section", "title"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        headings = metadata.get("headings")
        if isinstance(headings, list) and headings:
            for value in reversed(headings):
                if isinstance(value, str) and value.strip():
                    return value.strip()
                if isinstance(value, dict):
                    candidate = value.get("text") or value.get("title")
                    if isinstance(candidate, str) and candidate.strip():
                        return candidate.strip()
        return None

    @staticmethod
    def _extract_hierarchical_page_number(metadata: Any) -> Optional[int]:
        if not isinstance(metadata, dict):
            return None
        for key in ("page_no", "page_number", "page"):
            value = metadata.get(key)
            if isinstance(value, int):
                return value
            if isinstance(value, str) and value.strip().isdigit():
                return int(value.strip())
        return None

    def _build_chunks_from_hierarchical_assets(
        self,
        doc_dict: Dict[str, Any],
        paper_id: str,
    ) -> List[ChunkWithMetadata]:
        """Build paragraph-first chunks from Docling hierarchical export when present."""
        rows = self._read_hierarchical_chunks_from_assets(doc_dict)
        asset_paths = doc_dict.get("asset_paths") or {}
        figure_assets_by_page = self._build_figure_assets_by_page(doc_dict)
        if not rows:
            return []

        chunks: List[ChunkWithMetadata] = []
        char_offset = 0

        current_paragraphs: List[str] = []
        current_tokens = 0
        current_section_title: Optional[str] = None
        current_page_number: Optional[int] = None
        current_meta_rows: List[Dict[str, Any]] = []

        def flush_chunk() -> None:
            nonlocal char_offset, current_paragraphs, current_tokens, current_section_title, current_page_number, current_meta_rows
            if not current_paragraphs:
                return

            chunk_text = "\n\n".join(current_paragraphs).strip()
            if not chunk_text:
                current_paragraphs = []
                current_tokens = 0
                current_section_title = None
                current_page_number = None
                current_meta_rows = []
                return

            token_count = self.count_tokens(chunk_text)
            char_start = char_offset
            char_end = char_offset + len(chunk_text)
            figure_assets = figure_assets_by_page.get(current_page_number or -1, [])

            chunks.append(
                ChunkWithMetadata(
                    text=chunk_text,
                    token_count=token_count,
                    section_title=current_section_title,
                    page_number=current_page_number,
                    label="hierarchical_text",
                    level=None,
                    char_start=char_start,
                    char_end=char_end,
                    docling_metadata={
                        "source": "docling_hierarchical",
                        "hierarchical_meta": {
                            "paragraph_count": len(current_meta_rows),
                            "paragraph_meta": current_meta_rows[:20],
                        },
                        "assets_manifest_path": asset_paths.get("manifest_path"),
                        "figure_assets": figure_assets,
                        "retrieval_flags": {
                            "has_figure_assets": len(figure_assets) > 0,
                            "figure_retrieval_enabled": len(figure_assets) > 0,
                        },
                    },
                )
            )

            char_offset = char_end + 2
            current_paragraphs = []
            current_tokens = 0
            current_section_title = None
            current_page_number = None
            current_meta_rows = []

        for row in rows:
            raw = str(row.get("text") or "").strip()
            text = self._clean_noise_text(raw)
            if not text or self._is_boilerplate_text(text):
                continue

            metadata = row.get("metadata") if isinstance(row, dict) else None
            section_title = self._extract_hierarchical_section_title(metadata)
            page_number = self._extract_hierarchical_page_number(metadata)
            token_count = self.count_tokens(text)

            # Overlong paragraph: split only at sentence boundaries.
            if token_count > self.max_tokens:
                flush_chunk()
                split_chunks = self._split_section_with_overlap(
                    section_text=text,
                    section_header=section_title,
                    section_level=None,
                    section_page=page_number,
                    primary_label="hierarchical_text",
                    metadata_items={
                        "source": "docling_hierarchical",
                        "hierarchical_meta": metadata,
                        "assets_manifest_path": asset_paths.get("manifest_path"),
                        "figure_assets": figure_assets_by_page.get(page_number or -1, []),
                        "retrieval_flags": {
                            "has_figure_assets": len(figure_assets_by_page.get(page_number or -1, [])) > 0,
                            "figure_retrieval_enabled": len(figure_assets_by_page.get(page_number or -1, [])) > 0,
                        },
                    },
                    char_offset=char_offset,
                )
                chunks.extend(split_chunks)
                if split_chunks and split_chunks[-1].char_end is not None:
                    char_offset = split_chunks[-1].char_end + 2
                continue

            if current_paragraphs:
                section_changed = (
                    bool(section_title)
                    and bool(current_section_title)
                    and self._normalize_section_key(section_title)
                    != self._normalize_section_key(current_section_title)
                )
                page_changed = (
                    page_number is not None
                    and current_page_number is not None
                    and int(page_number) != int(current_page_number)
                )
                token_overflow = current_tokens + token_count > self.max_tokens

                if section_changed or page_changed or token_overflow:
                    flush_chunk()

            if current_section_title is None:
                current_section_title = section_title
            if current_page_number is None:
                current_page_number = page_number

            current_paragraphs.append(text)
            current_tokens += token_count
            current_meta_rows.append(metadata if isinstance(metadata, dict) else {})

        flush_chunk()

        if chunks:
            logger.info(
                f"[{paper_id}] Reused {len(chunks)} chunks from hierarchical export"
            )
        return chunks

    def _build_evidence_summary_text(self, raw_text: str, kind: str) -> str:
        """Build compact representation suitable for retrieval + generation."""
        if kind == "table":
            cleaned_lines = [re.sub(r"\s+", " ", line).strip() for line in raw_text.splitlines()]
            cleaned_lines = [line for line in cleaned_lines if line]
            cleaned = "\n".join(cleaned_lines)
        else:
            cleaned = re.sub(r"\s+", " ", raw_text).strip()

        lead = (
            "This chunk summarizes tabular evidence with key values and captions."
            if kind == "table"
            else (
                "This chunk summarizes formula/equation evidence and symbolic expressions."
                if kind == "formula"
                else "This chunk summarizes figure/chart evidence with key caption details."
            )
        )
        return f"{lead}\n\n{cleaned}"

    def _build_formula_chunks(
        self,
        formula_items: List[Dict[str, Any]],
        section_anchors: Dict[int, str],
        start_char_offset: int,
    ) -> List[ChunkWithMetadata]:
        """Create formula-focused chunks from Docling formula items."""
        chunks: List[ChunkWithMetadata] = []
        char_offset = start_char_offset

        for item in formula_items:
            raw_text = self._extract_text_item_content(item)
            if not raw_text:
                continue

            summary_text = self._build_evidence_summary_text(raw_text, "formula")
            token_count = self.count_tokens(summary_text)
            page_number = self._extract_page_number_from_item(item)
            section_anchor = section_anchors.get(page_number or -1)

            char_start = char_offset
            char_end = char_offset + len(summary_text)

            chunks.append(
                ChunkWithMetadata(
                    text=summary_text,
                    token_count=token_count,
                    section_title=(
                        f"{section_anchor} / Formula Evidence"
                        if section_anchor
                        else "Formula Evidence"
                    ),
                    page_number=page_number,
                    label="formula_evidence",
                    level=item.get("level"),
                    char_start=char_start,
                    char_end=char_end,
                    docling_metadata={
                        "is_multimodal_evidence": True,
                        "kind": "formula",
                        "source_label": self._normalize_docling_label(
                            item.get("label", "")
                        ),
                        "prov": item.get("prov"),
                        "bbox": item.get("bbox"),
                    },
                )
            )

            char_offset = char_end + 2

        return chunks

    def _build_table_evidence_chunks(
        self,
        doc_dict: Dict[str, Any],
        section_anchors: Dict[int, str],
        start_char_offset: int,
    ) -> List[ChunkWithMetadata]:
        """Build compact table chunks using Docling `tables[*].data.table_cells`."""
        chunks: List[ChunkWithMetadata] = []
        char_offset = start_char_offset

        tables = doc_dict.get("tables", []) or []
        texts = doc_dict.get("texts", []) or []
        manifest = self._read_assets_manifest(doc_dict)
        table_entries = manifest.get("tables", []) if isinstance(manifest, dict) else []

        for idx, table in enumerate(tables):
            data = table.get("data", {}) or {}
            cells = data.get("table_cells", []) or []
            if not cells:
                continue

            page_number = self._extract_page_number_from_item(table)
            section_anchor = section_anchors.get(page_number or -1)

            caption_texts: List[str] = []
            for caption_ref in table.get("captions", []) or []:
                text_idx = self._resolve_ref_index(caption_ref, "texts")
                if text_idx is not None and 0 <= text_idx < len(texts):
                    caption = self._extract_text_item_content(texts[text_idx])
                    if caption:
                        caption_texts.append(caption)

            rows: Dict[int, List[Dict[str, Any]]] = {}
            for cell in cells:
                row_idx = int(cell.get("start_row_offset_idx", 0) or 0)
                rows.setdefault(row_idx, []).append(cell)

            row_lines: List[str] = []
            for row_idx in sorted(rows.keys()):
                ordered = sorted(
                    rows[row_idx],
                    key=lambda c: int(c.get("start_col_offset_idx", 0) or 0),
                )
                values = [str(c.get("text") or "").strip() for c in ordered]
                values = [v for v in values if v]
                if values:
                    row_lines.append(" | ".join(values))

            if not row_lines and not caption_texts:
                continue

            raw_text = "\n".join(caption_texts + row_lines).strip()
            summary_text = self._build_evidence_summary_text(raw_text, "table")
            token_count = self.count_tokens(summary_text)

            char_start = char_offset
            char_end = char_offset + len(summary_text)

            chunks.append(
                ChunkWithMetadata(
                    text=summary_text,
                    token_count=token_count,
                    section_title=(
                        f"{section_anchor} / Table Evidence"
                        if section_anchor
                        else "Table Evidence"
                    ),
                    page_number=page_number,
                    label="table_evidence",
                    level=None,
                    char_start=char_start,
                    char_end=char_end,
                    docling_metadata={
                        "type": "table",
                        "source": "docling",
                        "table_index": idx,
                        "table_chunk_ref": f"table_{idx:03d}",
                        "path": self._resolve_manifest_link(
                            self._manifest_entry_by_index(table_entries, idx),
                            "path",
                        ),
                        "crop_path": self._resolve_manifest_link(
                            self._manifest_entry_by_index(table_entries, idx),
                            "crop_path",
                        ),
                        "links": {
                            "figure_paths": [],
                            "table_chunk_refs": [f"table_{idx:03d}"],
                        },
                    },
                )
            )
            char_offset = char_end + 2

        return chunks

    def _build_picture_evidence_chunks(
        self,
        doc_dict: Dict[str, Any],
        section_anchors: Dict[int, str],
        start_char_offset: int,
    ) -> List[ChunkWithMetadata]:
        """Build figure chunks from Docling `pictures[*]` captions."""
        chunks: List[ChunkWithMetadata] = []
        char_offset = start_char_offset

        pictures = doc_dict.get("pictures", []) or []
        texts = doc_dict.get("texts", []) or []
        manifest = self._read_assets_manifest(doc_dict)
        figure_entries = manifest.get("figures", []) if isinstance(manifest, dict) else []

        for idx, picture in enumerate(pictures):
            caption_texts: List[str] = []
            for caption_ref in picture.get("captions", []) or []:
                text_idx = self._resolve_ref_index(caption_ref, "texts")
                if text_idx is not None and 0 <= text_idx < len(texts):
                    caption = self._extract_text_item_content(texts[text_idx])
                    if caption:
                        caption_texts.append(caption)

            if not caption_texts:
                continue

            raw_text = "\n".join(caption_texts)
            summary_text = self._build_evidence_summary_text(raw_text, "figure")
            token_count = self.count_tokens(summary_text)
            page_number = self._extract_page_number_from_item(picture)
            section_anchor = section_anchors.get(page_number or -1)

            char_start = char_offset
            char_end = char_offset + len(summary_text)

            chunks.append(
                ChunkWithMetadata(
                    text=summary_text,
                    token_count=token_count,
                    section_title=(
                        f"{section_anchor} / Figure Evidence"
                        if section_anchor
                        else "Figure Evidence"
                    ),
                    page_number=page_number,
                    label="figure_evidence",
                    level=None,
                    char_start=char_start,
                    char_end=char_end,
                    docling_metadata={
                        "is_multimodal_evidence": True,
                        "kind": "figure",
                        "picture_index": idx,
                        "captions": caption_texts,
                        "prov": picture.get("prov"),
                        "assets": {
                            **self._manifest_entry_by_index(figure_entries, idx),
                            "metadata_path": self._resolve_manifest_link(
                                self._manifest_entry_by_index(figure_entries, idx),
                                "metadata_path",
                            ),
                            "image_path": self._resolve_manifest_link(
                                self._manifest_entry_by_index(figure_entries, idx),
                                "image_path",
                            ),
                            "crop_path": self._resolve_manifest_link(
                                self._manifest_entry_by_index(figure_entries, idx),
                                "crop_path",
                            ),
                        },
                    },
                )
            )
            char_offset = char_end + 2

        return chunks

    def _build_multimodal_evidence_chunks(
        self,
        doc_dict: Dict[str, Any],
        formula_items: List[Dict[str, Any]],
        section_anchors: Dict[int, str],
        start_char_offset: int,
    ) -> List[ChunkWithMetadata]:
        """Create multimodal evidence chunks. Current strategy keeps only table atomic chunks."""
        chunks: List[ChunkWithMetadata] = []

        current_offset = start_char_offset

        table_chunks = self._build_table_evidence_chunks(
            doc_dict, section_anchors, current_offset
        )
        chunks.extend(table_chunks)

        return chunks

    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.encoding.encode(text))

    def chunk_from_docling_structure(
        self, doc_dict: Dict[str, Any], paper_id: str
    ) -> List[ChunkWithMetadata]:
        """
        Chunk document from docling structure with SECTION-BOUNDARY-AWARE strategy.

        Strategy:
        - Never cross section boundaries
        - Keep entire abstract as one chunk (if <= max_tokens)
        - Keep entire section as one chunk (if <= max_tokens)
        - Split long sections with small overlap (10%) but maintain section boundaries
        - Chunks always end at section boundaries for clean semantic units

        Args:
            doc_dict: Docling document dictionary
            paper_id: Paper ID for logging

        Returns:
            List of ChunkWithMetadata tuples
        """
        chunks = []

        # Extract texts from docling
        texts = doc_dict.get("texts", [])
        asset_paths = doc_dict.get("asset_paths") or {}
        section_anchors = self._extract_section_anchor_by_page(texts)
        table_blocks_by_page = self._build_table_blocks_by_page(doc_dict)
        figure_assets_by_page = self._build_figure_assets_by_page(doc_dict)
        table_assets_by_page = self._build_table_assets_by_page(doc_dict)

        # Keep formulas inline in text chunks (no dedicated formula chunks).
        formula_items = self._collect_formula_items(texts)

        # Step 1: Group text items by section
        sections = []  # List of {"header": ..., "items": [...]}
        current_section = {"header": None, "level": None, "items": []}

        for text_item in texts:
            # Skip furniture (headers, footers, page numbers)
            if text_item.get("content_layer") == "furniture":
                continue

            text_content = self._extract_text_item_content(text_item)
            if not text_content:
                continue

            label = self._normalize_docling_label(text_item.get("label", ""))

            # If this is a section header, finalize previous section and start new one
            if label == "section_header":
                # Save previous section if it has content
                if current_section["items"]:
                    sections.append(current_section)

                # Start new section
                current_section = {
                    "header": text_content,
                    "level": text_item.get("level"),
                    "items": [],
                }
            else:
                # Add item to current section
                current_section["items"].append(text_item)

        # Add final section
        if current_section["items"]:
            sections.append(current_section)

        # Step 2: Chunk each section independently
        char_offset = 0
        for section in sections:
            section_header = section["header"]
            section_level = section["level"]
            section_items = section["items"]

            # Extract section text and metadata
            section_parts = []
            section_page = None

            for item in section_items:
                text_content = self._extract_text_item_content(item)
                if not text_content:
                    continue
                section_parts.append(text_content)

                # Track page number
                prov = item.get("prov", [])
                if prov and len(prov) > 0:
                    page_no = prov[0].get("page_no")
                    if page_no is not None:
                        section_page = page_no

            # Join section text
            section_pages_set: set[int] = set()
            for i in section_items:
                page_no = self._extract_page_number_from_item(i)
                if page_no is not None:
                    section_pages_set.add(int(page_no))
            section_pages = sorted(section_pages_set)

            for p in section_pages:
                for table_block in table_blocks_by_page.get(p, []):
                    section_parts.append(table_block)

            section_text = "\n\n".join(section_parts)
            section_tokens = self.count_tokens(section_text)
            section_metadata_summary = self._build_section_metadata_summary(
                section_items=section_items,
                asset_paths=asset_paths,
            )
            section_figure_assets: List[Dict[str, Any]] = []
            section_table_assets: List[Dict[str, Any]] = []
            for p in section_pages:
                section_figure_assets.extend(figure_assets_by_page.get(p, []))
                section_table_assets.extend(table_assets_by_page.get(p, []))

            figure_paths = [
                asset.get("crop_path") or asset.get("image_path")
                for asset in section_figure_assets
                if isinstance(asset, dict)
                and ((asset.get("crop_path") or asset.get("image_path")) is not None)
            ]
            table_chunk_refs = [
                asset.get("table_chunk_ref")
                for asset in section_table_assets
                if isinstance(asset, dict) and asset.get("table_chunk_ref")
            ]

            section_metadata_summary = {
                "type": "text",
                "source": "docling",
                "links": {
                    "figure_paths": figure_paths[:6],
                    "table_chunk_refs": table_chunk_refs[:10],
                },
            }

            # Determine primary label for this section
            # (use most common label, or "text" as default)
            labels = [item.get("label", "text") for item in section_items]
            primary_label = max(set(labels), key=labels.count) if labels else "text"

            # If section fits in one chunk, keep it whole
            if section_tokens <= self.max_tokens:
                char_start = char_offset
                char_end = char_offset + len(section_text)

                chunks.append(
                    ChunkWithMetadata(
                        text=section_text,
                        token_count=section_tokens,
                        section_title=section_header,
                        page_number=section_page,
                        label=primary_label,
                        level=section_level,
                        char_start=char_start,
                        char_end=char_end,
                        docling_metadata=section_metadata_summary,
                    )
                )

                char_offset = char_end + 2  # +2 for newlines between sections
            else:
                # Section too long - split with overlap but stay within section
                section_chunks = self._split_section_with_overlap(
                    section_text=section_text,
                    section_header=section_header,
                    section_level=section_level,
                    section_page=section_page,
                    primary_label=primary_label,
                    metadata_items=section_metadata_summary,
                    char_offset=char_offset,
                )
                chunks.extend(section_chunks)

                # Update char offset
                if section_chunks:
                    last_chunk_end = section_chunks[-1].char_end
                    char_offset = (
                        (last_chunk_end + 2)
                        if last_chunk_end is not None
                        else (char_offset + len(section_text) + 2)
                    )

        evidence_chunks = self._build_multimodal_evidence_chunks(
            doc_dict=doc_dict,
            formula_items=formula_items,
            section_anchors=section_anchors,
            start_char_offset=char_offset,
        )
        chunks.extend(evidence_chunks)

        logger.info(
            f"[{paper_id}] Created {len(chunks)} chunks from docling structure "
            f"({len(evidence_chunks)} table atomic chunks; formulas kept inline)"
        )
        return chunks

    def _split_section_with_overlap(
        self,
        section_text: str,
        section_header: Optional[str],
        section_level: Optional[int],
        section_page: Optional[int],
        primary_label: str,
        metadata_items: Dict[str, Any],
        char_offset: int,
    ) -> List[ChunkWithMetadata]:
        """
        Split a long section into multiple chunks with overlap.
        Chunks stay within section boundaries.

        Args:
            section_text: Full section text
            section_header: Section title
            section_level: Section hierarchy level
            section_page: Page number
            primary_label: Primary label for chunks
            metadata_items: Docling metadata
            char_offset: Starting character offset

        Returns:
            List of ChunkWithMetadata for this section
        """
        chunks: List[ChunkWithMetadata] = []

        blocks = [b.strip() for b in section_text.split("\n\n") if b.strip()]
        if not blocks:
            return chunks

        current_blocks: List[str] = []
        current_tokens = 0
        chunk_start_offset = char_offset

        def flush_chunk() -> None:
            nonlocal chunk_start_offset, current_blocks, current_tokens
            if not current_blocks:
                return
            chunk_text = "\n\n".join(current_blocks)
            chunk_end_offset = chunk_start_offset + len(chunk_text)
            chunks.append(
                ChunkWithMetadata(
                    text=chunk_text,
                    token_count=current_tokens,
                    section_title=section_header,
                    page_number=section_page,
                    label=primary_label,
                    level=section_level,
                    char_start=chunk_start_offset,
                    char_end=chunk_end_offset,
                    docling_metadata=metadata_items,
                )
            )

            overlap_tokens = int(self.max_tokens * self.overlap_ratio)
            overlap_blocks: List[str] = []
            overlap_count = 0
            for b in reversed(current_blocks):
                b_tokens = self.count_tokens(b)
                if overlap_count + b_tokens <= overlap_tokens:
                    overlap_blocks.insert(0, b)
                    overlap_count += b_tokens
                else:
                    break

            current_blocks = overlap_blocks
            current_tokens = overlap_count
            if overlap_blocks:
                overlap_text = "\n\n".join(overlap_blocks)
                chunk_start_offset = chunk_end_offset - len(overlap_text)
            else:
                chunk_start_offset = chunk_end_offset

        for block in blocks:
            block_tokens = self.count_tokens(block)

            if block_tokens > self.max_tokens:
                if "$$" in block or "\\(" in block or "\\[" in block:
                    if current_blocks:
                        flush_chunk()
                    current_blocks.append(block)
                    current_tokens += block_tokens
                    flush_chunk()
                    continue

                sentences = self.split_into_sentences(block)
                sentence_buf: List[str] = []
                sentence_tokens = 0
                for sentence in sentences:
                    st = self.count_tokens(sentence)
                    if sentence_tokens + st > self.max_tokens and sentence_buf:
                        current_blocks.append(" ".join(sentence_buf))
                        current_tokens += sentence_tokens
                        flush_chunk()
                        sentence_buf = []
                        sentence_tokens = 0
                    sentence_buf.append(sentence)
                    sentence_tokens += st
                if sentence_buf:
                    joined = " ".join(sentence_buf)
                    joined_tokens = self.count_tokens(joined)
                    if current_tokens + joined_tokens > self.max_tokens and current_blocks:
                        flush_chunk()
                    current_blocks.append(joined)
                    current_tokens += joined_tokens
                continue

            if current_tokens + block_tokens > self.max_tokens and current_blocks:
                flush_chunk()

            current_blocks.append(block)
            current_tokens += block_tokens

        if current_blocks:
            chunk_text = "\n\n".join(current_blocks)
            chunk_end_offset = chunk_start_offset + len(chunk_text)
            chunks.append(
                ChunkWithMetadata(
                    text=chunk_text,
                    token_count=current_tokens,
                    section_title=section_header,
                    page_number=section_page,
                    label=primary_label,
                    level=section_level,
                    char_start=chunk_start_offset,
                    char_end=chunk_end_offset,
                    docling_metadata=metadata_items,
                )
            )

        return chunks

    def chunk_from_tei_structure(
        self, tei_structure: Dict[str, Any], paper_id: str
    ) -> List[ChunkWithMetadata]:
        """
        Chunk document from TEI XML structure with SECTION-BOUNDARY-AWARE strategy.

        TEI structure format:
        {
            "title": str,
            "authors": List[{name, affiliation, email}],
            "abstract": str,
            "sections": List[{title, content}],
            "references": List[{raw_text}]
        }

        Strategy:
        - Keep entire abstract as one chunk (if <= max_tokens)
        - Keep entire section as one chunk (if <= max_tokens)
        - Split long sections with 10% overlap
        - Never cross section boundaries

        Args:
            tei_structure: TEI structure dictionary from extract_tei_xml_structure()
            paper_id: Paper ID for logging

        Returns:
            List of ChunkWithMetadata tuples
        """
        chunks = []
        char_offset = 0

        # Chunk abstract if present
        abstract = tei_structure.get("abstract", "").strip()
        if abstract:
            abstract_tokens = self.count_tokens(abstract)
            char_start = char_offset
            char_end = char_offset + len(abstract)

            if abstract_tokens <= self.max_tokens:
                # Keep entire abstract as one chunk
                chunks.append(
                    ChunkWithMetadata(
                        text=abstract,
                        token_count=abstract_tokens,
                        section_title="Abstract",
                        page_number=None,
                        label="abstract",
                        level=1,
                        char_start=char_start,
                        char_end=char_end,
                        docling_metadata={},
                    )
                )
            else:
                # Split long abstract with overlap
                abstract_chunks = self._split_section_with_overlap(
                    section_text=abstract,
                    section_header="Abstract",
                    section_level=1,
                    section_page=None,
                    primary_label="abstract",
                    metadata_items={"source": "tei", "part": "abstract"},
                    char_offset=char_start,
                )
                chunks.extend(abstract_chunks)

            char_offset = char_end + 2

        # Chunk sections
        sections = tei_structure.get("sections", [])
        for section in sections:
            section_title = section.get("title", "").strip()
            section_content = section.get("content", "").strip()

            if not section_content:
                continue

            section_tokens = self.count_tokens(section_content)
            char_start = char_offset
            char_end = char_offset + len(section_content)

            if section_tokens <= self.max_tokens:
                # Keep entire section as one chunk
                chunks.append(
                    ChunkWithMetadata(
                        text=section_content,
                        token_count=section_tokens,
                        section_title=section_title,
                        page_number=None,
                        label="section",
                        level=1,
                        char_start=char_start,
                        char_end=char_end,
                        docling_metadata={},
                    )
                )
            else:
                # Split long section with overlap
                section_chunks = self._split_section_with_overlap(
                    section_text=section_content,
                    section_header=section_title,
                    section_level=1,
                    section_page=None,
                    primary_label="section",
                    metadata_items={"source": "tei", "part": "section"},
                    char_offset=char_start,
                )
                chunks.extend(section_chunks)

            char_offset = char_end + 2

        logger.info(
            f"[{paper_id}] Created {len(chunks)} section-aware chunks from TEI structure"
        )
        return chunks

    def _split_text_into_chunks(
        self, text: str, section_title: Optional[str] = None
    ) -> List[Tuple[str, int]]:
        """
        Split text into overlapping chunks.

        Args:
            text: Text to split
            section_title: Section title for context

        Returns:
            List of (chunk_text, token_count) tuples
        """
        chunks = []
        sentences = self.split_into_sentences(text)

        current_chunk = []
        current_tokens = 0

        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)

            # If adding this sentence exceeds max_tokens, save current chunk
            if current_tokens + sentence_tokens > self.max_tokens and current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append((chunk_text, current_tokens))

                # Start new chunk with overlap
                # Calculate overlap tokens (10% of max_tokens)
                overlap_token_limit = int(self.max_tokens * self.overlap_ratio)

                # Keep last few sentences for context
                overlap_sentences = []
                overlap_tokens = 0
                for s in reversed(current_chunk):
                    s_tokens = self.count_tokens(s)
                    if overlap_tokens + s_tokens <= overlap_token_limit:
                        overlap_sentences.insert(0, s)
                        overlap_tokens += s_tokens
                    else:
                        break

                current_chunk = overlap_sentences
                current_tokens = overlap_tokens

            current_chunk.append(sentence)
            current_tokens += sentence_tokens

        # Add final chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append((chunk_text, current_tokens))

        return chunks

    def chunk_from_structure(
        self, doc_dict: Dict[str, Any], paper_id: str
    ) -> List[Tuple[str, int, Optional[str]]]:
        """
        DEPRECATED: Use chunk_from_docling_structure() instead.
        Kept for backward compatibility.
        """
        # Use new method and convert to old format
        new_chunks = self.chunk_from_docling_structure(doc_dict, paper_id)
        return [(c.text, c.token_count, c.section_title) for c in new_chunks]
        logger.info(
            f"Chunked paper {paper_id} into {len(chunks)} structure-aware chunks"
        )
        return chunks

    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting (can be improved with spaCy or NLTK)
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]

    def chunk_text(
        self, text: str, paper_id: str, preserve_sections: bool = True
    ) -> List[Tuple[str, int, Optional[str]]]:
        """
        DEPRECATED: Use chunk_from_docling_structure() instead for better section-aware chunking.\n
        Chunk text into overlapping pieces

        Args:
            text: Full text to chunk
            paper_id: Paper ID for logging
            preserve_sections: Try to preserve section boundaries

        Returns:
            List of (chunk_text, token_count, section_title) tuples
        """
        chunks = []

        # Try to split by sections first
        if preserve_sections:
            sections = self._split_into_sections(text)
        else:
            sections = [(text, None)]

        chunk_index = 0
        for section_text, section_title in sections:
            section_chunks = self._chunk_section(section_text, section_title)
            chunks.extend(section_chunks)

        logger.info(f"Chunked paper {paper_id} into {len(chunks)} chunks")
        return chunks

    def build_contextualized_embedding_text(self, chunk: ChunkWithMetadata) -> str:
        """Build embedding text with compact structural context."""
        header_parts: List[str] = []
        if chunk.section_title:
            header_parts.append(f"Section: {chunk.section_title}")

        if not header_parts:
            return chunk.text
        return "\n".join(header_parts) + "\n\n" + chunk.text

    def _split_into_sections(self, text: str) -> List[Tuple[str, Optional[str]]]:
        """
        Split text into sections based on headings.
        Enhanced to work with markdown headings from docling.

        Returns:
            List of (section_text, section_title) tuples
        """
        sections: List[Tuple[str, Optional[str]]] = []

        # Pattern to match both markdown headings and traditional section headings
        markdown_heading = r"^#{1,3}\s+(.+)$"
        traditional_heading = r"^(?:\d+\.?\s+)?([A-Z][A-Za-z\s]+)$"

        lines = text.split("\n")
        current_section = []
        current_title: Optional[str] = None

        for line in lines:
            stripped = line.strip()

            # Check for markdown heading (from docling)
            md_match = re.match(markdown_heading, stripped)
            if md_match:
                # Save previous section
                if current_section:
                    section_text = "\n".join(current_section)
                    sections.append((section_text, current_title))

                # Start new section
                current_title = md_match.group(1).strip()
                current_section = []
                continue

            # Check for traditional heading
            if len(stripped) < 100 and re.match(traditional_heading, stripped):
                # Save previous section
                if current_section:
                    section_text = "\n".join(current_section)
                    sections.append((section_text, current_title))

                # Start new section
                current_title = stripped
                current_section = []
            else:
                current_section.append(line)

        # Add last section
        if current_section:
            section_text = "\n".join(current_section)
            sections.append((section_text, current_title))

        # If no sections found, return entire text
        if not sections:
            sections = [(text, None)]

        return sections

    def _chunk_section(
        self, section_text: str, section_title: Optional[str]
    ) -> List[Tuple[str, int, Optional[str]]]:
        """
        Chunk a single section into overlapping pieces

        Returns:
            List of (chunk_text, token_count, section_title) tuples
        """
        chunks = []
        sentences = self.split_into_sentences(section_text)

        current_chunk = []
        current_tokens = 0

        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)

            # If adding this sentence exceeds max_tokens, save current chunk
            if current_tokens + sentence_tokens > self.max_tokens and current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append((chunk_text, current_tokens, section_title))

                # Start new chunk with overlap
                # Calculate overlap tokens (10% of max_tokens)
                overlap_token_limit = int(self.max_tokens * self.overlap_ratio)

                overlap_chunk = []
                overlap_tokens = 0

                # Add sentences from end of previous chunk for overlap
                for prev_sentence in reversed(current_chunk):
                    prev_tokens = self.count_tokens(prev_sentence)
                    if overlap_tokens + prev_tokens <= overlap_token_limit:
                        overlap_chunk.insert(0, prev_sentence)
                        overlap_tokens += prev_tokens
                    else:
                        break

                current_chunk = overlap_chunk
                current_tokens = overlap_tokens

            current_chunk.append(sentence)
            current_tokens += sentence_tokens

        # Add final chunk if it meets minimum size
        if current_chunk and current_tokens >= self.min_tokens:
            chunk_text = " ".join(current_chunk)
            chunks.append((chunk_text, current_tokens, section_title))
        elif current_chunk and chunks:
            # If final chunk is too small, append to last chunk
            chunk_text = " ".join(current_chunk)
            last_chunk, last_tokens, last_title = chunks[-1]
            combined_text = last_chunk + " " + chunk_text
            combined_tokens = self.count_tokens(combined_text)
            chunks[-1] = (combined_text, combined_tokens, last_title or section_title)
        elif current_chunk:
            # If only one small chunk, keep it anyway
            chunk_text = " ".join(current_chunk)
            chunks.append((chunk_text, current_tokens, section_title))

        return chunks

    def create_chunk_id(self, paper_id: str, chunk_index: int) -> str:
        """
        Create chunk ID in format P12345::C7

        Args:
            paper_id: Paper ID (e.g., P12345)
            chunk_index: 0-based chunk index

        Returns:
            Chunk ID (e.g., P12345::C7)
        """
        return f"{paper_id}::C{chunk_index}"
