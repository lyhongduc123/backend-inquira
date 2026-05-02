import io
import importlib
import json
import re
import unicodedata
import gc
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.pipeline_options import (
    ThreadedPdfPipelineOptions,
)
from docling.pipeline.threaded_standard_pdf_pipeline import ThreadedStandardPdfPipeline
from docling.datamodel.base_models import InputFormat
from docling_core.types.io import DocumentStream
from docling_core.transforms.chunker.hierarchical_chunker import HierarchicalChunker
from app.extensions.logger import create_logger
from app.modules.r2_storage import R2StorageService
import xml.etree.ElementTree as ET

fitz = None
try:  # pragma: no cover
    fitz = importlib.import_module("pymupdf")
except Exception:
    try:
        fitz = importlib.import_module("fitz")
    except Exception:
        fitz = None

logger = create_logger(__name__)

class ExtractorService:
    def __init__(
        self,
        use_cuda: bool = True,
        use_ocr: bool = False,
        assets_dir: Optional[str] = None,
        export_hierarchical_chunks: bool = False,
        enable_pymupdf_crops: bool = True,
        persist_local_assets: bool = True,
    ):
        """Initialize the extractor service with docling converter
        
        Args:
            use_cuda: Whether to attempt CUDA acceleration (falls back to CPU on error)
            use_ocr: Whether to enable OCR in Docling pipeline
            generate_picture_images: Whether to ask Docling to render picture images
            assets_dir: Optional directory to persist extracted assets (figures/tables)
            export_hierarchical_chunks: Whether to export Docling hierarchical chunks
            enable_pymupdf_crops: Whether to crop table/figure regions from raw PDF via PyMuPDF
            persist_local_assets: Whether to keep persisted assets on local disk after extraction
        """
        self._use_cuda = use_cuda
        self._use_ocr = use_ocr
        self._assets_dir = assets_dir
        self._export_hierarchical_chunks = export_hierarchical_chunks
        self._enable_pymupdf_crops = enable_pymupdf_crops
        self._persist_local_assets = persist_local_assets
        self._cuda_failed = False
        self.converter: Optional[DocumentConverter] = None
        self.hierarchical_chunker: Optional[HierarchicalChunker] = None
        self.r2_storage = R2StorageService()
        self._initialize_converter()

    def _initialize_converter(self):
        """Initialize the converter with appropriate device settings"""
        device = AcceleratorDevice.CUDA if (self._use_cuda and not self._cuda_failed) else AcceleratorDevice.CPU
        
        try:
            logger.debug(f"Initializing Docling converter with device: {device}")
            pipeline_options = ThreadedPdfPipelineOptions(
                accelerator_options=AcceleratorOptions(
                    device=device,
                ),
                do_ocr=self._use_ocr,
                # do_formula_enrichment=True, # 10-15x slower on 3050 laptop
            )
            
            self.converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_cls=ThreadedStandardPdfPipeline,
                        pipeline_options=pipeline_options
                    )
                }
            )
            self.converter.initialize_pipeline(InputFormat.PDF)
            logger.debug("Docling converter initialized successfully")
            
        except Exception as e:
            if device == AcceleratorDevice.CUDA and "CUDA" in str(e):
                logger.warning(f"CUDA initialization failed: {e}. Falling back to CPU")
                self._cuda_failed = True
                gc.collect()
                self._initialize_converter()  # Retry with CPU
            else:
                raise

    def extract_pdf_structure(self, pdf_bytes: bytes, paper_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract structured data from PDF bytes using docling.
        Returns a dictionary with document structure preserved.
        
        Args:
            pdf_bytes (bytes): The PDF file content in bytes.
            paper_id (Optional[str]): Paper ID used to persist optional assets.
        Returns:
            Dictionary containing structured document data with sections, paragraphs, tables, etc.
        """
        try:
            pdf_file = DocumentStream(name="input.pdf", stream=io.BytesIO(pdf_bytes))
            if self.converter is None:
                raise RuntimeError("Docling converter is not initialized")

            result = self.converter.convert(source=pdf_file)
            doc_dict = result.document.export_to_dict()

            if paper_id and (self._assets_dir or self.r2_storage.is_configured):
                self._persist_docling_assets(
                    paper_id=paper_id,
                    doc_dict=doc_dict,
                    document=result.document,
                    pdf_bytes=pdf_bytes,
                )

            logger.info(
                f"Successfully extracted structured document using docling"
            )
            return doc_dict

        except RuntimeError as e:
            # Handle CUDA errors with CPU fallback
            if ("CUDA" in str(e) or "CUBLAS" in str(e)) and not self._cuda_failed:
                logger.warning(f"CUDA error during PDF extraction: {e}. Reinitializing with CPU")
                self._cuda_failed = True
                gc.collect()
                self._initialize_converter()
                # Retry with CPU
                pdf_file = DocumentStream(name="input.pdf", stream=io.BytesIO(pdf_bytes))
                if self.converter is None:
                    raise RuntimeError("Docling converter is not initialized after CPU fallback")
                result = self.converter.convert(source=pdf_file)
                doc_dict = result.document.export_to_dict()
                if paper_id and (self._assets_dir or self.r2_storage.is_configured):
                    self._persist_docling_assets(
                        paper_id=paper_id,
                        doc_dict=doc_dict,
                        document=result.document,
                        pdf_bytes=pdf_bytes,
                    )
                logger.info("Successfully extracted document using CPU fallback")
                return doc_dict
            else:
                logger.error(f"Error extracting PDF structure with docling: {e}")
                raise Exception(f"Failed to extract PDF structure: {e}")
        except Exception as e:
            logger.error(f"Error extracting PDF structure with docling: {e}")
            raise Exception(f"Failed to extract PDF structure: {e}")

    def extract_pdf_structure_with_pymupdf(
        self,
        pdf_bytes: bytes,
        paper_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Extract a lightweight docling-like structure using PyMuPDF.

        This path is intended for live ingestion where latency is prioritized.
        The output shape remains compatible with chunking logic expecting
        `texts`, `tables`, and `pictures` keys.
        """
        if fitz is None:
            raise RuntimeError("PyMuPDF is not available for live PDF extraction")

        try:
            pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            text_items: List[Dict[str, Any]] = []

            for page_index in range(len(pdf_doc)):
                page = pdf_doc.load_page(page_index)
                page_no = page_index + 1

                blocks = page.get_text("blocks") or []
                for block in blocks:
                    if not isinstance(block, (list, tuple)) or len(block) < 5:
                        continue

                    x0, y0, x1, y1, text = block[:5]
                    if not isinstance(text, str):
                        continue

                    cleaned = self._fix_text_encoding(text)
                    if not cleaned:
                        continue

                    label = "section_header" if self._looks_like_section_header(cleaned) else "text"
                    text_items.append(
                        {
                            "text": cleaned,
                            "orig": cleaned,
                            "label": label,
                            "level": 1 if label == "section_header" else 2,
                            "content_layer": "body",
                            "prov": [
                                {
                                    "page_no": page_no,
                                    "bbox": {
                                        "l": float(x0),
                                        "t": float(y0),
                                        "r": float(x1),
                                        "b": float(y1),
                                        "coord_origin": "TOPLEFT",
                                    },
                                }
                            ],
                        }
                    )

            try:
                pdf_doc.close()
            except Exception:
                pass

            logger.info(
                "Successfully extracted lightweight document structure using PyMuPDF"
                + (f" for {paper_id}" if paper_id else "")
            )

            return {
                "texts": text_items,
                "tables": [],
                "pictures": [],
                "asset_paths": {},
                "extraction_backend": "pymupdf",
            }
        except Exception as e:
            logger.error(f"Error extracting PDF structure with PyMuPDF: {e}")
            raise Exception(f"Failed to extract PDF structure with PyMuPDF: {e}")

    @staticmethod
    def _looks_like_section_header(text: str) -> bool:
        candidate = (text or "").strip()
        if not candidate:
            return False

        # Heuristic: short heading-like lines are treated as section headers.
        if len(candidate) > 120:
            return False

        if "\n" in candidate:
            return False

        return bool(re.match(r"^(\d+(?:\.\d+)*)?\s*[A-Z][A-Za-z0-9\-,:()\s]{2,}$", candidate))

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
    def _extract_text_value(text_item: Dict[str, Any]) -> str:
        text = (text_item.get("text") or "").strip()
        if text:
            return text
        return (text_item.get("orig") or "").strip()

    def _persist_docling_assets(
        self,
        paper_id: str,
        doc_dict: Dict[str, Any],
        document: Any,
        pdf_bytes: Optional[bytes] = None,
    ) -> None:
        """Persist figures/tables and optional markdown/chunks into per-paper assets directory."""
        temporary_storage: Optional[tempfile.TemporaryDirectory[str]] = None
        # persist_local = bool(self._persist_local_assets and self._assets_dir)
        persist_local = False  # Enforced

        if persist_local:
            base_path = Path(self._assets_dir or "").expanduser().resolve() / paper_id
            base_path.mkdir(parents=True, exist_ok=True)
        else:
            temporary_storage = tempfile.TemporaryDirectory(prefix=f"docling_{paper_id}_")
            base_path = Path(temporary_storage.name) / paper_id
            base_path.mkdir(parents=True, exist_ok=True)

        assets_manifest: Dict[str, Any] = {
            "paper_id": paper_id,
            "base_path": str(base_path),
            "tables": [],
            "figures": [],
            "tables_crop_dir": str(base_path / "tables" / "crops"),
            "figures_crop_dir": str(base_path / "figures" / "crops"),
            "crop_backend": "pymupdf" if self._enable_pymupdf_crops else "disabled",
        }

        try:
            pdf_doc = None
            if self._enable_pymupdf_crops and pdf_bytes and fitz is not None:
                try:
                    pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                except Exception as open_error:
                    logger.warning(f"Failed to open PDF with PyMuPDF for {paper_id}: {open_error}")
            elif self._enable_pymupdf_crops and fitz is None:
                logger.warning("PyMuPDF is not available. Skipping visual crops.")

            self._persist_table_assets(
                base_path=base_path,
                doc_dict=doc_dict,
                manifest=assets_manifest,
                pdf_doc=pdf_doc,
            )
            self._persist_figure_assets(
                base_path=base_path,
                doc_dict=doc_dict,
                document=document,
                manifest=assets_manifest,
                paper_id=paper_id,
                pdf_doc=pdf_doc,
            )
            # self._persist_hierarchical_chunks(
            #     base_path=base_path,
            #     document=document,
            #     manifest=assets_manifest,
            # )

            if pdf_doc is not None:
                try:
                    pdf_doc.close()
                except Exception:
                    pass

            manifest_path = base_path / "assets_manifest.json"
            manifest_path.write_text(
                json.dumps(assets_manifest, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )

            doc_dict.setdefault("asset_paths", {})
            if persist_local:
                doc_dict["asset_paths"].update(
                    {
                        "base_path": str(base_path),
                        "manifest_path": str(manifest_path),
                        "tables_dir": str(base_path / "tables"),
                        "figures_dir": str(base_path / "figures"),
                        "tables_crop_dir": str(base_path / "tables" / "crops"),
                        "figures_crop_dir": str(base_path / "figures" / "crops"),
                    }
                )

            self._upload_assets_to_r2(
                paper_id=paper_id,
                base_path=base_path,
                manifest=assets_manifest,
                manifest_path=manifest_path,
                doc_dict=doc_dict,
            )

            doc_dict["asset_manifest"] = assets_manifest
        finally:
            if temporary_storage is not None:
                temporary_storage.cleanup()

    def _upload_assets_to_r2(
        self,
        paper_id: str,
        base_path: Path,
        manifest: Dict[str, Any],
        manifest_path: Path,
        doc_dict: Dict[str, Any],
    ) -> None:
        """Upload persisted local assets (tables/figures/crops/json) to Cloudflare R2 when enabled."""
        if not self.r2_storage.is_configured:
            return

        def upload_local_file(local_path_value: Optional[str]) -> Optional[Dict[str, Any]]:
            if not local_path_value:
                return None

            local_path = Path(local_path_value)
            if not local_path.exists() or not local_path.is_file():
                return None

            try:
                relative_path = local_path.relative_to(base_path)
            except Exception:
                relative_path = Path(local_path.name)

            key = self.r2_storage.build_asset_key(
                paper_id=paper_id,
                relative_path=str(relative_path).replace("\\", "/"),
            )
            return self.r2_storage.upload_file(local_path=local_path, key=key)

        try:
            for table_entry in manifest.get("tables", []) or []:
                if not isinstance(table_entry, dict):
                    continue
                path_upload = upload_local_file(table_entry.get("path"))
                crop_upload = upload_local_file(table_entry.get("crop_path"))
                table_entry["r2"] = {
                    "path": path_upload,
                    "crop_path": crop_upload,
                }
                if path_upload:
                    table_entry["path"] = path_upload.get("url") or path_upload.get("key")
                if crop_upload:
                    table_entry["crop_path"] = crop_upload.get("url") or crop_upload.get("key")

            for figure_entry in manifest.get("figures", []) or []:
                if not isinstance(figure_entry, dict):
                    continue
                metadata_upload = upload_local_file(figure_entry.get("metadata_path"))
                image_upload = upload_local_file(figure_entry.get("image_path"))
                crop_upload = upload_local_file(figure_entry.get("crop_path"))
                figure_entry["r2"] = {
                    "metadata_path": metadata_upload,
                    "image_path": image_upload,
                    "crop_path": crop_upload,
                }
                if metadata_upload:
                    figure_entry["metadata_path"] = (
                        metadata_upload.get("url") or metadata_upload.get("key")
                    )
                if image_upload:
                    figure_entry["image_path"] = image_upload.get("url") or image_upload.get("key")
                if crop_upload:
                    figure_entry["crop_path"] = crop_upload.get("url") or crop_upload.get("key")

            manifest_upload = upload_local_file(str(manifest_path))
            if manifest_upload:
                manifest["r2_manifest"] = manifest_upload
                doc_dict.setdefault("asset_paths", {})
                doc_dict["asset_paths"].update(
                    {
                        "manifest_r2_key": manifest_upload.get("key"),
                        "manifest_r2_url": manifest_upload.get("url"),
                    }
                )

            doc_dict["asset_manifest"] = manifest

            # Write back enriched manifest with R2 links.
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
        except Exception as error:
            logger.warning(f"Failed to upload assets to R2 for {paper_id}: {error}")

    @staticmethod
    def _bbox_from_prov(item: Dict[str, Any]) -> Optional[Tuple[int, Dict[str, Any]]]:
        prov = item.get("prov", []) or []
        if not prov:
            return None
        first = prov[0] or {}
        page_no = first.get("page_no")
        bbox = first.get("bbox")
        if page_no is None or not isinstance(bbox, dict):
            return None
        return int(page_no), bbox

    @staticmethod
    def _to_pymupdf_rect(page_height: float, bbox: Dict[str, Any], pad: float = 6.0):
        if fitz is None:
            return None
        l = float(bbox.get("l", 0.0))
        r = float(bbox.get("r", 0.0))
        t = float(bbox.get("t", 0.0))
        b = float(bbox.get("b", 0.0))
        origin = str(bbox.get("coord_origin") or "BOTTOMLEFT").upper()

        if origin == "BOTTOMLEFT":
            y0 = page_height - t
            y1 = page_height - b
        else:
            y0 = min(t, b)
            y1 = max(t, b)

        x0 = min(l, r)
        x1 = max(l, r)

        rect = fitz.Rect(x0 - pad, y0 - pad, x1 + pad, y1 + pad)
        return rect

    def _save_crop(
        self,
        pdf_doc: Any,
        page_no: int,
        bbox: Dict[str, Any],
        output_path: Path,
        zoom: float = 2.0,
    ) -> Optional[str]:
        if fitz is None or pdf_doc is None:
            return None
        try:
            page_index = max(page_no - 1, 0)
            page = pdf_doc.load_page(page_index)
            rect = self._to_pymupdf_rect(page.rect.height, bbox)
            if rect is None:
                return None
            clip = rect & page.rect
            if clip.is_empty:
                return None
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=clip, alpha=False)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            pix.save(str(output_path))
            return str(output_path)
        except Exception as crop_error:
            logger.warning(f"Failed to save crop at {output_path}: {crop_error}")
            return None

    def _persist_table_assets(
        self,
        base_path: Path,
        doc_dict: Dict[str, Any],
        manifest: Dict[str, Any],
        pdf_doc: Any = None,
    ) -> None:
        tables = doc_dict.get("tables", []) or []
        texts = doc_dict.get("texts", []) or []
        tables_dir = base_path / "tables"
        tables_dir.mkdir(parents=True, exist_ok=True)
        crop_dir = tables_dir / "crops"

        for idx, table in enumerate(tables):
            data = table.get("data", {}) or {}
            cells = data.get("table_cells", []) or []
            if not cells:
                continue

            rows: Dict[int, List[Dict[str, Any]]] = {}
            for cell in cells:
                row_idx = int(cell.get("start_row_offset_idx", 0) or 0)
                rows.setdefault(row_idx, []).append(cell)

            row_lines: List[str] = []
            for row_idx in sorted(rows.keys()):
                ordered_cells = sorted(rows[row_idx], key=lambda c: int(c.get("start_col_offset_idx", 0) or 0))
                values = [str(c.get("text") or "").replace("\n", " ").strip() for c in ordered_cells]
                row_lines.append("\t".join(values))

            table_file = tables_dir / f"table_{idx:03d}.tsv"
            table_file.write_text("\n".join(row_lines), encoding="utf-8")

            caption_texts: List[str] = []
            for caption_ref in table.get("captions", []) or []:
                text_idx = self._resolve_ref_index(caption_ref, "texts")
                if text_idx is not None and 0 <= text_idx < len(texts):
                    caption = self._extract_text_value(texts[text_idx])
                    if caption:
                        caption_texts.append(caption)

            manifest["tables"].append(
                {
                    "index": idx,
                    "path": str(table_file),
                    "caption": caption_texts,
                    "cell_count": len(cells),
                    "prov": table.get("prov"),
                    "crop_path": None,
                }
            )

            bbox_info = self._bbox_from_prov(table)
            if bbox_info and pdf_doc is not None:
                page_no, bbox = bbox_info
                crop_path = self._save_crop(
                    pdf_doc=pdf_doc,
                    page_no=page_no,
                    bbox=bbox,
                    output_path=crop_dir / f"table_{idx:03d}.png",
                )
                manifest["tables"][-1]["crop_path"] = crop_path

    def _persist_figure_assets(
        self,
        base_path: Path,
        doc_dict: Dict[str, Any],
        document: Any,
        manifest: Dict[str, Any],
        paper_id: str,
        pdf_doc: Any = None,
    ) -> None:
        pictures = doc_dict.get("pictures", []) or []
        texts = doc_dict.get("texts", []) or []

        figures_dir = base_path / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        crop_dir = figures_dir / "crops"

        doc_pictures = getattr(document, "pictures", None)

        for idx, picture in enumerate(pictures):
            caption_texts: List[str] = []
            for caption_ref in picture.get("captions", []) or []:
                text_idx = self._resolve_ref_index(caption_ref, "texts")
                if text_idx is not None and 0 <= text_idx < len(texts):
                    caption = self._extract_text_value(texts[text_idx])
                    if caption:
                        caption_texts.append(caption)

            figure_image_path: Optional[str] = None
            if doc_pictures and idx < len(doc_pictures):
                figure_obj = doc_pictures[idx]
                image_obj = getattr(figure_obj, "image", None)
                if image_obj is not None and hasattr(image_obj, "save"):
                    image_file = figures_dir / f"figure_{idx:03d}.png"
                    try:
                        image_obj.save(image_file)
                        figure_image_path = str(image_file)
                    except Exception as image_error:
                        logger.warning(f"Could not save figure image {idx} for {paper_id}: {image_error}")

            figure_meta_path = figures_dir / f"figure_{idx:03d}.json"
            figure_meta = {
                "index": idx,
                "captions": caption_texts,
                "annotations": picture.get("annotations") or [],
                "prov": picture.get("prov"),
                "image_path": figure_image_path,
            }
            figure_meta_path.write_text(
                json.dumps(figure_meta, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )

            manifest["figures"].append(
                {
                    "index": idx,
                    "metadata_path": str(figure_meta_path),
                    "image_path": figure_image_path,
                    "crop_path": None,
                }
            )

            bbox_info = self._bbox_from_prov(picture)
            if bbox_info and pdf_doc is not None:
                page_no, bbox = bbox_info
                crop_path = self._save_crop(
                    pdf_doc=pdf_doc,
                    page_no=page_no,
                    bbox=bbox,
                    output_path=crop_dir / f"figure_{idx:03d}.png",
                )
                manifest["figures"][-1]["crop_path"] = crop_path

    def _persist_hierarchical_chunks(
        self,
        base_path: Path,
        document: Any,
        manifest: Dict[str, Any],
    ) -> None:
        """Persist Docling hierarchical chunks for section-aware downstream chunking."""
        if not self._export_hierarchical_chunks:
            return

        try:
            hierarchical_chunker = self.hierarchical_chunker or HierarchicalChunker()
            self.hierarchical_chunker = hierarchical_chunker

            serialized_chunks: List[Dict[str, Any]] = []
            for idx, chunk in enumerate(hierarchical_chunker.chunk(dl_doc=document)):
                chunk_text = getattr(chunk, "text", None)
                if isinstance(chunk_text, str):
                    chunk_text = chunk_text.strip()

                if not chunk_text and hasattr(hierarchical_chunker, "contextualize"):
                    try:
                        chunk_text = hierarchical_chunker.contextualize(chunk=chunk)
                    except Exception:
                        chunk_text = ""

                meta = getattr(chunk, "meta", None)
                if meta is not None and hasattr(meta, "model_dump"):
                    try:
                        meta = meta.model_dump(mode="json")
                    except Exception:
                        meta = str(meta)
                elif meta is not None and hasattr(meta, "dict"):
                    try:
                        meta = meta.dict()
                    except Exception:
                        meta = str(meta)

                serialized_chunks.append(
                    {
                        "index": idx,
                        "text": chunk_text or "",
                        "metadata": meta,
                    }
                )

            if not serialized_chunks:
                return

            chunk_path = base_path / "hierarchical_chunks.json"
            chunk_path.write_text(
                json.dumps(serialized_chunks, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
            manifest["hierarchical_chunks_path"] = str(chunk_path)
        except Exception as chunk_error:
            logger.warning(
                f"Failed to export hierarchical chunks for assets at {base_path}: {chunk_error}"
            )

    def _fix_text_encoding(self, text: str) -> str:
        """
        Fix common PDF text encoding issues
        Args:
            text: Raw extracted text
        Returns:
            Cleaned text
        """
        # Normalize unicode characters
        text = unicodedata.normalize("NFKD", text)

        # Remove non-printable characters except newlines/tabs
        text = "".join(char for char in text if char.isprintable() or char in "\n\t")

        # Fix common ligatures that get mangled
        ligature_map = {
            "ﬁ": "fi",
            "ﬂ": "fl",
            "ﬀ": "ff",
            "ﬃ": "ffi",
            "ﬄ": "ffl",
            "ﬅ": "ft",
            "ﬆ": "st",
            "€": "e",
            "�": "",
        }

        for bad, good in ligature_map.items():
            text = text.replace(bad, good)

        # Fix multiple spaces
        text = re.sub(r" +", " ", text)

        # Fix multiple newlines (keep max 2)
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    def extract_tei_xml_structure(self, tei_xml: str) -> Dict[str, Any]:
        """
        Extract structured data from GROBID TEI XML.
        
        TEI (Text Encoding Initiative) XML from GROBID provides rich structured data:
        - Title, authors, affiliations
        - Abstract
        - Full text with section headers
        - References
        - Figures and tables metadata
        
        Args:
            tei_xml (str): TEI XML string from GROBID
            
        Returns:
            Dictionary with structured document data
        """
        try:
            root = ET.fromstring(tei_xml)
            
            # TEI namespace
            ns = {'tei': 'http://www.tei-c.org/ns/1.0'}
            
            # Extract metadata
            title_elem = root.find('.//tei:titleStmt/tei:title[@type="main"]', ns)
            title = title_elem.text if title_elem is not None else ""
            
            # Extract authors
            authors = []
            for author in root.findall('.//tei:sourceDesc//tei:author', ns):
                persName = author.find('.//tei:persName', ns)
                if persName is not None:
                    forename = persName.find('.//tei:forename[@type="first"]', ns)
                    surname = persName.find('.//tei:surname', ns)
                    
                    author_name = ""
                    if forename is not None and forename.text:
                        author_name = forename.text + " "
                    if surname is not None and surname.text:
                        author_name += surname.text
                    
                    # Extract affiliation
                    affiliation_elem = author.find('.//tei:affiliation/tei:orgName[@type="institution"]', ns)
                    affiliation = affiliation_elem.text if affiliation_elem is not None else None
                    
                    if author_name.strip():
                        authors.append({
                            'name': author_name.strip(),
                            'affiliation': affiliation
                        })
            
            # Extract abstract
            abstract_elem = root.find('.//tei:profileDesc/tei:abstract', ns)
            abstract = self._extract_text_from_element(abstract_elem, ns) if abstract_elem is not None else ""
            
            # Extract body sections
            sections = []
            body = root.find('.//tei:text/tei:body', ns)
            if body is not None:
                for div in body.findall('.//tei:div', ns):
                    head_elem = div.find('.//tei:head', ns)
                    section_title = head_elem.text if head_elem is not None else "Unknown Section"
                    
                    # Extract paragraphs in this section
                    paragraphs = []
                    for p in div.findall('.//tei:p', ns):
                        p_text = self._extract_text_from_element(p, ns)
                        if p_text.strip():
                            paragraphs.append(p_text.strip())
                    
                    if paragraphs:
                        sections.append({
                            'title': section_title,
                            'content': paragraphs
                        })
            
            # Extract references
            references = []
            for biblStruct in root.findall('.//tei:listBibl/tei:biblStruct', ns):
                ref_title_elem = biblStruct.find('.//tei:analytic/tei:title[@type="main"]', ns)
                if ref_title_elem is None:
                    ref_title_elem = biblStruct.find('.//tei:monogr/tei:title', ns)
                
                ref_title = ref_title_elem.text if ref_title_elem is not None else ""
                
                # Extract reference authors
                ref_authors = []
                for ref_author in biblStruct.findall('.//tei:author/tei:persName', ns):
                    ref_author_text = self._extract_text_from_element(ref_author, ns)
                    if ref_author_text.strip():
                        ref_authors.append(ref_author_text.strip())
                
                if ref_title and ref_title.strip():
                    references.append({
                        'title': ref_title.strip(),
                        'authors': ref_authors
                    })
            
            result = {
                'title': title,
                'authors': authors,
                'abstract': abstract,
                'sections': sections,
                'references': references
            }
            
            logger.info(f"Successfully extracted TEI XML structure: {len(sections)} sections, {len(references)} references")
            return result
            
        except Exception as e:
            logger.error(f"Error extracting TEI XML structure: {e}")
            raise Exception(f"Failed to extract TEI XML: {e}")
    
    def _extract_text_from_element(self, element: Optional[ET.Element], ns: Dict[str, str]) -> str:
        """
        Recursively extract text from an XML element and its children.
        
        Args:
            element: XML element
            ns: Namespace dictionary
            
        Returns:
            Extracted text
        """
        if element is None:
            return ""
        
        # Get text content
        text_parts = []
        
        # Add element's direct text
        if element.text:
            text_parts.append(element.text)
        
        # Add text from children
        for child in element:
            child_text = self._extract_text_from_element(child, ns)
            if child_text:
                text_parts.append(child_text)
            
            # Add tail text after child element
            if child.tail:
                text_parts.append(child.tail)
        
        return " ".join(text_parts)