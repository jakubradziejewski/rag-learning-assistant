import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import List

import yaml
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from docling.chunking import HierarchicalChunker


logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_embed_max_chars() -> int:
    env = os.getenv("EMBED_MAX_CHARS")
    if env:
        try:
            return int(env)
        except Exception:
            pass

    cfg_path = Path("config/config.yml")
    if cfg_path.exists():
        try:
            cfg = yaml.safe_load(cfg_path.read_text()) or {}
            # support top-level `embed_max_chars` or nested `embedding.max_chars`
            val = cfg.get("embed_max_chars") or (cfg.get("embedding") or {}).get("max_chars")
            if val is not None:
                return int(val)
        except Exception:
            pass

    return 512


def get_chunks_in_limit(text: str, max_chars: int | None = None) -> List[str]:
    """Split `text` into chunks not exceeding `max_chars` characters.

    `max_chars` defaults to the configured embedding limit (env `EMBED_MAX_CHARS`,
    `config/config.yml` or 512).
    """
    if max_chars is None:
        max_chars = _get_embed_max_chars()

    if len(text) <= max_chars:
        return [text]

    chunks: List[str] = []
    start = 0
    while start + max_chars < len(text):
        end = min(start + max_chars, len(text))
        chunks.append(text[start:end])
        start = end - 30

    # append the final tail
    if start < len(text):
        chunks.append(text[start:])

    return chunks


def parse_pdf(pdf_path: Path) -> list[dict]:
    logger.info("Parsing PDF started: path=%s", pdf_path)

    pipeline_options = PdfPipelineOptions(
        do_ocr=True,
        do_table_structure=True,
    )

    logger.info("Creating document converter")

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    logger.info("Converting PDF to document")
    result = converter.convert(str(pdf_path))
    doc = result.document

    logger.info("Chunking document")

    chunker = HierarchicalChunker(merge_peers=True)
    raw_chunks = list(chunker.chunk(doc))

    logger.info("Chunking finished: raw_chunks=%s", len(raw_chunks))

    chunks: list[dict] = []
    chunk_counter = 0
    for raw in raw_chunks:
        text = raw.text.strip()
        if not text:
            continue

        page_numbers = sorted(
            {prov.page_no for item in raw.meta.doc_items for prov in item.prov}
        ) if raw.meta.doc_items else []

        section_path = " > ".join(raw.meta.headings) if raw.meta.headings else ""

        # split long text into capped subchunks
        sub_texts = get_chunks_in_limit(text)
        for sub in sub_texts:
            chunks.append({
                "text": sub,
                "page_numbers": page_numbers,
                "section_path": section_path,
                "chunk_index": chunk_counter,
            })
            chunk_counter += 1

    logger.info("Parsing finished: usable_chunks=%s", len(chunks))

    return chunks