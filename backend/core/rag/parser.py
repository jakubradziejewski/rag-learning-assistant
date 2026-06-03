from pathlib import Path
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from docling.chunking import HierarchicalChunker


def _split_chunk_with_overlap(text: str, max_size: int = 300, overlap: int = 30) -> list[str]:
    """Split a chunk into smaller pieces with overlap at whitespace boundaries."""
    if len(text) <= max_size:
        return [text]
    
    chunks = []
    start = 0
    while start < len(text):
        # Find the end position (max_size from current start)
        end = min(start + max_size, len(text))
        
        # Split at whitespace
        if end < len(text):
            search_start = max(start, end - overlap)
            last_space = text.rfind(' ', search_start, end)
            if last_space > start:
                end = last_space + 1

        if start > 0:
            start = max(0, start - overlap)
            first_space = text.find(' ', start, start + overlap)
            if first_space != -1 and first_space < end:
                start = first_space + 1

        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # Move start position back by overlap amount
        start = end - overlap
        if start <= max(0, end - overlap):
            start = end
    
    return chunks


def _is_noise_chunk(text: str, min_size: int = 20, noise_threshold: float = 0.3) -> bool:
    """Check if a chunk should be filtered out as noise."""
    # Filter by minimum size
    if len(text) < min_size:
        return True
    
    # Filter by non-letter character ratio
    letter_count = sum(1 for c in text if c.isalpha())
    non_letter_ratio = 1 - (letter_count / len(text))
    if non_letter_ratio > noise_threshold:
        return True
    
    return False


def _postprocess_chunks(chunks: list[dict]) -> list[dict]:
    """Post-process chunks: split oversized ones, remove noise, update indices."""
    processed = []
    chunk_index = 0
    
    for raw_chunk in chunks:
        text = raw_chunk["text"]
        page_numbers = raw_chunk["page_numbers"]
        section_path = raw_chunk["section_path"]
        
        # Split large chunks
        sub_chunks = _split_chunk_with_overlap(text)
        
        # Filter noise and add to processed list
        for sub_text in sub_chunks:
            if not _is_noise_chunk(sub_text):
                processed.append({
                    "text": sub_text,
                    "page_numbers": page_numbers,
                    "section_path": section_path,
                    "chunk_index": chunk_index,
                })
                chunk_index += 1
    
    return processed


def parse_pdf(pdf_path: Path, do_ocr: bool, do_table_structure: bool = False) -> list[dict]:
    pipeline_options = PdfPipelineOptions(
        do_ocr=do_ocr,
        do_table_structure=do_table_structure,
    )

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    result = converter.convert(str(pdf_path))
    doc = result.document

    chunker = HierarchicalChunker(merge_peers=True)
    raw_chunks = list(chunker.chunk(doc))

    chunks = []
    for idx, raw in enumerate(raw_chunks):
        text = raw.text.strip()
        if not text:
            continue

        page_numbers = sorted(
            {prov.page_no for item in raw.meta.doc_items for prov in item.prov}
        ) if raw.meta.doc_items else []

        section_path = " > ".join(raw.meta.headings) if raw.meta.headings else ""

        chunks.append({
            "text": text,
            "page_numbers": page_numbers,
            "section_path": section_path,
            "chunk_index": idx,
        })

    # Post-process: split large chunks, filter noise
    chunks = _postprocess_chunks(chunks)
    
    return chunks