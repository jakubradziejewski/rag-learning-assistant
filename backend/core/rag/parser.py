from pathlib import Path
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from docling.chunking import HierarchicalChunker


def parse_pdf(pdf_path: Path) -> list[dict]:
    pipeline_options = PdfPipelineOptions(
        do_ocr=True,
        do_table_structure=True,
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

    return chunks