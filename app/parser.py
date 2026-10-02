from typing import Generator

from docling.chunking import HierarchicalChunker
from docling.document_converter import DocumentConverter

from app.logger import log
from app.schemas import ParsedChunk


class DocumentParser:
    def __init__(self):
        self.converter = DocumentConverter()
        self.chunker = HierarchicalChunker()

    @staticmethod
    def _extract_page_number(chunk, current_page: int) -> int:
        try:
            return chunk.meta.doc_items[0].prov[0].page_no
        except (AttributeError, IndexError, TypeError):
            return current_page

    def parse_pdf(self, file_path: str) -> Generator[ParsedChunk, None, None]:
        log.info(f"Starting ML parsing {file_path}...")
        chunk_index = 0
        current_page = 1

        try:
            result = self.converter.convert(file_path)
            chunks = self.chunker.chunk(result.document)

            for docling_chunk in chunks:
                if not docling_chunk.text.strip():
                    continue

                current_page = self._extract_page_number(docling_chunk, current_page)

                yield ParsedChunk(
                    page_number=current_page,
                    chunk_index=chunk_index,
                    text=docling_chunk.text.strip()
                )
                chunk_index += 1

        except Exception as e:
            log.error(f"Docling pipeline failed for {file_path}: {e}")
            raise
