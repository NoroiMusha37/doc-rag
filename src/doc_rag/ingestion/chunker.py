import logging
from pathlib import Path

from docling.chunking import HybridChunker
from docling_core.transforms.chunker.hierarchical_chunker import (
    ChunkingDocSerializer,
    ChunkingSerializerProvider,
    DocChunk,
)
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling_core.transforms.serializer.markdown import MarkdownTableSerializer
from docling_core.types.doc import DocItem, DocItemLabel, DoclingDocument, PictureItem

from src.doc_rag.config import settings
from src.doc_rag.db.models import ChunkKind
from src.doc_rag.ingestion.schemas import ParsedChunk

logger = logging.getLogger(__name__)

HEADING_SEPARATOR = " > "


class MarkdownTableSerializerProvider(ChunkingSerializerProvider):
    """Makes the chunker write tables as Markdown instead of Docling's cell triplets."""

    def get_serializer(self, doc: DoclingDocument) -> ChunkingDocSerializer:
        return ChunkingDocSerializer(doc=doc, table_serializer=MarkdownTableSerializer())


class DocumentChunker:
    """Splits a DoclingDocument into structure-aware chunks sized for the embedder."""

    def __init__(self) -> None:
        # Token counts must come from the embedder's own tokenizer to respect its limit.
        tokenizer = HuggingFaceTokenizer.from_pretrained(
            model_name=settings.EMBEDDING_MODEL, max_tokens=settings.CHUNK_MAX_TOKENS
        )
        self._chunker = HybridChunker(
            tokenizer=tokenizer, serializer_provider=MarkdownTableSerializerProvider()
        )

    def chunk(self, document: DoclingDocument, assets_dir: Path) -> list[ParsedChunk]:
        """Return the document's chunks in reading order; figure images go to `assets_dir`."""
        pictures = self._pictures_by_ref(document)
        chunks: list[ParsedChunk] = []
        for chunk in self._chunker.chunk(dl_doc=document):
            if not chunk.text.strip():
                continue
            # The chunker is typed as yielding BaseChunk; DocChunk carries items and headings.
            doc_chunk = DocChunk.model_validate(chunk)
            ordinal = len(chunks)
            picture = self._find_picture(doc_chunk.meta.doc_items, pictures)
            asset_path = None
            if picture is not None:
                asset_path = self._save_image(picture, document, assets_dir / f"{ordinal}.png")
            chunks.append(self._build(doc_chunk, ordinal, picture is not None, asset_path))
        logger.info(f"Chunked {document.name}: {len(chunks)} chunks")
        return chunks

    def _build(
        self, doc_chunk: DocChunk, ordinal: int, has_picture: bool, asset_path: str | None
    ) -> ParsedChunk:
        items = doc_chunk.meta.doc_items
        page_start, page_end = self._page_range(items)
        return ParsedChunk(
            ordinal=ordinal,
            kind=self._kind({item.label for item in items}, has_picture),
            text=doc_chunk.text,
            embed_text=self._chunker.contextualize(doc_chunk),
            heading_path=HEADING_SEPARATOR.join(doc_chunk.meta.headings or []),
            page_start=page_start,
            page_end=page_end,
            asset_path=asset_path,
        )

    @staticmethod
    def _pictures_by_ref(document: DoclingDocument) -> dict[str, PictureItem]:
        """Map each picture's own ref and its caption refs to the picture.

        A chunk lists generic item references, and a picture only reaches a chunk
        through its caption text, so captions are the link from chunk to image.
        """
        pictures: dict[str, PictureItem] = {}
        for picture in document.pictures:
            pictures[picture.self_ref] = picture
            for caption in picture.captions:
                pictures[caption.cref] = picture
        return pictures

    @staticmethod
    def _find_picture(items: list[DocItem], pictures: dict[str, PictureItem]) -> PictureItem | None:
        return next((pictures[item.self_ref] for item in items if item.self_ref in pictures), None)

    @staticmethod
    def _kind(labels: set[DocItemLabel], has_picture: bool) -> ChunkKind:
        # Items under one heading are merged into one chunk, so the richest content wins.
        if DocItemLabel.TABLE in labels:
            return ChunkKind.TABLE
        if has_picture:
            return ChunkKind.FIGURE
        if labels == {DocItemLabel.FORMULA}:
            return ChunkKind.FORMULA
        return ChunkKind.TEXT

    @staticmethod
    def _page_range(items: list[DocItem]) -> tuple[int | None, int | None]:
        pages = [prov.page_no for item in items for prov in item.prov]
        if not pages:
            return None, None
        return min(pages), max(pages)

    @staticmethod
    def _save_image(picture: PictureItem, document: DoclingDocument, path: Path) -> str | None:
        """Write the picture to `path`; return the path relative to STORAGE_DIR."""
        image = picture.get_image(document)
        if image is None:
            return None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            image.save(path)
        except OSError as exc:
            # A figure that cannot be written must not fail the whole document.
            logger.warning(f"Could not save figure {path.name}: {exc}")
            return None
        return path.relative_to(settings.STORAGE_DIR).as_posix()
