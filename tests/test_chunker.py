from pathlib import Path

import pytest
from docling_core.transforms.chunker.tokenizer.base import BaseTokenizer
from docling_core.types.doc import (
    BoundingBox,
    DocItemLabel,
    DoclingDocument,
    ImageRef,
    ProvenanceItem,
    TableCell,
    TableData,
)
from PIL import Image

from src.doc_rag.config import settings
from src.doc_rag.db.models import ChunkKind
from src.doc_rag.ingestion import chunker as chunker_module
from src.doc_rag.ingestion.chunker import DocumentChunker
from src.doc_rag.ingestion.schemas import ParsedChunk


class WordTokenizer(BaseTokenizer):
    """Counts words, so the chunker runs without downloading the real tokenizer."""

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def get_max_tokens(self) -> int:
        return 64

    def get_tokenizer(self):
        return self.count_tokens


def _prov(page: int) -> ProvenanceItem:
    return ProvenanceItem(page_no=page, bbox=BoundingBox(l=0, t=0, r=1, b=1), charspan=(0, 0))


def _cell(row: int, column: int, text: str, header: bool = False) -> TableCell:
    return TableCell(
        text=text,
        start_row_offset_idx=row,
        end_row_offset_idx=row + 1,
        start_col_offset_idx=column,
        end_col_offset_idx=column + 1,
        column_header=header,
    )


@pytest.fixture
def document() -> DoclingDocument:
    doc = DoclingDocument(name="sample")
    doc.add_heading("Black-Scholes", prov=_prov(1))
    doc.add_text(DocItemLabel.TEXT, "The model prices European options.", prov=_prov(1))
    doc.add_formula("C = S N(d_1) - K e^{-rT} N(d_2)", prov=_prov(2))
    doc.add_heading("Greeks", prov=_prov(2))
    doc.add_table(
        data=TableData(
            num_rows=2,
            num_cols=2,
            table_cells=[
                _cell(0, 0, "Greek", header=True),
                _cell(0, 1, "Meaning", header=True),
                _cell(1, 0, "Delta"),
                _cell(1, 1, "Price sensitivity"),
            ],
        ),
        prov=_prov(3),
    )
    caption = doc.add_text(DocItemLabel.CAPTION, "Figure 1: call payoff.", prov=_prov(4))
    doc.add_picture(
        image=ImageRef.from_pil(Image.new("RGB", (8, 8), "white"), dpi=72),
        caption=caption,
        prov=_prov(4),
    )
    return doc


@pytest.fixture
def chunks(
    document: DoclingDocument, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> list[ParsedChunk]:
    monkeypatch.setattr(settings, "STORAGE_DIR", str(tmp_path))
    monkeypatch.setattr(
        chunker_module.HuggingFaceTokenizer, "from_pretrained", lambda **kwargs: WordTokenizer()
    )
    return DocumentChunker().chunk(document, tmp_path / "assets" / "doc")


def test_chunks_are_numbered_in_reading_order(chunks: list[ParsedChunk]) -> None:
    assert [chunk.ordinal for chunk in chunks] == list(range(len(chunks)))
    assert [chunk.heading_path for chunk in chunks] == ["Black-Scholes", "Greeks"]


def test_text_chunk_keeps_formula_as_latex_and_spans_its_pages(chunks: list[ParsedChunk]) -> None:
    text_chunk = chunks[0]
    assert text_chunk.kind is ChunkKind.TEXT
    assert "$$C = S N(d_1) - K e^{-rT} N(d_2)$$" in text_chunk.text
    assert (text_chunk.page_start, text_chunk.page_end) == (1, 2)


def test_table_is_serialized_as_markdown(chunks: list[ParsedChunk]) -> None:
    table_chunk = chunks[1]
    assert table_chunk.kind is ChunkKind.TABLE
    assert "| Delta" in table_chunk.text
    assert "Price sensitivity |" in table_chunk.text


def test_embed_text_is_prefixed_with_the_heading(chunks: list[ParsedChunk]) -> None:
    for chunk in chunks:
        assert chunk.embed_text == f"{chunk.heading_path}\n{chunk.text}"


def test_a_later_batch_continues_numbering_and_inherits_the_open_section(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "STORAGE_DIR", str(tmp_path))
    monkeypatch.setattr(
        chunker_module.HuggingFaceTokenizer, "from_pretrained", lambda **kwargs: WordTokenizer()
    )
    # A batch that starts mid-section: text first, its own heading only afterwards.
    doc = DoclingDocument(name="batch")
    doc.add_text(DocItemLabel.TEXT, "The section goes on.", prov=_prov(11))
    doc.add_heading("Greeks", prov=_prov(12))
    doc.add_text(DocItemLabel.TEXT, "Delta measures sensitivity.", prov=_prov(12))

    chunks = DocumentChunker().chunk(
        doc, tmp_path / "assets" / "doc", first_ordinal=5, inherited_heading_path="Part > Options"
    )

    assert [chunk.ordinal for chunk in chunks] == [5, 6]
    assert [chunk.heading_path for chunk in chunks] == ["Part > Options", "Greeks"]
    assert chunks[0].embed_text == "Part\nOptions\nThe section goes on."
    assert chunks[0].page_start == 11


def test_picture_is_saved_even_when_merged_into_a_table_chunk(
    chunks: list[ParsedChunk], tmp_path: Path
) -> None:
    table_chunk = chunks[1]
    assert table_chunk.asset_path == "assets/doc/1.png"
    assert (tmp_path / table_chunk.asset_path).is_file()


@pytest.mark.parametrize(
    ("labels", "has_picture", "kind"),
    [
        ({DocItemLabel.TEXT}, False, ChunkKind.TEXT),
        ({DocItemLabel.TEXT, DocItemLabel.FORMULA}, False, ChunkKind.TEXT),
        ({DocItemLabel.FORMULA}, False, ChunkKind.FORMULA),
        ({DocItemLabel.TEXT, DocItemLabel.TABLE}, False, ChunkKind.TABLE),
        ({DocItemLabel.CAPTION}, True, ChunkKind.FIGURE),
        ({DocItemLabel.TABLE, DocItemLabel.CAPTION}, True, ChunkKind.TABLE),
    ],
)
def test_kind_prefers_table_then_figure_then_formula(
    labels: set[DocItemLabel], has_picture: bool, kind: ChunkKind
) -> None:
    assert DocumentChunker._kind(labels, has_picture) is kind


def test_page_range_is_empty_without_provenance() -> None:
    assert DocumentChunker._page_range([]) == (None, None)
