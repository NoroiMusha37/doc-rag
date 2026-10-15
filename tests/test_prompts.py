import uuid

import pytest

from src.doc_rag.db.models import ChunkKind
from src.doc_rag.llm.prompts import (
    NOT_FOUND_ANSWER,
    REWRITE_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_prompt,
    build_rewrite_prompt,
)
from src.doc_rag.retrieval.schemas import SearchResult


def _result(text: str, page_start: int | None, page_end: int | None, heading: str) -> SearchResult:
    return SearchResult(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        filename="paper.pdf",
        kind=ChunkKind.TEXT,
        text=text,
        heading_path=heading,
        page_start=page_start,
        page_end=page_end,
        asset_path=None,
        score=0.9,
    )


def test_prompt_numbers_the_sources_from_one_and_ends_with_the_question() -> None:
    prompt = build_prompt(
        "What is delta?",
        [_result("Delta is 0.5.", 3, 3, "Greeks"), _result("Gamma is 0.1.", 4, 5, "Greeks")],
    )

    assert prompt == (
        "Sources:\n\n"
        '[1] paper.pdf, page 3, section "Greeks"\nDelta is 0.5.\n\n'
        '[2] paper.pdf, pages 4-5, section "Greeks"\nGamma is 0.1.\n\n'
        "Question: What is delta?"
    )


@pytest.mark.parametrize(
    ("page_start", "page_end", "heading", "header"),
    [
        (7, 7, "", "[1] paper.pdf, page 7\n"),
        (7, None, "", "[1] paper.pdf, page 7\n"),
        (7, 9, "", "[1] paper.pdf, pages 7-9\n"),
        (None, None, "Intro", '[1] paper.pdf, page unknown, section "Intro"\n'),
    ],
)
def test_source_header_adapts_to_missing_pages_and_headings(
    page_start: int | None, page_end: int | None, heading: str, header: str
) -> None:
    assert header in build_prompt("q", [_result("text", page_start, page_end, heading)])


def test_system_prompt_requires_citations_and_the_exact_decline_sentence() -> None:
    assert "[1]" in SYSTEM_PROMPT
    assert NOT_FOUND_ANSWER in SYSTEM_PROMPT
    assert "only the sources" in SYSTEM_PROMPT


def test_rewrite_prompt_lists_the_question_and_every_query_already_tried() -> None:
    prompt = build_rewrite_prompt("how many heads?", ["how many heads?", "attention heads"])

    assert prompt.startswith("Question: how many heads?")
    assert prompt.endswith("- how many heads?\n- attention heads")


def test_rewrite_instructions_ask_for_the_query_alone() -> None:
    assert "query only" in REWRITE_SYSTEM_PROMPT
    assert "English" in REWRITE_SYSTEM_PROMPT
