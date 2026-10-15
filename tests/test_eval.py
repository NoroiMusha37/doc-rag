import json
from pathlib import Path

import httpx
import pytest

from eval.run_eval import (
    DEFAULT_MIN_SCORE,
    Answer,
    Question,
    Retrieval,
    ask,
    contains_all,
    first_hit_rank,
    is_relevant,
    load_questions,
    missing_documents,
    parse_events,
    retrieval_metrics,
    search,
)
from src.doc_rag.config import settings


def _question(**fields: object) -> Question:
    values = {
        "id": "q",
        "question": "What?",
        "kind": "table",
        "document": "paper.pdf",
        "pages": (4, 5),
    }
    return Question(**{**values, **fields})  # type: ignore[arg-type]


def _result(filename: str = "paper.pdf", start: int | None = 4, end: int | None = 4, score=0.9):
    return {"filename": filename, "page_start": start, "page_end": end, "score": score}


def test_the_question_file_is_valid_and_covers_every_kind() -> None:
    questions = load_questions()

    assert 20 <= len(questions) <= 30
    assert len({q.id for q in questions}) == len(questions)
    assert {q.kind for q in questions} == {"text", "table", "figure", "formula", "none"}
    assert {q.language for q in questions} == {"en", "uk"}
    assert any(not q.answerable for q in questions)


def test_the_default_threshold_matches_the_servers_setting() -> None:
    assert DEFAULT_MIN_SCORE == settings.CHAT_MIN_SCORE


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        pytest.param({"kind": "chart"}, "kind must be one of", id="unknown-kind"),
        pytest.param({"kind": "none"}, "only kind 'none'", id="none-with-document"),
        pytest.param({"document": None, "pages": ()}, "only kind 'none'", id="no-document"),
        pytest.param({"pages": ()}, "only kind 'none'", id="document-without-pages"),
    ],
)
def test_inconsistent_questions_are_rejected(fields: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _question(**fields)


def test_loading_reports_the_line_of_a_bad_question(tmp_path: Path) -> None:
    path = tmp_path / "questions.jsonl"
    good = {"id": "a", "question": "?", "kind": "none"}
    path.write_text(
        json.dumps(good) + "\n\n" + json.dumps({**good, "id": "b", "kind": "chart"}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"questions.jsonl, line 3: b: kind must be one of"):
        load_questions(path)


def test_loading_rejects_duplicate_ids_and_unknown_fields(tmp_path: Path) -> None:
    path = tmp_path / "questions.jsonl"
    line = json.dumps({"id": "a", "question": "?", "kind": "none"})
    path.write_text(f"{line}\n{line}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate id 'a'"):
        load_questions(path)

    path.write_text(json.dumps({"id": "a", "question": "?", "kind": "none", "page": 1}))
    with pytest.raises(ValueError, match="line 1"):
        load_questions(path)


@pytest.mark.parametrize(
    ("result", "relevant"),
    [
        pytest.param(_result(start=4, end=4), True, id="expected-page"),
        pytest.param(_result(start=5, end=5), True, id="other-expected-page"),
        pytest.param(_result(start=3, end=4), True, id="range-ends-on-an-expected-page"),
        pytest.param(_result(start=2, end=6), True, id="range-covers-the-pages"),
        pytest.param(_result(start=6, end=7), False, id="range-after-the-pages"),
        pytest.param(_result(start=1, end=2), False, id="range-before-the-pages"),
        pytest.param(_result(start=4, end=None), True, id="no-end-means-one-page"),
        pytest.param(_result(start=None, end=None), False, id="no-page"),
        pytest.param(_result(filename="other.pdf"), False, id="other-document"),
    ],
)
def test_a_result_is_relevant_when_it_covers_an_expected_page_of_the_document(
    result: dict, relevant: bool
) -> None:
    assert is_relevant(result, _question()) is relevant


def test_the_rank_is_the_position_of_the_first_relevant_result() -> None:
    results = [
        _result(start=9, end=9),
        _result(filename="other.pdf"),
        _result(),
        _result(start=5, end=5),
    ]

    assert first_hit_rank(results, _question()) == 3
    assert first_hit_rank(results[:2], _question()) is None
    assert first_hit_rank([], _question()) is None


def test_a_retrieval_knows_the_score_of_its_first_relevant_result() -> None:
    results = [_result(start=9, end=9, score=0.8), _result(score=0.3)]

    hit = Retrieval(_question(), results, 0.5, rank=2)
    miss = Retrieval(_question(), results[:1], 0.5, rank=None)

    assert (hit.hit_score, hit.top_score) == (0.3, 0.8)
    assert miss.hit_score is None


def test_hit_rates_and_mrr() -> None:
    metrics = retrieval_metrics([1, 2, 4, 7, None])

    assert metrics["hit@1"] == pytest.approx(1 / 5)
    assert metrics["hit@3"] == pytest.approx(2 / 5)
    assert metrics["hit@6"] == pytest.approx(3 / 5)
    assert metrics["MRR"] == pytest.approx((1 + 1 / 2 + 1 / 4 + 1 / 7) / 5)


@pytest.mark.parametrize(
    ("answer", "needles", "expected"),
    [
        pytest.param("The score is 28.4 BLEU [1].", ["28.4"], True, id="present"),
        pytest.param("ADAM with 4000 steps", ["Adam", "4000"], True, id="case-insensitive"),
        pytest.param("It uses Adam.", ["Adam", "4000"], False, id="one-missing"),
        pytest.param("six layers", ["6|six"], True, id="alternative-spelling"),
        pytest.param("seven layers", ["6|six"], False, id="no-alternative-matches"),
        pytest.param("anything", [], True, id="nothing-required"),
    ],
)
def test_answer_keywords(answer: str, needles: list[str], expected: bool) -> None:
    assert contains_all(answer, needles) is expected


def test_server_sent_events_are_parsed_and_comments_ignored() -> None:
    lines = [
        ": ping",
        "event: sources",
        'data: {"sources": []}',
        "",
        "event: token",
        'data: {"text":',
        'data: "hi"}',
        "",
    ]

    assert parse_events(lines) == [("sources", {"sources": []}), ("token", {"text": "hi"})]


def _api(handler) -> httpx.Client:
    return httpx.Client(base_url="http://api", transport=httpx.MockTransport(handler))


def test_search_sends_the_question_and_ranks_the_results() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/search"
        assert json.loads(request.content) == {"query": "What?", "top_k": 7}
        return httpx.Response(200, json={"results": [_result(start=9, end=9), _result()]})

    retrieval = search(_api(handler), _question(), top_k=7)

    assert retrieval.rank == 2
    assert retrieval.top_score == 0.9


def test_searching_an_unanswerable_question_has_no_rank() -> None:
    handler = lambda request: httpx.Response(200, json={"results": [_result(score=0.02)]})  # noqa: E731

    retrieval = search(_api(handler), _question(kind="none", document=None, pages=()), top_k=3)

    assert retrieval.rank is None
    assert retrieval.top_score == 0.02


def _chat_body(*events: tuple[str, dict]) -> str:
    return "".join(f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in events)


def test_ask_collects_sources_citations_and_the_answer() -> None:
    source = {"number": 1, **_result()}
    body = _chat_body(
        ("sources", {"sources": [source]}),
        ("token", {"text": "It is "}),
        ("token", {"text": "28.4 [1]."}),
        ("done", {"answer": "It is 28.4 [1].", "citations": [1]}),
    )
    handler = lambda request: httpx.Response(200, text=body)  # noqa: E731

    answer = ask(_api(handler), _question(answer_contains=("28.4",)))

    assert (answer.text, answer.citations, answer.error) == ("It is 28.4 [1].", [1], None)
    assert answer.correct


def test_ask_reports_an_error_event() -> None:
    body = _chat_body(("sources", {"sources": []}), ("error", {"message": "Gemini is overloaded"}))
    handler = lambda request: httpx.Response(200, text=body)  # noqa: E731

    answer = ask(_api(handler), _question())

    assert answer.error == "Gemini is overloaded"
    assert not answer.correct


@pytest.mark.parametrize(
    ("citations", "text", "correct"),
    [
        pytest.param([1], "It is 28.4 [1].", True, id="facts-and-the-right-page"),
        pytest.param([2], "It is 28.4 [2].", False, id="cites-the-wrong-page"),
        pytest.param([1], "It is 30 [1].", False, id="wrong-facts"),
        pytest.param([], "It is 28.4.", False, id="no-citation"),
    ],
)
def test_an_answer_needs_the_facts_and_a_citation_of_the_expected_page(
    citations: list[int], text: str, correct: bool
) -> None:
    sources = [{"number": 1, **_result(start=4)}, {"number": 2, **_result(start=9, end=9)}]
    question = _question(answer_contains=("28.4",))

    assert Answer(question, text, citations, sources, None, 1.0).correct is correct


def test_an_unanswerable_question_is_correct_only_when_nothing_is_cited() -> None:
    question = _question(kind="none", document=None, pages=())

    assert Answer(question, "Not in the documents.", [], [], None, 0.1).correct
    assert not Answer(question, "It is 5 [1].", [1], [{"number": 1}], None, 1.0).correct


def test_missing_documents_are_the_ones_that_are_not_ready() -> None:
    documents = [
        {"filename": "attention.pdf", "status": "ready"},
        {"filename": "bert.pdf", "status": "parsing"},
    ]
    handler = lambda request: httpx.Response(200, json=documents)  # noqa: E731
    questions = [
        _question(id="a", document="attention.pdf"),
        _question(id="b", document="bert.pdf"),
        _question(id="c", document="thesis.pdf"),
        _question(id="d", kind="none", document=None, pages=()),
    ]

    assert missing_documents(_api(handler), questions) == ["bert.pdf", "thesis.pdf"]
