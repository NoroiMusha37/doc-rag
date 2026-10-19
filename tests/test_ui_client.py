import json

import httpx
import pytest

from ui.client import ApiClient, ApiError

DOCUMENT = {"id": "7f3c", "filename": "paper.pdf", "status": "ready"}


def _client(handler) -> ApiClient:
    return ApiClient("http://api", transport=httpx.MockTransport(handler))


def _sse(*events: tuple[str, dict]) -> str:
    return "".join(f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in events)


def test_list_documents_returns_the_api_rows() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert (request.method, request.url.path) == ("GET", "/documents")
        return httpx.Response(200, json=[DOCUMENT])

    assert _client(handler).list_documents() == [DOCUMENT]


def test_upload_sends_the_file_as_a_pdf_form_field() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert (request.method, request.url.path) == ("POST", "/documents")
        body = request.content
        assert b'name="file"; filename="paper.pdf"' in body
        assert b"application/pdf" in body
        assert b"%PDF-1.7" in body
        return httpx.Response(202, json=DOCUMENT)

    assert _client(handler).upload("paper.pdf", b"%PDF-1.7") == DOCUMENT


def test_delete_calls_the_document_route() -> None:
    seen: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, request.url.path))
        return httpx.Response(204)

    _client(handler).delete("7f3c")

    assert seen == [("DELETE", "/documents/7f3c")]


def test_asset_fetches_the_image_at_the_path_a_source_carries() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/assets/7f3c/8.png"
        return httpx.Response(200, content=b"png-bytes")

    assert _client(handler).asset("assets/7f3c/8.png") == b"png-bytes"


def test_chat_yields_the_events_in_order() -> None:
    events = [
        ("sources", {"event": "sources", "sources": [{"number": 1}]}),
        ("token", {"event": "token", "text": "Delta "}),
        ("token", {"event": "token", "text": "is 0.5 [1]."}),
        ("done", {"event": "done", "answer": "Delta is 0.5 [1].", "citations": [1]}),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content) == {
            "question": "What is delta?",
            "document_ids": None,
            "mode": "simple",
        }
        return httpx.Response(
            200, text=_sse(*events), headers={"content-type": "text/event-stream"}
        )

    assert list(_client(handler).chat("What is delta?")) == events


def test_chat_passes_the_document_filter() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["document_ids"] == ["7f3c"]
        return httpx.Response(200, text="")

    assert list(_client(handler).chat("q", ["7f3c"])) == []


def test_chat_ignores_keep_alive_comments_and_joins_multi_line_data() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, text=': ping\n\nevent: token\ndata: {"text":\ndata: "hi"}\n\n: ping\n\n'
        )

    assert list(_client(handler).chat("q")) == [("token", {"text": "hi"})]


def test_an_api_error_carries_the_detail_text() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(415, json={"detail": "Only PDF files are supported"})

    with pytest.raises(ApiError, match="Only PDF files are supported"):
        _client(handler).upload("notes.txt", b"hello")


def test_a_validation_error_lists_what_was_rejected() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            422,
            json={
                "detail": [
                    {"loc": ["body", "question"], "msg": "String should have at least 1 character"}
                ]
            },
        )

    with pytest.raises(ApiError, match="at least 1 character"):
        list(_client(handler).chat(""))


def test_an_error_without_json_reports_the_status_code() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(502, text="Bad Gateway")

    with pytest.raises(ApiError, match="502"):
        _client(handler).list_documents()


@pytest.mark.parametrize("call", ["list", "chat"])
def test_an_unreachable_api_is_explained(call: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    client = _client(handler)

    with pytest.raises(ApiError, match="did not respond") as error:
        client.list_documents() if call == "list" else list(client.chat("q"))

    assert "http://api" in str(error.value)
