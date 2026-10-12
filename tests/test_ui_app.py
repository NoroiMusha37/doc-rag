import io

import pytest

# Streamlit is in the `ui` dependency group: run these with `uv run --group ui pytest`.
st = pytest.importorskip("streamlit")

from PIL import Image  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

import ui.client  # noqa: E402
from ui.client import ApiError  # noqa: E402

READY = {
    "id": "7f3c",
    "filename": "paper.pdf",
    "status": "ready",
    "error": None,
    "page_count": 15,
}
FAILED = {
    "id": "a1b2",
    "filename": "scan.pdf",
    "status": "failed",
    "error": "No text could be extracted from the document",
    "page_count": None,
}
SOURCE = {
    "number": 1,
    "filename": "paper.pdf",
    "kind": "text",
    "text": "Delta measures the sensitivity of the option price.",
    "heading_path": "3 Greeks",
    "page_start": 4,
    "page_end": 5,
    "asset_path": None,
    "score": 0.91,
}


class FakeClient:
    """Stands in for `ApiClient`; class attributes script what the API returns."""

    documents: list[dict] = []
    events: list[tuple[str, dict]] = []
    down = False
    asked: list[tuple[str, list[str] | None]] = []
    figure: bytes | None = None
    fetched: list[str] = []

    def __init__(self, base_url: str) -> None:
        pass

    def list_documents(self) -> list[dict]:
        if self.down:
            raise ApiError("The API at http://api did not respond")
        return self.documents

    def asset(self, asset_path: str) -> bytes:
        self.fetched.append(asset_path)
        if self.figure is None:
            raise ApiError("Not Found")
        return self.figure

    def chat(self, question: str, document_ids: list[str] | None = None):
        self.asked.append((question, document_ids))
        yield from self.events


@pytest.fixture
def api(monkeypatch: pytest.MonkeyPatch) -> type[FakeClient]:
    monkeypatch.setenv("API_URL", "http://api")
    monkeypatch.setattr(ui.client, "ApiClient", FakeClient)
    monkeypatch.setattr(FakeClient, "documents", [READY])
    monkeypatch.setattr(FakeClient, "events", [])
    monkeypatch.setattr(FakeClient, "down", False)
    monkeypatch.setattr(FakeClient, "asked", [])
    monkeypatch.setattr(FakeClient, "figure", None)
    monkeypatch.setattr(FakeClient, "fetched", [])
    st.cache_data.clear()
    # The page caches its client for the life of the process; each test needs its own.
    st.cache_resource.clear()
    return FakeClient


def _page() -> AppTest:
    return AppTest.from_file("../ui/app.py", default_timeout=10).run()


def _text(elements) -> str:
    return "\n".join(element.value for element in elements)


def test_the_sidebar_lists_documents_with_status_and_error(api) -> None:
    api.documents = [READY, FAILED]

    page = _page()

    assert not page.exception
    listed = _text(page.sidebar.markdown)
    assert "paper.pdf" in listed and "ready" in listed and "15 pages" in listed
    assert "scan.pdf" in listed and "failed" in listed
    assert "No text could be extracted" in _text(page.sidebar.caption)
    # Only documents that are ready can be picked as a filter.
    assert page.sidebar.multiselect[0].options == ["paper.pdf"]


def test_an_unreachable_api_shows_an_error_instead_of_crashing(api) -> None:
    api.down = True

    page = _page()

    assert not page.exception
    assert "did not respond" in page.sidebar.error[0].value


def test_a_question_shows_the_answer_and_its_sources(api) -> None:
    api.events = [
        ("sources", {"sources": [SOURCE]}),
        ("token", {"text": "Delta is the "}),
        ("token", {"text": "price sensitivity [1]."}),
        ("done", {"answer": "Delta is the price sensitivity [1].", "citations": [1]}),
    ]

    page = _page()
    page.chat_input[0].set_value("What is delta?").run()

    assert not page.exception
    assert api.asked == [("What is delta?", None)]
    user, assistant = page.chat_message
    assert user.markdown[0].value == "What is delta?"
    assert assistant.markdown[0].value == "Delta is the price sensitivity [1]."
    assert assistant.expander[0].label == "1. paper.pdf · pp. 4-5 · cited"
    assert "sensitivity of the option price" in _text(assistant.expander[0].markdown)

    # The conversation is kept when the page is redrawn.
    page.run()
    assert len(page.chat_message) == 2
    assert page.chat_message[1].markdown[0].value == "Delta is the price sensitivity [1]."


def test_the_filter_limits_the_question_to_the_chosen_documents(api) -> None:
    page = _page()
    page.sidebar.multiselect[0].select("7f3c").run()
    page.chat_input[0].set_value("What is delta?").run()

    assert api.asked == [("What is delta?", ["7f3c"])]


def test_an_error_event_is_shown_with_the_text_received_so_far(api) -> None:
    api.events = [
        ("sources", {"sources": [SOURCE]}),
        ("token", {"text": "Delta is"}),
        ("error", {"message": "Gemini is overloaded right now. Ask again in a moment."}),
    ]

    page = _page()
    page.chat_input[0].set_value("What is delta?").run()

    assert not page.exception
    assistant = page.chat_message[1]
    assert assistant.markdown[0].value == "Delta is"
    assert "overloaded" in assistant.error[0].value
    assert assistant.expander[0].label == "1. paper.pdf · pp. 4-5"


def _png() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), "white").save(buffer, format="PNG")
    return buffer.getvalue()


def test_a_figure_source_shows_its_image(api) -> None:
    api.figure = _png()
    api.events = [
        ("sources", {"sources": [SOURCE | {"kind": "figure", "asset_path": "assets/7f3c/8.png"}]}),
        ("token", {"text": "See the figure [1]."}),
        ("done", {"answer": "See the figure [1].", "citations": [1]}),
    ]

    page = _page()
    page.chat_input[0].set_value("What does the figure show?").run()

    assert not page.exception
    assert api.fetched == ["assets/7f3c/8.png"]
    assert len(page.chat_message[1].expander[0].get("image")) == 1


def test_a_figure_that_is_gone_does_not_break_the_answer(api) -> None:
    api.events = [
        ("sources", {"sources": [SOURCE | {"kind": "figure", "asset_path": "assets/7f3c/8.png"}]}),
        ("token", {"text": "See the figure [1]."}),
        ("done", {"answer": "See the figure [1].", "citations": [1]}),
    ]

    page = _page()
    page.chat_input[0].set_value("What does the figure show?").run()

    assert not page.exception
    assert "no longer available" in _text(page.chat_message[1].expander[0].caption)


def test_clearing_the_chat_removes_the_messages(api) -> None:
    api.events = [("token", {"text": "Hello."}), ("done", {"answer": "Hello.", "citations": []})]
    page = _page()
    page.chat_input[0].set_value("Hi").run()
    assert len(page.chat_message) == 2

    next(b for b in page.sidebar.button if b.label == "Clear the chat").click().run()

    assert len(page.chat_message) == 0
