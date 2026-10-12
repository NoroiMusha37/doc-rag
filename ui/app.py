import os
from collections.abc import Iterator

import streamlit as st

from ui.client import ApiClient, ApiError

# Statuses of a document that is still being ingested.
PROCESSING = ("pending", "parsing", "embedding")
STATUS_LABELS = {
    "pending": "⏳ waiting",
    "parsing": "⏳ reading the PDF",
    "embedding": "⏳ indexing",
    "ready": "✅ ready",
    "failed": "❌ failed",
}
# How often the document list refreshes itself while something is being ingested.
POLL_SECONDS = 3


@st.cache_resource
def get_client() -> ApiClient:
    # One client for all browser sessions; fails at once when API_URL is not set.
    return ApiClient(os.environ["API_URL"])


@st.cache_data(show_spinner=False)
def figure(asset_path: str) -> bytes:
    return get_client().asset(asset_path)


def page_label(source: dict) -> str:
    if source["page_start"] is None:
        return ""
    if source["page_end"] in (None, source["page_start"]):
        return f"p. {source['page_start']}"
    return f"pp. {source['page_start']}-{source['page_end']}"


def upload_form() -> None:
    # A form, so that the chosen files are cleared once they are sent.
    with st.form("upload", clear_on_submit=True, border=False):
        files = st.file_uploader("Add PDFs", type="pdf", accept_multiple_files=True)
        if not st.form_submit_button("Upload") or not files:
            return
    for file in files:
        try:
            get_client().upload(file.name, file.getvalue())
        except ApiError as exc:
            st.error(f"{file.name}: {exc}")


def documents_panel() -> None:
    """The document list. Runs as a fragment, so it can refresh without the whole page."""
    try:
        documents = get_client().list_documents()
    except ApiError as exc:
        st.error(str(exc))
        return
    if not documents:
        st.caption("No documents yet.")
    for document in documents:
        name, button = st.columns([5, 1], vertical_alignment="center")
        pages = f" · {document['page_count']} pages" if document["page_count"] else ""
        name.markdown(f"**{document['filename']}**  \n{STATUS_LABELS[document['status']]}{pages}")
        if document["error"]:
            name.caption(document["error"])
        if button.button("✕", key=f"delete-{document['id']}", help="Delete this document"):
            try:
                get_client().delete(document["id"])
            except ApiError as exc:
                st.error(str(exc))
            else:
                st.rerun()

    # When ingestion ends, redraw the whole page once: that stops the polling and puts
    # the new document into the filter. Not while an answer is being written, though,
    # because a redraw would cut it off.
    if (
        st.session_state.polling
        and not st.session_state.answering
        and not any(document["status"] in PROCESSING for document in documents)
    ):
        st.rerun()


def render_sources(sources: list[dict], citations: list[int]) -> None:
    for source in sources:
        cited = " · cited" if source["number"] in citations else ""
        with st.expander(f"{source['number']}. {source['filename']} · {page_label(source)}{cited}"):
            st.caption(
                f"{source['heading_path']} · {source['kind']} · relevance {source['score']:.2f}"
            )
            if source["asset_path"]:
                try:
                    st.image(figure(source["asset_path"]))
                except ApiError:
                    st.caption("The figure image is no longer available.")
            st.markdown(source["text"])


def render_message(message: dict) -> None:
    with st.chat_message(message["role"]):
        if message["content"]:
            st.markdown(message["content"])
        if message.get("error"):
            st.error(message["error"])
        render_sources(message.get("sources", []), message.get("citations", []))


def answer(question: str, document_ids: list[str] | None) -> dict:
    """Stream the answer into the page and return it as a chat message."""
    message: dict = {"role": "assistant", "content": "", "sources": [], "citations": []}

    def tokens() -> Iterator[str]:
        for event, data in get_client().chat(question, document_ids):
            if event == "sources":
                message["sources"] = data["sources"]
            elif event == "token":
                message["content"] += data["text"]
                yield data["text"]
            elif event == "done":
                message["citations"] = data["citations"]
            elif event == "error":
                raise ApiError(data["message"])

    st.session_state.answering = True
    try:
        with st.chat_message("assistant"):
            try:
                st.write_stream(tokens())
            except ApiError as exc:
                message["error"] = str(exc)
                st.error(message["error"])
            render_sources(message["sources"], message["citations"])
    finally:
        st.session_state.answering = False
    return message


st.set_page_config(page_title="doc-rag", page_icon="📚")
st.session_state.setdefault("messages", [])
st.session_state.setdefault("answering", False)

with st.sidebar:
    st.header("Documents")
    upload_form()
    try:
        documents = get_client().list_documents()
    except ApiError:
        documents = None  # the panel below shows the error and keeps trying
    st.session_state.polling = documents is None or any(
        document["status"] in PROCESSING for document in documents
    )
    st.fragment(run_every=POLL_SECONDS if st.session_state.polling else None)(documents_panel)()
    st.divider()
    ready = {
        document["id"]: document["filename"]
        for document in documents or []
        if document["status"] == "ready"
    }
    selected = st.multiselect(
        "Answer only from", list(ready), format_func=ready.get, placeholder="All documents"
    )
    if st.button("Clear the chat"):
        st.session_state.messages = []

st.title("Ask your documents")
for message in st.session_state.messages:
    render_message(message)

if question := st.chat_input("Ask a question about your documents"):
    st.session_state.messages.append({"role": "user", "content": question})
    render_message(st.session_state.messages[-1])
    st.session_state.messages.append(answer(question, selected or None))
