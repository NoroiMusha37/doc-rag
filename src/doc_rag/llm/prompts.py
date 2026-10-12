from collections.abc import Sequence

from src.doc_rag.retrieval.schemas import SearchResult

# Also returned by the chat pipeline itself when the search finds nothing relevant.
NOT_FOUND_ANSWER = (
    "The uploaded documents do not contain the information needed to answer this question."
)

SYSTEM_PROMPT = f"""\
You answer questions about the user's documents. You are given numbered sources, \
each an excerpt from one document, followed by a question.

Rules:
- Use only the sources. Do not add facts from your own knowledge, even if you are sure of them.
- After every statement, cite the sources that support it with their numbers in square \
brackets, for example [1] or [2][4]. Use only numbers that appear in the sources.
- Copy numbers, units, symbols and names exactly as the sources give them.
- If the sources do not contain the answer, reply only with: "{NOT_FOUND_ANSWER}" \
If they answer only part of the question, answer that part and say what is missing.
- Answer in the language of the question. Be concise and direct.
- Write formulas in LaTeX between dollar signs and tables in Markdown.
"""


def build_prompt(question: str, sources: Sequence[SearchResult]) -> str:
    """The numbered sources followed by the question.

    A source's number is its position in `sources`, starting at 1; the same numbers
    come back in the answer as citations.
    """
    return (
        "Sources:\n\n"
        + "\n\n".join(
            f"[{number}] {source.filename}, {_pages(source)}"
            + (f', section "{source.heading_path}"' if source.heading_path else "")
            + f"\n{source.text}"
            for number, source in enumerate(sources, start=1)
        )
        + f"\n\nQuestion: {question}"
    )


def _pages(source: SearchResult) -> str:
    if source.page_start is None:
        return "page unknown"
    if source.page_end in (None, source.page_start):
        return f"page {source.page_start}"
    return f"pages {source.page_start}-{source.page_end}"
