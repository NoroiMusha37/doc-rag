import json
from collections.abc import Iterator

import httpx

# Longest silence allowed while reading: an answer can wait on several Gemini models in
# turn before its first word arrives, and an upload is hashed and stored before the reply.
READ_TIMEOUT_SECONDS = 200.0
CONNECT_TIMEOUT_SECONDS = 5.0


class ApiError(Exception):
    """A failed API call; the message is written for the person using the page."""


class ApiClient:
    """Everything the page asks of the doc-rag API. Build once; it keeps a connection pool."""

    def __init__(self, base_url: str, transport: httpx.BaseTransport | None = None) -> None:
        self._http = httpx.Client(
            base_url=base_url,
            timeout=httpx.Timeout(READ_TIMEOUT_SECONDS, connect=CONNECT_TIMEOUT_SECONDS),
            transport=transport,
        )

    def list_documents(self) -> list[dict]:
        return self._request("GET", "/documents").json()

    def upload(self, filename: str, content: bytes) -> dict:
        return self._request(
            "POST", "/documents", files={"file": (filename, content, "application/pdf")}
        ).json()

    def delete(self, document_id: str) -> None:
        self._request("DELETE", f"/documents/{document_id}")

    def asset(self, asset_path: str) -> bytes:
        """The image of a figure; `asset_path` is the one a source carries."""
        return self._request("GET", f"/{asset_path}").content

    def chat(
        self, question: str, document_ids: list[str] | None = None
    ) -> Iterator[tuple[str, dict]]:
        """Ask a question; yields the answer's events as (name, data) while they arrive.

        Names, in order: `sources` once, `token` many times, then `done` or `error`.
        """
        try:
            with self._http.stream(
                "POST", "/chat", json={"question": question, "document_ids": document_ids}
            ) as response:
                if response.is_error:
                    response.read()
                    raise ApiError(self._detail(response))
                yield from self._events(response.iter_lines())
        except httpx.HTTPError as exc:
            raise ApiError(self._unreachable(exc)) from exc

    def _request(self, method: str, path: str, **kwargs: object) -> httpx.Response:
        try:
            response = self._http.request(method, path, **kwargs)
        except httpx.HTTPError as exc:
            raise ApiError(self._unreachable(exc)) from exc
        if response.is_error:
            raise ApiError(self._detail(response))
        return response

    def _unreachable(self, error: httpx.HTTPError) -> str:
        return (
            f"The API at {self._http.base_url} did not respond ({type(error).__name__}). "
            "It needs about a minute to load its models after starting."
        )

    @staticmethod
    def _events(lines: Iterator[str]) -> Iterator[tuple[str, dict]]:
        """Parse Server-Sent Events: `event:` and `data:` lines, ended by an empty line."""
        name, data = "", []
        for line in lines:
            if line.startswith("event:"):
                name = line.removeprefix("event:").strip()
            elif line.startswith("data:"):
                data.append(line.removeprefix("data:").strip())
            elif not line and data:
                yield name, json.loads("\n".join(data))
                name, data = "", []
            # Anything else is a comment (a keep-alive ping) and carries nothing.

    @staticmethod
    def _detail(response: httpx.Response) -> str:
        """The API's own explanation of an error, which FastAPI puts under `detail`."""
        try:
            detail = response.json()["detail"]
        except (ValueError, KeyError, TypeError):
            return f"The API answered with an error ({response.status_code})."
        if isinstance(detail, str):
            return detail
        # Validation errors are a list, one entry per rejected field.
        return "; ".join(str(item.get("msg", item)) for item in detail)
