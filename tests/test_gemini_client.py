import asyncio
from collections.abc import AsyncIterator
from types import SimpleNamespace

import pytest
from google.genai import errors

from src.doc_rag.config import settings
from src.doc_rag.llm.base import LLMError
from src.doc_rag.llm.gemini import GeminiClient


def _api_error(code: int) -> errors.APIError:
    return errors.APIError(code, {"error": {"code": code, "message": "provider detail"}})


@pytest.fixture
def client() -> GeminiClient:
    return GeminiClient()


@pytest.fixture
def respond(client: GeminiClient, monkeypatch: pytest.MonkeyPatch):
    """Make the SDK's streaming call yield the given texts, then raise the given error.

    `delay` is how long the call takes before the stream opens, in seconds.
    """
    calls: list[dict] = []

    def _respond(
        texts: list[str | None], error: Exception | None = None, delay: float = 0
    ) -> list[dict]:
        async def fake_stream(**kwargs) -> AsyncIterator[SimpleNamespace]:
            calls.append(kwargs)
            await asyncio.sleep(delay)

            async def chunks() -> AsyncIterator[SimpleNamespace]:
                for text in texts:
                    yield SimpleNamespace(text=text)
                if error is not None:
                    raise error

            return chunks()

        monkeypatch.setattr(client._client.aio.models, "generate_content_stream", fake_stream)
        return calls

    return _respond


async def _collect(client: GeminiClient) -> list[str]:
    return [text async for text in client.stream("system rules", "sources and question")]


async def test_stream_yields_the_text_pieces_and_skips_empty_ones(client, respond) -> None:
    respond(["Delta ", None, "", "is 0.5 [1]."])

    assert await _collect(client) == ["Delta ", "is 0.5 [1]."]


async def test_stream_sends_the_model_the_instructions_and_the_prompt(client, respond) -> None:
    calls = respond(["ok"])

    await _collect(client)

    (call,) = calls
    assert call["model"] == settings.GEMINI_MODEL
    assert call["contents"] == "sources and question"
    assert call["config"].system_instruction == "system rules"
    assert call["config"].temperature == settings.LLM_TEMPERATURE


async def test_a_rate_limit_error_tells_the_user_to_wait(client, respond) -> None:
    respond([], error=_api_error(429))

    with pytest.raises(LLMError, match="free-tier limit"):
        await _collect(client)


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(
            errors.APIError(400, {"error": {"code": 400, "message": "API key not valid."}}),
            id="gemini-style-400",
        ),
        pytest.param(_api_error(401), id="401"),
        pytest.param(_api_error(403), id="403"),
    ],
)
async def test_a_bad_api_key_tells_the_user_what_to_fix(client, respond, error) -> None:
    respond([], error=error)

    with pytest.raises(LLMError, match="GEMINI_API_KEY"):
        await _collect(client)


@pytest.mark.parametrize("code", [500, 503])
async def test_a_server_error_says_gemini_is_overloaded(client, respond, code: int) -> None:
    respond([], error=_api_error(code))

    with pytest.raises(LLMError, match="overloaded"):
        await _collect(client)


async def test_other_api_errors_report_the_code_without_provider_details(client, respond) -> None:
    respond([], error=_api_error(404))

    with pytest.raises(LLMError) as error:
        await _collect(client)

    assert "404" in str(error.value)
    assert "provider detail" not in str(error.value)


async def test_an_answer_that_does_not_start_in_time_is_an_error(
    client, respond, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "LLM_START_TIMEOUT_SECONDS", 0.05)
    respond(["too late"], delay=1)

    with pytest.raises(LLMError, match="did not start answering"):
        await _collect(client)


async def test_a_failure_after_the_answer_started_is_still_an_llm_error(client, respond) -> None:
    respond(["Delta is "], error=_api_error(503))
    received: list[str] = []

    with pytest.raises(LLMError, match="overloaded"):
        async for text in client.stream("system rules", "sources and question"):
            received.append(text)

    assert received == ["Delta is "]


async def test_an_answer_with_no_text_is_an_error(client, respond) -> None:
    respond([None])

    with pytest.raises(LLMError, match="empty answer"):
        await _collect(client)


def test_the_client_is_configured_to_retry_and_time_out(client: GeminiClient) -> None:
    options = client._client._api_client._http_options

    assert options.retry_options.attempts == settings.LLM_MAX_ATTEMPTS
    assert options.timeout == settings.LLM_ANSWER_TIMEOUT_SECONDS * 1000
