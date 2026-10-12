import asyncio
from collections.abc import AsyncIterator
from types import SimpleNamespace

import pytest
from google.genai import errors

from src.doc_rag.config import settings
from src.doc_rag.llm.base import LLMError
from src.doc_rag.llm.gemini import GeminiClient

BEST, SECOND, LAST = "model-best", "model-second", "model-last"


def _api_error(code: int) -> errors.APIError:
    return errors.APIError(code, {"error": {"code": code, "message": "provider detail"}})


def _quota_error(retry_delay: str = "40022s") -> errors.APIError:
    """A 429 shaped like Gemini's, naming the spent quota and when to retry."""
    return errors.APIError(
        429,
        {
            "error": {
                "code": 429,
                "message": "You exceeded your current quota.",
                "details": [
                    {
                        "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                        "violations": [
                            {
                                "quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier",
                                "quotaValue": "20",
                            }
                        ],
                    },
                    {
                        "@type": "type.googleapis.com/google.rpc.RetryInfo",
                        "retryDelay": retry_delay,
                    },
                ],
            }
        },
    )


class Reply:
    """What one model does when called: yield `texts`, then raise `error`, after `delay`."""

    def __init__(self, *texts: str | None, error: Exception | None = None, delay: float = 0):
        self.texts = texts
        self.error = error
        self.delay = delay


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> GeminiClient:
    monkeypatch.setattr(settings, "GEMINI_MODELS", f"{BEST}, {SECOND},{LAST}")
    return GeminiClient()


@pytest.fixture
def gemini(client: GeminiClient, monkeypatch: pytest.MonkeyPatch):
    """Script the SDK's streaming call per model; returns the list of calls it received."""
    calls: list[dict] = []

    def _script(**replies: Reply) -> list[dict]:
        by_model = {name.replace("_", "-"): reply for name, reply in replies.items()}

        async def fake_stream(**kwargs) -> AsyncIterator[SimpleNamespace]:
            calls.append(kwargs)
            reply = by_model[kwargs["model"]]
            await asyncio.sleep(reply.delay)

            async def chunks() -> AsyncIterator[SimpleNamespace]:
                for text in reply.texts:
                    yield SimpleNamespace(text=text)
                if reply.error is not None:
                    raise reply.error

            return chunks()

        monkeypatch.setattr(client._client.aio.models, "generate_content_stream", fake_stream)
        return calls

    return _script


async def _collect(client: GeminiClient) -> list[str]:
    return [text async for text in client.stream("system rules", "sources and question")]


def _models(calls: list[dict]) -> list[str]:
    return [call["model"] for call in calls]


async def test_the_first_model_answers_and_empty_pieces_are_skipped(client, gemini) -> None:
    calls = gemini(model_best=Reply("Delta ", None, "", "is 0.5 [1]."))

    assert await _collect(client) == ["Delta ", "is 0.5 [1]."]
    assert _models(calls) == [BEST]


async def test_the_model_gets_the_instructions_and_the_prompt(client, gemini) -> None:
    calls = gemini(model_best=Reply("ok"))

    await _collect(client)

    (call,) = calls
    assert call["model"] == BEST
    assert call["contents"] == "sources and question"
    assert call["config"].system_instruction == "system rules"
    assert call["config"].temperature == settings.LLM_TEMPERATURE


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(_quota_error(), id="quota-spent"),
        pytest.param(_api_error(503), id="overloaded"),
        pytest.param(_api_error(500), id="server-error"),
    ],
)
async def test_the_next_model_answers_when_one_has_no_quota_or_is_overloaded(
    client, gemini, error
) -> None:
    calls = gemini(model_best=Reply(error=error), model_second=Reply("from second"))

    assert await _collect(client) == ["from second"]
    assert _models(calls) == [BEST, SECOND]


async def test_a_model_that_does_not_start_in_time_is_skipped(
    client, gemini, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "LLM_START_TIMEOUT_SECONDS", 0.05)
    calls = gemini(model_best=Reply("too late", delay=1), model_second=Reply("on time"))

    assert await _collect(client) == ["on time"]
    assert _models(calls) == [BEST, SECOND]


async def test_a_model_whose_quota_is_spent_is_not_called_again(client, gemini) -> None:
    calls = gemini(model_best=Reply(error=_quota_error()), model_second=Reply("answer"))

    await _collect(client)
    await _collect(client)

    assert _models(calls) == [BEST, SECOND, SECOND]


async def test_a_rate_limited_model_is_used_again_after_the_stated_delay(
    client, gemini, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = [1000.0]
    monkeypatch.setattr("src.doc_rag.llm.gemini.time.monotonic", lambda: now[0])
    calls = gemini(model_best=Reply(error=_quota_error("40s")), model_second=Reply("answer"))

    await _collect(client)
    now[0] += 39
    await _collect(client)
    now[0] += 2
    await _collect(client)

    assert _models(calls) == [BEST, SECOND, SECOND, BEST, SECOND]


async def test_an_overloaded_model_is_tried_again_on_the_next_question(client, gemini) -> None:
    calls = gemini(model_best=Reply(error=_api_error(503)), model_second=Reply("answer"))

    await _collect(client)
    await _collect(client)

    assert _models(calls) == [BEST, SECOND, BEST, SECOND]


async def test_when_every_quota_is_spent_the_error_names_the_models(client, gemini) -> None:
    calls = gemini(
        model_best=Reply(error=_quota_error()),
        model_second=Reply(error=_quota_error()),
        model_last=Reply(error=_quota_error()),
    )

    with pytest.raises(LLMError) as error:
        await _collect(client)

    assert _models(calls) == [BEST, SECOND, LAST]
    for model in (BEST, SECOND, LAST):
        assert model in str(error.value)
    assert "midnight Pacific" in str(error.value)

    # Asking again makes no request at all.
    with pytest.raises(LLMError, match="used up"):
        await _collect(client)
    assert len(calls) == 3


async def test_when_every_model_is_overloaded_the_error_says_so(client, gemini) -> None:
    gemini(
        model_best=Reply(error=_api_error(503)),
        model_second=Reply(error=_api_error(503)),
        model_last=Reply(error=_api_error(503)),
    )

    with pytest.raises(LLMError, match="overloaded"):
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
async def test_a_bad_api_key_fails_at_once_without_trying_other_models(
    client, gemini, error
) -> None:
    calls = gemini(model_best=Reply(error=error), model_second=Reply("never used"))

    with pytest.raises(LLMError, match="GEMINI_API_KEY"):
        await _collect(client)
    assert _models(calls) == [BEST]


async def test_other_api_errors_report_the_code_without_provider_details(client, gemini) -> None:
    gemini(model_best=Reply(error=_api_error(404)))

    with pytest.raises(LLMError) as error:
        await _collect(client)

    assert "404" in str(error.value)
    assert "provider detail" not in str(error.value)


async def test_an_answer_with_no_text_is_an_error(client, gemini) -> None:
    gemini(model_best=Reply(None))

    with pytest.raises(LLMError, match="empty answer"):
        await _collect(client)


async def test_a_failure_after_the_answer_started_does_not_switch_models(client, gemini) -> None:
    calls = gemini(
        model_best=Reply("Delta is ", error=_api_error(503)), model_second=Reply("never used")
    )
    received: list[str] = []

    with pytest.raises(LLMError, match="overloaded"):
        async for text in client.stream("system rules", "sources and question"):
            received.append(text)

    assert received == ["Delta is "]
    assert _models(calls) == [BEST]


def test_the_client_retries_server_errors_but_never_rate_limits(client: GeminiClient) -> None:
    options = client._client._api_client._http_options

    assert options.retry_options.attempts == settings.LLM_MAX_ATTEMPTS
    assert options.timeout == settings.LLM_ANSWER_TIMEOUT_SECONDS * 1000
    assert 429 not in options.retry_options.http_status_codes
    assert 503 in options.retry_options.http_status_codes


def test_an_empty_model_list_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "GEMINI_MODELS", " , ")

    with pytest.raises(ValueError, match="GEMINI_MODELS"):
        GeminiClient()
