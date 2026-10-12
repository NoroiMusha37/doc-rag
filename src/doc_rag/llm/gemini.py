import asyncio
import logging
import time
from collections.abc import AsyncIterator

from google import genai
from google.genai import errors, types

from src.doc_rag.config import settings
from src.doc_rag.llm.base import LLMError

logger = logging.getLogger(__name__)

RATE_LIMITED = 429
# Gemini reports a bad key as 400 with an "API key not valid" message, not as 401/403.
UNAUTHORIZED = (401, 403)
SERVER_ERROR = 500
RETRIED_STATUS_CODES = (408, 500, 502, 503, 504)
# How long a rate-limited model is skipped when Gemini does not say when to retry.
DEFAULT_RETRY_SECONDS = 60.0


class GeminiClient:
    """`LLMClient` backed by the Gemini API. Build once; it keeps an HTTP connection pool.

    Models are tried in the order of GEMINI_MODELS. When one has no quota left or is
    overloaded, the next one answers, and a model whose quota is spent is skipped until
    the time Gemini said it can be used again.
    """

    def __init__(self) -> None:
        self._models = [name.strip() for name in settings.GEMINI_MODELS.split(",") if name.strip()]
        if not self._models:
            raise ValueError("GEMINI_MODELS must name at least one model")
        # Model name -> time.monotonic() value until which its quota is known to be spent.
        self._limited_until: dict[str, float] = {}
        self._client = genai.Client(
            api_key=settings.GEMINI_API_KEY,
            http_options=types.HttpOptions(
                # Also sent to Gemini as its own deadline for the whole answer, so this
                # must be the total time allowed, not the time to the first text.
                timeout=settings.LLM_ANSWER_TIMEOUT_SECONDS * 1000,
                # The SDK never retries unless told to. With these options it retries
                # timeouts and server errors with exponential backoff: 2s, 4s, 8s, ...
                # Rate limits (429) are left out on purpose: the free tier's daily quota is
                # small, retrying cannot succeed once it is spent, and every try counts.
                retry_options=types.HttpRetryOptions(
                    attempts=settings.LLM_MAX_ATTEMPTS,
                    initial_delay=2.0,
                    http_status_codes=list(RETRIED_STATUS_CODES),
                ),
            ),
        )
        logger.info(f"Gemini client ready (models, in order: {', '.join(self._models)})")

    async def stream(self, system: str, prompt: str) -> AsyncIterator[str]:
        failure: LLMError | None = None
        for model in self._models:
            if self._limited_until.get(model, 0) > time.monotonic():
                continue
            try:
                # One deadline for everything before the answer starts: the request, the
                # SDK's retries and the model's thinking. Without it, retries against an
                # overloaded model could keep a question waiting for minutes.
                async with asyncio.timeout(settings.LLM_START_TIMEOUT_SECONDS):
                    chunks = await self._client.aio.models.generate_content_stream(
                        model=model,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            system_instruction=system, temperature=settings.LLM_TEMPERATURE
                        ),
                    )
                    first = await self._first_text(chunks)
            except TimeoutError:
                logger.warning(f"{model} did not answer in {settings.LLM_START_TIMEOUT_SECONDS}s")
                failure = LLMError(
                    "Gemini did not start answering within "
                    f"{settings.LLM_START_TIMEOUT_SECONDS} seconds. "
                    "It may be overloaded; ask again in a moment."
                )
                continue
            except errors.APIError as exc:
                logger.warning(f"{model} failed with {exc.code}: {exc.message}")
                failure = LLMError(self._explain(exc))
                if exc.code == RATE_LIMITED:
                    self._limited_until[model] = time.monotonic() + self._retry_seconds(exc)
                if exc.code == RATE_LIMITED or exc.code >= SERVER_ERROR:
                    continue  # another model has its own quota and load
                raise failure from exc

            if first is None:
                # Happens when the response is blocked or cut off before any text.
                raise LLMError("Gemini returned an empty answer. Try rephrasing the question.")
            if model != self._models[0]:
                logger.info(f"Answering with fallback model {model}")
            yield first
            try:
                async for chunk in chunks:
                    if chunk.text:
                        yield chunk.text
            except errors.APIError as exc:
                # The answer has started, so switching models would repeat text.
                logger.warning(f"{model} failed mid-answer with {exc.code}: {exc.message}")
                raise LLMError(self._explain(exc)) from exc
            return

        if all(self._limited_until.get(model, 0) > time.monotonic() for model in self._models):
            raise LLMError(
                "The free limits of all configured Gemini models are used up for now "
                f"({', '.join(self._models)}). Daily limits reset at midnight Pacific time."
            )
        raise failure or LLMError("No Gemini model could answer.")

    async def close(self) -> None:
        await self._client.aio.aclose()

    @staticmethod
    async def _first_text(chunks: AsyncIterator[types.GenerateContentResponse]) -> str | None:
        """Read until the first piece of text; None if the stream ends without any."""
        async for chunk in chunks:
            if chunk.text:
                return chunk.text
        return None

    @staticmethod
    def _explain(error: errors.APIError) -> str:
        """A message for the user that says what to do, without provider internals."""
        if error.code == RATE_LIMITED:
            return "The Gemini free-tier limit was reached. Wait a minute and ask again."
        if error.code in UNAUTHORIZED or "API key" in (error.message or ""):
            return (
                "The Gemini API key is missing or invalid. "
                "Set GEMINI_API_KEY in .env and restart the application."
            )
        if error.code >= SERVER_ERROR:
            return "Gemini is overloaded right now. Ask again in a moment."
        return f"Gemini could not answer (error {error.code})."

    @staticmethod
    def _retry_seconds(error: errors.APIError) -> float:
        """How long Gemini says to wait before using the model again.

        A 429 carries a list of details; one of them has `retryDelay` like "40022s"
        (about 11 hours for a spent daily quota, under a minute for a per-minute limit).
        """
        if isinstance(error.details, dict):
            for detail in error.details.get("error", {}).get("details", []):
                delay = str(detail.get("retryDelay", "")).removesuffix("s")
                try:
                    return float(delay)
                except ValueError:
                    continue
        return DEFAULT_RETRY_SECONDS
