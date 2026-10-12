import asyncio
import logging
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


class GeminiClient:
    """`LLMClient` backed by the Gemini API. Build once; it keeps an HTTP connection pool."""

    def __init__(self) -> None:
        self._client = genai.Client(
            api_key=settings.GEMINI_API_KEY,
            http_options=types.HttpOptions(
                # Also sent to Gemini as its own deadline for the whole answer, so this
                # must be the total time allowed, not the time to the first text.
                timeout=settings.LLM_ANSWER_TIMEOUT_SECONDS * 1000,
                # The SDK never retries unless told to. With these options it retries rate
                # limits (429) and server errors with exponential backoff: 2s, 4s, 8s, ...
                retry_options=types.HttpRetryOptions(
                    attempts=settings.LLM_MAX_ATTEMPTS, initial_delay=2.0
                ),
            ),
        )
        logger.info(f"Gemini client ready (model: {settings.GEMINI_MODEL})")

    async def stream(self, system: str, prompt: str) -> AsyncIterator[str]:
        try:
            # One deadline for everything before the answer starts: the request, the
            # SDK's retries and the model's thinking. Without it, retries against an
            # overloaded model could keep a question waiting for minutes.
            async with asyncio.timeout(settings.LLM_START_TIMEOUT_SECONDS):
                chunks = await self._client.aio.models.generate_content_stream(
                    model=settings.GEMINI_MODEL,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system, temperature=settings.LLM_TEMPERATURE
                    ),
                )
                first = await self._first_text(chunks)
            if first is None:
                # Happens when the response is blocked or cut off before any text.
                raise LLMError("Gemini returned an empty answer. Try rephrasing the question.")
            yield first
            async for chunk in chunks:
                if chunk.text:
                    yield chunk.text
        except TimeoutError as exc:
            logger.warning(f"Gemini did not answer within {settings.LLM_START_TIMEOUT_SECONDS}s")
            raise LLMError(
                "Gemini did not start answering within "
                f"{settings.LLM_START_TIMEOUT_SECONDS} seconds. "
                "It may be overloaded; ask again in a moment."
            ) from exc
        except errors.APIError as exc:
            logger.warning(f"Gemini request failed with {exc.code}: {exc.message}")
            raise LLMError(self._explain(exc)) from exc

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
            return (
                "Gemini is overloaded right now. Ask again in a moment, "
                "or set GEMINI_MODEL to another free-tier model."
            )
        return f"Gemini could not answer (error {error.code})."
