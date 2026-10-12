from collections.abc import AsyncIterator
from typing import Protocol


class LLMError(Exception):
    """The LLM could not produce an answer; the message is safe to show to the user."""


class LLMClient(Protocol):
    """What the rest of the application needs from a language model.

    Any class with this method can be used (Gemini today, Ollama later); the
    application never imports a provider's SDK directly.
    """

    def stream(self, system: str, prompt: str) -> AsyncIterator[str]:
        """Yield the answer piece by piece as it is generated.

        `system` holds the standing instructions, `prompt` the sources and the question.
        Raises `LLMError` when no answer can be produced.
        """
        ...
