# src/doc_rag/llm

## Purpose

The only place that talks to a language model. Everything else depends on the `LLMClient` interface, never on a provider's SDK.

## Ownership

- `base.py` - `LLMClient` (a `Protocol` with one method, `stream(system, prompt)`) and `LLMError`
- `gemini.py` - `GeminiClient`, the `LLMClient` backed by the Gemini API (`google-genai`)
- `prompts.py` - `SYSTEM_PROMPT`, `NOT_FOUND_ANSWER`, `build_prompt(question, sources)`

## Local Contracts

- `LLMClient.stream(system, prompt)` is an async generator of text pieces. It raises `LLMError` when no answer can be produced; the message of an `LLMError` is written for the end user and is shown to them as-is, so it must not contain provider internals, keys or raw response bodies.
- Only `gemini.py` imports `google.genai`. Provider errors (`google.genai.errors.APIError`) are caught there and re-raised as `LLMError`; nothing outside this folder catches SDK exceptions.
- `GeminiClient` is built once in the lifespan (it owns an HTTP connection pool), reached through `get_llm`, and closed on shutdown. It holds no per-request state.
- Models: `GEMINI_MODELS` is a comma-separated list, best first (default `gemini-3.8-flash,gemini-3.6-flash,gemini-3.5-flash-lite`). `stream` tries them in order and moves to the next when a model returns 429 (quota), a 5xx, or does not start answering within `LLM_START_TIMEOUT_SECONDS`. It never switches once text has been yielded (that would repeat text), and it does not switch on errors another model would also hit (bad key, 4xx). Each model has its own quota, so this is what makes the free tier usable: the limits seen on 2026-10-12 were 20, 20 and 500 requests per day.
- A model that returned 429 is remembered in `_limited_until` and skipped until the `retryDelay` Gemini sent (about 11 hours for a spent daily quota, seconds for a per-minute limit; 60 s if none is given). This state is in memory: after a restart each spent model costs one refused request to rediscover. An overloaded model (5xx) is not remembered and is tried again on the next question.
- When every model is rate-limited the error names them and says daily limits reset at midnight Pacific time, and no request is made until one is usable again. Otherwise the last model's failure is reported. A fallback answer is logged at INFO (`Answering with fallback model ...`); the API does not tell the client which model answered.
- Retries are the SDK's own: `HttpRetryOptions(attempts=LLM_MAX_ATTEMPTS, initial_delay=2.0, http_status_codes=[408, 500, 502, 503, 504])` retries server errors and transport timeouts with exponential backoff and jitter, silently. 429 is deliberately not retried: a spent daily quota cannot recover and every try counts against it. Retries cover the request that opens the stream, not a failure after text has started.
- Two time limits. `LLM_START_TIMEOUT_SECONDS` (20) is one deadline, per model, for everything before the first piece of text: the request, all retries and the model's thinking (`asyncio.timeout` around opening the stream and reading up to the first text). Nothing is yielded inside the timeout block: a cancel scope must not be held across a `yield` in an async generator. `LLM_ANSWER_TIMEOUT_SECONDS` (180) is `HttpOptions.timeout`: the SDK uses it as the HTTP timeout and also sends it to Gemini as `X-Server-Timeout`, so Gemini aborts any answer still running at that point with 504 "Deadline expired". It must therefore be the total time allowed for an answer, never the time to the first text.
- Error messages by cause: no model started in time says Gemini may be overloaded; 429 says a limit was reached; a bad key tells the user to set `GEMINI_API_KEY` (Gemini reports it as 400 with "API key not valid", so the message text is checked as well as 401/403); 5xx says Gemini is overloaded; anything else reports only the status code.
- A stream that ends without any text (blocked or truncated response) is an `LLMError`, not an empty answer.
- Measured on 2026-10-12 with `gemini-3.8-flash` on the free tier and a 15k-character prompt: a healthy call starts answering in 3.6-4.8 s with about 400 thinking tokens; under load the same call returned 503 ("high demand"), hung, or stalled mid-answer until the server deadline (504), and with retries took 24-52 s. `thinking_level="minimal"` is rejected by this model (400).
- Every model in `GEMINI_MODELS` must have a free tier (see the pricing page linked in `config.py`). Per-model free limits are not published; they are shown per account at https://aistudio.google.com/rate-limit.
- `build_prompt` numbers sources from 1 by position; those numbers are the citation numbers the model is told to use, so callers must keep the same order when mapping citations back.
- `NOT_FOUND_ANSWER` is the exact sentence the model is told to reply with when the sources lack the answer, and the same sentence the chat pipeline returns itself when the search finds nothing relevant.

## Work Guidance

- A new provider (Ollama, phase 8) is a new file with a class that has the same `stream` method and raises `LLMError`; no other module changes except the lifespan line that picks the client.
- Change the prompt only together with a check on real questions; from phase 6 that check is the evaluation set.
- Free-tier inputs may be used by Google to improve its products; only retrieved excerpts and the question are sent.

## Verification

- `uv run pytest tests/test_gemini_client.py tests/test_prompts.py` - the client against a faked SDK call (text pieces, falling back on quota, overload and slow start, skipping a spent model until its retry time, all models spent, bad key, other errors, empty answer, a failure mid-answer, retry/timeout config) and the prompt format; no network or key needed
- Real behaviour needs a valid `GEMINI_API_KEY` in `.env` and is checked through `POST /chat` on the running stack

## Child DOX Index

None.
