# src/doc_rag/chat

## Purpose

Answers a question from the documents and streams the answer with its sources. `POST /chat` exposes it.

## Ownership

- `pipeline.py` - `ChatPipeline.run(question, top_k, document_ids)`: the hand-written flow
- `router.py` - `POST /chat`, Server-Sent Events
- `schemas.py` - `ChatRequest`, `Source`, and the event models `ChatSources`, `ChatToken`, `ChatDone`, `ChatError` (`ChatEvent` is their union)

`graph.py` (the LangGraph variant, phase 7) does not exist yet.

## Local Contracts

- Flow: `RetrievalService.search` -> drop results with `score < CHAT_MIN_SCORE` -> number the rest from 1 -> `build_prompt` -> `LLMClient.stream` -> collect the text and extract citations.
- Events, in order: one `sources` (possibly empty), zero or more `token`, then exactly one of `done` or `error`. Each event model carries its name in an `event` field; the router sends it as the SSE event name and the whole model as JSON data.
- `ChatPipeline.run` never raises. An `LLMError` becomes an `error` event with the error's own message; any other exception is logged with its traceback and becomes an `error` event with a generic message (internal details are never sent to the client).
- Declining: when no result reaches `CHAT_MIN_SCORE`, the pipeline answers `NOT_FOUND_ANSWER` itself and does not call the LLM (no quota spent). When sources exist but do not contain the answer, the system prompt makes the model reply with the same sentence. In both cases `citations` is empty.
- `citations` in `done` are the numbers written as `[n]` in the answer that match an existing source, ascending and without duplicates. A cited number that does not exist is ignored, not an error.
- `Source.number` is the position in the list sent in the `sources` event and the number used in the prompt; they must never be renumbered between the two.
- `ChatPipeline` is built per request (`get_chat_pipeline`) from the request's `RetrievalService` and the lifespan's `LLMClient`. It contains no search logic and no provider code.
- Each request is single-turn: there is no conversation history yet (phase 8).
- `CHAT_MIN_SCORE` (default 0.05) comes from manual tests in phase 3 (off-topic results scored at most 0.03, relevant ones at least 0.35); revisit it with the phase 6 evaluation.

## Work Guidance

- Keep the router a pass-through; anything that decides what to answer belongs in the pipeline.
- New consumers that need the answer without SSE (MCP tools, the evaluation script) should iterate `ChatPipeline.run` and read the `done` event, not parse the HTTP stream.

## Verification

- `uv run pytest tests/test_chat_pipeline.py tests/test_chat_api.py` - the flow with a fake search and a fake LLM (event order, score cutoff, declining, both failure paths, citation parsing) and the route's SSE output and validation
- With the stack running: `curl -N -H "Content-Type: application/json" -d '{"question": "..."}' http://localhost:8000/chat` (`-N` shows events as they arrive)

## Child DOX Index

None.
