# src/doc_rag/chat

## Purpose

Answers a question from the documents and streams the answer with its sources. `POST /chat` exposes it, with two interchangeable orchestrators over the same retrieval layer, chosen per request by `mode`.

## Ownership

- `pipeline.py` - `ChatPipeline.run(question, top_k, document_ids)`: the hand-written flow (`mode: "simple"`, the default). Also the shared answering code both orchestrators use: `number_sources`, `respond`, `extract_citations`, `guard`
- `graph.py` - `ChatGraph.run(...)`: the LangGraph flow (`mode: "graph"`). The only module that imports LangGraph
- `router.py` - `POST /chat`, Server-Sent Events; picks `ChatGraph` or `ChatPipeline` from `mode`
- `schemas.py` - `ChatRequest`, `Source`, and the event models `ChatStep`, `ChatSources`, `ChatToken`, `ChatDone`, `ChatError` (`ChatEvent` is their union)

## Local Contracts

- Simple flow: `RetrievalService.search` -> drop results with `score < CHAT_MIN_SCORE` -> number the rest from 1 -> `build_prompt` -> `LLMClient.stream` -> collect the text and extract citations.
- Graph flow (corrective): `retrieve` (the question as asked) -> `grade` -> good: `answer`; weak: `rewrite` -> `retrieve` -> `grade` -> `answer`. Results are weak when the best score is below `GRAPH_GOOD_SCORE` (0.3); at most `GRAPH_MAX_RETRIES` (1) extra searches are made, so a question costs one LLM request when the first results are good (the same as simple mode) and two when it is rewritten. The search whose top result scored higher is the one answered from. The plan listed "rewrite -> retrieve -> grade", but rewriting every question first would double the free-tier requests per question for results that were already good.
- The grader is the reranker's score, not the LLM: it is a model trained to judge relevance, it is free and repeatable, and an LLM grader would cost another request on every question.
- `rewrite` asks the LLM (`REWRITE_SYSTEM_PROMPT`, listing the queries already tried) for a differently worded, English search query, and cleans the reply to one short line. It changes nothing, and the graph moves on to `answer`, when the reply is empty, repeats a query already tried, or the LLM fails (`LLMError`): a question that would be declined anyway is still declined without an error, and a real LLM problem is reported by the answer step. The answer is always written for the user's own question, never for the rewritten query.
- The graph is compiled once at import (`compiled_graph`); per-request things (`RetrievalService`, `LLMClient`) travel in the run context (`GraphContext`, read by nodes through `Runtime.context`), never in module state, so concurrent requests cannot see each other. Nodes send events with `runtime.stream_writer`, and `ChatGraph.run` streams them with `stream_mode="custom"`.
- Events, in order: zero or more `step` (graph mode only: `message` says what was done, e.g. that the best score was low and which query is searched next), one `sources` (possibly empty), zero or more `token`, then exactly one of `done` or `error`. Each event model carries its name in an `event` field; the router sends it as the SSE event name and the whole model as JSON data. A client that does not know `step` can ignore it.
- `run` never raises, in either mode (`guard` in `pipeline.py`). An `LLMError` becomes an `error` event with the error's own message; any other exception is logged with its traceback and becomes an `error` event with a generic message (internal details are never sent to the client).
- Declining: when no result reaches `CHAT_MIN_SCORE`, `respond` answers `NOT_FOUND_ANSWER` itself and does not call the LLM (no quota spent). When sources exist but do not contain the answer, the system prompt makes the model reply with the same sentence. In both cases `citations` is empty.
- `citations` in `done` are the numbers written as `[n]` in the answer that match an existing source, ascending and without duplicates. A cited number that does not exist is ignored, not an error.
- `Source.number` is the position in the list sent in the `sources` event and the number used in the prompt; they must never be renumbered between the two.
- Both orchestrators are built per request (`get_chat_pipeline`, `get_chat_graph`) from the request's `RetrievalService` and the lifespan's `LLMClient`. Neither contains search logic or provider code. The MCP tools use `ChatPipeline` only.
- Each request is single-turn: there is no conversation history yet (phase 8).
- `CHAT_MIN_SCORE` (default 0.05) comes from manual tests in phase 3 (off-topic results scored at most 0.03, relevant ones at least 0.35); the phase 6 evaluation (`eval/AGENTS.md`) shows how far the unanswerable questions are from it: three scored 0.002-0.004, but a GPT-4 question scored 0.070. Re-measure with `eval/run_eval.py` before moving the threshold. `GRAPH_GOOD_SCORE` comes from the same evaluation (every relevant first result scored at least 0.319).

## Work Guidance

- Keep the router a pass-through; anything that decides what to answer belongs in an orchestrator.
- Anything both orchestrators do (numbering, prompt, streaming, citations, error handling) lives in `pipeline.py` and is imported by `graph.py`; do not copy it. Anything about searching belongs in `retrieval/`.
- LangGraph stays inside `graph.py`. Nodes are module-level functions that read the state and `Runtime[GraphContext]`, return only the keys they change, and are tested with fake retrieval and LLM classes.
- Consumers that run inside the application and need the answer without SSE (the MCP tools) should iterate `ChatPipeline.run` and read the `done` event, not parse the HTTP stream. The evaluation script is a separate HTTP client and reads `/chat` itself (see `eval/AGENTS.md`).
- Judge a change to either orchestrator, or to the thresholds, with `uv run python eval/run_eval.py --answers --mode simple|graph`.

## Verification

- `uv run pytest tests/test_chat_pipeline.py tests/test_chat_graph.py tests/test_chat_api.py` - the simple flow and the graph flow with a fake search and a fake LLM (event order, score cutoff, declining, rewriting and its limits, failed or useless rewrites, both failure paths, citation parsing) and the route's SSE output, mode selection and validation
- With the stack running: `curl -N -H "Content-Type: application/json" -d '{"question": "...", "mode": "graph"}' http://localhost:8000/chat` (`-N` shows events as they arrive)

## Child DOX Index

None.
