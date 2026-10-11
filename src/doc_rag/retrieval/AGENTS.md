# src/doc_rag/retrieval

## Purpose

Finds the chunks that best answer a question: dense and keyword search, rank fusion, reranking. `POST /search` exposes it; chat (phase 4+) reuses `RetrievalService`.

## Ownership

- `service.py` - `RetrievalService.search(query, top_k, document_ids)`: the whole flow
- `fusion.py` - `reciprocal_rank_fusion`: pure function, no I/O
- `router.py` - `POST /search`
- `schemas.py` - `SearchRequest`, `SearchResult`, `SearchResponse`

The SQL for both searches is in `db/repository.py` (`ChunksRepository.search_dense`, `search_lexical`); the reranker is `embedding/reranker.py` (`Reranker`).

## Local Contracts

- Flow: embed the query -> dense search and keyword search (`SEARCH_CANDIDATES` each) -> reciprocal rank fusion (`RRF_K`) -> the best `RERANK_CANDIDATES` go to the reranker -> `top_k` results sorted by reranker score.
- The reranker scores `Chunk.embed_text` (heading + text), the same text that was embedded. Results carry `Chunk.text`.
- `score` is the reranker's sigmoid relevance (0 to 1) and the only score returned; vector distances and keyword ranks are used for ordering inside the service and never exposed. Scores are absolute enough to threshold: on-topic hits were 0.5-1.0, off-topic or stop-word-only queries below 0.05 in manual tests.
- There is no minimum score and no empty result for off-topic questions: `/search` always returns up to `top_k` chunks if any chunk exists. Deciding that "the documents do not contain the answer" is the job of the answering step (phase 4), using `score`.
- `RetrievalService` is built per request (`get_retrieval_service`) from the request's `DBRepository` plus the `Embedder` and `Reranker` that live in the lifespan. Model calls run in `asyncio.to_thread`; both models hold a lock, so concurrent searches and ingestion queue on the GPU instead of overlapping.
- The keyword search uses `websearch_to_tsquery('english', ...)`: all words must occur in a chunk (stop words ignored, stemmed), `"quoted phrases"` and `-excluded` are supported. Natural-language questions therefore rarely match anything (0-2 hits in manual tests); it mainly helps short exact-term queries (`ByteNet`, `SQuAD v1.1 F1`). Dense search carries the long queries. A query of only stop words returns no keyword hits.
- `document_ids` restricts both searches to those documents; `None` searches everything. Dense search enables pgvector's iterative HNSW scan (`hnsw.iterative_scan = strict_order`, set per transaction) so a document filter still yields a full candidate list.
- The text-search configuration is `TEXT_SEARCH_CONFIG` in `db/models.py`; the generated `Chunk.tsv` column and the keyword query must always use the same one.
- Search never returns chunks of a document that is not `ready` because chunks and the `ready` status are written in one transaction.

## Work Guidance

- Add new retrieval behaviour here; `chat/pipeline.py` and `chat/graph.py` call `RetrievalService` and must not query chunks themselves.
- Tuning knobs are settings (`SEARCH_CANDIDATES`, `RERANK_CANDIDATES`, `SEARCH_TOP_K`, `RRF_K`, `RERANKER_BATCH_SIZE`); change them with evidence from the phase 6 evaluation, not by feel.
- Possible improvement once the evaluation exists: widen keyword matching (OR semantics, or a fallback when the AND query finds nothing) and compare hit rate; plain OR floods the candidate list with common words, so measure before adopting.

## Verification

- `uv run pytest tests/test_fusion.py tests/test_retrieval_service.py tests/test_search_api.py` - fusion maths, the service flow with a fake repository, embedder and reranker, and the route's validation
- With the stack running and a document ingested: `curl -H "Content-Type: application/json" -d '{"query": "..."}' http://localhost:8000/search`; an exact table header should return that table's page first, and a paraphrase of a concept should return the section that explains it

## Child DOX Index

None.
