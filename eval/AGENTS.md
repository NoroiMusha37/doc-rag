# eval

## Purpose

Measures how well the running application finds and answers questions, so that a change to retrieval, chunking, prompts or models can be judged by numbers instead of by feel. It is the regression check for every later change.

## Ownership

- `questions.jsonl` - the hand-written question set, one JSON object per line
- `run_eval.py` - the script: loads the questions, asks the API, prints the report, optionally writes an answers file
- `results/` - generated answers files; git-ignored

## Local Contracts

- The script is an HTTP client of the running API (`--api-url`, default `http://localhost:8000`) and imports nothing from `src/`. It therefore measures what is deployed, does not load the models a second time (they would compete for the 8 GB of VRAM), and can compare the `mode` of `/chat` once phase 7 adds it. The one value it cannot read from the server, `CHAT_MIN_SCORE`, is repeated as `DEFAULT_MIN_SCORE`; `tests/test_eval.py` fails if the two differ.
- A question has `id` (unique), `question`, `kind` (`text`, `table`, `figure` or `formula`: what holds the answer; `none` for a question the documents cannot answer), `document` (file name as uploaded), `pages` (every page that holds the answer, 1-based), optional `language` (`en` default, `uk`) and `answer_contains` (words the answer must contain, case-insensitive; `"a|b"` accepts either). A `none` question has no `document` and no `pages`; anything else needs both. The loader rejects inconsistent lines and names the line.
- Retrieval is scored with `POST /search` (`--top-k`, default 10): a result is relevant when it is from the expected document and its page range covers an expected page. Reported: hit@1, hit@3, hit@6 (share of questions with a relevant result in the top k) and MRR (mean of 1/rank of the first relevant result), overall and per `kind` and `language`; the questions without any relevant result and what ranked above the right page when it was not first; the lowest and median score of the first relevant result; search time. Retrieval-only runs spend no Gemini requests.
- Unanswerable questions are scored by the same threshold the chat pipeline uses: a top reranker score below `CHAT_MIN_SCORE` is declined without the LLM. The report prints each top score and the highest one, to compare with the lowest score of a relevant result when `CHAT_MIN_SCORE` is revisited.
- `--mode simple|graph` (default `simple`) is the `mode` sent to `/chat`, so one question set compares the two orchestrators. The retrieval report always uses `/search`, which is the simple flow's search; in graph mode the sources actually used are in the answer report (`relevant page among the sources`, `first source`, how many questions were rewritten). Graph mode spends one more request per rewritten question, and the report's request estimate does not include them.
- `--answers` also asks `/chat` for every selected question. It spends one Gemini request per question that reaches the LLM (the script prints the number first), so use `--ids` for a handful. An answerable question is correct when the answer contains all `answer_contains` words and cites a source on an expected page; an unanswerable one when it cites nothing. These are only word checks: read the Markdown file written to `results/` for the real verdict.
- The question set belongs to two public papers that must be uploaded under these exact names: `attention.pdf` (arXiv 1706.03762v7, 15 pages, https://arxiv.org/pdf/1706.03762v7) and `bert.pdf` (arXiv 1810.04805v2, 16 pages, https://arxiv.org/pdf/1810.04805v2). Another version has other page numbers. The script stops with a list of the documents that are missing or not `ready`. The PDFs are not stored in the repository.

## Work Guidance

- Label a new question from the document itself: read the page text (for example with `pypdfium2`, which Docling installs) and list every page that states the answer. Never take the expected page from the application's own search results, and never change a label to make a number better; fix it only when the PDF shows the label was wrong.
- Keep a mix of kinds, include questions in Ukrainian and questions the documents cannot answer, and keep the set between 20 and 30 questions so a run stays fast.
- Questions for the thesis documents can be added the same way (`document` is the uploaded file name). Compare runs only on the same documents.
- The two papers are small (about 100 chunks), so hit rates here are generous. A bigger or harder corpus will score lower; a change should be judged by the difference between runs, not by the absolute number.
- Run it before and after any retrieval, chunking, reranking or prompt change, with no other GPU load: a game running in the background starved the container's GPU work (searches took 50 s to minutes instead of 0.5 s). Scores do not depend on it, timings do.

## Baseline

Recorded 2026-10-15 with the GPU free, on `attention.pdf` and `bert.pdf` (94 chunks), `top_k` 10, defaults `SEARCH_CANDIDATES` 30, `RERANK_CANDIDATES` 30, `RRF_K` 60, 26 answerable questions. Two runs gave identical numbers.

| Group   | n  | hit@1 | hit@3 | hit@6 | MRR   |
|---------|----|-------|-------|-------|-------|
| all     | 26 | 92.3% | 100%  | 100%  | 0.955 |
| text    | 10 | 90.0% | 100%  | 100%  | 0.933 |
| table   | 12 | 91.7% | 100%  | 100%  | 0.958 |
| figure  | 2  | 100%  | 100%  | 100%  | 1.000 |
| formula | 2  | 100%  | 100%  | 100%  | 1.000 |
| en      | 23 | 91.3% | 100%  | 100%  | 0.949 |
| uk      | 3  | 100%  | 100%  | 100%  | 1.000 |

- Not first: `att-params-big` #2 (a BERT page about model size, 0.81, outranks the Attention paper's Table 3) and `bert-pretraining-setup` #3 (BERT pages 14 and 16 mention the steps and batch size but not the learning rate; only page 13 has all three).
- Score of the first relevant result: lowest 0.319, median 0.981. Unanswerable top scores: 0.002-0.004, and 0.070 for `none-gpt4-params`, which therefore reaches the LLM (`CHAT_MIN_SCORE` 0.05). Any threshold between 0.07 and 0.3 would separate this set; it is too small to justify moving it yet.
- Search time: median 0.43 s, slowest 0.82 s.
- Answers (`--answers` on 7 questions, Gemini fallback active, `gemini-3.8-flash` out of quota): formula, figure, table, Ukrainian and the weakest-ranked question all correct with a citation of the expected page; both unanswerable questions declined (the GPT-4 one by the LLM). One answer first failed because no model started within 20 s, and was correct on retry.
- The first run had five labels that were too narrow (other pages also state the answer: BERT p.8 for the parameter counts, Attention p.9 for heads and key size, BERT p.16 for the masking rates); they were corrected from the PDF text, which raised hit@1 from 76.9% to 92.3%.

## Orchestrator comparison (phase 7)

2026-10-15, both modes on the same documents, `/chat` pinned to `gemini-3.5-flash-lite` for both runs (a temporary compose override; `gemini-3.8-flash` was hanging and the fallback would have mixed models and 20-second waits into the timings). Answers are scored by word and citation checks, not by reading them, and the model is not deterministic, so a difference of one question is noise.

`questions.jsonl` (26 clearly worded answerable questions and 4 unanswerable):

| | simple | graph |
|---|---|---|
| answerable correct | 25/26 | 25/26 |
| relevant page among the sources / first | 26/26, 24/26 | 26/26, 24/26 |
| unanswerable declined | 4/4 | 4/4 |
| questions rewritten | 0 | only the 4 unanswerable ones (up to 4 extra requests) |

The one miss in both modes is `att-params-big`: a BERT page about model size scores 0.81 and takes first place, and the model declines to read the number out of the Attention paper's Table 3 chunk (0.32). The first search is "good", so graph mode does not rewrite.

`questions_vague.jsonl` (8 answerable questions phrased loosely, e.g. "how do they stop the decoder from cheating by looking at future words?"; `--questions eval/questions_vague.jsonl`):

| | simple | graph |
|---|---|---|
| first search (the same `/search` for both) finds the right page in the top 3 | 7/8 | 7/8 |
| median score of the first relevant result of that search | 0.020 | 0.020 |
| answerable correct | 2/8 | 5/8 |
| relevant page among the sources | 2/8 | 6/8 |
| questions rewritten (extra Gemini requests) | 0 | 7 |

- Loosely worded questions are the case graph mode exists for. The right page is usually found, but the reranker scores it 0.005-0.03, under `CHAT_MIN_SCORE`, so simple mode declines the question without asking Gemini. Graph mode rewrote seven of the eight and recovered three of the six that simple mode lost.
- The cutoff cannot simply be lowered: these scores (0.005-0.026) overlap those of unanswerable questions (0.002-0.004, and 0.070 for the GPT-4 question). A rewrite moves the score into a clearly separable range (0.8 for the masking example); a lower cutoff only lets noise through.
- The three questions graph mode still loses: the rewrite guessed generic machine-learning vocabulary (`vague-lost-signal`: "vanishing gradient … residual connections" still scored under the cutoff), drifted to another meaning (`vague-both-directions`: "reading speed comprehension"), or retrieval succeeded and the model still declined (`vague-hiding-words`: three relevant sources). The rewriter has never seen the documents; giving it their titles or headings is the obvious next step.
- Cost: a good question costs one request in either mode; a rewritten one costs two. On the clear set graph mode changes nothing except spending requests on questions the documents cannot answer, so `simple` stays the default.
- Timings are not comparable between the runs (Gemini's speed varied during the day); the graph's own work, two searches and a grade, is under a second.

## Verification

- `uv run pytest tests/test_eval.py` - question-file validity, relevance, metrics, the answer checks and the HTTP calls against a mock server
- `docker compose up -d`, then `uv run python eval/run_eval.py` (retrieval only)
- `uv run python eval/run_eval.py --answers --ids att-heads,none-gpt4-params` (answers for two questions)
- `uv run python eval/run_eval.py --answers --mode graph` (the whole set through the LangGraph flow)

## Child DOX Index

None.
