"""Measure how well a running doc-rag API finds and answers the questions in questions.jsonl.

    uv run python eval/run_eval.py                  retrieval only: no Gemini requests
    uv run python eval/run_eval.py --answers        also ask /chat and check the answers
    uv run python eval/run_eval.py --ids att-heads  only these questions (comma-separated)

Retrieval is scored with POST /search: a result is relevant when it comes from the
expected document and its page range covers an expected page. Answers are scored with
POST /chat; every question that reaches the LLM spends one Gemini request.
"""

import argparse
import json
import sys
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import httpx

QUESTIONS_PATH = Path(__file__).parent / "questions.jsonl"
RESULTS_DIR = Path(__file__).parent / "results"
KINDS = ("text", "table", "figure", "formula", "none")
HIT_KS = (1, 3, 6)
# Must equal CHAT_MIN_SCORE in src/doc_rag/config.py (tests/test_eval.py checks it). This
# script only talks HTTP, so it cannot read the server's settings.
DEFAULT_MIN_SCORE = 0.05
REQUEST_TIMEOUT_SECONDS = 200.0


@dataclass(frozen=True)
class Question:
    id: str
    question: str
    # What holds the answer: a text passage, a table, a figure or a formula; "none" for a
    # question the documents cannot answer.
    kind: str
    # Where the answer is. None, with no pages, for a question the documents cannot answer.
    document: str | None = None
    pages: tuple[int, ...] = ()
    language: str = "en"
    # Words the answer must contain (case-insensitive). "a|b" accepts either spelling.
    answer_contains: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"{self.id}: kind must be one of {KINDS}, not {self.kind!r}")
        if (self.document is None) != (self.kind == "none") or bool(self.pages) != (
            self.document is not None
        ):
            raise ValueError(f"{self.id}: only kind 'none' has no document, and then no pages")

    @property
    def answerable(self) -> bool:
        return self.document is not None


@dataclass
class Retrieval:
    question: Question
    results: list[dict]
    seconds: float
    # Position (1 is best) of the first relevant result; None when there is none.
    rank: int | None

    @property
    def top_score(self) -> float:
        return self.results[0]["score"] if self.results else 0.0

    @property
    def hit_score(self) -> float | None:
        """Reranker score of the first relevant result."""
        return None if self.rank is None else self.results[self.rank - 1]["score"]


@dataclass
class Answer:
    question: Question
    text: str
    citations: list[int]
    sources: list[dict]
    error: str | None
    seconds: float
    # What graph mode did before answering (one entry per rewritten search); empty otherwise.
    steps: list[str] = field(default_factory=list)

    @property
    def source_rank(self) -> int | None:
        """Position among the sources of the first relevant one; None if there is none."""
        return first_hit_rank(self.sources, self.question)

    @property
    def cites_expected(self) -> bool:
        return any(
            is_relevant(source, self.question)
            for source in self.sources
            if source["number"] in self.citations
        )

    @property
    def has_keywords(self) -> bool:
        return contains_all(self.text, self.question.answer_contains)

    @property
    def correct(self) -> bool:
        """An answerable question needs the facts and a citation of the right page; an
        unanswerable one needs to be declined, which the pipeline does by citing nothing."""
        if self.error:
            return False
        if not self.question.answerable:
            return not self.citations
        return self.has_keywords and self.cites_expected


def load_questions(path: Path = QUESTIONS_PATH) -> list[Question]:
    questions: list[Question] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
            question = Question(
                **{
                    **raw,
                    "pages": tuple(raw.get("pages", ())),
                    "answer_contains": tuple(raw.get("answer_contains", ())),
                }
            )
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{path.name}, line {number}: {exc}") from exc
        if any(question.id == other.id for other in questions):
            raise ValueError(f"{path.name}, line {number}: duplicate id {question.id!r}")
        questions.append(question)
    return questions


def is_relevant(result: dict, question: Question) -> bool:
    """A result counts when it is from the expected document and covers an expected page."""
    first = result["page_start"]
    if result["filename"] != question.document or first is None:
        return False
    last = result["page_end"] or first
    return any(first <= page <= last for page in question.pages)


def first_hit_rank(results: Sequence[dict], question: Question) -> int | None:
    return next(
        (rank for rank, result in enumerate(results, start=1) if is_relevant(result, question)),
        None,
    )


def retrieval_metrics(ranks: Sequence[int | None]) -> dict[str, float]:
    """hit@k: share of questions with a relevant result in the top k. MRR: mean of 1/rank."""
    return {
        **{f"hit@{k}": sum(r is not None and r <= k for r in ranks) / len(ranks) for k in HIT_KS},
        "MRR": sum(1 / r for r in ranks if r is not None) / len(ranks),
    }


def contains_all(answer: str, needles: Iterable[str]) -> bool:
    text = answer.lower()
    return all(
        any(option.strip().lower() in text for option in needle.split("|")) for needle in needles
    )


def parse_events(lines: Iterable[str]) -> list[tuple[str, dict]]:
    """Server-Sent Events: `event:` and `data:` lines, each event ended by an empty line."""
    events: list[tuple[str, dict]] = []
    name, data = "", []
    for line in lines:
        if line.startswith("event:"):
            name = line.removeprefix("event:").strip()
        elif line.startswith("data:"):
            data.append(line.removeprefix("data:").strip())
        elif not line and data:
            events.append((name, json.loads("\n".join(data))))
            name, data = "", []
    return events


def search(http: httpx.Client, question: Question, top_k: int) -> Retrieval:
    started = time.perf_counter()
    response = http.post("/search", json={"query": question.question, "top_k": top_k})
    response.raise_for_status()
    results = response.json()["results"]
    return Retrieval(
        question=question,
        results=results,
        seconds=time.perf_counter() - started,
        rank=first_hit_rank(results, question) if question.answerable else None,
    )


def ask(http: httpx.Client, question: Question, mode: str = "simple") -> Answer:
    started = time.perf_counter()
    response = http.post("/chat", json={"question": question.question, "mode": mode})
    response.raise_for_status()
    all_events = parse_events(response.text.splitlines())
    # Later events of the same name replace earlier ones; only `sources` and the last
    # `done` or `error` are needed, so the individual tokens are dropped.
    events = dict(all_events)
    done = events.get("done", {})
    return Answer(
        question=question,
        text=done.get("answer", ""),
        citations=done.get("citations", []),
        sources=events.get("sources", {}).get("sources", []),
        error=events.get("error", {}).get("message"),
        seconds=time.perf_counter() - started,
        steps=[data["message"] for name, data in all_events if name == "step"],
    )


def missing_documents(http: httpx.Client, questions: Iterable[Question]) -> list[str]:
    ready = {
        document["filename"]
        for document in http.get("/documents").raise_for_status().json()
        if document["status"] == "ready"
    }
    return sorted({q.document for q in questions if q.document and q.document not in ready})


def page_range(result: dict) -> str:
    first, last = result["page_start"], result["page_end"]
    return "?" if first is None else str(first) if last in (None, first) else f"{first}-{last}"


def percent(value: float) -> str:
    return f"{value:6.1%}"


def print_metrics(title: str, retrievals: list[Retrieval]) -> None:
    metrics = retrieval_metrics([r.rank for r in retrievals])
    print(
        f"  {title:<10} n={len(retrievals):<3}"
        + "".join(f"  {name} {percent(value)}" for name, value in metrics.items() if name != "MRR")
        + f"  MRR {metrics['MRR']:.3f}"
    )


def print_retrieval_report(retrievals: list[Retrieval], top_k: int, min_score: float) -> None:
    answerable = [r for r in retrievals if r.question.answerable]
    print(f"\nRetrieval: {len(answerable)} answerable questions, top_k={top_k}")
    if answerable:
        print_metrics("all", answerable)
        for kind in KINDS[:-1]:
            if group := [r for r in answerable if r.question.kind == kind]:
                print_metrics(kind, group)
        for language in sorted({r.question.language for r in answerable}):
            if group := [r for r in answerable if r.question.language == language]:
                print_metrics(language, group)
        scores = sorted(r.hit_score for r in answerable if r.hit_score is not None)
        if scores:
            print(
                f"  score of the first relevant result: lowest {scores[0]:.3f}, "
                f"median {scores[len(scores) // 2]:.3f}"
            )
        times = sorted(r.seconds for r in retrievals)
        print(f"  search time: median {times[len(times) // 2]:.2f}s, slowest {times[-1]:.2f}s")

    print("\n  Questions without a relevant result in the top results:")
    misses = [r for r in answerable if r.rank is None]
    for r in misses:
        found = ", ".join(
            f"{x['filename']} p.{page_range(x)} ({x['score']:.2f})" for x in r.results[:3]
        )
        print(
            f"    {r.question.id}: expected {r.question.document} p.{r.question.pages}; got {found}"
        )
    if not misses:
        print("    none")
    print("\n  Found, but not first (what ranked above the right page):")
    late = [(r, r.rank) for r in answerable if r.rank is not None and r.rank > 1]
    for r, rank in late:
        above = ", ".join(
            f"{x['filename']} p.{page_range(x)} ({x['score']:.2f})" for x in r.results[: rank - 1]
        )
        print(f"    {r.question.id}: #{rank}, above it: {above}")
    if not late:
        print("    none")

    unanswerable = [r for r in retrievals if not r.question.answerable]
    if unanswerable:
        print(f"\nUnanswerable questions (declined without the LLM below score {min_score}):")
        for r in unanswerable:
            outcome = "declined" if r.top_score < min_score else "reaches the LLM"
            print(f"    {r.question.id:<24} top score {r.top_score:.3f}  {outcome}")
        highest = max(r.top_score for r in unanswerable)
        print(f"  highest top score of an unanswerable question: {highest:.3f}")


def print_answer_report(answers: list[Answer], mode: str = "simple") -> None:
    answerable = [a for a in answers if a.question.answerable]
    print(f"\nAnswers ({mode} mode): {len(answers)} asked")
    for a in answers:
        if a.error:
            verdict = f"ERROR {a.error}"
        elif a.question.answerable:
            verdict = (
                f"facts {'yes' if a.has_keywords else 'NO'}, "
                f"cites expected page {'yes' if a.cites_expected else 'NO'}, "
                f"citations {a.citations}"
            )
        else:
            verdict = "declined" if not a.citations else f"ANSWERED, citing {a.citations}"
        rewrites = f"  [rewrote {len(a.steps)}x]" if a.steps else ""
        print(f"    {a.question.id:<24} {a.seconds:5.1f}s  {verdict}{rewrites}")
    if answerable:
        print(
            f"  answerable correct: {sum(a.correct for a in answerable)}/{len(answerable)} "
            f"(facts present: {sum(a.has_keywords for a in answerable)}, "
            f"cites the expected page: {sum(a.cites_expected for a in answerable)})"
        )
    if answerable:
        ranks = [a.source_rank for a in answerable]
        print(
            f"  relevant page among the sources: {sum(r is not None for r in ranks)}/{len(ranks)}, "
            f"first source: {sum(r == 1 for r in ranks)}/{len(ranks)}"
        )
    unanswerable = [a for a in answers if not a.question.answerable]
    if unanswerable:
        print(
            f"  unanswerable declined: {sum(a.correct for a in unanswerable)}/{len(unanswerable)}"
        )
    rewrites = sum(len(a.steps) for a in answers)
    print(f"  questions rewritten: {sum(bool(a.steps) for a in answers)} ({rewrites} rewrites)")
    print(f"  answer time: median {sorted(a.seconds for a in answers)[len(answers) // 2]:.1f}s")


def write_answers(answers: list[Answer], directory: Path, mode: str = "simple") -> Path:
    """A Markdown file to read the answers yourself; the checks above only look for words."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"answers-{mode}-{datetime.now():%Y%m%d-%H%M%S}.md"
    lines = [f"# Evaluation answers ({mode} mode)", ""]
    for a in answers:
        q = a.question
        expected = (
            f"{q.document} p.{list(q.pages)}, must mention {list(q.answer_contains)}"
            if q.document
            else "not in the documents"
        )
        lines += [
            f"## {q.id}",
            "",
            f"**Question:** {q.question}",
            "",
            f"**Expected:** {expected}",
            "",
        ]
        lines += [f"**Searched again:** {step}" for step in a.steps]
        lines += [f"**Error:** {a.error}" if a.error else f"**Answer:** {a.text}", ""]
        for source in a.sources:
            marks = ("cited " if source["number"] in a.citations else "") + (
                "relevant" if is_relevant(source, q) else ""
            )
            lines.append(
                f"- [{source['number']}] {source['filename']} p.{page_range(source)} "
                f"score {source['score']:.2f} {marks}".rstrip()
            )
        lines += [
            "",
            f"**Verdict:** {'correct' if a.correct else 'check this one'} ({a.seconds:.1f}s)",
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--top-k", type=int, default=10, help="results requested per search")
    parser.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE)
    parser.add_argument(
        "--answers", action="store_true", help="also ask /chat (spends Gemini requests)"
    )
    parser.add_argument(
        "--mode", choices=("simple", "graph"), default="simple", help="orchestrator for /chat"
    )
    parser.add_argument(
        "--questions", type=Path, default=QUESTIONS_PATH, help="JSON Lines file of questions"
    )
    parser.add_argument("--ids", help="comma-separated question ids to run")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR, help="folder for the answers file")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    args = parse_args(argv)
    questions = load_questions(args.questions)
    if args.ids:
        wanted = set(args.ids.split(","))
        if unknown := wanted - {q.id for q in questions}:
            print(f"Unknown question ids: {', '.join(sorted(unknown))}")
            return 2
        questions = [q for q in questions if q.id in wanted]

    timeout = httpx.Timeout(REQUEST_TIMEOUT_SECONDS, connect=5.0)
    try:
        with httpx.Client(base_url=args.api_url, timeout=timeout) as http:
            if missing := missing_documents(http, questions):
                print(f"Upload these documents first (see eval/AGENTS.md): {', '.join(missing)}")
                return 1
            retrievals = [search(http, q, args.top_k) for q in questions]
            print_retrieval_report(retrievals, args.top_k, args.min_score)
            if not args.answers:
                return 0
            reach_llm = sum(r.top_score >= args.min_score for r in retrievals)
            print(
                f"\nAsking /chat: {reach_llm} of {len(questions)} questions reach the LLM "
                "(one Gemini request each"
                + ("; graph mode adds one per rewrite)" if args.mode == "graph" else ")")
            )
            answers = [ask(http, q, args.mode) for q in questions]
    except httpx.HTTPError as exc:
        print(f"Could not talk to the API at {args.api_url}: {exc!r}")
        return 1
    print_answer_report(answers, args.mode)
    print(f"\nAnswers written to {write_answers(answers, args.out, args.mode)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
