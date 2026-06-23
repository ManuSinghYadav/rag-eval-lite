"""
rag_eval.py — Retrieval metrics evaluator for RAG pipelines.

Metrics supported:
  - Hit Rate
  - MRR (Mean Reciprocal Rank)
  - Precision@k
  - Recall@k
  - NDCG@k

Usage:
    evaluator = RAGEvaluator(k=5)
    results = evaluator.evaluate(queries_results)
    evaluator.report(results)
"""

import math
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class QueryResult:
    """
    A single query's retrieval result.

    Attributes:
        query_id:          Any identifier — string, int, whatever you prefer.
        retrieved_chunks:  Ordered list of chunk IDs returned by your retriever.
        relevant_chunks:   Set of chunk IDs that are relevant (from golden dataset).
    """
    query_id: str
    retrieved_chunks: list
    relevant_chunks: set


@dataclass
class QueryScore:
    """Per-query scores across all metrics, plus failure diagnostics."""
    query_id: str
    hit_rate: float
    mrr: float
    precision: float
    recall: float
    ndcg: float
    failures: list = field(default_factory=list)  # list of failure reason strings


@dataclass
class EvalReport:
    """Aggregated report across all queries."""
    k: int
    mean_hit_rate: float
    mean_mrr: float
    mean_precision: float
    mean_recall: float
    mean_ndcg: float
    per_query: list  # list of QueryScore
    failed_queries: list  # list of QueryScore where at least one metric is 0


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------

def _hit_rate(retrieved: list, relevant: set, k: int) -> float:
    return 1.0 if any(c in relevant for c in retrieved[:k]) else 0.0


def _reciprocal_rank(retrieved: list, relevant: set, k: int) -> float:
    for rank, chunk in enumerate(retrieved[:k], start=1):
        if chunk in relevant:
            return 1.0 / rank
    return 0.0


def _precision(retrieved: list, relevant: set, k: int) -> float:
    hits = sum(1 for c in retrieved[:k] if c in relevant)
    return hits / k


def _recall(retrieved: list, relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for c in retrieved[:k] if c in relevant)
    return hits / len(relevant)


def _dcg(retrieved: list, relevant: set, k: int) -> float:
    score = 0.0
    for rank, chunk in enumerate(retrieved[:k], start=1):
        if chunk in relevant:
            score += 1.0 / math.log2(rank + 1)
    return score


def _idcg(relevant: set, k: int) -> float:
    ideal_hits = min(len(relevant), k)
    return sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))


def _ndcg(retrieved: list, relevant: set, k: int) -> float:
    idcg = _idcg(relevant, k)
    if idcg == 0:
        return 0.0
    return _dcg(retrieved, relevant, k) / idcg


# ---------------------------------------------------------------------------
# Failure diagnostics
# ---------------------------------------------------------------------------

def _diagnose(retrieved: list, relevant: set, k: int, scores: dict) -> list:
    """
    Returns a list of human-readable failure reasons for a single query.
    """
    failures = []

    if scores["hit_rate"] == 0.0:
        failures.append(
            f"MISS — None of the {len(relevant)} relevant chunk(s) appeared in top-{k}. "
            f"Retrieved: {retrieved[:k]}, Expected any of: {relevant}"
        )
        return failures  # all other metrics will also be 0, no need to go further

    if scores["mrr"] < 0.5:
        first_hit_rank = next(
            (rank for rank, c in enumerate(retrieved[:k], 1) if c in relevant), None
        )
        failures.append(
            f"LOW MRR ({scores['mrr']:.2f}) — First relevant chunk found at rank {first_hit_rank}. "
            f"Consider re-ranking to push it higher."
        )

    if scores["recall"] < 1.0:
        found = {c for c in retrieved[:k] if c in relevant}
        missed = relevant - found
        failures.append(
            f"INCOMPLETE RECALL ({scores['recall']:.2f}) — "
            f"Missed {len(missed)} relevant chunk(s): {missed}"
        )

    if scores["precision"] < 0.5:
        noise = [c for c in retrieved[:k] if c not in relevant]
        failures.append(
            f"LOW PRECISION ({scores['precision']:.2f}) — "
            f"{len(noise)} irrelevant chunk(s) in top-{k}: {noise}"
        )

    if scores["ndcg"] < 0.5:
        failures.append(
            f"LOW NDCG ({scores['ndcg']:.2f}) — Relevant chunks are not ranked near the top. "
            f"Re-ranking should improve this."
        )

    return failures


# ---------------------------------------------------------------------------
# Main evaluator class
# ---------------------------------------------------------------------------

class RAGEvaluator:
    """
    Evaluates retrieval quality for a RAG pipeline across five metrics.

    Args:
        k: Number of top chunks to consider (default: 5).

    Example:
        evaluator = RAGEvaluator(k=5)

        queries_results = [
            QueryResult(
                query_id="q1",
                retrieved_chunks=["A", "C", "B", "D", "E"],
                relevant_chunks={"A", "B"}
            ),
            QueryResult(
                query_id="q2",
                retrieved_chunks=["X", "Y", "Z", "A", "B"],
                relevant_chunks={"A", "B", "F"}
            ),
        ]

        report = evaluator.evaluate(queries_results)
        evaluator.report(report)
    """

    def __init__(self, k: int = 5):
        self.k = k

    def _score_query(self, qr: QueryResult) -> QueryScore:
        retrieved = qr.retrieved_chunks
        relevant = qr.relevant_chunks
        k = self.k

        scores = {
            "hit_rate":  _hit_rate(retrieved, relevant, k),
            "mrr":       _reciprocal_rank(retrieved, relevant, k),
            "precision": _precision(retrieved, relevant, k),
            "recall":    _recall(retrieved, relevant, k),
            "ndcg":      _ndcg(retrieved, relevant, k),
        }

        failures = _diagnose(retrieved, relevant, k, scores)

        return QueryScore(
            query_id=qr.query_id,
            failures=failures,
            **scores,
        )

    def evaluate(self, queries_results: list) -> EvalReport:
        """
        Run evaluation on a list of QueryResult objects.

        Returns an EvalReport with mean scores and per-query breakdowns.
        """
        if not queries_results:
            raise ValueError("queries_results is empty.")

        per_query = [self._score_query(qr) for qr in queries_results]
        n = len(per_query)

        return EvalReport(
            k=self.k,
            mean_hit_rate=sum(q.hit_rate for q in per_query) / n,
            mean_mrr=sum(q.mrr for q in per_query) / n,
            mean_precision=sum(q.precision for q in per_query) / n,
            mean_recall=sum(q.recall for q in per_query) / n,
            mean_ndcg=sum(q.ndcg for q in per_query) / n,
            per_query=per_query,
            failed_queries=[q for q in per_query if q.failures],
        )

    def report(self, eval_report: EvalReport, verbose: bool = True) -> None:
        """
        Prints a formatted report to stdout.

        Args:
            eval_report: Output from evaluate().
            verbose:     If True, prints per-query failure details.
        """
        r = eval_report
        sep = "=" * 55

        print(f"\n{sep}")
        print(f"  RAG RETRIEVAL EVALUATION REPORT  (k={r.k})")
        print(sep)
        print(f"  {'Metric':<20} {'Score':>10}")
        print(f"  {'-'*30}")
        print(f"  {'Hit Rate':<20} {r.mean_hit_rate:>10.4f}")
        print(f"  {'MRR':<20} {r.mean_mrr:>10.4f}")
        print(f"  {'Precision@k':<20} {r.mean_precision:>10.4f}")
        print(f"  {'Recall@k':<20} {r.mean_recall:>10.4f}")
        print(f"  {'NDCG@k':<20} {r.mean_ndcg:>10.4f}")
        print(sep)
        print(f"  Queries evaluated : {len(r.per_query)}")
        print(f"  Queries with issues: {len(r.failed_queries)}")
        print(sep)

        if verbose and r.failed_queries:
            print("\n  FAILED / UNDERPERFORMING QUERIES")
            print(sep)
            for q in r.failed_queries:
                print(f"\n  Query ID : {q.query_id}")
                print(f"  Scores   → Hit:{q.hit_rate:.2f} MRR:{q.mrr:.2f} "
                      f"P:{q.precision:.2f} R:{q.recall:.2f} NDCG:{q.ndcg:.2f}")
                for reason in q.failures:
                    print(f"  ⚠  {reason}")
            print(f"\n{sep}\n")
        else:
            print()

    @classmethod
    def from_dict_list(cls, data: list[dict], k: int = 5) -> "EvalReport":
        """
        Accepts a list of dicts with keys:
        - 'question'         → used as query_id
        - 'relevent_chunks'  → list of relevant chunk IDs
        - 'retrieved_chunks' → list of retrieved chunk IDs

        Example:
            report = RAGEvaluator.from_dict_list(dataset, k=5)
        """
        evaluator = cls(k=k)
        queries_results = [
            QueryResult(
                query_id=item["question"],
                retrieved_chunks=item["retrieved_chunks"],
                relevant_chunks=set(item["relevent_chunks"]),
            )
            for item in data
        ]
        return evaluator.evaluate(queries_results)