import math
from typing import List, Dict, Any


# ---------- Metrics ----------

def hit_at_k(golden, retrieved, k):
    return 1.0 if any(cid in golden for cid in retrieved[:k]) else 0.0


def precision_at_k(golden, retrieved, k):
    if k == 0:
        return 0.0
    retrieved_k = retrieved[:k]
    relevant = sum(1 for cid in retrieved_k if cid in golden)
    return relevant / k


def recall_at_k(golden, retrieved, k):
    if not golden:
        return 0.0
    retrieved_k = retrieved[:k]
    relevant = sum(1 for cid in retrieved_k if cid in golden)
    return relevant / len(golden)


def mrr(golden, retrieved):
    for i, cid in enumerate(retrieved):
        if cid in golden:
            return 1.0 / (i + 1)
    return 0.0


def dcg_at_k(golden, retrieved, k):
    score = 0.0
    for i, cid in enumerate(retrieved[:k]):
        if cid in golden:
            score += 1.0 / math.log2(i + 2)
    return score


def ndcg_at_k(golden, retrieved, k):
    dcg = dcg_at_k(golden, retrieved, k)
    ideal_hits = min(len(golden), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


# ---------- Main evaluator ----------

def evaluate_dataset(
    data: List[Dict[str, Any]],
    k: int = 3,
    return_failures: bool = True
) -> Dict[str, Any]:

    hit_scores, precision_scores, recall_scores = [], [], []
    mrr_scores, ndcg_scores = [], []

    failures = []

    for idx, row in enumerate[Dict[str, Any]](data):
        golden = row.get("golden_chunk_ids", [])
        retrieved = row.get("retrieved_chunk_ids", [])

        q = row.get("question", "")
        qid = row.get("question_id", idx)

        h = hit_at_k(golden, retrieved, k)
        p = precision_at_k(golden, retrieved, k)
        r = recall_at_k(golden, retrieved, k)
        m = mrr(golden, retrieved)
        n = ndcg_at_k(golden, retrieved, k)

        hit_scores.append(h)
        precision_scores.append(p)
        recall_scores.append(r) 
        mrr_scores.append(m)
        ndcg_scores.append(n)

        # Define "failure"
        if h == 0:  # you can tweak this condition
            failures.append({
                "question_id": qid,
                "question": q,
                "golden_chunk_ids": golden,
                "retrieved_chunk_ids": retrieved,
                "metrics": {
                    "hit": h,
                    "precision": p,
                    "recall": r,
                    "mrr": m,
                    "ndcg": n
                }
            })

    def avg(lst):
        return sum(lst) / len(lst) if lst else 0.0

    result = {
        f"hit@{k}": avg(hit_scores),
        f"precision@{k}": avg(precision_scores),
        f"recall@{k}": avg(recall_scores),
        "mrr": avg(mrr_scores),
        f"ndcg@{k}": avg(ndcg_scores),
    }

    if return_failures:
        result["failures"] = failures
        result["num_failures"] = len(failures)

    return result
