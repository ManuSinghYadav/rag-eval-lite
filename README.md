# rag-eval-lite

Lightweight retrieval evaluation metrics for RAG systems.

---

## Install

```bash
pip install rag-eval-lite
```

---

## Usage

```python
from rag_eval import evaluate_dataset

results = evaluate_dataset(data, k=3)
print(results)
```

---

## Input Format

Each item in `data` must be:

```json
{
  "question": "string",
  "golden_chunk_ids": ["chunk_1"],
  "retrieved_chunk_ids": ["chunk_1", "chunk_3", "chunk_7"]
}
```

- `golden_chunk_ids`: ground truth relevant chunks  
- `retrieved_chunk_ids`: retriever output (ordered by rank)

---

## Output

```python
{
  "hit@k": float,
  "precision@k": float,
  "recall@k": float,
  "mrr": float,
  "ndcg@k": float,
  "num_failures": int,
  "failures": [...]
}
```

---

## Notes

- Order of retrieved chunks matters  
- Supports single and multi-chunk evaluation  
- Failures include full debug info