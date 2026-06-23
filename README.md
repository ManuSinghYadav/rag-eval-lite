### Structure

Three dataclasses + one evaluator class:

- **`QueryResult`** — your input. Pass `query_id`, `retrieved_chunks` (ordered list), `relevant_chunks` (set)
- **`QueryScore`** — per-query scores + a `failures` list with human-readable diagnostics
- **`EvalReport`** — mean scores across all queries + `failed_queries` list
- **`RAGEvaluator`** — the main class, just call `.evaluate()` then `.report()`

---

### The Failure Diagnostics Logic

Each query gets diagnosed in priority order:

| Failure | Meaning | Suggestion printed |
|---------|---------|-------------------|
| `MISS` | No relevant chunk in top-k at all | Shows what was retrieved vs what was expected |
| `LOW MRR` | First relevant chunk ranked too low | Shows exact rank, suggests re-ranking |
| `INCOMPLETE RECALL` | Some relevant chunks never retrieved | Shows exactly which chunk IDs were missed |
| `LOW PRECISION` | Too many irrelevant chunks in results | Shows the noisy chunk IDs |
| `LOW NDCG` | Relevant chunks not near the top | Suggests re-ranking |

---

### Usage in your package

```python
from rag_eval import RAGEvaluator

dataset = [
    {
        "question": "What is the significance of Apple's manufacturing...",
        "relevent_chunks": ["pdf_chunk_94", "pdf_chunk_17", "pdf_chunk_107"],
        "retrieved_chunks": ["pdf_chunk_94", "pdf_chunk_91", "pdf_chunk_95"]
    },
    ...
]

report = RAGEvaluator.from_dict_list(dataset, k=5)
RAGEvaluator(k=5).report(report)
```

You can also access `report.per_query` and `report.failed_queries` programmatically if you want to log them to MLflow or W&B Weave.