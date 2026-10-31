# DESIGN.md

Decision log for the RAG engine. Each entry: decision, alternatives rejected, errors hit, numbers.

## Milestone 1: Eval harness + naive BM25 baseline

### Decisions
- **Build tool: Maven, single module.** Rejected: Gradle (no benefit here), multi-module (premature split).
- **Java 21, JUnit 5, plugin versions pinned** for reproducible builds.
- **Package layout by component:** model, eval, bm25, retrieval now; hnsw, fusion, wal later.
- **Dataset: BEIR SciFact, test split.** Download is a script (scripts/download-scifact.ps1); data/ is gitignored. Loader keeps only queries that have judgments in the chosen split (queries.jsonl has 1109, the test split judges 300).
- **Jackson for JSON parsing only.** Parsing is not the retrieval layer, so it does not conflict with the from-scratch rule.
- **Recall@k = BEIR definition** (fraction of a query's relevant docs found in the top k), not "at least one hit". Rejected the hit-rate definition because the BEIR one is comparable with published numbers.
- **nDCG@k with linear gain** (gain = relevance score), as in BEIR's evaluator. **MRR@10** with a cutoff. All three tested against hand-computed values, including duplicates in a ranking counting once.
- **Queries that return nothing count as 0**, never skipped, so failures cannot inflate averages.
- **Retriever interface** (name, index, search). Every engine implements it, so the runner scores any of them unchanged. Scores are comparable only within one retriever, which is why fusion will use ranks (RRF) and not raw scores.
- **Naive BM25:** no inverted index, scores all docs per query, top-k via a min-heap. Lucene-style IDF ln(1 + (N - df + 0.5) / (df + 0.5)), k1 = 1.2, b = 0.75. Title and text indexed together.
- **Naive tokenizer:** lowercase, split on non-letter/digit. No stopwords, no stemming. Rejected for now, to be measured later as separate experiments against this baseline.

### Errors hit
- Git author email was set to a personal address; replaced with the GitHub noreply address before the first push, so public history never exposed it.

### Benchmark numbers (SciFact test, 300 queries, k=10)
| Retriever | Recall@10 | MRR@10 | nDCG@10 | index | p50 | p95 |
|---|---|---|---|---|---|---|
| bm25-naive | 0.7876 | 0.6309 | 0.6634 | ~300 ms | ~3.4-4.0 ms | ~6-7 ms |

- Published BM25 on SciFact is about 0.665 nDCG@10 (BEIR paper, quoted from memory; verify before citing). We are within about 0.002.
- Latency is a single run in one JVM, so it includes JIT warm-up noise. JMH will give precise numbers in the HNSW milestone.

### Reproduce
    powershell -ExecutionPolicy Bypass -File scripts\download-scifact.ps1
    mvn -q compile exec:java
