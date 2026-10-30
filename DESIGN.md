# DESIGN.md

Decision log for the RAG engine. Each entry: decision, alternatives rejected, errors hit, numbers.

## Milestone 1: Eval harness + naive BM25 baseline

### Decisions
- **Build tool: Maven, single module.** Rejected: Gradle (no benefit here), multi-module (premature split).
- **Java 21, JUnit 5, plugin versions pinned** for reproducible builds.
- **Package layout by component:** eval, m25 now; hnsw, usion, wal later.

### Errors hit
- Git author email was set to a personal address; replaced with the GitHub noreply address before pushing, so the public history never exposed it.

### Benchmark numbers
(none yet)
