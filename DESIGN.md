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

## Milestone 2: Inverted index + disk persistence

### Decisions
- **Inverted index:** term -> posting list of (docIndex, tf), plus doc lengths and average length. Scoring is a separate class (IndexedBm25Retriever), term-at-a-time into a per-document score array, top-k via min-heap. Same BM25 math as the naive version, so it had to reproduce 0.6634 exactly (it did).
- **docIndex is an int position, not the String doc id.** Smaller in memory and on disk. Ids live once in docs.bin.
- **Disk format v2, three files:** docs.bin (analyzer name, doc ids, lengths), terms.bin (sorted dictionary: term, docFreq, byte offset), postings.bin (delta-encoded varints). Dictionary and postings are split so postings can be memory-mapped later without loading the dictionary.
- **Every file has magic + version header and a CRC32 footer.** The reader verifies size, CRC, magic and version before trusting anything. Corruption, truncation and a missing file all fail loudly (tested). Rejected: no checksum (silent wrong results after a bad write).
- **Delta + varint postings.** Rejected: fixed 4-byte ints (3.7x bigger, see numbers).
- **Analyzer is pluggable and stored inside the index.** Index and query must use the same analyzer or recall silently drops, so the loader throws if the names differ. This forced format v1 -> v2; old files are rejected by the version check.
- **Own Porter stemmer** (~150 lines, rule tables for steps 2-4). Rejected: Lucene's stemmer, because Lucene is baseline-only. Stopword list = Lucene EnglishAnalyzer's 33 words, removed before stemming, so the comparison with lucene-english is fair.
- **Lucene as baseline only** (LuceneBm25Retriever, two analyzers). Nothing in the engine depends on it. Queries are built from analyzed tokens, not parsed, so ":" or "-" in a question cannot break them.

### Errors hit
- java -cp target\classes failed with NoClassDefFoundError (Jackson): target\classes has only our code. Fixed with mvn dependency:build-classpath into cp.txt (gitignored).
- RunLuceneCompare.java was never saved, so Maven compiled without it and the class was not found. Checked with Get-ChildItem -Recurse -Filter.
- RunPersist crashed with 'Unresolved compilation problem' although the source was fine: VS Code's Java compiler had written a broken stale class into target\classes. Fixed with mvn clean compile. Lesson: when a runtime error contradicts the source, clean-build before debugging code.

### Benchmark numbers (SciFact test, 300 queries, k=10)
| Retriever | Recall@10 | MRR@10 | nDCG@10 | p50 | p95 |
|---|---|---|---|---|---|
| bm25-naive | 0.7876 | 0.6309 | 0.6634 | 4.523 ms | 8.381 ms |
| bm25-indexed-simple | 0.7876 | 0.6309 | 0.6634 | 0.351 ms | 0.859 ms |
| bm25-indexed-stop | 0.7823 | 0.6361 | 0.6664 | 0.293 ms | 1.191 ms |
| bm25-indexed-stop-stem | 0.8168 | 0.6455 | 0.6827 | 0.241 ms | 0.567 ms |
| lucene-standard | 0.7843 | 0.6283 | 0.6606 | 1.035 ms | 2.975 ms |
| lucene-english | 0.8173 | 0.6464 | 0.6826 | 0.378 ms | 0.806 ms |

- Indexed vs naive: identical quality, about 13x faster at p50 and 10x at p95.
- Index vs Lucene: with comparable analysis we match it (simple vs lucene-standard: +0.3 points; stop-stem vs lucene-english: within 0.0001 nDCG).
- Stopwords alone: +0.3 nDCG points. Stemming is what closes the gap to lucene-english (+1.6 more).
- Latency is single-run, one JVM, JIT warm-up included, and Lucene ran later in a cold JVM. Treat all latency numbers as indicative only; precise numbers need JMH.

### Persistence numbers
- Restart (simple analyzer, format v2, rerun after clean rebuild): build 295 ms, save 193 ms, load cold 93 ms, load warm 48 ms (earlier v1 run: 328 / 162 / 93 / 28). Loaded index scored exactly 0.6634 in both.
- Size (stop-stem index): postings 513,401 entries = 4,107,208 bytes fixed-width vs 1,110,326 bytes varint+delta (2.16 bytes/posting, 3.7x smaller, 73% saved). terms.bin 540,247 bytes, docs.bin 70,705 bytes, about 1.72 MB total.
- Dictionary is about a third of the index (26,357 terms, each a full string + 12 bytes). Front-coding the sorted terms is a known future saving, not done.

### Known limits (honest)
- Index files are written in place: a crash mid-write leaves a broken index. The CRC detects it, but safe replacement and recovery are Milestone 6 (WAL).
- Loading reads whole files onto the heap. Memory-mapping postings.bin is not done; it needs a measurement before a claim.
- The index is immutable: no add or delete yet (Milestone 6).
- Our tokenizer splits "alzheimer's" into "alzheimer" + "s"; Lucene strips possessives. Likely the remaining 0.0005 recall difference. Not fixed.

### Reproduce
    mvn -q compile
    mvn -q dependency:build-classpath "-Dmdep.outputFile=cp.txt"
    $cp = "target\classes;" + (Get-Content cp.txt -Raw).Trim()
    java -cp $cp io.github.abdullah.rag.eval.RunCompare
    java -cp $cp io.github.abdullah.rag.eval.RunPersist
    java -cp $cp io.github.abdullah.rag.eval.RunLuceneCompare
    java -cp $cp io.github.abdullah.rag.eval.RunAnalyzers
    java -cp $cp io.github.abdullah.rag.eval.RunIndexSize
