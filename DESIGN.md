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

## Milestone 3: Chunking strategies

### Setup
- SciFact abstracts are short (~215 words on average), so chunking barely matters on the original corpus. Built long documents instead: shuffle with seed 42, pack 10 abstracts per long doc (5183 abstracts -> 519 long docs), record each abstract's char span.
- Scoring is at original-abstract level: a retrieved chunk credits every abstract whose span it overlaps, results deduplicated in rank order, then nDCG/Recall/MRR@10 against the original SciFact qrels. A chunk straddling two abstracts therefore costs ranking quality.
- Oracle chunker (cuts exactly at abstract boundaries) scores 0.6827 on test, identical to the M2 whole-abstract score. This validated the span mapping and the scoring bridge, and gives every chunker a ceiling.
- Two packing modes: hard (abstracts joined with a space, no structure) and paragraph (joined with a blank line).
- Tuned on the train split (809 queries), frozen, evaluated once on test (300 queries). Corpus and long-doc layout are the same in both splits; only queries are held out.
- Selection rule, fixed before the test run: per family and mode, highest train nDCG@10 subject to intact rate >= 0.5. The 0.5 threshold is a judgment call.

### Chunkers implemented
- Fixed-size: sliding window of N words with overlap.
- Recursive: paragraph, then line, then sentence, then word boundaries, packed greedily to N words.
- Semantic: TextTiling style. Embed each sentence, compare k sentences before vs after each gap, cut at local similarity minima below mean - alpha*std, with min and max word limits. Uses a hashed term-frequency embedder (lexical, no model) behind an `Embedder` interface, so the ONNX embedder can be swapped in at M4/M5.

### Decisions and alternatives rejected
- Hashed-TF embedder now, neural later: rejected pulling ONNX and a WordPiece tokenizer into a chunking milestone.
- Document-level scoring (any chunk of the right long doc counts): rejected, too lenient, every chunker would score the same.
- Choosing configs by nDCG alone: rejected after the sweep, because fixed-50 and recursive-50 had top nDCG with intact rate 0.000.
- Tuning on test: rejected, used train for tuning.
- Overlap: did not help in the sweep (o0 matched or beat o20 for recursive); not used in the selected configs except the fixed/recursive-50 rows, which only appear as failing-constraint references.

### Final results (test, 300 queries, K=10)
Hard mode (no structure):

| chunker | chunks | nDCG@10 | R@10 | MRR@10 | avg words | purity | boundary | intact |
|---|---|---|---|---|---|---|---|---|
| fixed-50-o10 (fails intact) | 27928 | 0.5916 | 0.7572 | 0.5452 | 49.6 | 0.796 | 0.000 | 0.000 |
| recursive-50-o10 (fails intact) | 29961 | 0.6130 | 0.7506 | 0.5763 | 38.4 | 0.891 | 0.305 | 0.000 |
| semantic k3 min100 max300 a0.0 | 6633 | 0.6085 | 0.7843 | 0.5595 | 167.7 | 0.804 | 0.716 | 0.557 |
| oracle | 5183 | 0.6827 | 0.8168 | 0.6455 | 214.6 | 1.000 | 1.000 | 1.000 |

Paragraph mode:

| chunker | chunks | nDCG@10 | R@10 | MRR@10 | avg words | purity | boundary | intact |
|---|---|---|---|---|---|---|---|---|
| fixed-50-o10 (fails intact) | 27928 | 0.5916 | 0.7572 | 0.5452 | 49.6 | 0.796 | 0.000 | 0.000 |
| recursive-300-o0 | 4951 | 0.6297 | 0.7977 | 0.5801 | 224.7 | 0.804 | 0.782 | 0.884 |
| semantic k3 min100 max300 a0.0 | 6635 | 0.6095 | 0.7832 | 0.5613 | 167.7 | 0.809 | 0.724 | 0.561 |
| oracle | 5183 | 0.6827 | 0.8168 | 0.6455 | 214.6 | 1.000 | 1.000 | 1.000 |

### Significance (paired bootstrap, 10k resamples, per-query nDCG)
- Every chunker loses to the oracle by 0.05 to 0.07 nDCG@10, and every such interval excludes 0.
- On nDCG, chunkers are not separable in hard mode (all intervals include 0). In paragraph mode only recursive-300 over fixed-50 is significant (+0.038, CI +0.011 to +0.064).
- Semantic vs recursive-300 in paragraph mode: -0.020, CI -0.042 to +0.001, not significant.
- Semantic Recall@10 lead over recursive-50 in hard mode (+0.034, p=0.017) is suggestive only: 12 comparisons were run, so it does not survive correction.
- Train-to-test shrinkage: semantic hard mode 0.6438 -> 0.6085; recursive-50 0.6212 -> 0.6130. Part of semantic's train lead was fit to the train queries.

### Title ablation (test, nDCG@10, locked configs)
Re-attaching an abstract's title to chunks that start mid-abstract: fixed-50 +0.058 and recursive-50 +0.054 (CIs exclude 0), recursive-300 +0.010, semantic +0.007 (hard, no detectable difference) and +0.009 (paragraph, borderline). Oracle +0.0000, as expected. Caveat: the title is attached using the known abstract spans, so this is an upper bound for "metadata on every chunk"; a real pipeline has one title per document.

### Conclusions I can defend
1. nDCG on 300 queries cannot rank the chunkers; intact rate and boundary integrity can. nDCG alone would have picked the 50-word configs, which cut every abstract.
2. With structure present, recursive-300 is the best tradeoff (intact 0.884, nDCG not worse than semantic) at the lowest complexity.
3. Without structure, semantic is the only chunker that keeps over half of the abstracts intact, at no detectable nDCG cost.
4. No chunker closes the gap to the oracle.
5. Small chunks need the document title re-attached; larger chunks barely do.

### Limitations
- Long docs are random concatenations, so topic shifts are sharper than in real text. This flatters semantic chunking.
- Hashed-TF is lexical; a neural embedder may change semantic's results (re-run at M4/M5).
- Words, not model tokens, are the size unit.
- A single sentence longer than max words stays one oversized chunk in the semantic chunker.
- No answer-span dataset, so "intact" is a proxy for "answer not cut in half".

### Errors hit
- Guessed the analyzer interface name (`Analyzer`); the real one is `TextAnalyzer`, and it is not a functional interface, so tests use `SimpleAnalyzer`.
- `-Dexec.mainClass` was ignored because the pom hardcodes the main class; runners are launched with `java -cp target\classes;<cp.txt>` instead.
- The first semantic grid capped max words at 200, below the abstract length, which understated its intact rate; widened to 300.