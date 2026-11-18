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
## Milestone 4: HNSW vector index + real embeddings

### Setup
- Data: SciFact test, 5183 docs, 300 judged queries, K=10. Vectors are all-MiniLM-L6-v2 (384-d), documents embedded as title + abstract.
- Embedding: one-off job, 211.2 s for the corpus (40.7 ms/doc), cached to disk. Every benchmark loads the cache.
- Query vectors are precomputed, so reported latency is index search time only. Live query embedding cost was NOT measured separately; add it before quoting an end-to-end latency.
- All indexes are single-threaded scalar Java (no SIMD), JVM warmed up with one pass before timing. Hardware: 12th Gen Intel(R) Core(TM) i5-12450H, 16 GB RAM, Windows 11, Java 21 (Temurin).

### Components implemented
- VectorMath (dot, distance = 1 - dot, normalize), VectorIndex interface, BruteForceIndex (exact, heap top-k), HnswIndex (own implementation), HnswConfig.
- WordPieceTokenizer (BERT uncased, hand-written), OnnxEmbedder (tokenizer + ONNX Runtime + mean pooling + L2 normalise), EmbeddingCache (binary file with model name and dimension in the header), DenseRetriever (any VectorIndex behind the Retriever interface).

### Decisions and alternatives rejected
- Embedding model: all-MiniLM-L6-v2 via ONNX. Rejected bge-small-en: it needs query prefixes and CLS pooling, which are extra variables while validating HNSW (can be added later as a measured comparison). Rejected an API model: costs money and is not reproducible offline.
- Tokenizer: wrote WordPiece by hand (about 150 lines) instead of adding a tokenizer library. Validated against known BERT token IDs (hello=7592, world=2088) and against reference similarities for the model: 0.6660 vs 0.666 and 0.1046 vs 0.105 (reference values recalled from the sentence-transformers docs, not re-fetched).
- Similarity: all vectors are unit length, so cosine = dot product, and distance = 1 - dot. One function is used by every index, so recall comparisons use the same metric.
- Ground truth: BruteForceIndex, tested against a full sort on random data. All ANN recall numbers are measured against it.
- HNSW internals: nodes addressed by int slot, not string id; graph is directed (pruning can leave one-way edges); level drawn from a seeded geometric distribution (test: P(level>=1) about 1/M, P(level>=2) about 1/M^2); neighbour selection uses the paper's diversity heuristic with keepPrunedConnections, and over-full neighbour lists are re-pruned with the same heuristic.
- efSearch is a query-time setter, not part of HnswConfig, so one built graph can be swept over many efSearch values.
- Cache header stores model name and dimension; loadFor() rejects a mismatch. This is the defence for "embedding model changes, old vectors become invalid".
- Documents truncated at 256 tokens (MiniLM training length). Some long abstracts are cut; not quantified.

### Benchmark numbers (SciFact test, 300 queries, k=10)

Dense baseline, exact search (BruteForceIndex over cached MiniLM vectors):

| retriever | Recall@10 | MRR@10 | nDCG@10 | p50 ms | p95 ms |
|---|---|---|---|---|---|
| bm25-naive (M1) | 0.7876 | 0.6309 | 0.6634 | 4.640 | 8.066 |
| dense-brute-force | 0.7883 | 0.6068 | 0.6484 | 1.199 | 1.645 |

The dense nDCG@10 of 0.6484 is in line with the published MiniLM-L6 SciFact figure of roughly 0.645 (recalled from memory, not re-checked), which validates tokenizer, ONNX call, pooling, cache and retriever end to end. Naive BM25 beats exact dense here, which is expected on a dataset full of exact scientific terms and is the motivation for hybrid retrieval in M5.

efSearch sweep (M=16, efC=200, seed 42; ANN recall is measured against BruteForceIndex top-10; saved in results/hnsw-sweep.csv):

| index | ANN recall@10 | nDCG@10 | p50 ms | p95 ms |
|---|---|---|---|---|
| brute-force | 1.0000 | 0.6484 | 1.322 | 2.341 |
| hnsw ef=10 | 0.9240 | 0.6217 | 0.113 | 0.159 |
| hnsw ef=20 | 0.9667 | 0.6422 | 0.162 | 0.230 |
| hnsw ef=40 | 0.9877 | 0.6484 | 0.270 | 0.410 |
| hnsw ef=80 | 0.9980 | 0.6484 | 0.464 | 0.630 |
| hnsw ef=160 | 0.9987 | 0.6484 | 0.772 | 1.025 |
| hnsw ef=320 | 1.0000 | 0.6484 | 1.509 | 2.613 |
| hnsw ef=640 | 1.0000 | 0.6484 | 2.129 | 3.510 |

Build time: 6.2 s for 5183 vectors (brute force: 2 ms). Plot: results/hnsw-recall-vs-latency.svg.

Run-to-run timing noise: an earlier identical run gave identical recall and nDCG (the build is seeded) but different latencies, e.g. hnsw ef=40 p50 0.350 ms and brute force 1.279 ms. So the ef=40 speedup over brute force is 3.7x in one run and 4.9x in the other; quote it as "about 4x".

Build-parameter sweep (single run, seed 42; recall = ANN recall@10 at efSearch 20 / 40 / 80; p50 is at efSearch 40):

| M | efC | build s | recall @20 | recall @40 | recall @80 | p50 ms @40 |
|---|---|---|---|---|---|---|
| 8 | 100 | 2.5 | 0.9147 | 0.9603 | 0.9883 | 0.189 |
| 8 | 200 | 3.4 | 0.9187 | 0.9627 | 0.9893 | 0.170 |
| 8 | 400 | 5.3 | 0.9183 | 0.9620 | 0.9880 | 0.180 |
| 16 | 100 | 4.5 | 0.9653 | 0.9853 | 0.9970 | 0.260 |
| 16 | 200 | 5.8 | 0.9667 | 0.9877 | 0.9980 | 0.280 |
| 16 | 400 | 8.8 | 0.9657 | 0.9873 | 0.9973 | 0.269 |
| 32 | 100 | 9.8 | 0.9857 | 0.9940 | 0.9980 | 0.404 |
| 32 | 200 | 25.9 (outlier) | 0.9873 | 0.9937 | 0.9987 | 0.438 |
| 32 | 400 | 15.3 | 0.9870 | 0.9943 | 0.9987 | 0.408 |

The 25.9 s build (M=32, efC=200) is system noise: efC=400 built faster (15.3 s), which is impossible for a real cost. It is not quoted anywhere as a result.

Diversity heuristic ablation (efC=100, mean over seeds 1, 2, 3; "nearest-M" = link to the M nearest candidates, no diversity check):

| selection | M | build s | recall @20 | recall @40 | recall @80 | p50 ms @20 / @40 / @80 |
|---|---|---|---|---|---|---|
| heuristic | 8 | 2.4 | 0.9153 | 0.9607 | 0.9879 | 0.099 / 0.162 / 0.311 |
| nearest-M | 8 | 1.6 | 0.8468 | 0.9089 | 0.9568 | 0.089 / 0.150 / 0.249 |
| heuristic | 16 | 4.1 | 0.9664 | 0.9858 | 0.9971 | 0.162 / 0.253 / 0.428 |
| nearest-M | 16 | 2.4 | 0.9333 | 0.9723 | 0.9912 | 0.138 / 0.222 / 0.386 |

### Conclusions I can defend
1. HNSW matches exact search on task quality at a fraction of the latency. At efSearch=40: ANN recall@10 0.9877, nDCG@10 identical to brute force (0.6484), p50 about 4x lower (3.7x and 4.9x in two runs). At efSearch=10: about 12x lower p50 for -0.027 nDCG.
2. Past efSearch=80 the extra latency buys almost nothing (recall 0.9980 -> 1.0000). At efSearch>=320, HNSW is no faster than brute force on this corpus.
3. ANN recall and task quality are different measurements. nDCG@10 is flat at 0.6484 from efSearch=40 even though ANN recall is still 0.9877 there: the missed neighbours are low-ranked ones that do not change which judged documents reach the top 10.
4. efConstruction above 100 gave no measurable recall gain at this corpus size (differences at most about 0.004, within noise) while build time grew (M=16: 4.5 s, 5.8 s, 8.8 s for efC 100, 200, 400). The default efC=200 was chosen before measuring and is not supported by this data; efC=100 is enough here.
5. M trades build time for query-time beam width. At matched recall (about 0.986) M=8/efSearch=80, M=16/efSearch=40 and M=32/efSearch=20 land at about 0.25 to 0.30 ms p50, but build time goes 2.5 s, 4.5 s, 9.8 s. M=16 is a reasonable middle, not a clear winner. Memory was not measured.
6. The diversity heuristic beats plain nearest-M neighbour selection: +3.1 to +6.9 recall points at M=8, +0.6 to +3.3 at M=16 (gap shrinks as M or efSearch grows). Cost: build is 1.5x to 1.7x slower, query latency at equal efSearch about 8 to 10% higher (differences of 0.01 to 0.04 ms, close to noise). At matched recall it is roughly 1.5x faster (heuristic M=16/efSearch=40: 0.9858 at 0.253 ms vs nearest-M M=16/efSearch=80: 0.9912 at 0.386 ms), and it lets a smaller M reach a given recall.
7. On SciFact, exact dense (nDCG@10 0.6484) is below naive BM25 (0.6634). The two should fail on different queries, which is the case for hybrid fusion in M5.

### Limitations
- Only 5183 vectors. Brute force is already about 1.3 ms there, so the speedups here may not extrapolate to millions of vectors, and the memory-vs-speed tradeoff only shows up at scale.
- Brute force is a plain scalar Java loop with no SIMD or multithreading, so every speedup is relative to a naive baseline, not an optimised one.
- Latencies are single runs of 300 queries; differences below about 0.1 ms are within noise. The build-parameter sweep uses one seed; only the ablation averages 3 seeds.
- The 0.98 recall target was picked after seeing the sweep, not fixed in advance.
- Query embedding time is excluded (query vectors are precomputed). End-to-end latency would add the ONNX inference cost.
- The HNSW graph is in memory only and rebuilt from the cached vectors on each run (about 6 s). No persistence, deletes or updates yet (M6); insertion is not thread-safe.
- One dataset (SciFact), one model (MiniLM-L6), no comparison yet against Lucene HNSW or pgvector.
- Documents are truncated at 256 tokens; how many abstracts that affects was not measured.

### Errors hit
- Ran mvn exec:java with -Dexec.mainClass again; the pom hardcodes the main class, so it silently ran RunBaseline instead of RunEmbedCorpus. Fixed by launching runners with java -cp target\classes;<cp.txt> (same trap as M3).
- Wrote a leftover empty loop in RunHnswSweep (harmless, removed).
- Predicted HNSW would not be meaningfully faster than brute force at 5k vectors; measured 3.7x to 4.9x at efSearch=40. The prediction was wrong.
- The build sweep produced a 25.9 s build for M=32/efC=200 while efC=400 took 15.3 s, which is impossible as a real cost. Treated as system noise and excluded from conclusions.
- Latency comparisons between two identical runs differed by up to 30% (brute force 1.279 vs 1.322 ms, hnsw ef=40 0.350 vs 0.270 ms). Quote ranges, not single figures.

### Reproduce
One-off setup: model files (gitignored) and classpath.

    New-Item -ItemType Directory -Force models\all-MiniLM-L6-v2
    $base = "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main"
    Invoke-WebRequest "$base/onnx/model.onnx" -OutFile models\all-MiniLM-L6-v2\model.onnx
    Invoke-WebRequest "$base/vocab.txt" -OutFile models\all-MiniLM-L6-v2\vocab.txt
    mvn -q compile
    mvn -q dependency:build-classpath "-Dmdep.outputFile=cp.txt"
    $cp = "target\classes;" + (Get-Content cp.txt)

Embed once (about 211 s), then run the benchmarks.

    java -cp $cp io.github.abdullah.rag.eval.RunEmbedCorpus
    java -cp $cp io.github.abdullah.rag.eval.RunDenseBaseline
    java -cp $cp io.github.abdullah.rag.eval.RunHnswSweep
    java -cp $cp io.github.abdullah.rag.eval.PlotRecallLatency
    java -cp $cp io.github.abdullah.rag.eval.RunHnswBuildSweep
    java -cp $cp io.github.abdullah.rag.eval.RunHnswHeuristicAblation

Only RunHnswSweep writes a CSV (results/hnsw-sweep.csv); the other two sweeps print to the console. Tests that need the model skip themselves if models/ is absent (as on CI).