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

## Milestone 5: Hybrid retrieval (RRF) + cross-encoder reranker

### Setup
- Data: SciFact, 5183 docs. Test = 300 queries, train = 809 queries. All tuning on train, config frozen, evaluated once on test.
- Children: BM25 (own index, stop + stem analyzer) and dense HNSW (MiniLM-L6, M=16, efC=200, efSearch=40, cached vectors). Query vectors are precomputed, so latencies exclude query embedding.
- Train-split query vectors embedded in a separate cache file (RunEmbedTrainQueries); the test cache is untouched, so M4 numbers stay reproducible.
- Hardware as in M4. All latencies are single runs in one JVM.

### Components implemented
- RrfFusion (pure function over ranked lists), HybridRetriever (any Retriever list behind the Retriever interface), RerankingRetriever (first stage + PairScorer), OnnxCrossEncoder, WordPieceTokenizer.encodePair.
- Experiment runners: RunHybrid, RunEmbedTrainQueries, RunHybridSweep, RunRerank, RunRerankCache, RunBlendSweep, RunBlendTest.

### Decisions and alternatives rejected
- **RRF on ranks only**, score = sum of 1/(c + rank). Rejected weighted score mixing: BM25 scores and cosine similarities are on different scales. A duplicate inside one list counts once; ties break by docId so runs are deterministic.
- **Parameters fixed before the first run:** c=60, candidate depth 100, rerank depth 50. This makes the first hybrid number an untuned result.
- **HybridRetriever has an indexChildren flag** so sweeps reuse one built HNSW graph (6 s to build) across configs.
- **Train/test discipline:** sweeps on train, selection rules written down before seeing numbers, one test run per frozen config. Rejected tuning on test and re-running other depths or models on test.
- **Reranker model:** cross-encoder/ms-marco-MiniLM-L-6-v2 via ONNX. It shares MiniLM's BERT uncased vocab (verified: vocab.txt files byte-identical), so the existing WordPiece tokenizer is reused. encodePair builds [CLS] query [SEP] doc [SEP] with segment ids 0/1; the query is capped at half the token budget. encode() was left untouched (a few duplicated lines) so the M4 reference-similarity check stays valid.
- **Sanity check before SciFact:** relevant vs irrelevant pairs scored +8.8 vs -11.2 and +7.7 vs -11.2. The model card's Berlin example is about +8.6 (recalled from memory, not re-checked).
- **Blend, not replace:** weighted RRF of the hybrid rank and the cross-encoder rank, (1-w)/(c+hybridRank) + w/(c+ceRank). Ranks, not logits, for the same scale reason. Cross-encoder scores cached to disk (results/rerank-scores-{train,test}.tsv) so the blend sweep is offline.

### Selection rules (written before the runs)
- Hybrid sweep: best train nDCG@10; within 0.002 of the best, lower depth; tie between configs at equal depth, higher train nDCG added before the test run.
- Blend sweep: best train nDCG@10; within 0.002 of the best, smaller w.

### Final results (SciFact test, 300 queries, k=10)
| Retriever | Recall@10 | MRR@10 | nDCG@10 | p50 | p95 |
|---|---|---|---|---|---|
| bm25-indexed-stop-stem | 0.8168 | 0.6455 | 0.6827 | 0.17-0.27 ms | 0.30-0.69 ms |
| dense-hnsw (efSearch 40) | 0.7883 | 0.6068 | 0.6484 | 0.29-0.43 ms | 0.36-0.69 ms |
| hybrid RRF c=60, depth 100 | 0.8503 | 0.6816 | 0.7166 | 1.0-1.1 ms | 1.5-2.1 ms |
| hybrid RRF c=1, depth 100 (train-tuned) | 0.8450 | 0.6771 | 0.7154 | 1.0 ms | 1.5 ms |
| hybrid + cross-encoder, replace (depth 50) | 0.8156 | 0.6558 | 0.6884 | 4383 ms | 19129 ms |
| hybrid + cross-encoder, frozen blend (c=1, w=0.6) | 0.8629 | 0.6860 | 0.7237 | n/m | n/m |

Latency ranges are across the runs in this milestone. The rerank latency is a cold, unbatched, single-threaded pass. The frozen-blend latency was not measured (computed offline from cached scores).

### Significance (paired bootstrap, 10k resamples, per-query nDCG@10)
- hybrid c=60 minus BM25: +0.0339 (CI +0.0103 to +0.0578, p=0.005). Minus dense: +0.0682 (CI +0.0450 to +0.0917).
- c=1 minus c=60 (both hybrid): -0.0012 (CI -0.0165 to +0.0141, p=0.86).
- rerank (replace) minus hybrid: -0.0282 (CI -0.0577 to +0.0006, p=0.054). Minus BM25: +0.0057 (CI includes 0).
- frozen blend minus hybrid: +0.0071 (CI -0.0135 to +0.0275, p=0.52). Minus BM25: +0.0409 (CI +0.0165 to +0.0654). Minus dense: +0.0753 (CI +0.0471 to +0.1043).
- Recall@10 and MRR@10 differences were not bootstrapped; they are not claimed as gains.

### Train sweeps (809 queries)
- Train baselines: BM25 0.6946, dense 0.6578, hybrid c=60/depth 100 0.7121 nDCG@10.
- Hybrid sweep (5 c values x 5 depths): every config beat both single retrievers (worst 0.7109). Smaller c was better (best c=1/depth 200, 0.7245), depth mattered little at c=1 (0.7191 to 0.7245). Frozen: c=1, depth 100 (0.7239). On test it was 0.7154 vs 0.7166 for the untuned default: the train gain did not transfer.
- Blend sweep (3 c values x 11 weights): nDCG rose from w=0, peaked near w=0.6 and fell toward w=1 for all three c. Frozen: c=1, w=0.6, train 0.7352 (+0.023 over hybrid, CI +0.011 to +0.036). On test +0.007, not significant.

### Conclusions I can defend
1. Hybrid RRF beats BM25-only by +0.034 and dense-only by +0.068 nDCG@10 on SciFact test, with intervals excluding 0, using untuned defaults. This is the M5 result.
2. The RRF constant and candidate depth do not matter on SciFact. The train-tuned c=1 gave -0.001 on test (null).
3. Replacing the hybrid ordering with MS MARCO cross-encoder scores did not help: -0.028 nDCG (borderline, p=0.054) at about 4 s of model time per query. The right documents were present (hybrid Recall@50 = 0.9403), so the loss is in the ordering, not the candidate pool.
4. Blending the cross-encoder in gave a small, statistically undetectable gain on test (+0.007, CI -0.014 to +0.028). It costs 50 model calls per query. Not adopted as the default pipeline.
5. Train gains from tuning shrank on test in both tuning sweeps, the same pattern as the semantic chunker in M3. Selecting on test would have hidden this.

### Limitations
- One dataset (SciFact), one dense model, one reranker model; no second dataset (FiQA) yet.
- Why the reranker underperforms is a hypothesis (trained on MS MARCO web QA, SciFact is claim verification); no domain-matched reranker was tried.
- Rerank latency is single-threaded, unbatched, up to 512 tokens per pair. Batching, threads and shorter inputs would cut it a lot; not done because there was no quality gain to justify it. The test-cache run (~63 min) was about 3x slower per query than train (~49 min), likely laptop throttling.
- Latencies are single runs; hybrid children run sequentially, not in parallel. Query embedding time is excluded.
- Pure rerank scored slightly above hybrid on train (0.7180 vs 0.7121) but below on test (0.6884 vs 0.7166), so train is not a reliable predictor for the reranker here.
- Blend sweep is 33 configs on one split; the best train number is optimistic by construction.

### Errors hit
- Predicted pure rerank would fall below hybrid on train; it scored slightly above (0.7180 vs 0.7121). It still lost on test.
- Estimated the test-cache run at 18 minutes from the train rate; it took about 63.
- Pasting several commands as one block merged lines, so Java received "train" as a main class (ClassNotFoundException). A fresh PowerShell session lost $cp, giving the same error for "test". Fix: rebuild cp.txt and $cp, check with $cp.Substring(0, 60), run one command at a time.

### Reproduce
    mvn -q compile
    mvn -q dependency:build-classpath "-Dmdep.outputFile=cp.txt"
    $cp = "target\classes;" + (Get-Content cp.txt -Raw).Trim()
    New-Item -ItemType Directory -Force models\ms-marco-MiniLM-L-6-v2
    $base = "https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2/resolve/main"
    Invoke-WebRequest "$base/onnx/model.onnx" -OutFile models\ms-marco-MiniLM-L-6-v2\model.onnx
    Invoke-WebRequest "$base/vocab.txt" -OutFile models\ms-marco-MiniLM-L-6-v2\vocab.txt
    java -cp $cp io.github.abdullah.rag.eval.RunEmbedTrainQueries
    java -cp $cp io.github.abdullah.rag.eval.RunHybrid
    java -cp $cp io.github.abdullah.rag.eval.RunHybridSweep
    java -cp $cp io.github.abdullah.rag.eval.RunRerank
    java -cp $cp io.github.abdullah.rag.eval.RunRerankCache train
    java -cp $cp io.github.abdullah.rag.eval.RunBlendSweep
    java -cp $cp io.github.abdullah.rag.eval.RunRerankCache test
    java -cp $cp io.github.abdullah.rag.eval.RunBlendTest

RunRerankCache takes about 50 to 65 minutes per split. Tests that need models skip themselves if models/ is absent.

## Milestone 6: WAL, crash recovery, updates, deletes

### Setup
- Data: SciFact test, 5183 docs, 300 queries, K=10. Pipeline and parameters as M5 (BM25 stop+stem, MiniLM HNSW M=16 efC=200 efSearch=40, RRF c=60, depth 100). Cached vectors, so latencies exclude query embedding. Hardware as M4. All latencies are single runs in one JVM.
- Gate: the live engine must reproduce the M5 hybrid score (0.7166 nDCG@10) after ingest, compaction and restart.

### Architecture
- Immutable base (BM25 index + HNSW) + in-memory delta (docs added or updated since the base) + a set of "touched" docIds. A search reads one consistent snapshot of the delta and the touched set, runs base and delta, and drops every base hit whose docId is touched.
- Directory layout: CURRENT (names generation N), base-N.snap (all docs with text and vectors), wal-N.log (changes since that base).
- Write path: append to WAL, fsync, then apply to memory. Open path: read CURRENT, load base snapshot, scan WAL, truncate torn tail, replay into the same state machine the live path uses, delete files of other generations.
- compact(): merge base + delta, write base-(N+1).snap (temp file, fsync, atomic rename), create wal-(N+1).log, switch CURRENT by atomic rename (the commit point), delete the old generation, rebuild indexes.

### Decisions and alternatives rejected
- **Never mutate the BM25 index or the HNSW graph.** Deletes and updates are tombstone + delta; compaction rebuilds. Rejected: in-place posting-list edits and HNSW node removal (graph connectivity repair is a research problem; rebuild is simple and testable).
- **Update = one ADD record for an existing docId** (upsert). Rejected: separate DELETE + ADD records, because a crash between them would lose the doc. One record is atomic by construction.
- **ADD records carry the embedding vector and the model name.** Replay never runs ONNX (about 40 ms/doc), and a vector from another model is rejected per record. Cost: about 3.1 KB per record (16,074,476 bytes for 5183 docs, mostly the 1,536-byte vector).
- **Record format:** length | seq | type | payload | CRC32. Reader stops at the first truncated, corrupt or out-of-sequence record and truncates the file there, so new appends never follow garbage. Rejected: skipping a bad record and continuing (cannot know what was lost).
- **Fsync on every append.** Rejected: group commit (more code, no measured need at 1.14 ms/put).
- **BM25 across segments uses combined N, df and average length**, so base and delta scores are comparable (checked: a corpus split across base and delta scores identically to one index). Rejected: scoring segments separately and merging by raw score.
- **Dense over-fetch:** ask the base for k + (number of touched ids), then filter. Rejected: filter after top-k (returns nothing when the nearest vectors are all deleted; tested).
- **Delta dense search is brute force**, delta BM25 is a small InvertedIndex rebuilt whenever the state version changes.
- **Snapshot and CURRENT are CRC-protected and atomically replaced.** A damaged base or CURRENT fails loudly instead of guessing. A WAL that does not start right after the base's last seq is rejected.
- **Read-write lock in LiveEngine:** searches share a read lock, compaction takes the write lock. Without it a search could pair the new generation's empty touched set with the old base index and return deleted docs.

### Benchmark numbers (SciFact test, 300 queries, K=10)
| state | Recall@10 | MRR@10 | nDCG@10 | p50 ms | p95 ms |
|---|---|---|---|---|---|
| M5 hybrid (reference) | 0.8503 | 0.6816 | 0.7166 | 1.0-1.1 | 1.5-2.1 |
| live, all 5183 docs in the delta | 0.8536 | 0.6817 | 0.7173 | 7.935 | 17.310 |
| live, after compaction | 0.8503 | 0.6816 | 0.7166 | 1.165 | 2.447 |
| live, after restart | 0.8503 | 0.6816 | 0.7166 | 0.997 | 1.862 |

- Compacted and restarted engines reproduce M5 exactly. The delta-only engine is +0.0007 nDCG; not bootstrapped, not claimed as a gain (hypothesis: exact dense search in the delta vs approximate HNSW).
- Ingest: 5183 puts in 5911 ms = 877 docs/s, 1.14 ms per put with fsync on every put (one run, one laptop disk; it may reflect a drive that acknowledges flush from a volatile cache).
- Recovery: reopen with a full 5183-record WAL replay and empty base: 141 ms (the first query then builds the delta index). Reopen after compaction: 8138 ms, dominated by rebuilding HNSW (about 6 s in M4) and BM25 from the snapshot.
- Compaction of 5183 docs: 8.6 s, searches blocked while it runs.

Churn (518 docs deleted, 518 docs updated with identical content, no compaction; scored on 267 queries that still have a surviving relevant doc, with deleted docs removed from the judgments; not comparable with the 300-query numbers above). Two identical runs:

| engine | nDCG@10 | deleted docs returned | p50 / p95 ms, run 1 | p50 / p95 ms, run 2 |
|---|---|---|---|---|
| live, churned, no compaction | 0.6947 | 0 | 5.246 / 9.453 | 11.493 / 14.672 |
| reference, built from scratch on survivors | 0.6974 | 0 | 1.424 / 1.908 | 2.383 / 3.658 |
| live, compacted | 0.6974 | 0 | 1.017 / 1.855 | 2.480 / 3.349 |

- Churned vs reference: nDCG -0.0027 (not bootstrapped); top-10 differs on 115 of 300 queries (46 different doc set, 69 same docs in a different order). Compacted vs reference: top-10 identical on all 300 queries.
- Compaction after churn: 7.4 s and 14.4 s in the two runs. Both runs used the same code, so the 2x spread in every timing is machine noise (likely throttling); quote ranges.

### Tests
- Crash by truncation: the log is cut at every byte offset (WalReaderTest, DurableStoreTest); recovery keeps exactly the complete records and later writes survive another restart.
- Corruption: every single-bit flip in a record, snapshot and CURRENT is rejected; every proper prefix is rejected.
- Compaction crash windows: debris before the CURRENT switch is ignored and cleaned; old files left after the switch are cleaned and the new generation is used.
- Real kill: a child JVM writes in a loop, prints ACK only after the op returned, and is killed with destroyForcibly. Six rounds (45 to 154 acked ops): recovered = acked + 1 every time, never fewer, and the recovered state equals the operation history.
- Concurrency: one writer (puts, deletes, updates), one compactor and two searchers; 7984 searches overlapped 20 compactions with no deleted doc returned and no duplicate ids; exact live set at the end, after a final compaction, and after restart.
- Equivalence: an engine that went through updates, deletes and compaction returns the same ids and scores as one built from scratch on the same final docs.

### Conclusions I can defend
1. Crash recovery is exact: every acknowledged operation survives a process kill, and recovery from any torn tail stops at the last complete record.
2. Deleted docs are never returned: 0 across the churn benchmark, the stress test, after compaction and after restart.
3. Compaction is invisible to users except for speed: the compacted engine returns identical top-10 lists to a from-scratch build (0 of 300 differ).
4. Un-compacted churn costs quality and latency, not correctness: -0.0027 nDCG and about 3.7x to 4.8x slower search at about 20% churn. The likely cause of the latency is dense over-fetch (about 1,100 candidates asked of HNSW); not directly measured.
5. There is a real tradeoff between restart and search time: a log-only index restarts in 141 ms but searches in about 8 ms p50; a compacted index searches in about 1 ms but restarts in about 8 s because HNSW is rebuilt. Persisting the graph would remove the restart cost; not done.

### Limitations
- Crash tests kill the process or cut files. The OS page cache survives a process kill, so this does not prove fsync reaches stable storage; power-loss testing needs a VM or hardware. Torn tails come only from the byte-cut tests: in the six kill rounds truncatedBytes was 0 every time (a record is one small write).
- The snapshot file is fsynced but its directory entry is not (Java on Windows cannot fsync a directory). On Linux the directory should be fsynced after each rename.
- Compaction rebuilds BM25 and HNSW from scratch and blocks all searches for 7 to 14 s at 5183 docs. A background build followed by a swap would fix it; not done.
- Base docs (text and vectors) stay on the heap next to the indexes built from them, a second copy of the corpus.
- The 46 churned queries with a different doc set are not explained by measurement. Candidates: BM25 statistics skewed by hidden docs, a base graph that still contains deleted vectors, and the large over-fetch. No ablation was run.
- The stress test uses an exact in-memory fake vector index, so it shows lock correctness, not compaction latency with real HNSW. A stress test can fail but cannot prove a lock-based design correct.
- Updates in the churn benchmark re-put identical content, so they test the replace path but not changed-content ranking (that is covered by unit tests only).
- One dataset (SciFact), one machine, single-run latencies, nothing measured beyond about 5k docs. Multi-process access is not handled: one process per directory.

### Errors hit
- Guessed package names twice (ScoredDoc and TextAnalyzer under the wrong packages; VectorIndex under hnsw instead of vector). Fixed by asking for the real package lines before writing more code.
- Predicted the kill test would show torn tails; it showed none (0 truncated bytes in 6 rounds).
- Predicted 3 to 15 ms per fsync'd put; measured 1.14 ms.
- Predicted reopen after compaction at about 6 s; measured 8.1 s.
- Predicted the churned top-10 would differ on single digits to low tens of queries; it differed on 115 (46 by doc set).
- Churn and compaction timings doubled between two identical runs (machine noise).

### Reproduce
    mvn -q compile
    mvn -q dependency:build-classpath "-Dmdep.outputFile=cp.txt"
    $cp = "target\classes;" + (Get-Content cp.txt -Raw).Trim()
    mvn -q test "-Dtest=WalRecordTest,WalWriterTest,WalReaderTest,WalPayloadsTest,LiveDocsTest,SnapshotFileTest,CurrentPointerTest,DurableStoreTest"
    mvn -q test "-Dtest=SegmentedBm25Test,SegmentedDenseTest,SegmentedEngineTest,LiveEngineTest"
    mvn -q test "-Dtest=CrashRecoveryTest,ConcurrencyStressTest"
    java -cp $cp io.github.abdullah.rag.eval.RunLiveBench
    java -cp $cp io.github.abdullah.rag.eval.RunChurnBench

The benchmarks recreate results\live-bench and results\churn-live / churn-ref on each run. Tests that need the ONNX models skip themselves if models/ is absent.