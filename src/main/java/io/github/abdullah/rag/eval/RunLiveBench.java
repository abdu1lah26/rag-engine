package io.github.abdullah.rag.eval;

import io.github.abdullah.rag.analysis.StemmingAnalyzer;
import io.github.abdullah.rag.embedding.EmbeddingCache;
import io.github.abdullah.rag.fusion.RrfFusion;
import io.github.abdullah.rag.model.Document;
import io.github.abdullah.rag.retrieval.LiveEngine;
import io.github.abdullah.rag.retrieval.Retriever;
import io.github.abdullah.rag.retrieval.ScoredDoc;
import io.github.abdullah.rag.vector.HnswConfig;
import io.github.abdullah.rag.vector.HnswIndex;
import io.github.abdullah.rag.wal.DurableStore.CompactionResult;
import io.github.abdullah.rag.wal.WalPayloads.AddOp;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.stream.Stream;

/**
 * M6 benchmark, part 1: ingest every SciFact doc through the WAL, then score the same
 * hybrid pipeline in three states: everything in the delta, after compaction, after restart.
 * Parameters are the M5 ones (RRF c=60, depth 100, efSearch 40); M5 hybrid scored 0.7166 nDCG@10.
 */
public final class RunLiveBench {

    private static final String MODEL = "onnx-all-MiniLM-L6-v2";
    private static final int DIM = 384;
    private static final int K = 10;
    private static final int DEPTH = 100;
    private static final int RRF_C = 60;
    private static final int EF_SEARCH = 40;

    private RunLiveBench() {}

    /** Lets EvalRunner score a LiveEngine. index() is a no-op: the data is already in the store. */
    private static final class LiveRetriever implements Retriever {
        private final String name;
        private final LiveEngine engine;

        LiveRetriever(String name, LiveEngine engine) {
            this.name = name;
            this.engine = engine;
        }

        public String name() {
            return name;
        }

        public void index(List<Document> corpus) {}

        public List<ScoredDoc> search(String queryText, int k) {
            return engine.search(queryText, k);
        }
    }

    static LiveEngine open(Path dir, Map<String, float[]> queryVectors) throws IOException {
        return new LiveEngine(dir, new StemmingAnalyzer(), MODEL,
                q -> {
                    float[] v = queryVectors.get(q);
                    if (v == null) {
                        throw new IllegalArgumentException("no precomputed vector for query: " + q);
                    }
                    return v;
                },
                () -> {
                    HnswIndex h = new HnswIndex(DIM, HnswConfig.defaults());
                    h.setEfSearch(EF_SEARCH);
                    return h;
                },
                new RrfFusion(RRF_C), DEPTH);
    }

    static void deleteTree(Path root) throws IOException {
        if (!Files.exists(root)) {
            return;
        }
        try (Stream<Path> s = Files.walk(root)) {
            for (Path p : (Iterable<Path>) s.sorted(Comparator.reverseOrder())::iterator) {
                Files.delete(p);
            }
        }
    }

    private static double ms(long startNanos) {
        return (System.nanoTime() - startNanos) / 1e6;
    }

    public static void main(String[] args) throws Exception {
        Path dataDir = Path.of("data", "scifact");
        BeirDataset dataset = BeirLoader.load(dataDir, "test");
        System.out.println("Loaded " + dataset);
        Map<String, float[]> queryVectors = RunDenseBaseline.loadQueryVectors(dataDir, dataset);

        EmbeddingCache.Loaded cache = EmbeddingCache.loadFor(
                dataDir.resolve("embeddings-" + MODEL + "-docs.bin"), MODEL, DIM);
        Map<String, float[]> docVectors = new HashMap<>();
        for (int i = 0; i < cache.ids().size(); i++) {
            docVectors.put(cache.ids().get(i), cache.vectors().get(i));
        }

        Path dir = Path.of("results", "live-bench");
        deleteTree(dir);

        // 1. Ingest: one WAL record per doc, fsync on every put.
        List<Document> corpus = dataset.corpus();
        LiveEngine engine = open(dir, queryVectors);
        long t = System.nanoTime();
        for (Document d : corpus) {
            float[] v = docVectors.get(d.id());
            if (v == null) {
                throw new IllegalStateException("no cached vector for doc " + d.id());
            }
            engine.put(new AddOp(d.id(), d.title() == null ? "" : d.title(), d.text(), MODEL, v));
        }
        double ingestMs = ms(t);
        System.out.println(String.format(Locale.ROOT,
                "ingest: %d docs in %.0f ms (%.1f docs/s, %.2f ms per put, fsync each)",
                corpus.size(), ingestMs, corpus.size() / (ingestMs / 1000.0), ingestMs / corpus.size()));
        System.out.println(String.format(Locale.ROOT, "wal size: %d bytes",
                Files.size(dir.resolve("wal-000000.log"))));

        // 2. Everything still in the delta (base empty): BM25 over the delta, dense by brute force.
        System.out.println("--- all docs in the delta, nothing compacted");
        System.out.println(EvalRunner.run(new LiveRetriever("live-delta-only", engine), dataset, K));
        engine.close();

        // 3. Restart with a full-size WAL.
        t = System.nanoTime();
        engine = open(dir, queryVectors);
        System.out.println(String.format(Locale.ROOT,
                "reopen, WAL replay of %d records, empty base: %.0f ms", corpus.size(), ms(t)));

        // 4. Compaction.
        t = System.nanoTime();
        CompactionResult r = engine.compact();
        System.out.println(String.format(Locale.ROOT,
                "compact: generation %d, %d docs, %.0f ms (includes HNSW and BM25 rebuild)",
                r.generation(), r.docs(), ms(t)));
        System.out.println("--- after compaction: everything in the base");
        System.out.println(EvalRunner.run(new LiveRetriever("live-compacted", engine), dataset, K));
        engine.close();

        // 5. Restart after compaction: load the snapshot, rebuild indexes, empty WAL.
        t = System.nanoTime();
        engine = open(dir, queryVectors);
        System.out.println(String.format(Locale.ROOT,
                "reopen after compaction (snapshot load + index rebuild, empty WAL): %.0f ms", ms(t)));
        System.out.println("--- after restart");
        System.out.println(EvalRunner.run(new LiveRetriever("live-restarted", engine), dataset, K));
        engine.close();
    }
}