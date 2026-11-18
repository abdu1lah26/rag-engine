package io.github.abdullah.rag.eval;

import io.github.abdullah.rag.embedding.EmbeddingCache;
import io.github.abdullah.rag.model.Query;
import io.github.abdullah.rag.retrieval.ScoredDoc;
import io.github.abdullah.rag.vector.BruteForceIndex;
import io.github.abdullah.rag.vector.HnswConfig;
import io.github.abdullah.rag.vector.HnswIndex;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/** Sweeps HNSW efSearch on real SciFact embeddings and prints the recall/latency curve. */
public final class RunHnswSweep {

    private static final String MODEL = "onnx-all-MiniLM-L6-v2";
    private static final int DIM = 384;
    private static final int K = 10;
    private static final int[] EF_VALUES = {10, 20, 40, 80, 160, 320, 640};

    private RunHnswSweep() {
    }

    public static void main(String[] args) throws Exception {
        Path dataDir = Path.of("data", "scifact");
        BeirDataset dataset = BeirLoader.load(dataDir, "test");
        EmbeddingCache.Loaded docs = EmbeddingCache.loadFor(
                dataDir.resolve("embeddings-" + MODEL + "-docs.bin"), MODEL, DIM);
        Map<String, float[]> qByText = RunDenseBaseline.loadQueryVectors(dataDir, dataset);

        HnswConfig config = HnswConfig.defaults();
        BruteForceIndex exact = new BruteForceIndex(DIM);
        HnswIndex hnsw = new HnswIndex(DIM, config);

        long t0 = System.nanoTime();
        for (int i = 0; i < docs.ids().size(); i++) {
            exact.add(docs.ids().get(i), docs.vectors().get(i));
        }
        double exactBuildMs = (System.nanoTime() - t0) / 1e6;

        long t1 = System.nanoTime();
        for (int i = 0; i < docs.ids().size(); i++) {
            hnsw.add(docs.ids().get(i), docs.vectors().get(i));
        }
        double hnswBuildMs = (System.nanoTime() - t1) / 1e6;
        System.out.printf("Loaded %s%n", dataset);
        System.out.printf("%s built in %.0f ms (brute force: %.0f ms)%n",
                hnsw.name(), hnswBuildMs, exactBuildMs);

        List<Query> queries = dataset.queries();
        List<float[]> qVecs = new ArrayList<>();
        List<Set<String>> truth = new ArrayList<>();
        for (Query q : queries) {
            float[] v = qByText.get(q.text());
            qVecs.add(v);
            Set<String> t = new HashSet<>();
            for (ScoredDoc s : exact.search(v, K)) {
                t.add(s.docId());
            }
            truth.add(t);
        }

        System.out.printf("%-12s %-12s %-10s %-10s %-10s%n",
                "index", "annRecall@10", "nDCG@10", "p50 ms", "p95 ms");

        // Reference row: exact search
        printRow("brute-force", queries, qVecs, truth, dataset, exact, null);

        for (int ef : EF_VALUES) {
            hnsw.setEfSearch(ef);
            printRow("hnsw ef=" + ef, queries, qVecs, truth, dataset, null, hnsw);
        }
    }

    private static void printRow(String label, List<Query> queries, List<float[]> qVecs,
            List<Set<String>> truth, BeirDataset dataset, BruteForceIndex exact, HnswIndex hnsw) {
        // Warmup pass: let the JIT compile the hot loops before timing anything.
        for (float[] v : qVecs) {
            search(exact, hnsw, v);
        }

        double[] lat = new double[queries.size()];
        double recallSum = 0;
        double ndcgSum = 0;
        for (int i = 0; i < queries.size(); i++) {
            long s = System.nanoTime();
            List<ScoredDoc> r = search(exact, hnsw, qVecs.get(i));
            lat[i] = (System.nanoTime() - s) / 1e6;

            List<String> ranked = new ArrayList<>();
            int hit = 0;
            for (ScoredDoc d : r) {
                ranked.add(d.docId());
                if (truth.get(i).contains(d.docId())) {
                    hit++;
                }
            }
            recallSum += hit / (double) truth.get(i).size();
            ndcgSum += Metrics.ndcgAtK(ranked, dataset.qrels().forQuery(queries.get(i).id()), K);
        }
        int n = queries.size();
        System.out.printf("%-12s %-12.4f %-10.4f %-10.3f %-10.3f%n", label,
                recallSum / n, ndcgSum / n,
                EvalRunner.percentile(lat, 50), EvalRunner.percentile(lat, 95));
    }

    private static List<ScoredDoc> search(BruteForceIndex exact, HnswIndex hnsw, float[] v) {
        return exact != null ? exact.search(v, K) : hnsw.search(v, K);
    }
}