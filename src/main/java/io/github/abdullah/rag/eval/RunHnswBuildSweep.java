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

/** Sweeps HNSW build parameters (M, efConstruction) on real SciFact embeddings. */
public final class RunHnswBuildSweep {

    private static final String MODEL = "onnx-all-MiniLM-L6-v2";
    private static final int DIM = 384;
    private static final int K = 10;
    private static final int[] M_VALUES = {8, 16, 32};
    private static final int[] EFC_VALUES = {100, 200, 400};
    private static final int[] EF_SEARCH = {20, 40, 80};

    private RunHnswBuildSweep() {
    }

    public static void main(String[] args) throws Exception {
        Path dataDir = Path.of("data", "scifact");
        BeirDataset dataset = BeirLoader.load(dataDir, "test");
        EmbeddingCache.Loaded docs = EmbeddingCache.loadFor(
                dataDir.resolve("embeddings-" + MODEL + "-docs.bin"), MODEL, DIM);
        Map<String, float[]> qByText = RunDenseBaseline.loadQueryVectors(dataDir, dataset);

        BruteForceIndex exact = new BruteForceIndex(DIM);
        for (int i = 0; i < docs.ids().size(); i++) {
            exact.add(docs.ids().get(i), docs.vectors().get(i));
        }

        List<float[]> qVecs = new ArrayList<>();
        List<Set<String>> truth = new ArrayList<>();
        for (Query q : dataset.queries()) {
            float[] v = qByText.get(q.text());
            qVecs.add(v);
            Set<String> t = new HashSet<>();
            for (ScoredDoc s : exact.search(v, K)) {
                t.add(s.docId());
            }
            truth.add(t);
        }

        System.out.println("Loaded " + dataset);
        System.out.printf("%-4s %-6s %-9s %-9s %-13s %-8s%n",
                "M", "efC", "build s", "efSearch", "annRecall@10", "p50 ms");

        for (int m : M_VALUES) {
            for (int efc : EFC_VALUES) {
                HnswIndex hnsw = new HnswIndex(DIM, new HnswConfig(m, efc, 42L));
                long t0 = System.nanoTime();
                for (int i = 0; i < docs.ids().size(); i++) {
                    hnsw.add(docs.ids().get(i), docs.vectors().get(i));
                }
                double buildSec = (System.nanoTime() - t0) / 1e9;

                for (int ef : EF_SEARCH) {
                    hnsw.setEfSearch(ef);
                    for (float[] v : qVecs) { // warmup
                        hnsw.search(v, K);
                    }
                    double[] lat = new double[qVecs.size()];
                    double recallSum = 0;
                    for (int i = 0; i < qVecs.size(); i++) {
                        long s = System.nanoTime();
                        List<ScoredDoc> r = hnsw.search(qVecs.get(i), K);
                        lat[i] = (System.nanoTime() - s) / 1e6;
                        int hit = 0;
                        for (ScoredDoc d : r) {
                            if (truth.get(i).contains(d.docId())) {
                                hit++;
                            }
                        }
                        recallSum += hit / (double) truth.get(i).size();
                    }
                    System.out.printf("%-4d %-6d %-9.1f %-9d %-13.4f %-8.3f%n",
                            m, efc, buildSec, ef, recallSum / qVecs.size(),
                            EvalRunner.percentile(lat, 50));
                }
            }
        }
    }
}