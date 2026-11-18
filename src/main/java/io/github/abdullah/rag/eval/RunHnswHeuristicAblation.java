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

/** Ablation: diversity heuristic vs plain "M nearest" neighbour selection. */
public final class RunHnswHeuristicAblation {

    private static final String MODEL = "onnx-all-MiniLM-L6-v2";
    private static final int DIM = 384;
    private static final int K = 10;
    private static final int EFC = 100;
    private static final int[] M_VALUES = {8, 16};
    private static final int[] EF_SEARCH = {20, 40, 80};
    private static final long[] SEEDS = {1L, 2L, 3L};

    private RunHnswHeuristicAblation() {
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

        System.out.println("Loaded " + dataset + "  (efC=" + EFC + ", mean over "
                + SEEDS.length + " seeds)");
        System.out.printf("%-10s %-4s %-9s %-13s %-8s %-8s%n",
                "selection", "M", "efSearch", "annRecall@10", "p50 ms", "build s");

        for (int m : M_VALUES) {
            for (boolean heuristic : new boolean[] {true, false}) {
                double[] recall = new double[EF_SEARCH.length];
                double[] p50 = new double[EF_SEARCH.length];
                double buildSum = 0;

                for (long seed : SEEDS) {
                    HnswIndex hnsw = new HnswIndex(DIM, new HnswConfig(m, EFC, seed));
                    hnsw.setUseHeuristic(heuristic);
                    long t0 = System.nanoTime();
                    for (int i = 0; i < docs.ids().size(); i++) {
                        hnsw.add(docs.ids().get(i), docs.vectors().get(i));
                    }
                    buildSum += (System.nanoTime() - t0) / 1e9;

                    for (int e = 0; e < EF_SEARCH.length; e++) {
                        hnsw.setEfSearch(EF_SEARCH[e]);
                        for (float[] v : qVecs) { // warmup
                            hnsw.search(v, K);
                        }
                        double[] lat = new double[qVecs.size()];
                        double rSum = 0;
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
                            rSum += hit / (double) truth.get(i).size();
                        }
                        recall[e] += rSum / qVecs.size() / SEEDS.length;
                        p50[e] += EvalRunner.percentile(lat, 50) / SEEDS.length;
                    }
                }
                for (int e = 0; e < EF_SEARCH.length; e++) {
                    System.out.printf("%-10s %-4d %-9d %-13.4f %-8.3f %-8.1f%n",
                            heuristic ? "heuristic" : "nearest-M", m, EF_SEARCH[e],
                            recall[e], p50[e], buildSum / SEEDS.length);
                }
            }
        }
    }
}