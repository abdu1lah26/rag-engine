package io.github.abdullah.rag.eval;

import io.github.abdullah.rag.embedding.EmbeddingCache;
import io.github.abdullah.rag.model.Query;
import io.github.abdullah.rag.retrieval.DenseRetriever;
import io.github.abdullah.rag.vector.BruteForceIndex;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;

/** Exact (brute-force) dense retrieval on SciFact with cached MiniLM vectors. */
public final class RunDenseBaseline {

    private static final String MODEL = "onnx-all-MiniLM-L6-v2";
    private static final int DIM = 384;

    private RunDenseBaseline() {
    }

    /** Loads cached query vectors and keys them by query text. */
        static Map<String, float[]> loadQueryVectors(Path dataDir, BeirDataset dataset)
            throws Exception {
        return loadQueryVectors(dataDir, dataset, "embeddings-" + MODEL + "-queries.bin");
    }

    /** Same as above, but reads the named cache file (e.g. the train-split queries). */
    static Map<String, float[]> loadQueryVectors(Path dataDir, BeirDataset dataset,
            String cacheFileName) throws Exception {
        EmbeddingCache.Loaded loaded = EmbeddingCache.loadFor(
                dataDir.resolve(cacheFileName), MODEL, DIM);
        Map<String, float[]> byId = new HashMap<>();
        for (int i = 0; i < loaded.ids().size(); i++) {
            byId.put(loaded.ids().get(i), loaded.vectors().get(i));
        }
        Map<String, float[]> byText = new HashMap<>();
        for (Query q : dataset.queries()) {
            float[] v = byId.get(q.id());
            if (v == null) {
                throw new IllegalStateException("no cached vector for query " + q.id());
            }
            byText.put(q.text(), v);
        }
        return byText;
    }

    public static void main(String[] args) throws Exception {
        Path dataDir = Path.of("data", "scifact");
        BeirDataset dataset = BeirLoader.load(dataDir, "test");
        System.out.println("Loaded " + dataset);

        DenseRetriever retriever = new DenseRetriever(
                new BruteForceIndex(DIM),
                dataDir.resolve("embeddings-" + MODEL + "-docs.bin"),
                MODEL,
                loadQueryVectors(dataDir, dataset));

        System.out.println(EvalRunner.run(retriever, dataset, 10));
    }
}