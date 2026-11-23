package io.github.abdullah.rag.eval;

import io.github.abdullah.rag.embedding.EmbeddingCache;
import io.github.abdullah.rag.embedding.OnnxEmbedder;
import io.github.abdullah.rag.model.Query;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * One-off job: embeds the SciFact train-split queries with MiniLM into a
 * separate cache file, so hybrid parameters can be tuned on train and the test
 * queries stay untouched. The document cache is shared (same corpus in both splits).
 */
public final class RunEmbedTrainQueries {

    private RunEmbedTrainQueries() {
    }

    public static void main(String[] args) throws Exception {
        Path dataDir = Path.of("data", "scifact");
        Path modelDir = Path.of("models", "all-MiniLM-L6-v2");
        BeirDataset train = BeirLoader.load(dataDir, "train");
        System.out.println("Loaded " + train);

        try (OnnxEmbedder embedder = new OnnxEmbedder(modelDir, 256)) {
            List<String> ids = new ArrayList<>();
            List<float[]> vecs = new ArrayList<>();
            for (Query q : train.queries()) {
                ids.add(q.id());
                vecs.add(embedder.embed(q.text()));
            }
            Path file = dataDir.resolve("embeddings-" + embedder.name() + "-queries-train.bin");
            EmbeddingCache.save(file, embedder.name(), embedder.dimension(), ids, vecs);
            System.out.printf("Embedded %d train queries -> %s%n", ids.size(), file);
        }
    }
}