package io.github.abdullah.rag.eval;

import io.github.abdullah.rag.embedding.EmbeddingCache;
import io.github.abdullah.rag.embedding.OnnxEmbedder;
import io.github.abdullah.rag.model.Document;
import io.github.abdullah.rag.model.Query;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * One-off job: embeds the SciFact corpus and test queries with MiniLM and
 * writes them to disk, so benchmarks never have to re-run the model.
 *
 * Usage: mvn exec:java -Dexec.mainClass=io.github.abdullah.rag.eval.RunEmbedCorpus
 */
public final class RunEmbedCorpus {

    private RunEmbedCorpus() {
    }

    public static void main(String[] args) throws Exception {
        Path dataDir = Path.of("data", "scifact");
        Path modelDir = Path.of("models", "all-MiniLM-L6-v2");
        BeirDataset dataset = BeirLoader.load(dataDir, "test");
        System.out.println("Loaded " + dataset);

        try (OnnxEmbedder embedder = new OnnxEmbedder(modelDir, 256)) {
            // Documents
            List<String> docIds = new ArrayList<>();
            List<float[]> docVecs = new ArrayList<>();
            long start = System.nanoTime();
            int done = 0;
            for (Document d : dataset.corpus()) {
                docIds.add(d.id());
                docVecs.add(embedder.embed(d.indexableText()));
                done++;
                if (done % 500 == 0) {
                    double secs = (System.nanoTime() - start) / 1e9;
                    System.out.printf("  docs %d/%d  (%.0f s elapsed)%n",
                            done, dataset.corpus().size(), secs);
                }
            }
            double docSecs = (System.nanoTime() - start) / 1e9;
            Path docFile = dataDir.resolve("embeddings-" + embedder.name() + "-docs.bin");
            EmbeddingCache.save(docFile, embedder.name(), embedder.dimension(), docIds, docVecs);
            System.out.printf("Embedded %d docs in %.1f s (%.1f ms/doc) -> %s%n",
                    docIds.size(), docSecs, docSecs * 1000 / docIds.size(), docFile);

            // Queries
            List<String> qIds = new ArrayList<>();
            List<float[]> qVecs = new ArrayList<>();
            for (Query q : dataset.queries()) {
                qIds.add(q.id());
                qVecs.add(embedder.embed(q.text()));
            }
            Path qFile = dataDir.resolve("embeddings-" + embedder.name() + "-queries.bin");
            EmbeddingCache.save(qFile, embedder.name(), embedder.dimension(), qIds, qVecs);
            System.out.printf("Embedded %d queries -> %s%n", qIds.size(), qFile);
        }
    }
}