package io.github.abdullah.rag.retrieval;

import io.github.abdullah.rag.embedding.EmbeddingCache;
import io.github.abdullah.rag.model.Document;
import io.github.abdullah.rag.vector.VectorIndex;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Dense (embedding) retrieval over any VectorIndex.
 *
 * Document vectors are loaded from the on-disk cache, and query vectors are
 * looked up by query text from a precomputed map. So search() time is pure
 * index time and excludes the embedding model's inference cost.
 */
public final class DenseRetriever implements Retriever {

    private final VectorIndex index;
    private final Path docCacheFile;
    private final String modelName;
    private final Map<String, float[]> queryVectors;

    /**
     * @param index        empty index to fill (BruteForceIndex, HnswIndex, ...)
     * @param docCacheFile cache file written by RunEmbedCorpus
     * @param modelName    embedder name the cache must have been built with
     * @param queryVectors query text -> precomputed query vector
     */
    public DenseRetriever(VectorIndex index, Path docCacheFile, String modelName,
            Map<String, float[]> queryVectors) {
        this.index = index;
        this.docCacheFile = docCacheFile;
        this.modelName = modelName;
        this.queryVectors = queryVectors;
    }

    @Override
    public String name() {
        return "dense-" + index.name();
    }

    @Override
    public void index(List<Document> corpus) {
        EmbeddingCache.Loaded loaded;
        try {
            loaded = EmbeddingCache.loadFor(docCacheFile, modelName, index.dimension());
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
        Map<String, float[]> byId = new HashMap<>();
        for (int i = 0; i < loaded.ids().size(); i++) {
            byId.put(loaded.ids().get(i), loaded.vectors().get(i));
        }
        for (Document d : corpus) {
            float[] v = byId.get(d.id());
            if (v == null) {
                throw new IllegalStateException("no cached vector for doc " + d.id());
            }
            index.add(d.id(), v);
        }
    }

    @Override
    public List<ScoredDoc> search(String queryText, int k) {
        float[] v = queryVectors.get(queryText);
        if (v == null) {
            throw new IllegalArgumentException("no precomputed vector for query: " + queryText);
        }
        return index.search(v, k);
    }
}