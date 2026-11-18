package io.github.abdullah.rag.vector;

import io.github.abdullah.rag.retrieval.ScoredDoc;
import java.util.List;

/**
 * A searchable collection of unit-length vectors keyed by id.
 * Implemented by the exact brute-force index (ground truth) and by HNSW
 * (approximate), so both can be swapped in the same benchmark code.
 */
public interface VectorIndex {

    /** Short name for logs and benchmark tables, e.g. "brute-force", "hnsw-M16-ef200". */
    String name();

    /** Dimension every vector in this index must have. */
    int dimension();

    /** Number of vectors currently stored. */
    int size();

    /** Adds a vector under the given id. Throws if the id already exists or the dimension is wrong. */
    void add(String id, float[] vector);

    /**
     * Returns up to k nearest vectors to the query, best first.
     * ScoredDoc.score is cosine similarity (higher = closer), i.e. 1 - distance.
     */
    List<ScoredDoc> search(float[] query, int k);
}