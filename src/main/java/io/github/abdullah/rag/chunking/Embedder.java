package io.github.abdullah.rag.chunking;

/**
 * Turns text into a dense vector. Implementations must be deterministic
 * and return L2-normalised vectors of a fixed dimension, so cosine similarity
 * is just a dot product.
 */
public interface Embedder {

    /** Short name for logs and benchmark tables, e.g. "hashed-tf-1024". */
    String name();

    /** The dimension of the embedding vectors. */
    int dimension();

    float[] embed(String text);
}