package io.github.abdullah.rag.chunking;

import io.github.abdullah.rag.analysis.TextAnalyzer;
import java.util.List;

/**
 * Lexical "embedding" via feature hashing: each term is hashed into one of `dim` buckets
 * with a +/- sign, weighted by 1 + log(tf), then L2-normalised.
 * No model needed. Similar vocabulary gives a similar vector.
 */
public final class HashedTfEmbedder implements Embedder {

    private final TextAnalyzer analyzer;
    private final int dim;

    public HashedTfEmbedder(TextAnalyzer analyzer, int dim) {
        if (dim <= 0) throw new IllegalArgumentException("dim must be > 0");
        this.analyzer = analyzer;
        this.dim = dim;
    }

    @Override
    public String name() {
        return "hashed-tf-" + dim;
    }

    @Override
    public int dimension() {
        return dim;
    }

    @Override
    public float[] embed(String text) {
        List<String> terms = analyzer.analyze(text);
        float[] raw = new float[dim];

        // term counts folded straight into buckets, sign chosen by a second hash bit
        java.util.Map<String, Integer> tf = new java.util.HashMap<>();
        for (String t : terms) tf.merge(t, 1, Integer::sum);

        for (var e : tf.entrySet()) {
            int h = mix(e.getKey().hashCode());
            int bucket = Math.floorMod(h, dim);
            float sign = ((h >>> 31) == 0) ? 1f : -1f;
            raw[bucket] += sign * (float) (1.0 + Math.log(e.getValue()));
        }

        double norm = 0;
        for (float v : raw) norm += (double) v * v;
        norm = Math.sqrt(norm);
        if (norm > 0) {
            for (int i = 0; i < dim; i++) raw[i] /= (float) norm;
        }
        return raw;
    }

    /** Murmur3 finalizer: spreads String.hashCode bits so bucket and sign are independent. */
    private static int mix(int h) {
        h ^= h >>> 16;
        h *= 0x85ebca6b;
        h ^= h >>> 13;
        h *= 0xc2b2ae35;
        h ^= h >>> 16;
        return h;
    }
}