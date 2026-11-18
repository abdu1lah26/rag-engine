package io.github.abdullah.rag.vector;

/**
 * Build-time parameters for HNSW.
 *
 * @param m              max neighbours per node on layers above 0 (layer 0 allows 2*m)
 * @param efConstruction beam width used while inserting; higher = better graph, slower build
 * @param seed           seed for level assignment, so builds are reproducible
 *
 * efSearch is deliberately NOT here: it is a query-time knob that must be
 * changeable without rebuilding the index (needed for the recall-vs-latency sweep).
 */
public record HnswConfig(int m, int efConstruction, long seed) {

    public HnswConfig {
        if (m < 2) {
            throw new IllegalArgumentException("m must be >= 2");
        }
        if (efConstruction < m) {
            throw new IllegalArgumentException("efConstruction must be >= m");
        }
    }

    /** Max neighbours on layer 0. */
    public int maxNeighboursLayer0() {
        return 2 * m;
    }

    /** Level normalisation factor from the paper: mL = 1 / ln(M). */
    public double levelMultiplier() {
        return 1.0 / Math.log(m);
    }

    /** Sensible defaults: M=16, efConstruction=200, fixed seed. */
    public static HnswConfig defaults() {
        return new HnswConfig(16, 200, 42L);
    }
}