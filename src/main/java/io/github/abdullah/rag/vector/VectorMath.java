package io.github.abdullah.rag.vector;

/**
 * Basic vector operations shared by the brute-force index and HNSW.
 * All vectors are assumed L2-normalised (see Embedder), so cosine
 * similarity equals the dot product.
 */
public final class VectorMath {

    private VectorMath() {
    }

    /** Dot product. For unit vectors this is the cosine similarity. */
    public static float dot(float[] a, float[] b) {
        if (a.length != b.length) {
            throw new IllegalArgumentException(
                    "dimension mismatch: " + a.length + " vs " + b.length);
        }
        float sum = 0f;
        for (int i = 0; i < a.length; i++) {
            sum += a[i] * b[i];
        }
        return sum;
    }

    /** Distance used by the indexes: smaller = closer. Range [0, 2] for unit vectors. */
    public static float distance(float[] a, float[] b) {
        return 1f - dot(a, b);
    }

    /** L2 norm (length) of a vector. */
    public static float norm(float[] v) {
        return (float) Math.sqrt(dot(v, v));
    }

    /** Returns a unit-length copy of v. Throws on the zero vector. */
    public static float[] normalize(float[] v) {
        float n = norm(v);
        if (n == 0f) {
            throw new IllegalArgumentException("cannot normalise the zero vector");
        }
        float[] out = new float[v.length];
        for (int i = 0; i < v.length; i++) {
            out[i] = v[i] / n;
        }
        return out;
    }
}