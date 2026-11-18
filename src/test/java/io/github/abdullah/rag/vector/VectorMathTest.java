package io.github.abdullah.rag.vector;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;

class VectorMathTest {

    private static final float EPS = 1e-6f;

    @Test
    void dotOfOrthogonalVectorsIsZero() {
        assertEquals(0f, VectorMath.dot(new float[] {1, 0}, new float[] {0, 1}), EPS);
    }

    @Test
    void dotComputesSumOfProducts() {
        assertEquals(32f, VectorMath.dot(new float[] {1, 2, 3}, new float[] {4, 5, 6}), EPS);
    }

    @Test
    void dotRejectsDimensionMismatch() {
        assertThrows(IllegalArgumentException.class,
                () -> VectorMath.dot(new float[] {1, 2}, new float[] {1, 2, 3}));
    }

    @Test
    void distanceOfIdenticalUnitVectorsIsZero() {
        float[] v = VectorMath.normalize(new float[] {3, 4});
        assertEquals(0f, VectorMath.distance(v, v), EPS);
    }

    @Test
    void distanceOfOppositeUnitVectorsIsTwo() {
        assertEquals(2f, VectorMath.distance(new float[] {1, 0}, new float[] {-1, 0}), EPS);
    }

    @Test
    void normalizeProducesUnitLength() {
        float[] v = VectorMath.normalize(new float[] {3, 4});
        assertEquals(1f, VectorMath.norm(v), EPS);
        assertEquals(0.6f, v[0], EPS);
        assertEquals(0.8f, v[1], EPS);
    }

    @Test
    void normalizeRejectsZeroVector() {
        assertThrows(IllegalArgumentException.class,
                () -> VectorMath.normalize(new float[] {0, 0, 0}));
    }
}