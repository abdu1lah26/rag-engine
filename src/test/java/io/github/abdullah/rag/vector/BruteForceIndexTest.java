package io.github.abdullah.rag.vector;

import static org.junit.jupiter.api.Assertions.*;

import io.github.abdullah.rag.retrieval.ScoredDoc;
import java.util.List;
import java.util.Random;
import org.junit.jupiter.api.Test;

class BruteForceIndexTest {

    private static float[] unit(float... v) {
        return VectorMath.normalize(v);
    }

    @Test
    void returnsNearestFirst() {
        BruteForceIndex idx = new BruteForceIndex(2);
        idx.add("east", unit(1, 0));
        idx.add("northeast", unit(1, 1));
        idx.add("north", unit(0, 1));

        List<ScoredDoc> r = idx.search(unit(1, 0.1f), 3);

        assertEquals(3, r.size());
        assertEquals("east", r.get(0).docId());
        assertEquals("northeast", r.get(1).docId());
        assertEquals("north", r.get(2).docId());
    }

    @Test
    void respectsK() {
        BruteForceIndex idx = new BruteForceIndex(2);
        idx.add("a", unit(1, 0));
        idx.add("b", unit(0, 1));
        idx.add("c", unit(-1, 0));

        assertEquals(2, idx.search(unit(1, 0), 2).size());
    }

    @Test
    void kLargerThanSizeReturnsEverything() {
        BruteForceIndex idx = new BruteForceIndex(2);
        idx.add("a", unit(1, 0));
        idx.add("b", unit(0, 1));

        assertEquals(2, idx.search(unit(1, 0), 10).size());
    }

    @Test
    void emptyIndexReturnsEmpty() {
        BruteForceIndex idx = new BruteForceIndex(2);
        assertTrue(idx.search(unit(1, 0), 5).isEmpty());
    }

    @Test
    void scoreIsCosineSimilarity() {
        BruteForceIndex idx = new BruteForceIndex(2);
        idx.add("same", unit(1, 0));
        idx.add("opposite", unit(-1, 0));

        List<ScoredDoc> r = idx.search(unit(1, 0), 2);

        assertEquals(1.0, r.get(0).score(), 1e-6);
        assertEquals(-1.0, r.get(1).score(), 1e-6);
    }

    @Test
    void rejectsDuplicateId() {
        BruteForceIndex idx = new BruteForceIndex(2);
        idx.add("a", unit(1, 0));
        assertThrows(IllegalArgumentException.class, () -> idx.add("a", unit(0, 1)));
    }

    @Test
    void rejectsWrongDimension() {
        BruteForceIndex idx = new BruteForceIndex(3);
        assertThrows(IllegalArgumentException.class, () -> idx.add("a", unit(1, 0)));
        assertThrows(IllegalArgumentException.class, () -> idx.search(unit(1, 0), 1));
    }

    @Test
    void matchesFullSortOnRandomData() {
        // The heap-based top-k must equal "score everything, sort, take k".
        Random rnd = new Random(42);
        int dim = 16;
        BruteForceIndex idx = new BruteForceIndex(dim);
        float[][] data = new float[200][];
        for (int i = 0; i < data.length; i++) {
            data[i] = randomUnit(rnd, dim);
            idx.add("v" + i, data[i]);
        }
        float[] q = randomUnit(rnd, dim);

        List<ScoredDoc> got = idx.search(q, 10);

        List<ScoredDoc> expected = new java.util.ArrayList<>();
        for (int i = 0; i < data.length; i++) {
            expected.add(new ScoredDoc("v" + i, VectorMath.dot(q, data[i])));
        }
        expected.sort((a, b) -> Double.compare(b.score(), a.score()));

        for (int i = 0; i < 10; i++) {
            assertEquals(expected.get(i).docId(), got.get(i).docId());
        }
    }

    private static float[] randomUnit(Random rnd, int dim) {
        float[] v = new float[dim];
        for (int i = 0; i < dim; i++) {
            v[i] = (float) rnd.nextGaussian();
        }
        return VectorMath.normalize(v);
    }
}