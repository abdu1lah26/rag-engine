package io.github.abdullah.rag.vector;

import static org.junit.jupiter.api.Assertions.*;

import io.github.abdullah.rag.retrieval.ScoredDoc;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;
import org.junit.jupiter.api.Test;

class HnswIndexSearchTest {

    private static float[] randomUnit(Random rnd, int dim) {
        float[] v = new float[dim];
        for (int i = 0; i < dim; i++) {
            v[i] = (float) rnd.nextGaussian();
        }
        return VectorMath.normalize(v);
    }

    private static double recallAtK(List<ScoredDoc> truth, List<ScoredDoc> got, int k) {
        Set<String> t = new HashSet<>();
        for (int i = 0; i < Math.min(k, truth.size()); i++) {
            t.add(truth.get(i).docId());
        }
        int hit = 0;
        for (int i = 0; i < Math.min(k, got.size()); i++) {
            if (t.contains(got.get(i).docId())) {
                hit++;
            }
        }
        return hit / (double) Math.max(1, t.size());
    }

    @Test
    void emptyIndexReturnsEmpty() {
        HnswIndex idx = new HnswIndex(4, HnswConfig.defaults());
        assertTrue(idx.search(randomUnit(new Random(1), 4), 5).isEmpty());
    }

    @Test
    void singleNodeIsFound() {
        HnswIndex idx = new HnswIndex(4, HnswConfig.defaults());
        float[] v = randomUnit(new Random(1), 4);
        idx.add("only", v);

        List<ScoredDoc> r = idx.search(v, 5);

        assertEquals(1, r.size());
        assertEquals("only", r.get(0).docId());
        assertEquals(1.0, r.get(0).score(), 1e-5);
    }

    @Test
    void exactMatchIsRankedFirst() {
        Random rnd = new Random(2);
        HnswIndex idx = new HnswIndex(16, HnswConfig.defaults());
        float[] target = null;
        for (int i = 0; i < 500; i++) {
            float[] v = randomUnit(rnd, 16);
            if (i == 123) target = v;
            idx.add("v" + i, v);
        }
        idx.setEfSearch(100);

        assertEquals("v123", idx.search(target, 5).get(0).docId());
    }

    @Test
    void rejectsDuplicateIdAndWrongDimension() {
        HnswIndex idx = new HnswIndex(4, HnswConfig.defaults());
        Random rnd = new Random(3);
        idx.add("a", randomUnit(rnd, 4));
        assertThrows(IllegalArgumentException.class, () -> idx.add("a", randomUnit(rnd, 4)));
        assertThrows(IllegalArgumentException.class, () -> idx.add("b", randomUnit(rnd, 3)));
        assertThrows(IllegalArgumentException.class, () -> idx.search(randomUnit(rnd, 3), 1));
    }

    @Test
    void smallIndexWithLargeEfMatchesBruteForceExactly() {
        // With ef >= n the beam can reach every reachable node, so results must be exact.
        Random rnd = new Random(4);
        int dim = 16;
        HnswIndex hnsw = new HnswIndex(dim, new HnswConfig(8, 100, 42L));
        BruteForceIndex exact = new BruteForceIndex(dim);
        for (int i = 0; i < 100; i++) {
            float[] v = randomUnit(rnd, dim);
            hnsw.add("v" + i, v);
            exact.add("v" + i, v);
        }
        hnsw.setEfSearch(100);

        for (int q = 0; q < 20; q++) {
            float[] query = randomUnit(rnd, dim);
            List<ScoredDoc> truth = exact.search(query, 10);
            List<ScoredDoc> got = hnsw.search(query, 10);
            for (int i = 0; i < 10; i++) {
                assertEquals(truth.get(i).docId(), got.get(i).docId());
            }
        }
    }

    @Test
    void recallAt10IsHighOnRandomData() {
        Random rnd = new Random(5);
        int dim = 32;
        int n = 3000;
        HnswIndex hnsw = new HnswIndex(dim, new HnswConfig(16, 200, 42L));
        BruteForceIndex exact = new BruteForceIndex(dim);
        for (int i = 0; i < n; i++) {
            float[] v = randomUnit(rnd, dim);
            hnsw.add("v" + i, v);
            exact.add("v" + i, v);
        }
        hnsw.setEfSearch(100);

        double total = 0;
        int queries = 100;
        for (int q = 0; q < queries; q++) {
            float[] query = randomUnit(rnd, dim);
            total += recallAtK(exact.search(query, 10), hnsw.search(query, 10), 10);
        }
        double recall = total / queries;
        System.out.println("recall@10 (n=3000, d=32, M=16, efC=200, efS=100) = " + recall);
        assertTrue(recall >= 0.90, "recall too low: " + recall);
    }

    @Test
    void higherEfSearchDoesNotLowerRecall() {
        Random rnd = new Random(6);
        int dim = 32;
        HnswIndex hnsw = new HnswIndex(dim, new HnswConfig(8, 100, 42L));
        BruteForceIndex exact = new BruteForceIndex(dim);
        for (int i = 0; i < 2000; i++) {
            float[] v = randomUnit(rnd, dim);
            hnsw.add("v" + i, v);
            exact.add("v" + i, v);
        }
        float[][] queries = new float[50][];
        for (int i = 0; i < queries.length; i++) {
            queries[i] = randomUnit(rnd, dim);
        }

        double low = avgRecall(hnsw, exact, queries, 10);
        double high = avgRecall(hnsw, exact, queries, 200);
        System.out.println("recall@10 efSearch=10: " + low + "  efSearch=200: " + high);
        assertTrue(high >= low);
    }

    private static double avgRecall(HnswIndex hnsw, BruteForceIndex exact, float[][] qs, int ef) {
        hnsw.setEfSearch(ef);
        double total = 0;
        for (float[] q : qs) {
            total += recallAtK(exact.search(q, 10), hnsw.search(q, 10), 10);
        }
        return total / qs.length;
    }
}