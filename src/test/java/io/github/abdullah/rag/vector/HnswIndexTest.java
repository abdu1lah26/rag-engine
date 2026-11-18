package io.github.abdullah.rag.vector;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;
import java.util.List;

class HnswIndexTest {

    @Test
    void randomLevelIsNeverNegative() {
        HnswIndex idx = new HnswIndex(8, HnswConfig.defaults());
        for (int i = 0; i < 100_000; i++) {
            assertTrue(idx.randomLevel() >= 0);
        }
    }

    @Test
    void randomLevelFollowsGeometricDistribution() {
        // With M=16, P(level >= 1) should be about 1/16 = 0.0625,
        // and P(level >= 2) about 1/256 = 0.0039.
        HnswIndex idx = new HnswIndex(8, new HnswConfig(16, 200, 7L));
        int n = 200_000;
        int atLeast1 = 0;
        int atLeast2 = 0;
        for (int i = 0; i < n; i++) {
            int level = idx.randomLevel();
            if (level >= 1) atLeast1++;
            if (level >= 2) atLeast2++;
        }
        double p1 = atLeast1 / (double) n;
        double p2 = atLeast2 / (double) n;
        assertEquals(1.0 / 16, p1, 0.005);
        assertEquals(1.0 / 256, p2, 0.002);
    }

    @Test
    void sameSeedGivesSameLevels() {
        HnswIndex a = new HnswIndex(8, new HnswConfig(16, 200, 99L));
        HnswIndex b = new HnswIndex(8, new HnswConfig(16, 200, 99L));
        for (int i = 0; i < 1000; i++) {
            assertEquals(a.randomLevel(), b.randomLevel());
        }
    }

        /** Five nodes at 0, 20, 40, 60, 80 degrees on the unit circle, linked as a chain. */
    private static HnswIndex chain() {
        HnswIndex idx = new HnswIndex(2, HnswConfig.defaults());
        for (int i = 0; i < 5; i++) {
            double a = Math.toRadians(20 * i);
            int slot = idx.addRawNode("n" + i,
                    new float[] {(float) Math.cos(a), (float) Math.sin(a)}, 0);
            if (i > 0) {
                idx.link(i - 1, slot, 0);
            }
        }
        return idx;
    }

    private static float[] atDegrees(double deg) {
        double a = Math.toRadians(deg);
        return new float[] {(float) Math.cos(a), (float) Math.sin(a)};
    }

    @Test
    void searchLayerWalksTheChainToTheNearestNode() {
        HnswIndex idx = chain();
        // Query at 85 degrees, entering at node 0 (far end). Must hop 0->1->2->3->4.
        List<HnswIndex.Candidate> r = idx.searchLayer(atDegrees(85), 0, 3, 0);

        assertEquals(3, r.size());
        assertEquals(4, r.get(0).slot());
        assertEquals(3, r.get(1).slot());
        assertEquals(2, r.get(2).slot());
    }

    @Test
    void searchLayerWithEf1IsGreedyAndStillArrives() {
        HnswIndex idx = chain();
        List<HnswIndex.Candidate> r = idx.searchLayer(atDegrees(85), 0, 1, 0);

        assertEquals(1, r.size());
        assertEquals(4, r.get(0).slot());
    }

    @Test
    void searchLayerSurvivesCycles() {
        HnswIndex idx = chain();
        idx.link(0, 4, 0); // close the chain into a ring
        List<HnswIndex.Candidate> r = idx.searchLayer(atDegrees(85), 0, 5, 0);

        assertEquals(5, r.size()); // each node exactly once, and it terminated
        assertEquals(4, r.get(0).slot());
    }

    @Test
    void searchLayerResultsAreSortedNearestFirst() {
        HnswIndex idx = chain();
        List<HnswIndex.Candidate> r = idx.searchLayer(atDegrees(30), 0, 5, 0);

        for (int i = 1; i < r.size(); i++) {
            assertTrue(r.get(i - 1).dist() <= r.get(i).dist());
        }
    }

        private static HnswIndex.Candidate cand(HnswIndex idx, float[] base, double deg, String id) {
        float[] v = atDegrees(deg);
        int slot = idx.addRawNode(id, v, 0);
        return new HnswIndex.Candidate(slot, VectorMath.distance(base, v));
    }

    @Test
    void selectNeighboursSkipsCoveredCandidateAndKeepsDiverseOne() {
        HnswIndex idx = new HnswIndex(2, HnswConfig.defaults());
        float[] base = atDegrees(0);
        HnswIndex.Candidate a = cand(idx, base, 10, "A");
        HnswIndex.Candidate b = cand(idx, base, 12, "B");
        HnswIndex.Candidate c = cand(idx, base, -60, "C");

        List<Integer> chosen = idx.selectNeighbours(List.of(a, b, c), 2);

        assertEquals(List.of(a.slot(), c.slot()), chosen);
    }

    @Test
    void selectNeighboursTopsUpWithDiscardedWhenRoomRemains() {
        HnswIndex idx = new HnswIndex(2, HnswConfig.defaults());
        float[] base = atDegrees(0);
        HnswIndex.Candidate a = cand(idx, base, 10, "A");
        HnswIndex.Candidate b = cand(idx, base, 12, "B");
        HnswIndex.Candidate c = cand(idx, base, -60, "C");

        List<Integer> chosen = idx.selectNeighbours(List.of(a, b, c), 3);

        // A and C pass the heuristic, B is topped up afterwards.
        assertEquals(List.of(a.slot(), c.slot(), b.slot()), chosen);
    }

    @Test
    void selectNeighboursReturnsAllWhenFewerThanMax() {
        HnswIndex idx = new HnswIndex(2, HnswConfig.defaults());
        float[] base = atDegrees(0);
        HnswIndex.Candidate a = cand(idx, base, 10, "A");
        HnswIndex.Candidate c = cand(idx, base, -60, "C");

        assertEquals(2, idx.selectNeighbours(List.of(a, c), 8).size());
    }

    @Test
    void selectNeighboursNeverExceedsMax() {
        HnswIndex idx = new HnswIndex(2, HnswConfig.defaults());
        float[] base = atDegrees(0);
        List<HnswIndex.Candidate> all = new java.util.ArrayList<>();
        for (int i = 1; i <= 10; i++) {
            all.add(cand(idx, base, i * 15, "n" + i));
        }

        assertEquals(4, idx.selectNeighbours(all, 4).size());
    }
}