package io.github.abdullah.rag.retrieval;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;

import io.github.abdullah.rag.vector.VectorIndex;
import io.github.abdullah.rag.wal.WalPayloads.AddOp;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.junit.jupiter.api.Test;

class SegmentedDenseTest {

    /** Exact brute-force stand-in for HnswIndex, so the test isolates the segment logic. */
    static final class FakeIndex implements VectorIndex {
        private final Map<String, float[]> vecs = new LinkedHashMap<>();
        public String name() { return "fake"; }
        public int dimension() { return 2; }
        public int size() { return vecs.size(); }
        public void add(String id, float[] v) { vecs.put(id, v); }
        public List<ScoredDoc> search(float[] q, int k) {
            List<ScoredDoc> out = new ArrayList<>();
            vecs.forEach((id, v) -> out.add(new ScoredDoc(id, v[0] * q[0] + v[1] * q[1])));
            out.sort(Comparator.comparingDouble(ScoredDoc::score).reversed().thenComparing(ScoredDoc::docId));
            return out.size() <= k ? out : new ArrayList<>(out.subList(0, k));
        }
    }

    private static final float[] Q = {1f, 0f};

    private static FakeIndex base() {
        FakeIndex b = new FakeIndex();
        b.add("a", new float[] {0.9f, 0.1f});
        b.add("b", new float[] {0.8f, 0.2f});
        b.add("c", new float[] {0.7f, 0.3f});
        b.add("d", new float[] {0.1f, 0.9f});
        return b;
    }

    private static AddOp op(String id, float x, float y) {
        return new AddOp(id, "t", "txt", "m", new float[] {x, y});
    }

    private static List<String> ids(List<ScoredDoc> r) {
        return r.stream().map(ScoredDoc::docId).toList();
    }

    @Test
    void noChangesMatchesBase() {
        SegmentedDense sd = new SegmentedDense("m");
        List<ScoredDoc> r = sd.search(base(), List.of(), id -> true, 0, Q, 3);
        assertEquals(List.of("a", "b", "c"), ids(r));
    }

    @Test
    void deletedNearestDocIsNeverReturned() {
        SegmentedDense sd = new SegmentedDense("m");
        List<ScoredDoc> r = sd.search(base(), List.of(), id -> !id.equals("a"), 1, Q, 3);
        assertFalse(ids(r).contains("a"));
        assertEquals(List.of("b", "c", "d"), ids(r));
    }

    @Test
    void overFetchKeepsResultsFullWhenTopHitsAreAllHidden() {
        SegmentedDense sd = new SegmentedDense("m");
        Set<String> hidden = Set.of("a", "b", "c");
        List<ScoredDoc> r = sd.search(base(), List.of(), id -> !hidden.contains(id), hidden.size(), Q, 1);
        assertEquals(List.of("d"), ids(r));
    }

    @Test
    void withoutOverFetchTheSameQueryWouldReturnNothing() {
        SegmentedDense sd = new SegmentedDense("m");
        Set<String> hidden = Set.of("a", "b", "c");
        List<ScoredDoc> r = sd.search(base(), List.of(), id -> !hidden.contains(id), 0, Q, 1);
        assertEquals(0, r.size()); // documents why hiddenCount is needed
    }

    @Test
    void updatedDocUsesNewVectorOnlyAndAppearsOnce() {
        SegmentedDense sd = new SegmentedDense("m");
        // "d" was far from Q in base; its update moves it right next to Q.
        List<ScoredDoc> r = sd.search(base(), List.of(op("d", 1f, 0f)), id -> !id.equals("d"), 1, Q, 4);
        assertEquals("d", r.get(0).docId());
        assertEquals(1, ids(r).stream().filter(x -> x.equals("d")).count());
    }

    @Test
    void newDocInDeltaCompetesWithBase() {
        SegmentedDense sd = new SegmentedDense("m");
        List<ScoredDoc> r = sd.search(base(), List.of(op("new", 0.85f, 0.15f)), id -> true, 0, Q, 3);
        assertEquals(List.of("a", "new", "b"), ids(r));
    }

    @Test
    void emptyBaseWorks() {
        SegmentedDense sd = new SegmentedDense("m");
        List<ScoredDoc> r = sd.search(new FakeIndex(), List.of(op("x", 1f, 0f)), id -> true, 0, Q, 5);
        assertEquals(List.of("x"), ids(r));
    }

    @Test
    void wrongModelIsRejected() {
        SegmentedDense sd = new SegmentedDense("m");
        AddOp other = new AddOp("x", "t", "txt", "other-model", new float[] {1f, 0f});
        assertThrows(IllegalStateException.class,
                () -> sd.search(base(), List.of(other), id -> true, 0, Q, 3));
    }

    @Test
    void wrongDimensionIsRejected() {
        SegmentedDense sd = new SegmentedDense("m");
        AddOp bad = new AddOp("x", "t", "txt", "m", new float[] {1f, 0f, 0f});
        assertThrows(IllegalArgumentException.class,
                () -> sd.search(base(), List.of(bad), id -> true, 0, Q, 3));
    }
}