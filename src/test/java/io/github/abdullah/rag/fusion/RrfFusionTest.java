package io.github.abdullah.rag.fusion;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.abdullah.rag.retrieval.ScoredDoc;
import java.util.List;
import org.junit.jupiter.api.Test;

class RrfFusionTest {

    private static final double EPS = 1e-12;

    private static ScoredDoc d(String id) {
        // The input score is irrelevant to RRF, so use a junk value on purpose.
        return new ScoredDoc(id, 999.0);
    }

    @Test
    void twoLists_handComputed() {
        // k = 60. A: a, b, c    B: b, d, a
        // a = 1/61 + 1/63, b = 1/62 + 1/61, c = 1/63, d = 1/62
        List<ScoredDoc> a = List.of(d("a"), d("b"), d("c"));
        List<ScoredDoc> b = List.of(d("b"), d("d"), d("a"));
        List<ScoredDoc> out = new RrfFusion(60).fuse(List.of(a, b), 10);

        assertEquals(List.of("b", "a", "d", "c"), out.stream().map(ScoredDoc::docId).toList());
        assertEquals(1.0 / 62 + 1.0 / 61, out.get(0).score(), EPS);
        assertEquals(1.0 / 61 + 1.0 / 63, out.get(1).score(), EPS);
        assertEquals(1.0 / 62, out.get(2).score(), EPS);
        assertEquals(1.0 / 63, out.get(3).score(), EPS);
    }

    @Test
    void truncatesToK() {
        List<ScoredDoc> a = List.of(d("a"), d("b"), d("c"));
        List<ScoredDoc> out = new RrfFusion(60).fuse(List.of(a), 2);
        assertEquals(List.of("a", "b"), out.stream().map(ScoredDoc::docId).toList());
    }

    @Test
    void duplicateInOneListCountsOnce() {
        // a is repeated; the second copy is ignored and b keeps rank 2.
        List<ScoredDoc> a = List.of(d("a"), d("a"), d("b"));
        List<ScoredDoc> out = new RrfFusion(60).fuse(List.of(a), 10);
        assertEquals(2, out.size());
        assertEquals(1.0 / 61, out.get(0).score(), EPS);
        assertEquals(1.0 / 62, out.get(1).score(), EPS);
    }

    @Test
    void tiesAreBrokenByDocIdRegardlessOfInputOrder() {
        RrfFusion f = new RrfFusion(60);
        List<ScoredDoc> xy = f.fuse(List.of(List.of(d("x")), List.of(d("y"))), 10);
        List<ScoredDoc> yx = f.fuse(List.of(List.of(d("y")), List.of(d("x"))), 10);
        assertEquals(List.of("x", "y"), xy.stream().map(ScoredDoc::docId).toList());
        assertEquals(List.of("x", "y"), yx.stream().map(ScoredDoc::docId).toList());
    }

    @Test
    void emptyInputGivesEmptyOutput() {
        RrfFusion f = new RrfFusion(60);
        assertTrue(f.fuse(List.of(), 10).isEmpty());
        assertTrue(f.fuse(List.of(List.of(), List.of()), 10).isEmpty());
    }

    @Test
    void rejectsBadArguments() {
        assertThrows(IllegalArgumentException.class, () -> new RrfFusion(0));
        assertThrows(IllegalArgumentException.class, () -> new RrfFusion(60).fuse(List.of(), -1));
    }
}