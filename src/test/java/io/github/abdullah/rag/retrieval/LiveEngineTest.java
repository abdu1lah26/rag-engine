package io.github.abdullah.rag.retrieval;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.abdullah.rag.analysis.SimpleAnalyzer;
import io.github.abdullah.rag.fusion.RrfFusion;
import io.github.abdullah.rag.retrieval.SegmentedDenseTest.FakeIndex;
import io.github.abdullah.rag.wal.WalPayloads.AddOp;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class LiveEngineTest {

    private static final String[] QUERIES = {"apple", "cherry", "mango zebra", "banana", "date"};

    @TempDir
    Path dir;

    private static LiveEngine open(Path d) throws Exception {
        return new LiveEngine(d, new SimpleAnalyzer(), "m", q -> new float[] {1f, 0f},
                FakeIndex::new, new RrfFusion(60), 50);
    }

    private static AddOp op(String id, String text, float x, float y) {
        return new AddOp(id, "", text, "m", new float[] {x, y});
    }

    /** Final surviving docs: a, c ("cherry zebra"), d. Doc b was deleted. */
    private static void loadScenario(LiveEngine e) throws Exception {
        e.put(op("a", "apple banana", 0.9f, 0.1f));
        e.put(op("b", "banana cherry", 0.5f, 0.5f));
        e.put(op("c", "cherry date", 0.1f, 0.9f));
        e.compact();
        e.put(op("d", "apple mango", 0.8f, 0.2f));
        e.delete("b");
        e.put(op("c", "cherry zebra", 0.2f, 0.8f));
    }

    private static void assertSame(List<ScoredDoc> expected, List<ScoredDoc> actual, String q) {
        assertEquals(expected.size(), actual.size(), "size for " + q);
        for (int i = 0; i < expected.size(); i++) {
            assertEquals(expected.get(i).docId(), actual.get(i).docId(), "id " + i + " for " + q);
            assertEquals(expected.get(i).score(), actual.get(i).score(), 1e-9, "score " + i + " for " + q);
        }
    }

    private static List<String> ids(List<ScoredDoc> r) {
        return r.stream().map(ScoredDoc::docId).toList();
    }

    @Test
    void compactedEngineEqualsOneBuiltFromScratch() throws Exception {
        try (LiveEngine e = open(dir.resolve("live")); LiveEngine ref = open(dir.resolve("ref"))) {
            loadScenario(e);
            e.compact();
            ref.put(op("a", "apple banana", 0.9f, 0.1f));
            ref.put(op("c", "cherry zebra", 0.2f, 0.8f));
            ref.put(op("d", "apple mango", 0.8f, 0.2f));
            for (String q : QUERIES) {
                assertSame(ref.search(q, 10), e.search(q, 10), q);
            }
        }
    }

    @Test
    void deletedDocIsNeverReturnedBeforeOrAfterCompactionOrRestart() throws Exception {
        Path d = dir.resolve("live");
        try (LiveEngine e = open(d)) {
            loadScenario(e);
            for (String q : QUERIES) {
                assertFalse(ids(e.search(q, 10)).contains("b"), "before compaction: " + q);
            }
            e.compact();
            for (String q : QUERIES) {
                assertFalse(ids(e.search(q, 10)).contains("b"), "after compaction: " + q);
            }
        }
        try (LiveEngine e = open(d)) {
            for (String q : QUERIES) {
                assertFalse(ids(e.search(q, 10)).contains("b"), "after restart: " + q);
            }
        }
    }

    @Test
    void restartAfterCompactionGivesIdenticalResults() throws Exception {
        Path d = dir.resolve("live");
        try (LiveEngine e = open(d)) {
            loadScenario(e);
            e.compact();
        }
        try (LiveEngine first = open(d)) {
            try (LiveEngine second = open(d.resolveSibling("copy-unused"))) {
                assertEquals(0, second.generation());
            }
            for (String q : QUERIES) {
                List<ScoredDoc> r = first.search(q, 10);
                assertFalse(ids(r).contains("b"));
                assertEquals(ids(r), ids(first.search(q, 10)));
            }
            assertEquals(2, first.generation());
        }
    }

    @Test
    void writesAfterCompactionAreSearchableAndSurviveRestart() throws Exception {
        Path d = dir.resolve("live");
        try (LiveEngine e = open(d)) {
            loadScenario(e);
            e.compact();
            e.put(op("n", "papaya", 0.3f, 0.7f));
            assertEquals("n", e.search("papaya", 5).get(0).docId());
            e.delete("a");
            assertFalse(ids(e.search("apple", 5)).contains("a"));
        }
        try (LiveEngine e = open(d)) {
            assertEquals("n", e.search("papaya", 5).get(0).docId());
            assertFalse(ids(e.search("apple", 5)).contains("a"));
            assertTrue(ids(e.search("apple", 5)).contains("d"));
        }
    }

    @Test
    void wrongModelIsRejectedAtWriteTime() throws Exception {
        try (LiveEngine e = open(dir.resolve("live"))) {
            assertThrows(IllegalArgumentException.class,
                    () -> e.put(new AddOp("x", "", "txt", "other-model", new float[] {1f, 0f})));
        }
    }
}