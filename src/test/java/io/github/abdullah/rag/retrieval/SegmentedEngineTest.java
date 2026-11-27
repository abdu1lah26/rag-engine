package io.github.abdullah.rag.retrieval;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.abdullah.rag.analysis.SimpleAnalyzer;
import io.github.abdullah.rag.analysis.TextAnalyzer;
import io.github.abdullah.rag.fusion.RrfFusion;
import io.github.abdullah.rag.index.InvertedIndex;
import io.github.abdullah.rag.model.Document;
import io.github.abdullah.rag.retrieval.SegmentedDenseTest.FakeIndex;
import io.github.abdullah.rag.wal.DurableStore;
import io.github.abdullah.rag.wal.WalPayloads.AddOp;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class SegmentedEngineTest {

    private static final TextAnalyzer AN = new SimpleAnalyzer();

    @TempDir
    Path dir;

    private static FakeIndex baseVectors() {
        FakeIndex v = new FakeIndex();
        v.add("a", new float[] {0.9f, 0.1f});
        v.add("b", new float[] {0.5f, 0.5f});
        v.add("c", new float[] {0.1f, 0.9f});
        return v;
    }

    private static InvertedIndex baseBm25() {
        return InvertedIndex.build(List.of(
                new Document("a", "", "apple banana"),
                new Document("b", "", "banana cherry"),
                new Document("c", "", "cherry date")), AN);
    }

    private SegmentedEngine engine(DurableStore store) {
        return new SegmentedEngine(baseBm25(), baseVectors(), AN, "m", q -> new float[] {1f, 0f},
                store, new RrfFusion(60), 50);
    }

    private static List<String> ids(List<ScoredDoc> r) {
        return r.stream().map(ScoredDoc::docId).toList();
    }

    @Test
    void untouchedBaseWorks() throws Exception {
        try (DurableStore s = DurableStore.open(dir.resolve("s"))) {
            assertEquals("a", engine(s).search("apple", 3).get(0).docId());
        }
    }

    @Test
    void deletedDocIsGoneFromBothRankings() throws Exception {
        try (DurableStore s = DurableStore.open(dir.resolve("s"))) {
            SegmentedEngine e = engine(s);
            s.delete("a");
            List<String> r = ids(e.search("apple", 5));
            assertFalse(r.contains("a"));
            assertTrue(r.containsAll(List.of("b", "c")));
        }
    }

    @Test
    void updatedDocAppearsOnceAndIsFoundByNewText() throws Exception {
        try (DurableStore s = DurableStore.open(dir.resolve("s"))) {
            SegmentedEngine e = engine(s);
            s.put(new AddOp("a", "", "zebra", "m", new float[] {0f, 1f}));
            assertEquals(1, ids(e.search("apple", 5)).stream().filter(x -> x.equals("a")).count());
            assertEquals("a", e.search("zebra", 5).get(0).docId());
        }
    }

    @Test
    void newDocIsSearchableImmediately() throws Exception {
        try (DurableStore s = DurableStore.open(dir.resolve("s"))) {
            SegmentedEngine e = engine(s);
            assertFalse(ids(e.search("mango", 5)).contains("n"));
            s.put(new AddOp("n", "", "mango", "m", new float[] {0.2f, 0.8f}));
            assertEquals("n", e.search("mango", 5).get(0).docId());
        }
    }

    @Test
    void afterRestartResultsAreIdentical() throws Exception {
        Path d = dir.resolve("s");
        List<ScoredDoc> before;
        try (DurableStore s = DurableStore.open(d)) {
            s.put(new AddOp("n", "", "mango apple", "m", new float[] {0.95f, 0.05f}));
            s.delete("b");
            s.put(new AddOp("a", "", "zebra apple", "m", new float[] {0.3f, 0.7f}));
            before = engine(s).search("apple", 5);
        }
        try (DurableStore s = DurableStore.open(d)) {
            List<ScoredDoc> after = engine(s).search("apple", 5);
            assertEquals(before.size(), after.size());
            for (int i = 0; i < before.size(); i++) {
                assertEquals(before.get(i).docId(), after.get(i).docId());
                assertEquals(before.get(i).score(), after.get(i).score(), 1e-12);
            }
            assertFalse(ids(after).contains("b"));
        }
    }
}