package io.github.abdullah.rag.bm25;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.abdullah.rag.analysis.SimpleAnalyzer;
import io.github.abdullah.rag.analysis.TextAnalyzer;
import io.github.abdullah.rag.index.IndexedBm25Retriever;
import io.github.abdullah.rag.index.InvertedIndex;
import io.github.abdullah.rag.model.Document;
import io.github.abdullah.rag.retrieval.ScoredDoc;
import java.util.HashSet;
import java.util.List;
import java.util.stream.Collectors;
import org.junit.jupiter.api.Test;

class SegmentedBm25Test {

    private static final TextAnalyzer AN = new SimpleAnalyzer();

    private static final List<Document> CORPUS = List.of(
            new Document("d1", "", "apple banana"),
            new Document("d2", "", "apple apple cherry"),
            new Document("d3", "", "banana cherry date"),
            new Document("d4", "", "date elder fig"),
            new Document("d5", "", "fig grape apple"),
            new Document("d6", "", "grape grape grape banana"));

    private static final String[] QUERIES = {"apple", "banana cherry", "grape fig", "zzz"};

    private static InvertedIndex index(List<Document> docs) {
        return InvertedIndex.build(docs, AN);
    }

    private static void assertSameRanking(List<ScoredDoc> expected, List<ScoredDoc> actual, String q) {
        assertEquals(expected.size(), actual.size(), "size for " + q);
        for (int i = 0; i < expected.size(); i++) {
            assertEquals(expected.get(i).score(), actual.get(i).score(), 1e-9, "score " + i + " for " + q);
        }
        assertEquals(
                expected.stream().map(ScoredDoc::docId).collect(Collectors.toSet()),
                actual.stream().map(ScoredDoc::docId).collect(Collectors.toSet()),
                "ids for " + q);
    }

    @Test
    void wholeCorpusInBaseMatchesIndexedBm25() {
        IndexedBm25Retriever ref = new IndexedBm25Retriever(AN);
        ref.index(CORPUS);
        SegmentedBm25 seg = new SegmentedBm25(AN);
        InvertedIndex base = index(CORPUS);
        InvertedIndex emptyDelta = index(List.of());
        for (String q : QUERIES) {
            assertSameRanking(ref.search(q, 10), seg.search(base, emptyDelta, id -> true, q, 10), q);
        }
    }

    @Test
    void splitAcrossBaseAndDeltaMatchesIndexedBm25() {
        IndexedBm25Retriever ref = new IndexedBm25Retriever(AN);
        ref.index(CORPUS);
        SegmentedBm25 seg = new SegmentedBm25(AN);
        InvertedIndex base = index(CORPUS.subList(0, 3));
        InvertedIndex delta = index(CORPUS.subList(3, 6));
        for (String q : QUERIES) {
            assertSameRanking(ref.search(q, 10), seg.search(base, delta, id -> true, q, 10), q);
        }
    }

    @Test
    void hiddenBaseDocIsNeverReturned() {
        SegmentedBm25 seg = new SegmentedBm25(AN);
        InvertedIndex base = index(CORPUS);
        InvertedIndex emptyDelta = index(List.of());
        List<ScoredDoc> res = seg.search(base, emptyDelta, id -> !id.equals("d6"), "grape", 10);
        assertFalse(res.stream().anyMatch(d -> d.docId().equals("d6")));
        assertTrue(res.stream().anyMatch(d -> d.docId().equals("d5")));
    }

    @Test
    void updatedDocIsFoundByNewTextOnlyAndOnlyOnce() {
        SegmentedBm25 seg = new SegmentedBm25(AN);
        InvertedIndex base = index(CORPUS);
        InvertedIndex delta = index(List.of(new Document("d6", "", "cherry")));
        java.util.function.Predicate<String> visible = id -> !id.equals("d6");

        List<ScoredDoc> grape = seg.search(base, delta, visible, "grape", 10);
        assertFalse(grape.stream().anyMatch(d -> d.docId().equals("d6")), "old text must not match");

        List<ScoredDoc> cherry = seg.search(base, delta, visible, "cherry", 10);
        long d6Hits = cherry.stream().filter(d -> d.docId().equals("d6")).count();
        assertEquals(1, d6Hits);
        assertEquals(cherry.size(), new HashSet<>(cherry.stream().map(ScoredDoc::docId).toList()).size());
    }
}