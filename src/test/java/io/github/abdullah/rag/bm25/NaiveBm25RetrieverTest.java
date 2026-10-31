package io.github.abdullah.rag.bm25;

import io.github.abdullah.rag.model.Document;
import io.github.abdullah.rag.retrieval.ScoredDoc;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class NaiveBm25RetrieverTest {

    private static final double EPS = 1e-9;

    // d1: "cat dog" (len 2), d2: "cat cat bird" (len 3), d3: "fish bird" (len 2)
    // N = 3, average length = 7/3
    private static NaiveBm25Retriever indexed() {
        NaiveBm25Retriever r = new NaiveBm25Retriever();
        r.index(List.of(
                new Document("d1", "", "cat dog"),
                new Document("d2", "", "cat cat bird"),
                new Document("d3", "", "fish bird")));
        return r;
    }

    @Test
    void higherTermFrequencyRanksHigher() {
        List<ScoredDoc> results = indexed().search("cat", 10);

        assertEquals(2, results.size(), "d3 has no 'cat' and must not appear");
        assertEquals("d2", results.get(0).docId()); // tf=2
        assertEquals("d1", results.get(1).docId()); // tf=1
    }

    @Test
    void scoreMatchesHandComputedBm25() {
        // query "cat": df = 2, so idf = ln(1 + (3 - 2 + 0.5) / (2 + 0.5))
        double idf = Math.log(1.0 + (3 - 2 + 0.5) / (2 + 0.5));
        // d1: tf = 1, docLen = 2
        double lengthNorm = 1.0 - 0.75 + 0.75 * 2 / (7.0 / 3);
        double expected = idf * 1 * (1.2 + 1.0) / (1 + 1.2 * lengthNorm);

        ScoredDoc d1 = indexed().search("cat", 10).stream()
                .filter(r -> r.docId().equals("d1"))
                .findFirst().orElseThrow();

        assertEquals(expected, d1.score(), EPS);
    }

    @Test
    void rareTermOutweighsCommonTerm() {
        // "dog" appears in 1 doc (rare), "cat" in 2 docs; d1 matches both
        List<ScoredDoc> results = indexed().search("cat dog", 10);

        assertEquals("d1", results.get(0).docId());
    }

    @Test
    void unknownOrEmptyQueryReturnsNothing() {
        NaiveBm25Retriever r = indexed();
        assertTrue(r.search("zebra", 10).isEmpty());
        assertTrue(r.search("", 10).isEmpty());
    }

    @Test
    void respectsK() {
        List<ScoredDoc> results = indexed().search("cat", 1);

        assertEquals(1, results.size());
        assertEquals("d2", results.get(0).docId());
    }
}