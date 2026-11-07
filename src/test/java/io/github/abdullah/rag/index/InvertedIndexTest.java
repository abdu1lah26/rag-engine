package io.github.abdullah.rag.index;

import io.github.abdullah.rag.model.Document;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class InvertedIndexTest {

    private static InvertedIndex sample() {
        return InvertedIndex.build(List.of(
                new Document("d1", "", "apple banana apple"),
                new Document("d2", "", "banana cherry"),
                new Document("d3", "", "durian")));
    }

    @Test
    void postingsHoldDocIndexAndTermFrequency() {
        InvertedIndex idx = sample();
        assertEquals(List.of(new Posting(0, 2)), idx.postings("apple"));
        assertEquals(List.of(new Posting(0, 1), new Posting(1, 1)), idx.postings("banana"));
    }

    @Test
    void docFreqAndUnknownTerm() {
        InvertedIndex idx = sample();
        assertEquals(2, idx.docFreq("banana"));
        assertEquals(0, idx.docFreq("zebra"));
        assertTrue(idx.postings("zebra").isEmpty());
    }

    @Test
    void statsAreCorrect() {
        InvertedIndex idx = sample();
        assertEquals(3, idx.numDocs());
        assertEquals(4, idx.numTerms());
        assertEquals(3, idx.docLength(0));
        assertEquals(2.0, idx.avgDocLength(), 1e-9);
        assertEquals("d2", idx.docId(1));
    }
}