package io.github.abdullah.rag.index;

import io.github.abdullah.rag.bm25.NaiveBm25Retriever;
import io.github.abdullah.rag.model.Document;
import io.github.abdullah.rag.retrieval.ScoredDoc;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class IndexedBm25RetrieverTest {

    private static final List<Document> CORPUS = List.of(
            new Document("d1", "Fruit", "apple banana apple"),
            new Document("d2", "", "banana cherry banana banana"),
            new Document("d3", "Exotic", "durian smells strongly"),
            new Document("d4", "", "apple pie with cherry filling"),
            new Document("d5", "", "nothing relevant here at all"));

    private static final List<String> QUERIES = List.of(
            "apple", "banana cherry", "apple apple pie", "durian", "cherry banana apple", "zebra", "");

    @Test
    void matchesNaiveRetrieverOnEveryQuery() {
        NaiveBm25Retriever naive = new NaiveBm25Retriever();
        IndexedBm25Retriever indexed = new IndexedBm25Retriever();
        naive.index(CORPUS);
        indexed.index(CORPUS);

        for (String q : QUERIES) {
            List<ScoredDoc> a = naive.search(q, 10);
            List<ScoredDoc> b = indexed.search(q, 10);
            assertEquals(a.size(), b.size(), "result count for query: " + q);
            for (int i = 0; i < a.size(); i++) {
                assertEquals(a.get(i).docId(), b.get(i).docId(), "doc at rank " + i + " for query: " + q);
                assertEquals(a.get(i).score(), b.get(i).score(), 1e-9, "score at rank " + i + " for query: " + q);
            }
        }
    }

    @Test
    void respectsK() {
        IndexedBm25Retriever indexed = new IndexedBm25Retriever();
        indexed.index(CORPUS);
        assertEquals(2, indexed.search("apple banana cherry", 2).size());
    }

    @Test
    void unknownQueryReturnsEmpty() {
        IndexedBm25Retriever indexed = new IndexedBm25Retriever();
        indexed.index(CORPUS);
        assertTrue(indexed.search("zebra", 5).isEmpty());
    }

    @Test
    void rejectsNonPositiveK() {
        IndexedBm25Retriever indexed = new IndexedBm25Retriever();
        indexed.index(CORPUS);
        assertThrows(IllegalArgumentException.class, () -> indexed.search("apple", 0));
    }
}