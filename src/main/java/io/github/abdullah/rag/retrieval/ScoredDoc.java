package io.github.abdullah.rag.retrieval;

import java.util.Objects;

/**
 * One search result: a document id and the retriever's score for it (higher =
 * better).
 */
public record ScoredDoc(String docId, double score) {

    public ScoredDoc {
        Objects.requireNonNull(docId, "docId");
    }
}