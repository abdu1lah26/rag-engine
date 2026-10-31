package io.github.abdullah.rag.retrieval;

import io.github.abdullah.rag.model.Document;

import java.util.List;

/**
 * The contract every retrieval method implements.
 * Scores are only comparable within one retriever, never across retrievers
 * (BM25 scores and cosine similarities live on different scales).
 */
public interface Retriever {

    /** Short label used in eval reports, e.g. "bm25-naive". */
    String name();

    /** Build the index over the corpus. Called once before any search. */
    void index(List<Document> corpus);

    /** Top-k results for the query, best first. At most k items. */
    List<ScoredDoc> search(String queryText, int k);
}