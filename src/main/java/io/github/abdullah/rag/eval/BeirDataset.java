package io.github.abdullah.rag.eval;

import io.github.abdullah.rag.model.Document;
import io.github.abdullah.rag.model.Qrels;
import io.github.abdullah.rag.model.Query;

import java.util.List;

/**
 * A loaded BEIR dataset split: the documents to search, the questions to ask,
 * and the answer key. Queries are already filtered to those that have judgments
 * in the chosen split, so every query here can be scored.
 */
public record BeirDataset(String name, List<Document> corpus, List<Query> queries, Qrels qrels) {

    public BeirDataset {
        corpus = List.copyOf(corpus);
        queries = List.copyOf(queries);
    }

    @Override
    public String toString() {
        return name + " [docs=" + corpus.size()
                + ", queries=" + queries.size()
                + ", judged queries=" + qrels.size() + "]";
    }
}