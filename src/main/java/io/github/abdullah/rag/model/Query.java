package io.github.abdullah.rag.model;

import java.util.Objects;

/**
 * One evaluation question, as loaded from a BEIR dataset's queries.jsonl.
 */
public record Query(String id, String text) {

    public Query {
        Objects.requireNonNull(id, "id");
        Objects.requireNonNull(text, "text");
    }
}