package io.github.abdullah.rag.model;

import java.util.Objects;

/**
 * One document in the corpus, as loaded from a BEIR dataset.
 * Immutable: a record gives us final fields, equals/hashCode and toString for
 * free.
 */
public record Document(String id, String title, String text) {

    public Document {
        Objects.requireNonNull(id, "id");
        title = (title == null) ? "" : title;
        text = (text == null) ? "" : text;
    }

    /**
     * The string we index: title and body together, since titles carry strong
     * signal.
     */
    public String indexableText() {
        return title.isEmpty() ? text : title + ". " + text;
    }
}