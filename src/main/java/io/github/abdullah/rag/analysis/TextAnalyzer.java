package io.github.abdullah.rag.analysis;

import java.util.List;

/**
 * Turns raw text into the terms that get indexed and searched.
 * The same analyzer MUST be used at index time and query time, otherwise
 * "mutations" is indexed as one term and searched as another.
 */
public interface TextAnalyzer {

    /** Short label for reports and for the saved index, e.g. "simple". */
    String name();

    List<String> analyze(String text);
}