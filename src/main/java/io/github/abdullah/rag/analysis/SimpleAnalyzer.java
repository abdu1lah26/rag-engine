package io.github.abdullah.rag.analysis;

import io.github.abdullah.rag.bm25.Tokenizer;

import java.util.List;

/**
 * The analysis we have used so far: lowercase, split on non-letters/digits.
 * No stopwords, no stemming. Kept as the reference point so every later
 * analyzer is measured against 0.6634.
 */
public final class SimpleAnalyzer implements TextAnalyzer {

    @Override
    public String name() {
        return "simple";
    }

    @Override
    public List<String> analyze(String text) {
        return Tokenizer.tokenize(text);
    }
}