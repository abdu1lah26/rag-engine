package io.github.abdullah.rag.analysis;

import io.github.abdullah.rag.bm25.Tokenizer;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;

/**
 * SimpleAnalyzer plus stopword removal. The 33-word list is the same one
 * Lucene's EnglishAnalyzer uses by default, so the comparison with
 * lucene-english stays fair.
 * Note: avgDocLength and doc lengths now count only the kept terms.
 */
public final class StopwordAnalyzer implements TextAnalyzer {

    static final Set<String> STOPWORDS = Set.of(
            "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", "into", "is", "it",
            "no", "not", "of", "on", "or", "such", "that", "the", "their", "then", "there", "these",
            "they", "this", "to", "was", "will", "with");

    @Override
    public String name() {
        return "stop";
    }

    @Override
    public List<String> analyze(String text) {
        List<String> kept = new ArrayList<>();
        for (String token : Tokenizer.tokenize(text)) {
            if (!STOPWORDS.contains(token)) {
                kept.add(token);
            }
        }
        return kept;
    }
}