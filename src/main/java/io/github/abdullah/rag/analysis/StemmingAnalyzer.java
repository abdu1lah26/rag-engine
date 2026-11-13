package io.github.abdullah.rag.analysis;

import io.github.abdullah.rag.bm25.Tokenizer;

import java.util.ArrayList;
import java.util.List;

/**
 * Lowercase + split, drop stopwords, then Porter-stem what is left.
 * Stopwords are removed BEFORE stemming, same order as Lucene's EnglishAnalyzer.
 */
public final class StemmingAnalyzer implements TextAnalyzer {

    @Override
    public String name() {
        return "stop-stem";
    }

    @Override
    public List<String> analyze(String text) {
        List<String> terms = new ArrayList<>();
        for (String token : Tokenizer.tokenize(text)) {
            if (!StopwordAnalyzer.STOPWORDS.contains(token)) {
                terms.add(PorterStemmer.stem(token));
            }
        }
        return terms;
    }
}