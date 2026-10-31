package io.github.abdullah.rag.bm25;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class TokenizerTest {

    @Test
    void lowercasesAndSplitsOnPunctuation() {
        assertEquals(List.of("hello", "world", "2024"), Tokenizer.tokenize("Hello, World-2024!"));
    }

    @Test
    void collapsesRepeatedSeparators() {
        assertEquals(List.of("a", "b", "c"), Tokenizer.tokenize("  a  ,, b\t\nc  "));
    }

    @Test
    void handlesEmptyAndNull() {
        assertTrue(Tokenizer.tokenize("").isEmpty());
        assertTrue(Tokenizer.tokenize(null).isEmpty());
        assertTrue(Tokenizer.tokenize("   ...  ").isEmpty());
    }

    @Test
    void keepsLastTokenWithoutTrailingSeparator() {
        assertEquals(List.of("brca1", "mutation"), Tokenizer.tokenize("BRCA1 mutation"));
    }

    @Test
    void splitsHyphenatedTermsAndKeepsDigits() {
        assertEquals(List.of("covid", "19", "il", "6"), Tokenizer.tokenize("COVID-19 IL-6"));
    }
}