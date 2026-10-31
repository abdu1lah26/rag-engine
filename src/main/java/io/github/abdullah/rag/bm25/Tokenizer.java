package io.github.abdullah.rag.bm25;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * Naive tokenizer: lowercase, then split on every run of non-letter, non-digit
 * characters.
 * "Hello, World-2024!" -> [hello, world, 2024]
 */
public final class Tokenizer {

    private Tokenizer() {
    }

    public static List<String> tokenize(String text) {
        List<String> tokens = new ArrayList<>();
        if (text == null || text.isEmpty()) {
            return tokens;
        }
        String lower = text.toLowerCase(Locale.ROOT);
        int start = -1;
        for (int i = 0; i < lower.length(); i++) {
            boolean wordChar = Character.isLetterOrDigit(lower.charAt(i));
            if (wordChar && start < 0) {
                start = i;
            } else if (!wordChar && start >= 0) {
                tokens.add(lower.substring(start, i));
                start = -1;
            }
        }
        if (start >= 0) {
            tokens.add(lower.substring(start));
        }
        return tokens;
    }
}