package io.github.abdullah.rag.embedding;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

class WordPieceTokenizerTest {

    private static final Path VOCAB = Path.of("models", "all-MiniLM-L6-v2", "vocab.txt");
    private static WordPieceTokenizer tok;

    @BeforeAll
    static void load() throws IOException {
        // The model files are gitignored, so skip (not fail) where they are absent, e.g. CI.
        assumeTrue(Files.exists(VOCAB), "vocab.txt not downloaded");
        tok = new WordPieceTokenizer(VOCAB, 256);
    }

    private static long[] ids(String text) {
        return tok.encode(text).inputIds();
    }

    @Test
    void helloWorldMatchesKnownIds() {
        assertArrayEquals(new long[] {101, 7592, 2088, 102}, ids("hello world"));
    }

    @Test
    void punctuationBecomesSeparateTokens() {
        assertArrayEquals(new long[] {101, 7592, 1010, 2088, 1012, 102}, ids("Hello, world."));
    }

    @Test
    void lowercasesInput() {
        assertArrayEquals(ids("hello world"), ids("HELLO WORLD"));
    }

    @Test
    void stripsAccents() {
        assertArrayEquals(ids("hello"), ids("h\u00e9llo"));
    }

    @Test
    void collapsesExtraWhitespace() {
        assertArrayEquals(ids("hello world"), ids("  hello \t\n  world  "));
    }

    @Test
    void emptyTextGivesOnlyClsAndSep() {
        assertArrayEquals(new long[] {101, 102}, ids(""));
    }

    @Test
    void overlongWordBecomesUnk() {
        assertArrayEquals(new long[] {101, 100, 102}, ids("a".repeat(101)));
    }

    @Test
    void longWordsAreSplitIntoSubwordPieces() {
        // A rare word must split into several pieces, never collapse to one [UNK].
        long[] out = ids("electroencephalography");
        assertTrue(out.length > 3, "expected multiple wordpieces");
        for (long id : out) {
            assertNotEquals(100L, id);
        }
    }

    @Test
    void truncatesToMaxLengthAndKeepsSepAtTheEnd() throws IOException {
        WordPieceTokenizer small = new WordPieceTokenizer(VOCAB, 8);
        long[] out = small.encode("the quick brown fox jumps over the lazy dog again and again").inputIds();
        assertEquals(8, out.length);
        assertEquals(101L, out[0]);
        assertEquals(102L, out[7]);
    }

    @Test
    void maskAndTypeIdsMatchInputLength() {
        WordPieceTokenizer.Encoding e = tok.encode("hello world");
        assertEquals(e.inputIds().length, e.attentionMask().length);
        assertEquals(e.inputIds().length, e.tokenTypeIds().length);
        for (long m : e.attentionMask()) {
            assertEquals(1L, m);
        }
        for (long t : e.tokenTypeIds()) {
            assertEquals(0L, t);
        }
    }
}