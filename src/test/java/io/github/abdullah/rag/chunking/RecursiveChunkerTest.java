package io.github.abdullah.rag.chunking;

import static org.junit.jupiter.api.Assertions.*;

import java.util.List;
import org.junit.jupiter.api.Test;

class RecursiveChunkerTest {

    private static String words(int n) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < n; i++) sb.append("w").append(i).append(i < n - 1 ? " " : "");
        return sb.toString();
    }

    private static int wordCount(String s) {
        return s.trim().isEmpty() ? 0 : s.trim().split("\\s+").length;
    }

    @Test
    void packsWholeParagraphsAndNeverCutsInsideThem() {
        String text = "a b c\n\nd e f\n\ng h i";
        List<Chunk> chunks = new RecursiveChunker(6, 0).chunk("d", text);
        assertEquals(2, chunks.size());
        assertEquals("a b c\n\nd e f", chunks.get(0).text());
        assertEquals("g h i", chunks.get(1).text());
    }

    @Test
    void fallsBackToSentenceBoundaries() {
        List<Chunk> chunks = new RecursiveChunker(3, 0).chunk("d", "One two three. Four five six.");
        assertEquals(2, chunks.size());
        assertEquals("One two three.", chunks.get(0).text());
        assertEquals("Four five six.", chunks.get(1).text());
    }

    @Test
    void overlapRepeatsTrailingSentence() {
        String text = "a1 a2. b1 b2. c1 c2. d1 d2.";
        List<Chunk> chunks = new RecursiveChunker(4, 2).chunk("d", text);
        assertEquals(3, chunks.size());
        assertEquals("a1 a2. b1 b2.", chunks.get(0).text());
        assertEquals("b1 b2. c1 c2.", chunks.get(1).text());
        assertEquals("c1 c2. d1 d2.", chunks.get(2).text());
    }

    @Test
    void fallsBackToWordsWhenNoPunctuation() {
        // 50 words, size 10, overlap 2 -> step 8 -> starts at 0,8,16,24,32,40
        List<Chunk> chunks = new RecursiveChunker(10, 2).chunk("d", words(50));
        assertEquals(6, chunks.size());
        for (Chunk c : chunks) assertTrue(wordCount(c.text()) <= 10);
        assertEquals("w8", chunks.get(1).text().split(" ")[0]);
    }

    @Test
    void offsetsPointBackIntoOriginalText() {
        String text = "Para one is here. It has two sentences.\n\nPara two follows.\n"
                + "With a line break. And more words here to force splits.";
        List<Chunk> chunks = new RecursiveChunker(6, 2).chunk("d", text);
        assertFalse(chunks.isEmpty());
        for (Chunk c : chunks) assertEquals(c.text(), text.substring(c.startChar(), c.endChar()));
    }

    @Test
    void shortDocGivesOneChunk() {
        List<Chunk> chunks = new RecursiveChunker(200, 40).chunk("d", words(5));
        assertEquals(1, chunks.size());
        assertEquals("d#0", chunks.get(0).chunkId());
    }

    @Test
    void emptyTextGivesNoChunks() {
        assertTrue(new RecursiveChunker(4, 1).chunk("d", "  \n\n ").isEmpty());
    }

    @Test
    void invalidParamsRejected() {
        assertThrows(IllegalArgumentException.class, () -> new RecursiveChunker(0, 0));
        assertThrows(IllegalArgumentException.class, () -> new RecursiveChunker(4, 4));
    }
}