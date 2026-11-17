package io.github.abdullah.rag.chunking;

import static org.junit.jupiter.api.Assertions.*;

import java.util.List;
import org.junit.jupiter.api.Test;

class FixedSizeChunkerTest {

    private static String words(int n) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < n; i++) sb.append("w").append(i).append(i < n - 1 ? " " : "");
        return sb.toString();
    }

    @Test
    void windowsOverlapCorrectly() {
        // 10 words, size 4, overlap 1 -> step 3 -> windows: 0-3, 3-6, 6-9
        List<Chunk> chunks = new FixedSizeChunker(4, 1).chunk("d", words(10));
        assertEquals(3, chunks.size());
        assertEquals("w0 w1 w2 w3", chunks.get(0).text());
        assertEquals("w3 w4 w5 w6", chunks.get(1).text());
        assertEquals("w6 w7 w8 w9", chunks.get(2).text());
    }

    @Test
    void offsetsPointBackIntoOriginalText() {
        String text = words(10);
        for (Chunk c : new FixedSizeChunker(4, 1).chunk("d", text)) {
            assertEquals(c.text(), text.substring(c.startChar(), c.endChar()));
        }
    }

    @Test
    void shortDocGivesOneChunk() {
        List<Chunk> chunks = new FixedSizeChunker(200, 40).chunk("d", words(5));
        assertEquals(1, chunks.size());
        assertEquals("d#0", chunks.get(0).chunkId());
    }

    @Test
    void emptyTextGivesNoChunks() {
        assertTrue(new FixedSizeChunker(4, 1).chunk("d", "   ").isEmpty());
    }

    @Test
    void noRedundantTailChunk() {
        // 8 words, size 4, overlap 1 -> windows 0-3, 3-6, 6-7; last must not repeat
        List<Chunk> chunks = new FixedSizeChunker(4, 1).chunk("d", words(8));
        assertEquals(3, chunks.size());
        assertEquals("w6 w7", chunks.get(2).text());
    }

    @Test
    void invalidParamsRejected() {
        assertThrows(IllegalArgumentException.class, () -> new FixedSizeChunker(0, 0));
        assertThrows(IllegalArgumentException.class, () -> new FixedSizeChunker(4, 4));
    }
}