package io.github.abdullah.rag.chunking;

import static org.junit.jupiter.api.Assertions.*;

import io.github.abdullah.rag.analysis.SimpleAnalyzer;
import java.util.List;
import org.junit.jupiter.api.Test;

class SemanticChunkerTest {

    private final Embedder emb = new HashedTfEmbedder(new SimpleAnalyzer(), 1024);

    // three sentences about glucose, then three about football: one clear topic shift
    private static final String TWO_TOPICS =
            "insulin regulates blood glucose levels. "
          + "blood glucose levels rise after meals. "
          + "insulin lowers blood glucose levels. "
          + "the football stadium hosted a final. "
          + "the football final drew a stadium crowd. "
          + "the stadium crowd cheered the football final.";

    private static int wordCount(String s) {
        return s.trim().isEmpty() ? 0 : s.trim().split("\\s+").length;
    }

    @Test
    void splitsAtTopicShift() {
        List<Chunk> chunks = new SemanticChunker(emb, 1, 5, 100, 0.8).chunk("d", TWO_TOPICS);
        assertEquals(2, chunks.size());
        assertTrue(chunks.get(0).text().startsWith("insulin regulates"));
        assertTrue(chunks.get(1).text().startsWith("the football"));
        assertFalse(chunks.get(0).text().contains("football"));
        assertFalse(chunks.get(1).text().contains("glucose"));
    }

    @Test
    void minWordsSuppressesEarlyCut() {
        // first topic is 16 words, below minWords 20, so the boundary is skipped
        List<Chunk> chunks = new SemanticChunker(emb, 1, 20, 100, 0.8).chunk("d", TWO_TOPICS);
        assertEquals(1, chunks.size());
    }

    @Test
    void maxWordsForcesSplitEvenWithoutTopicShift() {
        String s = "alpha beta gamma delta epsilon. ";
        String text = (s + s + s + s + s + s).trim(); // 6 identical 5-word sentences
        // alpha=100 disables semantic cuts, so only the hard cap splits: 2 sentences per chunk
        List<Chunk> chunks = new SemanticChunker(emb, 1, 1, 12, 100).chunk("d", text);
        assertEquals(3, chunks.size());
        for (Chunk c : chunks) assertTrue(wordCount(c.text()) <= 12);
    }

    @Test
    void offsetsPointBackIntoOriginalText() {
        for (Chunk c : new SemanticChunker(emb, 1, 5, 100, 0.8).chunk("d", TWO_TOPICS)) {
            assertEquals(c.text(), TWO_TOPICS.substring(c.startChar(), c.endChar()));
        }
    }

    @Test
    void chunkIdsAreSequential() {
        List<Chunk> chunks = new SemanticChunker(emb, 1, 5, 100, 0.8).chunk("doc9", TWO_TOPICS);
        assertEquals("doc9#0", chunks.get(0).chunkId());
        assertEquals("doc9#1", chunks.get(1).chunkId());
    }

    @Test
    void singleSentenceGivesOneChunk() {
        List<Chunk> chunks = new SemanticChunker(emb, 1, 5, 100, 0.8).chunk("d", "just one sentence here.");
        assertEquals(1, chunks.size());
    }

    @Test
    void emptyTextGivesNoChunks() {
        assertTrue(new SemanticChunker(emb, 1, 5, 100, 0.8).chunk("d", "   ").isEmpty());
    }

    @Test
    void invalidParamsRejected() {
        assertThrows(IllegalArgumentException.class, () -> new SemanticChunker(emb, 0, 5, 100, 0.8));
        assertThrows(IllegalArgumentException.class, () -> new SemanticChunker(emb, 1, 50, 10, 0.8));
        assertThrows(IllegalArgumentException.class, () -> new SemanticChunker(emb, 1, 5, 100, -1));
        assertThrows(IllegalArgumentException.class, () -> new SemanticChunker(null, 1, 5, 100, 0.8));
    }
}