package io.github.abdullah.rag.chunking;

import static org.junit.jupiter.api.Assertions.*;

import io.github.abdullah.rag.analysis.SimpleAnalyzer;
import io.github.abdullah.rag.analysis.TextAnalyzer;
import org.junit.jupiter.api.Test;

class HashedTfEmbedderTest {

        // no stemming or stopwords, keeps this test independent of the stemmer
    private static final TextAnalyzer SIMPLE = new SimpleAnalyzer();
    private final HashedTfEmbedder emb = new HashedTfEmbedder(SIMPLE, 1024);

    private static double dot(float[] a, float[] b) {
        double s = 0;
        for (int i = 0; i < a.length; i++) s += (double) a[i] * b[i];
        return s;
    }

    @Test
    void vectorHasFixedDimensionAndUnitLength() {
        float[] v = emb.embed("insulin regulates blood glucose levels");
        assertEquals(1024, v.length);
        assertEquals(1.0, dot(v, v), 1e-5);
    }

    @Test
    void identicalTextGivesCosineOne() {
        float[] a = emb.embed("gene expression in tumor cells");
        float[] b = emb.embed("gene expression in tumor cells");
        assertEquals(1.0, dot(a, b), 1e-6);
    }

    @Test
    void sharedVocabularyScoresHigherThanDisjoint() {
        float[] q = emb.embed("insulin regulates blood glucose");
        float[] close = emb.embed("blood glucose is regulated by insulin");
        float[] far = emb.embed("the stadium hosted a football final");
        assertTrue(dot(q, close) > dot(q, far) + 0.1);
    }

    @Test
    void emptyTextGivesZeroVector() {
        float[] v = emb.embed("   ");
        assertEquals(0.0, dot(v, v), 0.0);
    }

    @Test
    void deterministicAcrossInstances() {
        float[] a = new HashedTfEmbedder(SIMPLE, 256).embed("same input");
        float[] b = new HashedTfEmbedder(SIMPLE, 256).embed("same input");
        assertArrayEquals(a, b);
    }

    @Test
    void invalidDimRejected() {
        assertThrows(IllegalArgumentException.class, () -> new HashedTfEmbedder(SIMPLE, 0));
    }
}