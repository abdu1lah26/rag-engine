package io.github.abdullah.rag.embedding;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.github.abdullah.rag.vector.VectorMath;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

class OnnxEmbedderTest {

    private static final Path MODEL_DIR = Path.of("models", "all-MiniLM-L6-v2");
    private static OnnxEmbedder emb;

    @BeforeAll
    static void load() throws Exception {
        assumeTrue(Files.exists(MODEL_DIR.resolve("model.onnx")), "model not downloaded");
        emb = new OnnxEmbedder(MODEL_DIR, 256);
    }

    @AfterAll
    static void close() throws Exception {
        if (emb != null) {
            emb.close();
        }
    }

    private static float sim(String a, String b) {
        return VectorMath.dot(emb.embed(a), emb.embed(b));
    }

    @Test
    void dimensionIs384() {
        assertEquals(384, emb.dimension());
        assertEquals(384, emb.embed("anything").length);
    }

    @Test
    void outputIsUnitLength() {
        assertEquals(1f, VectorMath.norm(emb.embed("The cat sat on the mat.")), 1e-4f);
    }

    @Test
    void isDeterministic() {
        assertArrayEquals(emb.embed("same input"), emb.embed("same input"), 1e-6f);
    }

    @Test
    void relatedSentencesScoreHigherThanUnrelated() {
        float related = sim("A man is eating food.", "A man is eating a piece of bread.");
        float unrelated = sim("A man is eating food.", "The stock market fell sharply today.");
        assertTrue(related > unrelated + 0.3f, "related=" + related + " unrelated=" + unrelated);
    }

    @Test
    void matchesReferenceSimilaritiesFromSbertDocs() {
        String a = "The weather is lovely today.";
        String b = "It's so sunny outside!";
        String c = "He drove to the stadium.";
        float ab = sim(a, b);
        float ac = sim(a, c);
        System.out.println("sim(a,b)=" + ab + " (reference ~0.666)  sim(a,c)=" + ac
                + " (reference ~0.105)");
        assertEquals(0.666f, ab, 0.02f);
        assertEquals(0.105f, ac, 0.02f);
    }

    @Test
    void handlesEmptyString() {
        float[] v = emb.embed("");
        assertEquals(384, v.length);
        assertEquals(1f, VectorMath.norm(v), 1e-4f);
    }
}