package io.github.abdullah.rag.embedding;

import static org.junit.jupiter.api.Assertions.*;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class EmbeddingCacheTest {

    @TempDir
    Path tmp;

    private final List<String> ids = List.of("d1", "d2", "d3");
    private final List<float[]> vectors = List.of(
            new float[] {0.1f, 0.2f, 0.3f},
            new float[] {-1f, 0f, 1f},
            new float[] {0.5f, 0.5f, 0.5f});

    @Test
    void roundTripPreservesEverything() throws IOException {
        Path file = tmp.resolve("cache.bin");
        EmbeddingCache.save(file, "test-model", 3, ids, vectors);

        EmbeddingCache.Loaded loaded = EmbeddingCache.load(file);

        assertEquals("test-model", loaded.modelName());
        assertEquals(3, loaded.dimension());
        assertEquals(ids, loaded.ids());
        assertEquals(3, loaded.vectors().size());
        for (int i = 0; i < 3; i++) {
            assertArrayEquals(vectors.get(i), loaded.vectors().get(i), 0f);
        }
    }

    @Test
    void createsMissingParentDirectories() throws IOException {
        Path file = tmp.resolve("a").resolve("b").resolve("cache.bin");
        EmbeddingCache.save(file, "m", 3, ids, vectors);
        assertTrue(Files.exists(file));
    }

    @Test
    void loadForRejectsDifferentModel() throws IOException {
        Path file = tmp.resolve("cache.bin");
        EmbeddingCache.save(file, "model-a", 3, ids, vectors);
        assertThrows(IOException.class, () -> EmbeddingCache.loadFor(file, "model-b", 3));
    }

    @Test
    void loadForRejectsDifferentDimension() throws IOException {
        Path file = tmp.resolve("cache.bin");
        EmbeddingCache.save(file, "model-a", 3, ids, vectors);
        assertThrows(IOException.class, () -> EmbeddingCache.loadFor(file, "model-a", 384));
    }

    @Test
    void loadForAcceptsMatchingModel() throws IOException {
        Path file = tmp.resolve("cache.bin");
        EmbeddingCache.save(file, "model-a", 3, ids, vectors);
        assertEquals(3, EmbeddingCache.loadFor(file, "model-a", 3).ids().size());
    }

    @Test
    void loadRejectsGarbageFile() throws IOException {
        Path file = tmp.resolve("junk.bin");
        Files.writeString(file, "this is not a cache file at all");
        assertThrows(IOException.class, () -> EmbeddingCache.load(file));
    }

    @Test
    void saveRejectsWrongVectorDimension() {
        Path file = tmp.resolve("cache.bin");
        assertThrows(IllegalArgumentException.class,
                () -> EmbeddingCache.save(file, "m", 4, ids, vectors));
    }

    @Test
    void saveRejectsMismatchedListSizes() {
        Path file = tmp.resolve("cache.bin");
        assertThrows(IllegalArgumentException.class,
                () -> EmbeddingCache.save(file, "m", 3, ids, vectors.subList(0, 2)));
    }
}
