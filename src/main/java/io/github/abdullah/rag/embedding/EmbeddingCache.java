package io.github.abdullah.rag.embedding;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Binary on-disk cache of (id, vector) pairs produced by one embedding model.
 *
 * Layout: magic int | model name (UTF) | dimension | count | then per entry:
 * id (UTF) followed by `dimension` floats.
 */
public final class EmbeddingCache {

    private static final int MAGIC = 0x454D4244; // "EMBD"

    /** Everything read back from a cache file. */
    public record Loaded(String modelName, int dimension, List<String> ids, List<float[]> vectors) {
    }

    private EmbeddingCache() {
    }

    public static void save(Path file, String modelName, int dimension,
            List<String> ids, List<float[]> vectors) throws IOException {
        if (ids.size() != vectors.size()) {
            throw new IllegalArgumentException("ids and vectors must have the same size");
        }
        if (file.getParent() != null) {
            Files.createDirectories(file.getParent());
        }
        try (DataOutputStream out = new DataOutputStream(
                new BufferedOutputStream(Files.newOutputStream(file)))) {
            out.writeInt(MAGIC);
            out.writeUTF(modelName);
            out.writeInt(dimension);
            out.writeInt(ids.size());
            for (int i = 0; i < ids.size(); i++) {
                float[] v = vectors.get(i);
                if (v.length != dimension) {
                    throw new IllegalArgumentException(
                            "vector " + ids.get(i) + " has dimension " + v.length);
                }
                out.writeUTF(ids.get(i));
                for (float f : v) {
                    out.writeFloat(f);
                }
            }
        }
    }

    public static Loaded load(Path file) throws IOException {
        try (DataInputStream in = new DataInputStream(
                new BufferedInputStream(Files.newInputStream(file)))) {
            if (in.readInt() != MAGIC) {
                throw new IOException("not an embedding cache file: " + file);
            }
            String modelName = in.readUTF();
            int dimension = in.readInt();
            int count = in.readInt();
            List<String> ids = new ArrayList<>(count);
            List<float[]> vectors = new ArrayList<>(count);
            for (int i = 0; i < count; i++) {
                ids.add(in.readUTF());
                float[] v = new float[dimension];
                for (int d = 0; d < dimension; d++) {
                    v[d] = in.readFloat();
                }
                vectors.add(v);
            }
            return new Loaded(modelName, dimension, ids, vectors);
        }
    }

    /** Like load, but fails loudly if the file was produced by a different model. */
    public static Loaded loadFor(Path file, String expectedModelName, int expectedDimension)
            throws IOException {
        Loaded loaded = load(file);
        if (!loaded.modelName().equals(expectedModelName)
                || loaded.dimension() != expectedDimension) {
            throw new IOException("cache " + file + " was built with " + loaded.modelName()
                    + "/" + loaded.dimension() + " but expected " + expectedModelName
                    + "/" + expectedDimension + "; delete it and re-embed");
        }
        return loaded;
    }
}