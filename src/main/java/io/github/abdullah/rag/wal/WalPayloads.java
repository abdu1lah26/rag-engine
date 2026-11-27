package io.github.abdullah.rag.wal;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.util.Objects;

/** Byte layouts for the payload of ADD and DELETE records. */
public final class WalPayloads {

    private WalPayloads() {}

    /** A document to add, or to replace if docId already exists. Carries its embedding. */
    public record AddOp(String docId, String title, String text, String model, float[] vector) {
        public AddOp {
            Objects.requireNonNull(docId, "docId");
            Objects.requireNonNull(title, "title");
            Objects.requireNonNull(text, "text");
            Objects.requireNonNull(model, "model");
            Objects.requireNonNull(vector, "vector");
            vector = vector.clone();
        }
    }

    // ADD layout: str docId | str title | str text | str model | int dim | float[dim]
    // str = int byteLength + UTF-8 bytes
    public static byte[] encodeAdd(AddOp op) {
        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        DataOutputStream out = new DataOutputStream(bos);
        try {
            writeString(out, op.docId());
            writeString(out, op.title());
            writeString(out, op.text());
            writeString(out, op.model());
            float[] v = op.vector();
            out.writeInt(v.length);
            for (float f : v) {
                out.writeFloat(f);
            }
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
        return bos.toByteArray();
    }

    public static AddOp decodeAdd(byte[] payload) {
        ByteArrayInputStream bin = new ByteArrayInputStream(payload);
        DataInputStream in = new DataInputStream(bin);
        try {
            String docId = readString(in, bin);
            String title = readString(in, bin);
            String text = readString(in, bin);
            String model = readString(in, bin);
            int dim = in.readInt();
            if (dim < 0 || (long) dim * 4 > bin.available()) {
                throw new IllegalArgumentException("bad vector dimension " + dim);
            }
            float[] vector = new float[dim];
            for (int i = 0; i < dim; i++) {
                vector[i] = in.readFloat();
            }
            if (bin.available() != 0) {
                throw new IllegalArgumentException("trailing bytes in ADD payload");
            }
            return new AddOp(docId, title, text, model, vector);
        } catch (IOException e) {
            throw new IllegalArgumentException("malformed ADD payload", e);
        }
    }

    // DELETE layout: str docId
    public static byte[] encodeDelete(String docId) {
        Objects.requireNonNull(docId, "docId");
        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        try {
            writeString(new DataOutputStream(bos), docId);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
        return bos.toByteArray();
    }

    public static String decodeDelete(byte[] payload) {
        ByteArrayInputStream bin = new ByteArrayInputStream(payload);
        try {
            String docId = readString(new DataInputStream(bin), bin);
            if (bin.available() != 0) {
                throw new IllegalArgumentException("trailing bytes in DELETE payload");
            }
            return docId;
        } catch (IOException e) {
            throw new IllegalArgumentException("malformed DELETE payload", e);
        }
    }

    private static void writeString(DataOutputStream out, String s) throws IOException {
        byte[] b = s.getBytes(StandardCharsets.UTF_8);
        out.writeInt(b.length);
        out.write(b);
    }

    private static String readString(DataInputStream in, ByteArrayInputStream bin) throws IOException {
        int len = in.readInt();
        if (len < 0 || len > bin.available()) {
            throw new IllegalArgumentException("bad string length " + len);
        }
        byte[] b = new byte[len];
        in.readFully(b);
        return new String(b, StandardCharsets.UTF_8);
    }
}