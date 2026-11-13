package io.github.abdullah.rag.index;

import java.io.IOException;
import java.io.OutputStream;
import java.nio.ByteBuffer;

/**
 * Variable-length encoding for non-negative ints: 7 data bits per byte, high
 * bit = "more bytes follow". Values below 128 take 1 byte, below 16384 take 2.
 * Most doc-id gaps and term frequencies are small, so this is much smaller
 * than a fixed 4-byte int.
 */
public final class VarInt {

    private VarInt() {
    }

    public static void write(OutputStream out, int value) throws IOException {
        if (value < 0) {
            throw new IllegalArgumentException("varint must be non-negative, got " + value);
        }
        while ((value & ~0x7F) != 0) {
            out.write((value & 0x7F) | 0x80);
            value >>>= 7;
        }
        out.write(value);
    }

    public static int read(ByteBuffer buf) {
        int result = 0;
        int shift = 0;
        while (true) {
            if (shift > 28) {
                throw new IllegalStateException("varint too long: corrupted data");
            }
            byte b = buf.get();
            result |= (b & 0x7F) << shift;
            if ((b & 0x80) == 0) {
                return result;
            }
            shift += 7;
        }
    }
}