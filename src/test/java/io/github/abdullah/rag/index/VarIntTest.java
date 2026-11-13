package io.github.abdullah.rag.index;

import org.junit.jupiter.api.Test;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class VarIntTest {

    private static byte[] encode(int value) throws IOException {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        VarInt.write(out, value);
        return out.toByteArray();
    }

    @Test
    void roundTripsBoundaryValues() throws IOException {
        int[] values = {0, 1, 127, 128, 16383, 16384, 2097151, 2097152, 268435455, 268435456, Integer.MAX_VALUE};
        for (int v : values) {
            assertEquals(v, VarInt.read(ByteBuffer.wrap(encode(v))), "value " + v);
        }
    }

    @Test
    void usesExpectedByteLengths() throws IOException {
        assertEquals(1, encode(0).length);
        assertEquals(1, encode(127).length);
        assertEquals(2, encode(128).length);
        assertEquals(2, encode(16383).length);
        assertEquals(3, encode(16384).length);
        assertEquals(5, encode(Integer.MAX_VALUE).length);
    }

    @Test
    void readsSeveralValuesBackToBack() throws IOException {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        VarInt.write(out, 5);
        VarInt.write(out, 300);
        VarInt.write(out, 70000);
        ByteBuffer buf = ByteBuffer.wrap(out.toByteArray());
        assertEquals(5, VarInt.read(buf));
        assertEquals(300, VarInt.read(buf));
        assertEquals(70000, VarInt.read(buf));
        assertEquals(0, buf.remaining());
    }

    @Test
    void rejectsNegativeValues() {
        assertThrows(IllegalArgumentException.class, () -> encode(-1));
    }

    @Test
    void rejectsOverlongVarint() {
        byte[] bad = {(byte) 0x80, (byte) 0x80, (byte) 0x80, (byte) 0x80, (byte) 0x80, (byte) 0x80};
        assertThrows(IllegalStateException.class, () -> VarInt.read(ByteBuffer.wrap(bad)));
    }
}