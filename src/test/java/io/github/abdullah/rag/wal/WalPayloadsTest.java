package io.github.abdullah.rag.wal;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import io.github.abdullah.rag.wal.WalPayloads.AddOp;
import java.util.Arrays;
import org.junit.jupiter.api.Test;

class WalPayloadsTest {

    private static AddOp sample() {
        return new AddOp("doc-42", "T\u00edtulo", "na\u00efve \u03b1\u03b2\u03b3 \u65e5\u672c\u8a9e \uD83D\uDE00 text",
                "all-MiniLM-L6-v2", new float[] {0.25f, -1.5f, 3.0e-7f, 0f});
    }

    @Test
    void addRoundTrip() {
        AddOp op = sample();
        AddOp back = WalPayloads.decodeAdd(WalPayloads.encodeAdd(op));
        assertEquals(op.docId(), back.docId());
        assertEquals(op.title(), back.title());
        assertEquals(op.text(), back.text());
        assertEquals(op.model(), back.model());
        assertArrayEquals(op.vector(), back.vector());
    }

    @Test
    void emptyStringsAndEmptyVectorRoundTrip() {
        AddOp op = new AddOp("", "", "", "", new float[0]);
        AddOp back = WalPayloads.decodeAdd(WalPayloads.encodeAdd(op));
        assertEquals("", back.docId());
        assertEquals(0, back.vector().length);
    }

    @Test
    void deleteRoundTrip() {
        assertEquals("doc-\u65e5\u672c", WalPayloads.decodeDelete(WalPayloads.encodeDelete("doc-\u65e5\u672c")));
    }

    @Test
    void everyProperPrefixOfAddIsRejected() {
        byte[] full = WalPayloads.encodeAdd(sample());
        for (int len = 0; len < full.length; len++) {
            byte[] prefix = Arrays.copyOf(full, len);
            assertThrows(IllegalArgumentException.class, () -> WalPayloads.decodeAdd(prefix), "prefix " + len);
        }
    }

    @Test
    void trailingBytesAreRejected() {
        byte[] full = WalPayloads.encodeAdd(sample());
        byte[] extra = Arrays.copyOf(full, full.length + 1);
        assertThrows(IllegalArgumentException.class, () -> WalPayloads.decodeAdd(extra));

        byte[] del = WalPayloads.encodeDelete("x");
        byte[] delExtra = Arrays.copyOf(del, del.length + 1);
        assertThrows(IllegalArgumentException.class, () -> WalPayloads.decodeDelete(delExtra));
    }

    @Test
    void hugeLengthFieldIsRejectedWithoutAllocating() {
        byte[] bogus = {0x7f, (byte) 0xff, (byte) 0xff, (byte) 0xff};
        assertThrows(IllegalArgumentException.class, () -> WalPayloads.decodeDelete(bogus));
    }
}