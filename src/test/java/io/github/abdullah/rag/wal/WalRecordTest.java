package io.github.abdullah.rag.wal;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;

import io.github.abdullah.rag.wal.WalRecord.DecodeResult;
import io.github.abdullah.rag.wal.WalRecord.Status;
import io.github.abdullah.rag.wal.WalRecord.Type;
import java.nio.charset.StandardCharsets;
import org.junit.jupiter.api.Test;

class WalRecordTest {

    private static WalRecord sample() {
        return new WalRecord(42L, Type.ADD, "hello wal".getBytes(StandardCharsets.UTF_8));
    }

    @Test
    void roundTrip() {
        WalRecord r = sample();
        byte[] bytes = r.encode();
        DecodeResult res = WalRecord.decode(bytes, 0, bytes.length);
        assertEquals(Status.OK, res.status());
        assertEquals(r, res.record());
        assertEquals(bytes.length, res.size());
        assertEquals(r.encodedSize(), bytes.length);
    }

    @Test
    void emptyPayloadRoundTrips() {
        WalRecord r = new WalRecord(0L, Type.CHECKPOINT, new byte[0]);
        byte[] bytes = r.encode();
        DecodeResult res = WalRecord.decode(bytes, 0, bytes.length);
        assertEquals(Status.OK, res.status());
        assertEquals(r, res.record());
    }

    @Test
    void emptyInputIsEnd() {
        assertEquals(Status.END, WalRecord.decode(new byte[0], 0, 0).status());
    }

    @Test
    void everyProperPrefixIsTruncated() {
        byte[] bytes = sample().encode();
        for (int len = 1; len < bytes.length; len++) {
            assertEquals(Status.TRUNCATED, WalRecord.decode(bytes, 0, len).status(), "prefix length " + len);
        }
    }

    @Test
    void anySingleBitFlipIsRejected() {
        byte[] original = sample().encode();
        for (int i = 0; i < original.length; i++) {
            for (int bit = 0; bit < 8; bit++) {
                byte[] copy = original.clone();
                copy[i] ^= (byte) (1 << bit);
                assertNotEquals(Status.OK, WalRecord.decode(copy, 0, copy.length).status(),
                        "flip byte " + i + " bit " + bit);
            }
        }
    }

    @Test
    void twoRecordsBackToBack() {
        WalRecord a = new WalRecord(1L, Type.ADD, new byte[] {1, 2, 3});
        WalRecord b = new WalRecord(2L, Type.DELETE, new byte[] {9});
        byte[] ab = a.encode();
        byte[] bb = b.encode();
        byte[] all = new byte[ab.length + bb.length];
        System.arraycopy(ab, 0, all, 0, ab.length);
        System.arraycopy(bb, 0, all, ab.length, bb.length);

        DecodeResult first = WalRecord.decode(all, 0, all.length);
        DecodeResult second = WalRecord.decode(all, first.size(), all.length);
        DecodeResult third = WalRecord.decode(all, first.size() + second.size(), all.length);

        assertEquals(a, first.record());
        assertEquals(b, second.record());
        assertEquals(Status.END, third.status());
    }
}