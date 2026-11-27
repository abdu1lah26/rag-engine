package io.github.abdullah.rag.wal;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.abdullah.rag.wal.WalReader.ScanResult;
import io.github.abdullah.rag.wal.WalRecord.Status;
import io.github.abdullah.rag.wal.WalRecord.Type;
import java.io.ByteArrayOutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class WalReaderTest {

    @TempDir
    Path dir;

    private static byte[] concat(WalRecord... records) {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        for (WalRecord r : records) {
            out.writeBytes(r.encode());
        }
        return out.toByteArray();
    }

    @Test
    void missingFileIsEmptyLog() throws Exception {
        ScanResult res = WalReader.scan(dir.resolve("nope.log"));
        assertEquals(0, res.records().size());
        assertEquals(Status.END, res.stop());
    }

    @Test
    void crashAtEveryByteOffsetKeepsExactlyTheCompleteRecords() throws Exception {
        WalRecord a = new WalRecord(1L, Type.ADD, new byte[] {1, 2, 3});
        WalRecord b = new WalRecord(2L, Type.DELETE, new byte[] {4});
        WalRecord c = new WalRecord(3L, Type.ADD, new byte[] {5, 6, 7, 8, 9});
        byte[] full = concat(a, b, c);
        long end1 = a.encodedSize();
        long end2 = end1 + b.encodedSize();
        long end3 = end2 + c.encodedSize();
        Path file = dir.resolve("wal.log");

        for (int cut = 0; cut <= full.length; cut++) {
            Files.write(file, Arrays.copyOf(full, cut));
            ScanResult res = WalReader.scan(file);

            int expectedRecords = cut >= end3 ? 3 : cut >= end2 ? 2 : cut >= end1 ? 1 : 0;
            long expectedValid = cut >= end3 ? end3 : cut >= end2 ? end2 : cut >= end1 ? end1 : 0;
            boolean onBoundary = cut == 0 || cut == end1 || cut == end2 || cut == end3;

            assertEquals(expectedRecords, res.records().size(), "records at cut " + cut);
            assertEquals(expectedValid, res.validBytes(), "validBytes at cut " + cut);
            assertEquals(onBoundary ? Status.END : Status.TRUNCATED, res.stop(), "stop at cut " + cut);
        }
    }

    @Test
    void corruptMiddleRecordStopsTheScan() throws Exception {
        WalRecord a = new WalRecord(1L, Type.ADD, new byte[] {1, 2, 3});
        WalRecord b = new WalRecord(2L, Type.ADD, new byte[] {4, 5, 6});
        WalRecord c = new WalRecord(3L, Type.ADD, new byte[] {7, 8, 9});
        byte[] data = concat(a, b, c);
        data[a.encodedSize() + 13] ^= 0x01; // flip a payload bit inside record b
        Path file = dir.resolve("wal.log");
        Files.write(file, data);

        ScanResult res = WalReader.scan(file);
        assertEquals(1, res.records().size());
        assertEquals(a.encodedSize(), res.validBytes());
        assertEquals(Status.CORRUPT, res.stop());
        assertTrue(res.hasGarbageTail());
    }

    @Test
    void sequenceGapStopsTheScan() throws Exception {
        WalRecord a = new WalRecord(1L, Type.ADD, new byte[] {1});
        WalRecord gap = new WalRecord(5L, Type.ADD, new byte[] {2});
        Path file = dir.resolve("wal.log");
        Files.write(file, concat(a, gap));

        ScanResult res = WalReader.scan(file);
        assertEquals(1, res.records().size());
        assertEquals(Status.CORRUPT, res.stop());
    }

    @Test
    void truncateThenAppendYieldsACleanLog() throws Exception {
        WalRecord a = new WalRecord(1L, Type.ADD, new byte[] {1});
        WalRecord b = new WalRecord(2L, Type.ADD, new byte[] {2});
        byte[] good = concat(a, b);
        byte[] torn = Arrays.copyOf(good, good.length + 3);
        Path file = dir.resolve("wal.log");
        Files.write(file, torn);

        ScanResult before = WalReader.scan(file);
        assertTrue(before.hasGarbageTail());
        WalReader.truncateTo(file, before.validBytes());

        long nextSeq = before.records().get(before.records().size() - 1).seq() + 1;
        try (WalWriter w = new WalWriter(file, nextSeq, true)) {
            w.append(Type.DELETE, new byte[] {9});
        }

        ScanResult after = WalReader.scan(file);
        assertEquals(3, after.records().size());
        assertEquals(Status.END, after.stop());
        assertFalse(after.hasGarbageTail());
    }
}