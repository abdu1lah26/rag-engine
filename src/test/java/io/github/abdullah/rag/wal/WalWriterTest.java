package io.github.abdullah.rag.wal;

import static org.junit.jupiter.api.Assertions.assertEquals;

import io.github.abdullah.rag.wal.WalRecord.DecodeResult;
import io.github.abdullah.rag.wal.WalRecord.Status;
import io.github.abdullah.rag.wal.WalRecord.Type;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class WalWriterTest {

    @TempDir
    Path dir;

    @Test
    void appendedRecordsDecodeInOrder() throws Exception {
        Path file = dir.resolve("wal.log");
        try (WalWriter w = new WalWriter(file, 1L, true)) {
            w.append(Type.ADD, new byte[] {1});
            w.append(Type.DELETE, new byte[] {2, 3});
            assertEquals(3L, w.nextSeq());
        }
        byte[] data = Files.readAllBytes(file);
        DecodeResult a = WalRecord.decode(data, 0, data.length);
        DecodeResult b = WalRecord.decode(data, a.size(), data.length);
        DecodeResult end = WalRecord.decode(data, a.size() + b.size(), data.length);

        assertEquals(1L, a.record().seq());
        assertEquals(Type.ADD, a.record().type());
        assertEquals(2L, b.record().seq());
        assertEquals(Type.DELETE, b.record().type());
        assertEquals(Status.END, end.status());
    }

    @Test
    void reopenAppendsAfterExistingRecords() throws Exception {
        Path file = dir.resolve("wal.log");
        try (WalWriter w = new WalWriter(file, 1L, true)) {
            w.append(Type.ADD, new byte[] {1});
        }
        try (WalWriter w = new WalWriter(file, 2L, true)) {
            w.append(Type.ADD, new byte[] {2});
        }
        byte[] data = Files.readAllBytes(file);
        DecodeResult a = WalRecord.decode(data, 0, data.length);
        DecodeResult b = WalRecord.decode(data, a.size(), data.length);
        DecodeResult end = WalRecord.decode(data, a.size() + b.size(), data.length);

        assertEquals(1L, a.record().seq());
        assertEquals(2L, b.record().seq());
        assertEquals(Status.END, end.status());
    }
}