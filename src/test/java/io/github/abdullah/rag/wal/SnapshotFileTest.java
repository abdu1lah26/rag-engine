package io.github.abdullah.rag.wal;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;

import io.github.abdullah.rag.wal.SnapshotFile.Contents;
import io.github.abdullah.rag.wal.WalPayloads.AddOp;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class SnapshotFileTest {

    @TempDir
    Path dir;

    private static List<AddOp> docs() {
        return List.of(
                new AddOp("a", "T\u00edtulo", "na\u00efve \u65e5\u672c\u8a9e text", "m", new float[] {0.5f, -0.5f}),
                new AddOp("b", "", "", "m", new float[] {1f, 0f}));
    }

    @Test
    void roundTrip() throws Exception {
        Path f = dir.resolve("base.snap");
        SnapshotFile.write(f, docs(), 17L);
        Contents c = SnapshotFile.read(f);
        assertEquals(17L, c.baseSeq());
        assertEquals(2, c.docs().size());
        assertEquals("a", c.docs().get(0).docId());
        assertEquals("na\u00efve \u65e5\u672c\u8a9e text", c.docs().get(0).text());
        assertArrayEquals(new float[] {0.5f, -0.5f}, c.docs().get(0).vector());
        assertFalse(Files.exists(dir.resolve("base.snap.tmp")));
    }

    @Test
    void emptySnapshotRoundTrips() throws Exception {
        Path f = dir.resolve("base.snap");
        SnapshotFile.write(f, List.of(), 0L);
        Contents c = SnapshotFile.read(f);
        assertEquals(0, c.docs().size());
        assertEquals(0L, c.baseSeq());
    }

    @Test
    void rewriteReplacesTheOldFile() throws Exception {
        Path f = dir.resolve("base.snap");
        SnapshotFile.write(f, docs(), 1L);
        SnapshotFile.write(f, docs().subList(0, 1), 2L);
        Contents c = SnapshotFile.read(f);
        assertEquals(1, c.docs().size());
        assertEquals(2L, c.baseSeq());
    }

    @Test
    void everyProperPrefixIsRejected() throws Exception {
        Path f = dir.resolve("base.snap");
        SnapshotFile.write(f, docs(), 5L);
        byte[] full = Files.readAllBytes(f);
        Path cut = dir.resolve("cut.snap");
        for (int len = 0; len < full.length; len++) {
            Files.write(cut, Arrays.copyOf(full, len));
            assertThrows(IOException.class, () -> SnapshotFile.read(cut), "prefix " + len);
        }
    }

    @Test
    void anySingleBitFlipIsRejected() throws Exception {
        Path f = dir.resolve("base.snap");
        SnapshotFile.write(f, docs(), 5L);
        byte[] full = Files.readAllBytes(f);
        Path bad = dir.resolve("bad.snap");
        for (int i = 0; i < full.length; i++) {
            byte[] copy = full.clone();
            copy[i] ^= 0x01;
            Files.write(bad, copy);
            assertThrows(IOException.class, () -> SnapshotFile.read(bad), "flip byte " + i);
        }
    }

    @Test
    void trailingBytesAreRejected() throws Exception {
        Path f = dir.resolve("base.snap");
        SnapshotFile.write(f, docs(), 5L);
        byte[] full = Files.readAllBytes(f);
        Files.write(f, Arrays.copyOf(full, full.length + 1));
        assertThrows(IOException.class, () -> SnapshotFile.read(f));
    }

    @Test
    void missingFileIsAnIoError() {
        assertThrows(IOException.class, () -> SnapshotFile.read(dir.resolve("nope.snap")));
    }
}