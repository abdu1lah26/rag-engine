package io.github.abdullah.rag.wal;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class CurrentPointerTest {

    @TempDir
    Path dir;

    @Test
    void missingFileIsEmpty() throws Exception {
        assertTrue(CurrentPointer.read(dir).isEmpty());
    }

    @Test
    void writeThenReadRoundTrips() throws Exception {
        CurrentPointer.write(dir, 7);
        assertEquals(7L, CurrentPointer.read(dir).getAsLong());
        CurrentPointer.write(dir, 8);
        assertEquals(8L, CurrentPointer.read(dir).getAsLong());
        assertFalse(Files.exists(dir.resolve("CURRENT.tmp")));
    }

    @Test
    void everyProperPrefixIsRejected() throws Exception {
        byte[] full = CurrentPointer.encode(12).getBytes(java.nio.charset.StandardCharsets.US_ASCII);
        for (int len = 0; len < full.length; len++) {
            Files.write(dir.resolve("CURRENT"), Arrays.copyOf(full, len));
            assertThrows(IOException.class, () -> CurrentPointer.read(dir), "prefix " + len);
        }
    }

    @Test
    void anySingleBitFlipIsRejected() throws Exception {
        byte[] full = CurrentPointer.encode(12).getBytes(java.nio.charset.StandardCharsets.US_ASCII);
        for (int i = 0; i < full.length; i++) {
            for (int bit = 0; bit < 8; bit++) {
                byte[] copy = full.clone();
                copy[i] ^= (byte) (1 << bit);
                Files.write(dir.resolve("CURRENT"), copy);
                assertThrows(IOException.class, () -> CurrentPointer.read(dir), "byte " + i + " bit " + bit);
            }
        }
    }

    @Test
    void trailingBytesAreRejected() throws Exception {
        Files.writeString(dir.resolve("CURRENT"), CurrentPointer.encode(3) + "x");
        assertThrows(IOException.class, () -> CurrentPointer.read(dir));
    }

    @Test
    void staleTmpFileDoesNotAffectRead() throws Exception {
        CurrentPointer.write(dir, 4);
        Files.writeString(dir.resolve("CURRENT.tmp"), "garbage from a crash");
        assertEquals(4L, CurrentPointer.read(dir).getAsLong());
    }

    @Test
    void cleanupKeepsOnlyTheCurrentGeneration() throws Exception {
        for (String n : List.of("base-000001.snap", "base-000002.snap", "base-000002.snap.tmp",
                "base-000003.snap", "wal-000001.log", "wal-000002.log", "wal-000003.log",
                "CURRENT", "CURRENT.tmp", "notes.txt")) {
            Files.writeString(dir.resolve(n), "x");
        }
        List<String> deleted = CurrentPointer.deleteOtherGenerations(dir, 2);
        assertEquals(Set.of("base-000001.snap", "base-000002.snap.tmp", "base-000003.snap",
                "wal-000001.log", "wal-000003.log", "CURRENT.tmp"), Set.copyOf(deleted));
        try (Stream<Path> s = Files.list(dir)) {
            assertEquals(Set.of("base-000002.snap", "wal-000002.log", "CURRENT", "notes.txt"),
                    s.map(p -> p.getFileName().toString()).collect(Collectors.toSet()));
        }
    }

    @Test
    void fileNamesAreZeroPaddedPerGeneration() {
        assertEquals("base-000005.snap", CurrentPointer.baseFile(dir, 5).getFileName().toString());
        assertEquals("wal-000005.log", CurrentPointer.walFile(dir, 5).getFileName().toString());
    }
}