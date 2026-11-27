package io.github.abdullah.rag.wal;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.abdullah.rag.wal.DurableStore.CompactionResult;
import io.github.abdullah.rag.wal.WalPayloads.AddOp;
import io.github.abdullah.rag.wal.WalReader.ScanResult;
import io.github.abdullah.rag.wal.WalRecord.Status;
import io.github.abdullah.rag.wal.WalRecord.Type;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.util.Arrays;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class DurableStoreTest {

    @TempDir
    Path dir;

    private static AddOp op(String id, String text) {
        return new AddOp(id, "t", text, "m", new float[] {1f, 2f});
    }

    private static Set<String> baseIds(DurableStore s) {
        return s.baseDocs().stream().map(AddOp::docId).collect(Collectors.toSet());
    }

    private static String baseText(DurableStore s, String id) {
        return s.baseDocs().stream().filter(o -> o.docId().equals(id)).findFirst().orElseThrow().text();
    }

    @Test
    void stateSurvivesRestart() throws Exception {
        Path d = dir.resolve("s1");
        try (DurableStore s = DurableStore.open(d)) {
            s.put(op("a", "v1"));
            s.put(op("b", "v1"));
            s.delete("a");
            s.put(op("b", "v2"));
        }
        try (DurableStore s = DurableStore.open(d)) {
            assertEquals(4, s.recovery().replayed());
            assertEquals(0, s.recovery().truncatedBytes());
            assertNull(s.state().get("a"));
            assertEquals("v2", s.state().get("b").text());
            assertEquals(4, s.state().lastSeq());
            s.put(op("c", "v1"));
            assertEquals(5, s.state().lastSeq());
        }
    }

    @Test
    void garbageTailIsTruncatedAndNewWritesSurvive() throws Exception {
        Path d = dir.resolve("s2");
        try (DurableStore s = DurableStore.open(d)) {
            s.put(op("a", "v1"));
        }
        Path wal = CurrentPointer.walFile(d, 0);
        long good = Files.size(wal);
        Files.write(wal, new byte[] {0, 0, 0, 40, 1, 2, 3}, StandardOpenOption.APPEND);

        try (DurableStore s = DurableStore.open(d)) {
            assertEquals(7, s.recovery().truncatedBytes());
            assertEquals(good, Files.size(wal));
            s.put(op("b", "v1"));
        }
        try (DurableStore s = DurableStore.open(d)) {
            assertEquals(2, s.recovery().replayed());
            assertEquals(Status.END, s.recovery().stop());
            assertEquals(Set.of("a", "b"), s.state().liveDocs().keySet());
        }
    }

    @Test
    void crashAtEveryByteOffsetRecoversExactlyTheCompleteOps() throws Exception {
        Path src = dir.resolve("src");
        try (DurableStore s = DurableStore.open(src)) {
            s.put(op("a", "v1"));
            s.put(op("b", "v1"));
            s.delete("a");
            s.put(op("b", "v2"));
        }
        Path srcWal = CurrentPointer.walFile(src, 0);
        byte[] full = Files.readAllBytes(srcWal);
        ScanResult scan = WalReader.scan(srcWal);
        long[] ends = new long[scan.records().size()];
        long acc = 0;
        for (int i = 0; i < ends.length; i++) {
            acc += scan.records().get(i).encodedSize();
            ends[i] = acc;
        }
        @SuppressWarnings("unchecked")
        Set<String>[] expectedIds = new Set[] {Set.of(), Set.of("a"), Set.of("a", "b"), Set.of("b"), Set.of("b")};

        for (int cut = 0; cut <= full.length; cut++) {
            int complete = 0;
            long boundary = 0;
            for (int i = 0; i < ends.length; i++) {
                if (cut >= ends[i]) {
                    complete = i + 1;
                    boundary = ends[i];
                }
            }
            Path d = dir.resolve("cut" + cut);
            Files.createDirectories(d);
            Files.copy(src.resolve("CURRENT"), d.resolve("CURRENT"));
            Files.copy(CurrentPointer.baseFile(src, 0), CurrentPointer.baseFile(d, 0));
            Files.write(CurrentPointer.walFile(d, 0), Arrays.copyOf(full, cut));

            try (DurableStore s = DurableStore.open(d)) {
                assertEquals(complete, s.recovery().replayed(), "replayed at cut " + cut);
                assertEquals(cut - boundary, s.recovery().truncatedBytes(), "truncated at cut " + cut);
                assertEquals(expectedIds[complete], s.state().liveDocs().keySet(), "live ids at cut " + cut);
                assertEquals(complete, s.state().lastSeq(), "lastSeq at cut " + cut);
                if (complete == 4) {
                    assertEquals("v2", s.state().get("b").text());
                }
                s.put(op("new", "x"));
            }
            try (DurableStore s = DurableStore.open(d)) {
                assertEquals(complete + 1, s.recovery().replayed(), "reopen replayed at cut " + cut);
                assertTrue(s.state().liveDocs().containsKey("new"), "new doc survived at cut " + cut);
            }
        }
    }

    @Test
    void compactionFoldsDeltaIntoBaseAndSurvivesRestart() throws Exception {
        Path d = dir.resolve("c1");
        try (DurableStore s = DurableStore.open(d)) {
            s.put(op("a", "v1"));
            s.put(op("b", "v1"));
            s.delete("a");
            s.put(op("c", "v1"));
            s.put(op("b", "v2"));
            CompactionResult r = s.compact();
            assertEquals(1, r.generation());
            assertEquals(2, r.docs());
            assertEquals(5, r.seq());
            assertEquals(Set.of("b", "c"), baseIds(s));
            assertEquals("v2", baseText(s, "b"));
            assertTrue(s.state().touchedIds().isEmpty());
            assertTrue(s.state().liveDocs().isEmpty());
            assertEquals(5, s.state().lastSeq());
            s.put(op("e", "v1"));
            assertEquals(6, s.state().lastSeq());
        }
        assertTrue(Files.exists(CurrentPointer.baseFile(d, 1)));
        assertTrue(Files.exists(CurrentPointer.walFile(d, 1)));
        assertFalse(Files.exists(CurrentPointer.baseFile(d, 0)));
        assertFalse(Files.exists(CurrentPointer.walFile(d, 0)));

        try (DurableStore s = DurableStore.open(d)) {
            assertEquals(1, s.generation());
            assertEquals(Set.of("b", "c"), baseIds(s));
            assertEquals(Set.of("e"), s.state().liveDocs().keySet());
            assertEquals(1, s.recovery().replayed());
            assertEquals(6, s.state().lastSeq());
        }
    }

    @Test
    void deleteOfBaseDocSurvivesLaterCompaction() throws Exception {
        Path d = dir.resolve("c2");
        try (DurableStore s = DurableStore.open(d)) {
            s.put(op("a", "v1"));
            s.put(op("b", "v1"));
            s.compact();
            s.delete("a");
            assertFalse(s.state().baseVisible("a"));
            s.compact();
            assertEquals(Set.of("b"), baseIds(s));
            assertEquals(2, s.generation());
        }
        try (DurableStore s = DurableStore.open(d)) {
            assertEquals(Set.of("b"), baseIds(s));
            assertTrue(s.state().liveDocs().isEmpty());
            assertTrue(s.state().touchedIds().isEmpty());
        }
    }

    @Test
    void updateOfBaseDocReplacesIt() throws Exception {
        Path d = dir.resolve("c3");
        try (DurableStore s = DurableStore.open(d)) {
            s.put(op("a", "v1"));
            s.compact();
            s.put(op("a", "v2"));
            s.compact();
            assertEquals(1, s.baseDocs().size());
            assertEquals("v2", baseText(s, "a"));
        }
    }

    @Test
    void crashBeforeSwitchKeepsOldGeneration() throws Exception {
        Path d = dir.resolve("c4");
        try (DurableStore s = DurableStore.open(d)) {
            s.put(op("a", "v1"));
        }
        // Debris of a compaction that died before CURRENT was switched.
        SnapshotFile.write(CurrentPointer.baseFile(d, 1), List.of(op("zzz", "x")), 99L);
        Files.writeString(d.resolve("base-000002.snap.tmp"), "garbage");
        Files.writeString(CurrentPointer.walFile(d, 1), "");
        Files.writeString(d.resolve("CURRENT.tmp"), "garbage");

        try (DurableStore s = DurableStore.open(d)) {
            assertEquals(0, s.generation());
            assertEquals(Set.of("a"), s.state().liveDocs().keySet());
            assertTrue(baseIds(s).isEmpty());
        }
        assertFalse(Files.exists(CurrentPointer.baseFile(d, 1)));
        assertFalse(Files.exists(CurrentPointer.walFile(d, 1)));
        assertFalse(Files.exists(d.resolve("base-000002.snap.tmp")));
        assertFalse(Files.exists(d.resolve("CURRENT.tmp")));
    }

    @Test
    void crashAfterSwitchBeforeCleanupUsesNewGeneration() throws Exception {
        Path d = dir.resolve("c5");
        Path saved = dir.resolve("saved");
        Files.createDirectories(saved);
        try (DurableStore s = DurableStore.open(d)) {
            s.put(op("a", "v1"));
            s.put(op("b", "v1"));
        }
        Files.copy(CurrentPointer.baseFile(d, 0), saved.resolve("base-000000.snap"));
        Files.copy(CurrentPointer.walFile(d, 0), saved.resolve("wal-000000.log"));
        try (DurableStore s = DurableStore.open(d)) {
            s.compact();
        }
        // Put the old generation back: as if the process died before deleting it.
        Files.copy(saved.resolve("base-000000.snap"), CurrentPointer.baseFile(d, 0), StandardCopyOption.REPLACE_EXISTING);
        Files.copy(saved.resolve("wal-000000.log"), CurrentPointer.walFile(d, 0), StandardCopyOption.REPLACE_EXISTING);

        try (DurableStore s = DurableStore.open(d)) {
            assertEquals(1, s.generation());
            assertEquals(Set.of("a", "b"), baseIds(s));
            assertEquals(0, s.recovery().replayed());
        }
        assertFalse(Files.exists(CurrentPointer.baseFile(d, 0)));
        assertFalse(Files.exists(CurrentPointer.walFile(d, 0)));
    }

    @Test
    void walThatDoesNotFollowTheBaseIsRejected() throws Exception {
        Path d = dir.resolve("c6");
        try (DurableStore s = DurableStore.open(d)) {
            s.put(op("a", "v1"));
            s.compact(); // base ends at seq 1
        }
        byte[] wrong = new WalRecord(7L, Type.ADD, WalPayloads.encodeAdd(op("x", "v"))).encode();
        Files.write(CurrentPointer.walFile(d, 1), wrong);
        assertThrows(IOException.class, () -> DurableStore.open(d));
    }

    @Test
    void damagedBaseFailsLoudly() throws Exception {
        Path d = dir.resolve("c7");
        try (DurableStore s = DurableStore.open(d)) {
            s.put(op("a", "v1"));
            s.compact();
        }
        Path base = CurrentPointer.baseFile(d, 1);
        byte[] bytes = Files.readAllBytes(base);
        bytes[bytes.length / 2] ^= 0x01;
        Files.write(base, bytes);
        assertThrows(IOException.class, () -> DurableStore.open(d));
    }

    @Test
    void walWithoutCurrentIsRejected() throws Exception {
        Path d = dir.resolve("c8");
        Files.createDirectories(d);
        Files.writeString(CurrentPointer.walFile(d, 0), "orphan");
        assertThrows(IOException.class, () -> DurableStore.open(d));
    }
}