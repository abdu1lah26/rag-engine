package io.github.abdullah.rag.wal;

import io.github.abdullah.rag.wal.WalPayloads.AddOp;
import io.github.abdullah.rag.wal.WalReader.ScanResult;
import io.github.abdullah.rag.wal.WalRecord.Status;
import io.github.abdullah.rag.wal.WalRecord.Type;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.OptionalLong;
import java.util.TreeMap;

/**
 * Directory layout: CURRENT -> generation N, base-N.snap (docs), wal-N.log (changes since the base).
 * Write path: append to WAL, fsync, then apply to memory.
 * Open path: read CURRENT, load base, scan WAL, truncate torn tail, replay, delete stray files.
 * compact(): write base-(N+1).snap from base + delta, switch CURRENT atomically, start wal-(N+1).log.
 */
public final class DurableStore implements AutoCloseable {

    /** What recovery found when the store was opened. */
    public record Recovery(int replayed, long truncatedBytes, Status stop) {}

    public record CompactionResult(long generation, int docs, long seq) {}

    private final Path dir;
    private final Recovery recovery;
    private WalWriter writer;
    private volatile LiveDocs state;
    private Map<String, AddOp> baseDocs;
    private long generation;

    private DurableStore(Path dir, long generation, WalWriter writer, LiveDocs state,
                         Map<String, AddOp> baseDocs, Recovery recovery) {
        this.dir = dir;
        this.generation = generation;
        this.writer = writer;
        this.state = state;
        this.baseDocs = baseDocs;
        this.recovery = recovery;
    }

    public static DurableStore open(Path dir) throws IOException {
        Files.createDirectories(dir);

        long gen;
        OptionalLong cur = CurrentPointer.read(dir);
        if (cur.isPresent()) {
            gen = cur.getAsLong();
        } else {
            if (Files.exists(CurrentPointer.walFile(dir, 0))) {
                throw new IOException("a WAL exists but CURRENT is missing in " + dir);
            }
            gen = 0;
            SnapshotFile.write(CurrentPointer.baseFile(dir, 0), List.of(), 0L);
            CurrentPointer.write(dir, 0);
        }

        // CURRENT is only switched after the base is fully on disk, so a bad base is real damage: fail loudly.
        SnapshotFile.Contents base = SnapshotFile.read(CurrentPointer.baseFile(dir, gen));
        Map<String, AddOp> baseDocs = new TreeMap<>();
        for (AddOp op : base.docs()) {
            baseDocs.put(op.docId(), op);
        }

        Path walPath = CurrentPointer.walFile(dir, gen);
        ScanResult scan = WalReader.scan(walPath);
        if (!scan.records().isEmpty() && scan.records().get(0).seq() != base.baseSeq() + 1) {
            throw new IOException("WAL starts at seq " + scan.records().get(0).seq()
                    + " but the base ends at seq " + base.baseSeq());
        }
        long truncated = scan.fileBytes() - scan.validBytes();
        if (scan.hasGarbageTail()) {
            WalReader.truncateTo(walPath, scan.validBytes());
        }

        // A CRC-valid record with a bad payload is a bug, not a crash artifact: apply() throws.
        LiveDocs state = new LiveDocs(base.baseSeq());
        for (WalRecord r : scan.records()) {
            state.apply(r);
        }

        WalWriter writer = new WalWriter(walPath, state.lastSeq() + 1, true);
        CurrentPointer.deleteOtherGenerations(dir, gen); // leftovers from an interrupted compaction
        return new DurableStore(dir, gen, writer, state, baseDocs,
                new Recovery(scan.records().size(), truncated, scan.stop()));
    }

    /** Add a doc, or replace it if the id exists. Returns after the record is on disk. */
    public synchronized void put(AddOp op) throws IOException {
        WalRecord rec = writer.append(Type.ADD, WalPayloads.encodeAdd(op));
        state.apply(rec);
    }

    public synchronized void delete(String docId) throws IOException {
        WalRecord rec = writer.append(Type.DELETE, WalPayloads.encodeDelete(docId));
        state.apply(rec);
    }

    /**
     * Folds base + delta into a new base and starts an empty WAL.
     * Commit point = the atomic rename of CURRENT. After it returns, any SegmentedEngine
     * built over the previous base is stale and must be rebuilt from baseDocs().
     */
    public synchronized CompactionResult compact() throws IOException {
        LiveDocs.Snapshot snap = state.snapshot();
        Map<String, AddOp> merged = new TreeMap<>(baseDocs);
        for (String id : snap.touched()) {
            merged.remove(id);
        }
        merged.putAll(snap.live());

        long newGen = generation + 1;
        long seq = snap.seq();
        SnapshotFile.write(CurrentPointer.baseFile(dir, newGen), merged.values(), seq);
        WalWriter newWriter = new WalWriter(CurrentPointer.walFile(dir, newGen), seq + 1, true);
        try {
            CurrentPointer.write(dir, newGen); // commit point
        } catch (IOException e) {
            newWriter.close();
            throw e; // old generation is still the truth; stray new files are cleaned on next open
        }

        WalWriter old = writer;
        writer = newWriter;
        state = new LiveDocs(seq);
        baseDocs = merged;
        generation = newGen;
        try {
            old.close();
        } catch (IOException ignored) {
            // the commit already happened; a failed close of the old log changes nothing
        }
        try {
            CurrentPointer.deleteOtherGenerations(dir, newGen);
        } catch (IOException ignored) {
            // best effort; open() repeats this cleanup
        }
        return new CompactionResult(newGen, merged.size(), seq);
    }

    public LiveDocs state() {
        return state;
    }

    /** The docs in the current base, sorted by id. */
    public synchronized List<AddOp> baseDocs() {
        return new ArrayList<>(baseDocs.values());
    }

    public synchronized long generation() {
        return generation;
    }

    public Recovery recovery() {
        return recovery;
    }

    @Override
    public synchronized void close() throws IOException {
        writer.close();
    }
}