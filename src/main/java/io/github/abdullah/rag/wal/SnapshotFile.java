package io.github.abdullah.rag.wal;

import io.github.abdullah.rag.wal.WalPayloads.AddOp;
import io.github.abdullah.rag.wal.WalRecord.DecodeResult;
import io.github.abdullah.rag.wal.WalRecord.Status;
import io.github.abdullah.rag.wal.WalRecord.Type;
import java.io.BufferedOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.channels.Channels;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;

/**
 * A complete base: every doc as an ADD record, then a CHECKPOINT footer.
 * Written to a temp file, fsynced, then renamed into place, so a reader
 * sees either the whole old file or the whole new one.
 */
public final class SnapshotFile {

    private static final byte[] MAGIC = "RAGSNAP1".getBytes(StandardCharsets.US_ASCII);

    private SnapshotFile() {}

    public record Contents(long baseSeq, List<AddOp> docs) {}

    public static void write(Path path, Collection<AddOp> docs, long baseSeq) throws IOException {
        Path tmp = path.resolveSibling(path.getFileName() + ".tmp");
        try (FileChannel ch = FileChannel.open(tmp, StandardOpenOption.CREATE, StandardOpenOption.WRITE,
                StandardOpenOption.TRUNCATE_EXISTING)) {
            OutputStream out = new BufferedOutputStream(Channels.newOutputStream(ch), 1 << 16);
            out.write(MAGIC);
            long seq = 0;
            for (AddOp op : docs) {
                out.write(new WalRecord(seq++, Type.ADD, WalPayloads.encodeAdd(op)).encode());
            }
            byte[] footer = ByteBuffer.allocate(12).putLong(baseSeq).putInt(docs.size()).array();
            out.write(new WalRecord(seq, Type.CHECKPOINT, footer).encode());
            out.flush();
            ch.force(true);
        }
        Files.move(tmp, path, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
    }

    /** Throws IOException unless the file is complete and every record is valid. */
    public static Contents read(Path path) throws IOException {
        byte[] data = Files.readAllBytes(path);
        if (data.length < MAGIC.length || !Arrays.equals(Arrays.copyOf(data, MAGIC.length), MAGIC)) {
            throw new IOException("not a snapshot file (bad magic): " + path);
        }
        int off = MAGIC.length;
        long expectSeq = 0;
        List<AddOp> docs = new ArrayList<>();
        while (true) {
            DecodeResult res = WalRecord.decode(data, off, data.length);
            if (res.status() != Status.OK) {
                throw new IOException("snapshot damaged at byte " + off + " (" + res.status() + "): " + path);
            }
            WalRecord rec = res.record();
            if (rec.seq() != expectSeq++) {
                throw new IOException("snapshot record out of order at byte " + off);
            }
            off += res.size();
            if (rec.type() == Type.CHECKPOINT) {
                byte[] p = rec.payload();
                if (p.length != 12) {
                    throw new IOException("bad snapshot footer");
                }
                ByteBuffer fb = ByteBuffer.wrap(p);
                long baseSeq = fb.getLong();
                int count = fb.getInt();
                if (off != data.length) {
                    throw new IOException("trailing bytes after snapshot footer");
                }
                if (count != docs.size()) {
                    throw new IOException("footer says " + count + " docs but file has " + docs.size());
                }
                return new Contents(baseSeq, List.copyOf(docs));
            }
            if (rec.type() != Type.ADD) {
                throw new IOException("unexpected record type in snapshot: " + rec.type());
            }
            try {
                docs.add(WalPayloads.decodeAdd(rec.payload()));
            } catch (IllegalArgumentException e) {
                throw new IOException("bad doc payload in snapshot", e);
            }
        }
    }
}