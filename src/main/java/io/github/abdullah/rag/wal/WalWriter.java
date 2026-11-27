package io.github.abdullah.rag.wal;

import io.github.abdullah.rag.wal.WalRecord.Type;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;

/**
 * Appends records to the log file. A record is acknowledged (append returns)
 * only after its bytes are written and, if syncEachAppend is on, fsynced.
 */
public final class WalWriter implements AutoCloseable {

    private final FileChannel channel;
    private final boolean syncEachAppend;
    private long nextSeq;
    private boolean broken;

    /**
     * @param nextSeq sequence number given to the next appended record
     * @param syncEachAppend true = fsync after every append (durable); false = caller calls sync()
     */
    public WalWriter(Path path, long nextSeq, boolean syncEachAppend) throws IOException {
        this.channel = FileChannel.open(path, StandardOpenOption.CREATE, StandardOpenOption.WRITE);
        this.channel.position(channel.size());
        this.nextSeq = nextSeq;
        this.syncEachAppend = syncEachAppend;
    }

    public synchronized WalRecord append(Type type, byte[] payload) throws IOException {
        if (broken) {
            throw new IOException("WAL writer is broken after an earlier I/O failure");
        }
        WalRecord record = new WalRecord(nextSeq, type, payload);
        try {
            ByteBuffer buf = ByteBuffer.wrap(record.encode());
            while (buf.hasRemaining()) {
                channel.write(buf);
            }
            if (syncEachAppend) {
                channel.force(false);
            }
        } catch (IOException e) {
            broken = true; // file may now end in a partial record; recovery must truncate it
            throw e;
        }
        nextSeq++;
        return record;
    }

    public synchronized void sync() throws IOException {
        channel.force(false);
    }

    public synchronized long nextSeq() {
        return nextSeq;
    }

    @Override
    public synchronized void close() throws IOException {
        channel.close();
    }
}