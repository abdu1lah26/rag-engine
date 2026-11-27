package io.github.abdullah.rag.wal;

import io.github.abdullah.rag.wal.WalRecord.DecodeResult;
import io.github.abdullah.rag.wal.WalRecord.Status;
import java.io.IOException;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.List;

/** Reads a log front to back and stops at the first record that is not valid. */
public final class WalReader {

    private WalReader() {}

    /**
     * @param records    valid records, in order
     * @param validBytes length of the valid prefix of the file
     * @param fileBytes  total file length
     * @param stop       END (clean), TRUNCATED (torn tail), or CORRUPT (bad bytes or bad seq)
     */
    public record ScanResult(List<WalRecord> records, long validBytes, long fileBytes, Status stop) {
        public boolean hasGarbageTail() {
            return validBytes < fileBytes;
        }
    }

    /** A missing file is an empty log. Records must have consecutive seq numbers. */
    public static ScanResult scan(Path path) throws IOException {
        if (!Files.exists(path)) {
            return new ScanResult(List.of(), 0, 0, Status.END);
        }
        byte[] data = Files.readAllBytes(path);
        List<WalRecord> records = new ArrayList<>();
        int offset = 0;
        Status stop;
        while (true) {
            DecodeResult res = WalRecord.decode(data, offset, data.length);
            if (res.status() != Status.OK) {
                stop = res.status();
                break;
            }
            WalRecord r = res.record();
            if (!records.isEmpty() && r.seq() != records.get(records.size() - 1).seq() + 1) {
                stop = Status.CORRUPT; // CRC was fine but the sequence is broken
                break;
            }
            records.add(r);
            offset += res.size();
        }
        return new ScanResult(List.copyOf(records), offset, data.length, stop);
    }

    /** Cuts the file back to its valid prefix so new appends follow good data. */
    public static void truncateTo(Path path, long validBytes) throws IOException {
        try (FileChannel ch = FileChannel.open(path, StandardOpenOption.WRITE)) {
            ch.truncate(validBytes);
            ch.force(true);
        }
    }
}