package io.github.abdullah.rag.wal;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.List;
import java.util.OptionalLong;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.zip.CRC32;

/**
 * Names the active generation. File content: "RAGCUR1 <gen> <crc32 hex>\n".
 * Written to CURRENT.tmp, fsynced, then atomically renamed over CURRENT.
 */
public final class CurrentPointer {

    static final String FILE = "CURRENT";
    private static final String TMP = "CURRENT.tmp";
    private static final String PREFIX = "RAGCUR1";
    private static final Pattern BASE = Pattern.compile("base-(\\d+)\\.snap(\\.tmp)?");
    private static final Pattern WAL = Pattern.compile("wal-(\\d+)\\.log");

    private CurrentPointer() {}

    public static Path baseFile(Path dir, long gen) {
        return dir.resolve(String.format("base-%06d.snap", gen));
    }

    public static Path walFile(Path dir, long gen) {
        return dir.resolve(String.format("wal-%06d.log", gen));
    }

    static String encode(long gen) {
        String body = PREFIX + " " + gen;
        CRC32 crc = new CRC32();
        crc.update(body.getBytes(StandardCharsets.US_ASCII));
        return body + " " + String.format("%08x", crc.getValue()) + "\n";
    }

    /** Empty if there is no CURRENT file. A present but invalid file throws: never guess. */
    public static OptionalLong read(Path dir) throws IOException {
        Path f = dir.resolve(FILE);
        if (!Files.exists(f)) {
            return OptionalLong.empty();
        }
        String s = new String(Files.readAllBytes(f), StandardCharsets.US_ASCII);
        String[] parts = s.split(" ");
        if (parts.length != 3 || !parts[0].equals(PREFIX)) {
            throw new IOException("CURRENT is damaged: " + f);
        }
        long gen;
        try {
            gen = Long.parseLong(parts[1]);
        } catch (NumberFormatException e) {
            throw new IOException("CURRENT is damaged (bad generation): " + f, e);
        }
        if (gen < 0 || !s.equals(encode(gen))) {
            throw new IOException("CURRENT is damaged (checksum mismatch): " + f);
        }
        return OptionalLong.of(gen);
    }

    public static void write(Path dir, long gen) throws IOException {
        if (gen < 0) {
            throw new IllegalArgumentException("gen must be >= 0");
        }
        Path tmp = dir.resolve(TMP);
        try (FileChannel ch = FileChannel.open(tmp, StandardOpenOption.CREATE, StandardOpenOption.WRITE,
                StandardOpenOption.TRUNCATE_EXISTING)) {
            ByteBuffer buf = ByteBuffer.wrap(encode(gen).getBytes(StandardCharsets.US_ASCII));
            while (buf.hasRemaining()) {
                ch.write(buf);
            }
            ch.force(true);
        }
        Files.move(tmp, dir.resolve(FILE), StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
    }

    /**
     * Deletes generation files that are not the current one, plus any .tmp leftovers.
     * Call only after CURRENT is known good. Returns the names removed.
     */
    public static List<String> deleteOtherGenerations(Path dir, long gen) throws IOException {
        List<String> deleted = new ArrayList<>();
        try (DirectoryStream<Path> ds = Files.newDirectoryStream(dir)) {
            for (Path p : ds) {
                String name = p.getFileName().toString();
                boolean remove = false;
                Matcher b = BASE.matcher(name);
                Matcher w = WAL.matcher(name);
                if (b.matches()) {
                    remove = b.group(2) != null || Long.parseLong(b.group(1)) != gen;
                } else if (w.matches()) {
                    remove = Long.parseLong(w.group(1)) != gen;
                } else if (name.equals(TMP)) {
                    remove = true;
                }
                if (remove) {
                    Files.delete(p);
                    deleted.add(name);
                }
            }
        }
        return deleted;
    }
}