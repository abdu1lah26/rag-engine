package io.github.abdullah.rag.index;

import java.io.ByteArrayInputStream;
import java.io.DataInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.zip.CRC32;

/**
 * Loads an index directory written by IndexWriter. Every file is verified
 * (size, CRC32, magic, version) before any of its content is trusted.
 */
public final class IndexReader {

    private static final int HEADER_BYTES = 2 * Integer.BYTES;
    private static final int FOOTER_BYTES = Long.BYTES;

    private IndexReader() {
    }

    public static InvertedIndex load(Path dir) throws IOException {
        // docs.bin
        // docs.bin
        String analyzerName;
        List<String> docIds = new ArrayList<>();
        int[] docLengths;
        try (DataInputStream in = bodyStream(openVerified(dir.resolve(IndexFormat.DOCS_FILE)))) {
            analyzerName = in.readUTF();
            int numDocs = in.readInt();
            docLengths = new int[numDocs];
            for (int i = 0; i < numDocs; i++) {
                docIds.add(in.readUTF());
                docLengths[i] = in.readInt();
            }
        }

        // terms.bin
        List<String> terms = new ArrayList<>();
        List<Integer> docFreqs = new ArrayList<>();
        List<Long> offsets = new ArrayList<>();
        try (DataInputStream in = bodyStream(openVerified(dir.resolve(IndexFormat.TERMS_FILE)))) {
            int numTerms = in.readInt();
            for (int i = 0; i < numTerms; i++) {
                terms.add(in.readUTF());
                docFreqs.add(in.readInt());
                offsets.add(in.readLong());
            }
        }

        // postings.bin: jump to each term's offset and decode its gaps back to doc
        // indexes
        ByteBuffer postingsBuf = openVerified(dir.resolve(IndexFormat.POSTINGS_FILE));
        Map<String, List<Posting>> postings = new HashMap<>();
        for (int i = 0; i < terms.size(); i++) {
            postingsBuf.position(Math.toIntExact(offsets.get(i)));
            int df = docFreqs.get(i);
            List<Posting> list = new ArrayList<>(df);
            int docIndex = 0;
            for (int j = 0; j < df; j++) {
                docIndex += VarInt.read(postingsBuf);
                int tf = VarInt.read(postingsBuf);
                list.add(new Posting(docIndex, tf));
            }
            postings.put(terms.get(i), list);
        }
        return new InvertedIndex(analyzerName, docIds, docLengths, postings);
    }

    /**
     * Reads the file, checks it, returns a buffer positioned at the body start,
     * limited to the body end.
     */
    private static ByteBuffer openVerified(Path path) throws IOException {
        byte[] bytes = Files.readAllBytes(path);
        if (bytes.length < HEADER_BYTES + FOOTER_BYTES) {
            throw new IOException("corrupted index file (too short): " + path);
        }
        int bodyEnd = bytes.length - FOOTER_BYTES;
        CRC32 crc = new CRC32();
        crc.update(bytes, 0, bodyEnd);

        ByteBuffer buf = ByteBuffer.wrap(bytes);
        if (crc.getValue() != buf.getLong(bodyEnd)) {
            throw new IOException("corrupted index file (checksum mismatch): " + path);
        }
        if (buf.getInt(0) != IndexFormat.MAGIC) {
            throw new IOException("not an index file (bad magic number): " + path);
        }
        if (buf.getInt(Integer.BYTES) != IndexFormat.VERSION) {
            throw new IOException("unsupported index format version in: " + path);
        }
        buf.position(HEADER_BYTES);
        buf.limit(bodyEnd);
        return buf;
    }

    private static DataInputStream bodyStream(ByteBuffer buf) {
        return new DataInputStream(new ByteArrayInputStream(buf.array(), buf.position(), buf.remaining()));
    }
}