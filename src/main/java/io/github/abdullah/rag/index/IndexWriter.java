package io.github.abdullah.rag.index;

import java.io.BufferedOutputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.zip.CRC32;
import java.util.zip.CheckedOutputStream;

/**
 * Writes an InvertedIndex to a directory in the layout described by
 * IndexFormat. Every file = header + body + CRC32 footer.
 */
public final class IndexWriter {

    private IndexWriter() {
    }

    private record TermEntry(String term, int docFreq, long offset) {
    }

    @FunctionalInterface
    private interface Body {
        void write(DataOutputStream out) throws IOException;
    }

    public static void write(InvertedIndex index, Path dir) throws IOException {
        Files.createDirectories(dir);
        writeDocs(index, dir.resolve(IndexFormat.DOCS_FILE));
        // postings first: only after writing them do we know each term's offset
        List<TermEntry> entries = writePostings(index, dir.resolve(IndexFormat.POSTINGS_FILE));
        writeTerms(entries, dir.resolve(IndexFormat.TERMS_FILE));
    }

    private static void writeDocs(InvertedIndex index, Path path) throws IOException {
        writeFile(path, out -> {
            out.writeUTF(index.analyzerName());
            out.writeInt(index.numDocs());
            for (int i = 0; i < index.numDocs(); i++) {
                out.writeUTF(index.docId(i));
                out.writeInt(index.docLength(i));
            }
        });
    }

    private static List<TermEntry> writePostings(InvertedIndex index, Path path) throws IOException {
        List<TermEntry> entries = new ArrayList<>();
        writeFile(path, out -> {
            long pos = 2L * Integer.BYTES; // header already written
            List<String> terms = new ArrayList<>(index.allPostings().keySet());
            Collections.sort(terms);
            for (String term : terms) {
                List<Posting> list = index.postings(term);
                ByteArrayOutputStream buf = new ByteArrayOutputStream();
                int prev = 0;
                for (Posting p : list) {
                    VarInt.write(buf, p.docIndex() - prev); // gap, not absolute id
                    VarInt.write(buf, p.tf());
                    prev = p.docIndex();
                }
                entries.add(new TermEntry(term, list.size(), pos));
                out.write(buf.toByteArray());
                pos += buf.size();
            }
        });
        return entries;
    }

    private static void writeTerms(List<TermEntry> entries, Path path) throws IOException {
        writeFile(path, out -> {
            out.writeInt(entries.size());
            for (TermEntry e : entries) {
                out.writeUTF(e.term());
                out.writeInt(e.docFreq());
                out.writeLong(e.offset());
            }
        });
    }

    /** Header, then the body, then a CRC32 of everything before the footer. */
    private static void writeFile(Path path, Body body) throws IOException {
        try (OutputStream raw = new BufferedOutputStream(Files.newOutputStream(path))) {
            CheckedOutputStream checked = new CheckedOutputStream(raw, new CRC32());
            DataOutputStream out = new DataOutputStream(checked);
            out.writeInt(IndexFormat.MAGIC);
            out.writeInt(IndexFormat.VERSION);
            body.write(out);
            out.flush();
            // footer goes to raw, not through the checksum, so it isn't counted
            new DataOutputStream(raw).writeLong(checked.getChecksum().getValue());
        }
    }
}