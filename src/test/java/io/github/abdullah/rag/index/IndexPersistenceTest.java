package io.github.abdullah.rag.index;

import io.github.abdullah.rag.model.Document;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class IndexPersistenceTest {

    private static InvertedIndex sample() {
        List<Document> docs = new ArrayList<>();
        docs.add(new Document("d1", "Fruit", "apple banana apple"));
        docs.add(new Document("d2", "", "banana cherry banana banana"));
        docs.add(new Document("d3", "Exotic", "durian smells strongly"));
        // 300 filler docs so doc-index gaps cross the 127/128 varint boundary
        for (int i = 0; i < 300; i++) {
            docs.add(new Document("f" + i, "", "filler text number " + i));
        }
        docs.add(new Document("last", "", "apple zürich café"));
        return InvertedIndex.build(docs);
    }

    private static void assertSameIndex(InvertedIndex a, InvertedIndex b) {
        assertEquals(a.numDocs(), b.numDocs());
        assertEquals(a.numTerms(), b.numTerms());
        assertEquals(a.avgDocLength(), b.avgDocLength(), 1e-12);
        for (int i = 0; i < a.numDocs(); i++) {
            assertEquals(a.docId(i), b.docId(i));
            assertEquals(a.docLength(i), b.docLength(i));
        }
        for (String term : a.allPostings().keySet()) {
            assertEquals(a.postings(term), b.postings(term), "postings for term: " + term);
        }
    }

    @Test
    void roundTripPreservesEverything(@TempDir Path dir) throws IOException {
        InvertedIndex original = sample();
        IndexWriter.write(original, dir);
        assertSameIndex(original, IndexReader.load(dir));
    }

    @Test
    void emptyIndexRoundTrips(@TempDir Path dir) throws IOException {
        InvertedIndex empty = InvertedIndex.build(List.of());
        IndexWriter.write(empty, dir);
        assertEquals(0, IndexReader.load(dir).numDocs());
    }

    @Test
    void flippedByteIsDetected(@TempDir Path dir) throws IOException {
        IndexWriter.write(sample(), dir);
        Path postings = dir.resolve(IndexFormat.POSTINGS_FILE);
        byte[] bytes = Files.readAllBytes(postings);
        bytes[bytes.length / 2] ^= 0x01;
        Files.write(postings, bytes);
        assertThrows(IOException.class, () -> IndexReader.load(dir));
    }

    @Test
    void truncatedFileIsDetected(@TempDir Path dir) throws IOException {
        IndexWriter.write(sample(), dir);
        Path terms = dir.resolve(IndexFormat.TERMS_FILE);
        byte[] bytes = Files.readAllBytes(terms);
        Files.write(terms, java.util.Arrays.copyOf(bytes, bytes.length - 20));
        assertThrows(IOException.class, () -> IndexReader.load(dir));
    }

    @Test
    void missingFileIsDetected(@TempDir Path dir) throws IOException {
        IndexWriter.write(sample(), dir);
        Files.delete(dir.resolve(IndexFormat.DOCS_FILE));
        assertThrows(IOException.class, () -> IndexReader.load(dir));
    }
}