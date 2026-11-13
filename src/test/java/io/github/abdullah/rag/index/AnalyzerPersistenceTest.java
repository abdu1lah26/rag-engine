package io.github.abdullah.rag.index;

import io.github.abdullah.rag.analysis.SimpleAnalyzer;
import io.github.abdullah.rag.analysis.TextAnalyzer;
import io.github.abdullah.rag.bm25.Tokenizer;
import io.github.abdullah.rag.model.Document;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class AnalyzerPersistenceTest {

    /** A second analyzer with a different name, same behavior: enough to test the name check. */
    private static final TextAnalyzer OTHER = new TextAnalyzer() {
        @Override
        public String name() {
            return "other";
        }

        @Override
        public List<String> analyze(String text) {
            return Tokenizer.tokenize(text);
        }
    };

    private static final List<Document> CORPUS = List.of(
            new Document("d1", "", "apple banana"),
            new Document("d2", "", "banana cherry"));

    @Test
    void analyzerNameSurvivesRoundTrip(@TempDir Path dir) throws IOException {
        IndexWriter.write(InvertedIndex.build(CORPUS, OTHER), dir);
        assertEquals("other", IndexReader.load(dir).analyzerName());
    }

    @Test
    void loadingWithWrongAnalyzerFails(@TempDir Path dir) throws IOException {
        IndexWriter.write(InvertedIndex.build(CORPUS, new SimpleAnalyzer()), dir);
        assertThrows(IllegalArgumentException.class, () -> IndexedBm25Retriever.load(dir, OTHER));
    }

    @Test
    void loadingWithMatchingAnalyzerWorks(@TempDir Path dir) throws IOException {
        IndexWriter.write(InvertedIndex.build(CORPUS, OTHER), dir);
        IndexedBm25Retriever loaded = IndexedBm25Retriever.load(dir, OTHER);
        assertEquals("d1", loaded.search("apple", 5).get(0).docId());
    }
}