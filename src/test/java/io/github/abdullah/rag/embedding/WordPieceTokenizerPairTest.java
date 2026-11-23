package io.github.abdullah.rag.embedding;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class WordPieceTokenizerPairTest {

    @TempDir
    Path tmp;

    private WordPieceTokenizer tokenizer(int maxLength) throws IOException {
        Path vocab = tmp.resolve("vocab.txt");
        Files.write(vocab, List.of("[PAD]", "[UNK]", "[CLS]", "[SEP]",
                "hello", "world", "foo", "bar"));
        return new WordPieceTokenizer(vocab, maxLength);
    }

    @Test
    void pairLayoutAndSegmentIds() throws IOException {
        WordPieceTokenizer.Encoding e = tokenizer(32).encodePair("hello world", "foo");
        // [CLS] hello world [SEP] foo [SEP]
        assertArrayEquals(new long[] {2, 4, 5, 3, 6, 3}, e.inputIds());
        assertArrayEquals(new long[] {0, 0, 0, 0, 1, 1}, e.tokenTypeIds());
        assertArrayEquals(new long[] {1, 1, 1, 1, 1, 1}, e.attentionMask());
    }

    @Test
    void longQueryAndDocAreBothCut() throws IOException {
        // maxLength 9 -> budget 6 -> query capped at 3, document gets 3
        WordPieceTokenizer.Encoding e = tokenizer(9).encodePair(
                "hello hello hello hello hello", "foo foo foo foo foo");
        assertArrayEquals(new long[] {2, 4, 4, 4, 3, 6, 6, 6, 3}, e.inputIds());
        assertArrayEquals(new long[] {0, 0, 0, 0, 0, 1, 1, 1, 1}, e.tokenTypeIds());
    }

    @Test
    void shortQueryLeavesRoomForDocument() throws IOException {
        // query uses 1 of its 3 slots, so the document may use 5
        WordPieceTokenizer.Encoding e = tokenizer(9).encodePair(
                "hello", "foo foo foo foo foo foo foo foo foo foo");
        assertEquals(9, e.inputIds().length);
        assertArrayEquals(new long[] {2, 4, 3, 6, 6, 6, 6, 6, 3}, e.inputIds());
        assertArrayEquals(new long[] {0, 0, 0, 1, 1, 1, 1, 1, 1}, e.tokenTypeIds());
    }

    @Test
    void singleTextEncodingIsUnchanged() throws IOException {
        WordPieceTokenizer.Encoding e = tokenizer(32).encode("hello");
        assertArrayEquals(new long[] {2, 4, 3}, e.inputIds());
        assertArrayEquals(new long[] {0, 0, 0}, e.tokenTypeIds());
    }
}