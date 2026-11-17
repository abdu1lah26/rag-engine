package io.github.abdullah.rag.longdoc;

import io.github.abdullah.rag.chunking.Chunk;
import io.github.abdullah.rag.chunking.Chunker;
import io.github.abdullah.rag.longdoc.LongDocBuilder.LongDocs;
import io.github.abdullah.rag.longdoc.LongDocBuilder.Span;
import io.github.abdullah.rag.model.Document;
import java.util.List;

/**
 * Retrieval-independent chunk quality, measured against the known abstract spans:
 *  - purity:            share of chunks that overlap exactly one abstract (no mixing of topics)
 *  - boundaryIntegrity: share of abstract-to-abstract boundaries that no chunk crosses
 *  - intactRate:        share of abstracts that fit whole inside at least one chunk
 *                       (the "answer not cut in half" number)
 */
public final class ChunkQuality {

    public record Result(String chunker, int chunks, double avgWords,
                         double purity, double boundaryIntegrity, double intactRate) {}

    private ChunkQuality() {}

    public static Result measure(Chunker chunker, LongDocs longDocs) {
        int chunks = 0, pure = 0, boundaries = 0, crossed = 0, abstracts = 0, intact = 0;
        long words = 0;

        for (Document doc : longDocs.docs()) {
            List<Span> spans = longDocs.spans().get(doc.id());
            List<Chunk> list = chunker.chunk(doc.id(), doc.text());

            for (Chunk c : list) {
                chunks++;
                words += countWords(c.text());
                int overlapping = 0;
                for (Span s : spans) {
                    if (c.startChar() < s.end() && s.start() < c.endChar()) overlapping++;
                }
                if (overlapping == 1) pure++;
            }

            for (int i = 1; i < spans.size(); i++) {
                boundaries++;
                Span a = spans.get(i - 1);
                Span b = spans.get(i);
                for (Chunk c : list) {
                    if (c.startChar() < a.end() && c.endChar() > b.start()) {
                        crossed++;
                        break;
                    }
                }
            }

            for (Span s : spans) {
                abstracts++;
                for (Chunk c : list) {
                    if (c.startChar() <= s.start() && c.endChar() >= s.end()) {
                        intact++;
                        break;
                    }
                }
            }
        }

        return new Result(
                chunker.name(),
                chunks,
                chunks == 0 ? 0 : (double) words / chunks,
                chunks == 0 ? 0 : (double) pure / chunks,
                boundaries == 0 ? 1 : 1.0 - (double) crossed / boundaries,
                abstracts == 0 ? 0 : (double) intact / abstracts);
    }

    private static int countWords(String s) {
        int count = 0;
        boolean inWord = false;
        for (int i = 0; i < s.length(); i++) {
            boolean ws = Character.isWhitespace(s.charAt(i));
            if (!ws && !inWord) count++;
            inWord = !ws;
        }
        return count;
    }
}