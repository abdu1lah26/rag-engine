package io.github.abdullah.rag.longdoc;

import io.github.abdullah.rag.chunking.Chunk;
import io.github.abdullah.rag.chunking.Chunker;
import io.github.abdullah.rag.longdoc.LongDocBuilder.Span;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Upper bound: cuts exactly at the known abstract boundaries (it cheats by reading the spans).
 * Not a real chunker; it exists to validate the harness and give every other row a ceiling.
 */
public final class OracleChunker implements Chunker {

    private final Map<String, List<Span>> spans;

    public OracleChunker(Map<String, List<Span>> spans) {
        this.spans = spans;
    }

    @Override
    public String name() {
        return "oracle (abstract boundaries)";
    }

    @Override
    public List<Chunk> chunk(String docId, String text) {
        List<Chunk> out = new ArrayList<>();
        int index = 0;
        for (Span s : spans.get(docId)) {
            out.add(Chunk.of(docId, index++, text.substring(s.start(), s.end()), s.start(), s.end()));
        }
        return out;
    }
}