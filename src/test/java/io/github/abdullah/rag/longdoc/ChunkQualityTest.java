package io.github.abdullah.rag.longdoc;

import static org.junit.jupiter.api.Assertions.*;

import io.github.abdullah.rag.chunking.FixedSizeChunker;
import io.github.abdullah.rag.longdoc.ChunkQuality.Result;
import io.github.abdullah.rag.longdoc.LongDocBuilder.LongDocs;
import io.github.abdullah.rag.longdoc.LongDocBuilder.Span;
import io.github.abdullah.rag.model.Document;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

class ChunkQualityTest {

    /** One long doc "L0" with abstracts of exactly 4 words each, joined by a single space. */
    private static LongDocs abstracts(String... parts) {
        StringBuilder sb = new StringBuilder();
        List<Span> spans = new ArrayList<>();
        for (int i = 0; i < parts.length; i++) {
            if (i > 0) sb.append(" ");
            int start = sb.length();
            sb.append(parts[i]);
            spans.add(new Span("x" + i, start, sb.length()));
        }
        Map<String, List<Span>> map = new HashMap<>();
        map.put("L0", spans);
        return new LongDocs(List.of(new Document("L0", "", sb.toString())), map);
    }

    private static LongDocs three() {
        return abstracts("a1 a2 a3 a4", "b1 b2 b3 b4", "c1 c2 c3 c4");
    }

    @Test
    void oracleIsPerfect() {
        LongDocs ld = three();
        Result r = ChunkQuality.measure(new OracleChunker(ld.spans()), ld);
        assertEquals(3, r.chunks());
        assertEquals(4.0, r.avgWords(), 1e-9);
        assertEquals(1.0, r.purity(), 1e-9);
        assertEquals(1.0, r.boundaryIntegrity(), 1e-9);
        assertEquals(1.0, r.intactRate(), 1e-9);
    }

    @Test
    void chunksAlignedWithAbstractsArePerfect() {
        Result r = ChunkQuality.measure(new FixedSizeChunker(4, 0), three());
        assertEquals(1.0, r.purity(), 1e-9);
        assertEquals(1.0, r.boundaryIntegrity(), 1e-9);
        assertEquals(1.0, r.intactRate(), 1e-9);
    }

    @Test
    void tinyChunksArePureButNeverKeepAnAbstractIntact() {
        Result r = ChunkQuality.measure(new FixedSizeChunker(2, 0), three());
        assertEquals(6, r.chunks());
        assertEquals(2.0, r.avgWords(), 1e-9);
        assertEquals(1.0, r.purity(), 1e-9);
        assertEquals(1.0, r.boundaryIntegrity(), 1e-9);
        assertEquals(0.0, r.intactRate(), 1e-9);
    }

    @Test
    void chunksStraddlingBoundariesScorePoorly() {
        // size 6 -> chunk0 = a1..b2, chunk1 = b3..c4: both cross a boundary, only a and c stay whole
        Result r = ChunkQuality.measure(new FixedSizeChunker(6, 0), three());
        assertEquals(2, r.chunks());
        assertEquals(6.0, r.avgWords(), 1e-9);
        assertEquals(0.0, r.purity(), 1e-9);
        assertEquals(0.0, r.boundaryIntegrity(), 1e-9);
        assertEquals(2.0 / 3.0, r.intactRate(), 1e-9);
    }

    @Test
    void singleAbstractHasNoBoundariesToCross() {
        Result r = ChunkQuality.measure(new FixedSizeChunker(2, 0), abstracts("a1 a2 a3 a4"));
        assertEquals(1.0, r.boundaryIntegrity(), 1e-9);
    }
}