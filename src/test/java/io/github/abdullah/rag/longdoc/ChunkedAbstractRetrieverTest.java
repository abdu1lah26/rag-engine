package io.github.abdullah.rag.longdoc;

import static org.junit.jupiter.api.Assertions.*;

import io.github.abdullah.rag.analysis.SimpleAnalyzer;
import io.github.abdullah.rag.chunking.FixedSizeChunker;
import io.github.abdullah.rag.chunking.RecursiveChunker;
import io.github.abdullah.rag.longdoc.LongDocBuilder.LongDocs;
import io.github.abdullah.rag.longdoc.LongDocBuilder.Span;
import io.github.abdullah.rag.model.Document;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

class ChunkedAbstractRetrieverTest {

    // four abstracts, exactly 8 words each
    private static final String[] TEXTS = {
        "insulin regulates blood glucose levels in diabetic patients.",
        "football stadium crowds cheered the final match loudly.",
        "quantum entanglement enables secure communication between distant photons.",
        "volcanic eruptions release ash clouds across wide regions."
    };

    /** One long doc "L0" holding all four abstracts, with exact spans. */
    private static LongDocs longDocs(String separator) {
        StringBuilder sb = new StringBuilder();
        List<Span> spans = new ArrayList<>();
        for (int i = 0; i < TEXTS.length; i++) {
            if (i > 0) sb.append(separator);
            int start = sb.length();
            sb.append(TEXTS[i]);
            spans.add(new Span("a" + (i + 1), start, sb.length()));
        }
        Map<String, List<Span>> map = new HashMap<>();
        map.put("L0", spans);
        return new LongDocs(List.of(new Document("L0", "", sb.toString())), map);
    }

    @Test
    void cleanBoundariesReturnTheRightAbstractFirst() {
        // paragraph separator + size 8 -> one chunk per abstract
        var r = new ChunkedAbstractRetriever(new RecursiveChunker(8, 0), new SimpleAnalyzer(), 100);
        r.index(longDocs("\n\n"));
        assertEquals(4, r.chunkCount());
        assertEquals("a1", r.search("insulin glucose", 10).get(0));
        assertEquals("a4", r.search("volcanic ash", 10).get(0));
    }

    @Test
    void straddlingChunkCreditsEveryAbstractItOverlaps() {
        // 32 words, windows of 12 -> chunk0 = a1 + first half of a2
        var r = new ChunkedAbstractRetriever(new FixedSizeChunker(12, 0), new SimpleAnalyzer(), 100);
        r.index(longDocs(" "));
        assertEquals(3, r.chunkCount());
        assertEquals(List.of("a1", "a2"), r.search("insulin", 10));
    }

    @Test
    void resultsAreDeduplicatedAcrossChunks() {
        var r = new ChunkedAbstractRetriever(new FixedSizeChunker(12, 0), new SimpleAnalyzer(), 100);
        r.index(longDocs(" "));
        // "insulin" hits chunk0 (a1,a2), "final" hits chunk1 (a2,a3): a2 must appear once
        List<String> res = r.search("insulin final", 10);
        assertEquals(3, res.size());
        assertEquals(3, new HashSet<>(res).size());
        assertTrue(res.containsAll(List.of("a1", "a2", "a3")));
    }

    @Test
    void kLimitsTheResult() {
        var r = new ChunkedAbstractRetriever(new FixedSizeChunker(12, 0), new SimpleAnalyzer(), 100);
        r.index(longDocs(" "));
        assertEquals(List.of("a1"), r.search("insulin", 1));
    }

    @Test
    void unknownQueryGivesEmptyResult() {
        var r = new ChunkedAbstractRetriever(new RecursiveChunker(8, 0), new SimpleAnalyzer(), 100);
        r.index(longDocs("\n\n"));
        assertTrue(r.search("zzzzqqq", 10).isEmpty());
    }

    @Test
    void invalidChunkDepthRejected() {
        assertThrows(IllegalArgumentException.class,
                () -> new ChunkedAbstractRetriever(new RecursiveChunker(8, 0), new SimpleAnalyzer(), 0));
    }
}