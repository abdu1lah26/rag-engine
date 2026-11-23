package io.github.abdullah.rag.retrieval;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import io.github.abdullah.rag.fusion.RrfFusion;
import io.github.abdullah.rag.model.Document;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;

class HybridRetrieverTest {

    /** Returns a fixed ranking and records how it was called. */
    private static final class Stub implements Retriever {
        private final String name;
        private final List<String> ids;
        int indexCalls = 0;
        int lastK = -1;

        Stub(String name, String... ids) {
            this.name = name;
            this.ids = List.of(ids);
        }

        @Override
        public String name() {
            return name;
        }

        @Override
        public void index(List<Document> corpus) {
            indexCalls++;
        }

        @Override
        public List<ScoredDoc> search(String queryText, int k) {
            lastK = k;
            List<ScoredDoc> out = new ArrayList<>();
            for (String id : ids) {
                if (out.size() == k) {
                    break;
                }
                out.add(new ScoredDoc(id, 1.0));
            }
            return out;
        }
    }

    private static List<String> ids(List<ScoredDoc> r) {
        return r.stream().map(ScoredDoc::docId).toList();
    }

    @Test
    void fusesChildrenAndCutsToK() {
        Stub a = new Stub("a", "x", "y", "z");
        Stub b = new Stub("b", "y", "w", "x");
        HybridRetriever h = new HybridRetriever(List.of(a, b), new RrfFusion(60), 3, true);
        // y = 1/62 + 1/61, x = 1/61 + 1/63, then w (1/62) and z (1/63)
        assertEquals(List.of("y", "x"), ids(h.search("q", 2)));
    }

    @Test
    void asksChildrenForCandidateDepthNotK() {
        Stub a = new Stub("a", "x", "y", "z");
        Stub b = new Stub("b", "y", "w", "x");
        HybridRetriever h = new HybridRetriever(List.of(a, b), new RrfFusion(60), 50, true);
        h.search("q", 10);
        assertEquals(50, a.lastK);
        assertEquals(50, b.lastK);
    }

    @Test
    void indexChildrenFlagControlsIndexing() {
        Stub a = new Stub("a", "x");
        Stub b = new Stub("b", "y");
        new HybridRetriever(List.of(a, b), new RrfFusion(60), 10, true).index(List.of());
        assertEquals(1, a.indexCalls);
        assertEquals(1, b.indexCalls);

        Stub c = new Stub("c", "x");
        new HybridRetriever(List.of(c), new RrfFusion(60), 10, false).index(List.of());
        assertEquals(0, c.indexCalls);
    }

    @Test
    void nameListsChildrenAndParameters() {
        HybridRetriever h = new HybridRetriever(
                List.of(new Stub("bm25", "x"), new Stub("dense", "y")), new RrfFusion(60), 50, false);
        assertEquals("hybrid-rrf[bm25+dense, c=60, depth=50]", h.name());
    }

    @Test
    void rejectsBadArguments() {
        assertThrows(IllegalArgumentException.class,
                () -> new HybridRetriever(List.of(), new RrfFusion(60), 10, true));
        assertThrows(IllegalArgumentException.class,
                () -> new HybridRetriever(List.of(new Stub("a", "x")), new RrfFusion(60), 0, true));
    }
}