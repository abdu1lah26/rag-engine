package io.github.abdullah.rag.retrieval;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import io.github.abdullah.rag.model.Document;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;

class RerankingRetrieverTest {

    /** First stage with a fixed ranking; records how it was called. */
    private static final class Stub implements Retriever {
        private final List<String> ids;
        int indexCalls = 0;
        int lastK = -1;

        Stub(String... ids) {
            this.ids = List.of(ids);
        }

        @Override
        public String name() {
            return "stub";
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

    private static final List<Document> CORPUS = List.of(
            new Document("a", "", "text-a"),
            new Document("b", "", "text-b"),
            new Document("c", "", "text-c"),
            new Document("d", "", "text-d"));

    private static List<String> ids(List<ScoredDoc> r) {
        return r.stream().map(ScoredDoc::docId).toList();
    }

    @Test
    void reordersByScorerAndCutsToK() {
        // The scorer prefers c, then a, then b: the opposite of first-stage order.
        RerankingRetriever.PairScorer scorer = (q, d) -> switch (d) {
            case "text-c" -> 3f;
            case "text-a" -> 2f;
            case "text-b" -> 1f;
            default -> 0f;
        };
        RerankingRetriever r = new RerankingRetriever(new Stub("a", "b", "c"), scorer, "s", 3, true);
        r.index(CORPUS);
        assertEquals(List.of("c", "a"), ids(r.search("q", 2)));
    }

    @Test
    void onlyRerankDepthCandidatesAreScored() {
        // d is ranked 4th by the first stage and the scorer loves it, but depth is 3.
        RerankingRetriever.PairScorer scorer = (q, d) -> d.equals("text-d") ? 100f : 1f;
        Stub first = new Stub("a", "b", "c", "d");
        RerankingRetriever r = new RerankingRetriever(first, scorer, "s", 3, true);
        r.index(CORPUS);
        assertEquals(List.of("a", "b", "c"), ids(r.search("q", 10)));
        assertEquals(3, first.lastK);
    }

    @Test
    void scorerSeesQueryAndIndexableText() {
        List<String> seen = new ArrayList<>();
        RerankingRetriever.PairScorer scorer = (q, d) -> {
            seen.add(q + "|" + d);
            return 0f;
        };
        RerankingRetriever r = new RerankingRetriever(new Stub("t"), scorer, "s", 5, true);
        r.index(List.of(new Document("t", "Title", "Body")));
        r.search("my query", 5);
        assertEquals(List.of("my query|Title. Body"), seen);
    }

    @Test
    void equalScoresTieBreakByDocId() {
        RerankingRetriever r = new RerankingRetriever(new Stub("b", "a"), (q, d) -> 1f, "s", 5, true);
        r.index(CORPUS);
        assertEquals(List.of("a", "b"), ids(r.search("q", 5)));
    }

    @Test
    void indexFirstStageFlagControlsIndexing() {
        Stub built = new Stub("a");
        new RerankingRetriever(built, (q, d) -> 0f, "s", 5, true).index(CORPUS);
        assertEquals(1, built.indexCalls);

        Stub reused = new Stub("a");
        new RerankingRetriever(reused, (q, d) -> 0f, "s", 5, false).index(CORPUS);
        assertEquals(0, reused.indexCalls);
    }

    @Test
    void missingDocumentTextFailsLoudly() {
        RerankingRetriever r = new RerankingRetriever(new Stub("zzz"), (q, d) -> 0f, "s", 5, true);
        r.index(CORPUS);
        assertThrows(IllegalStateException.class, () -> r.search("q", 5));
    }

    @Test
    void rejectsBadDepth() {
        assertThrows(IllegalArgumentException.class,
                () -> new RerankingRetriever(new Stub("a"), (q, d) -> 0f, "s", 0, true));
    }
}