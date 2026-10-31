package io.github.abdullah.rag.eval;

import io.github.abdullah.rag.model.Document;
import io.github.abdullah.rag.model.Qrels;
import io.github.abdullah.rag.model.Query;
import io.github.abdullah.rag.retrieval.Retriever;
import io.github.abdullah.rag.retrieval.ScoredDoc;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class EvalRunnerTest {

    private static final double EPS = 1e-9;

    /** Returns a fixed ranking per query text; ignores the corpus. */
    private static final class FakeRetriever implements Retriever {
        private final Map<String, List<String>> rankings;
        boolean indexed = false;

        FakeRetriever(Map<String, List<String>> rankings) {
            this.rankings = rankings;
        }

        @Override public String name() { return "fake"; }

        @Override public void index(List<Document> corpus) { indexed = true; }

        @Override public List<ScoredDoc> search(String queryText, int k) {
            List<ScoredDoc> out = new ArrayList<>();
            List<String> ids = rankings.getOrDefault(queryText, List.of());
            for (int i = 0; i < Math.min(k, ids.size()); i++) {
                out.add(new ScoredDoc(ids.get(i), ids.size() - i));
            }
            return out;
        }
    }

    private static BeirDataset tinyDataset() {
        List<Document> corpus = List.of(
                new Document("a", "A", "doc a"),
                new Document("b", "B", "doc b"),
                new Document("x", "X", "doc x"));
        List<Query> queries = List.of(new Query("q1", "first"), new Query("q2", "second"));
        Qrels qrels = new Qrels(Map.of(
                "q1", Map.of("a", 1),
                "q2", Map.of("b", 1)));
        return new BeirDataset("tiny/test", corpus, queries, qrels);
    }

    @Test
    void averagesMetricsAcrossQueries() {
        // q1: relevant a at rank 1 -> recall 1, mrr 1,   ndcg 1
        // q2: relevant b at rank 2 -> recall 1, mrr 0.5, ndcg 1/log2(3)
        FakeRetriever fake = new FakeRetriever(Map.of(
                "first", List.of("a", "x"),
                "second", List.of("x", "b")));

        EvalReport report = EvalRunner.run(fake, tinyDataset(), 10);

        assertTrue(fake.indexed, "runner must call index()");
        assertEquals("fake", report.retriever());
        assertEquals(2, report.queries());
        assertEquals(1.0, report.recall(), EPS);
        assertEquals(0.75, report.mrr(), EPS);
        double ndcgQ2 = Math.log(2) / Math.log(3);
        assertEquals((1.0 + ndcgQ2) / 2, report.ndcg(), EPS);
    }

    @Test
    void queryWithNoResultsCountsAsZeroNotSkipped() {
        // q1 perfect, q2 returns nothing -> averages are exactly half of q1's scores
        FakeRetriever fake = new FakeRetriever(Map.of("first", List.of("a")));

        EvalReport report = EvalRunner.run(fake, tinyDataset(), 10);

        assertEquals(0.5, report.recall(), EPS);
        assertEquals(0.5, report.mrr(), EPS);
        assertEquals(0.5, report.ndcg(), EPS);
    }

    @Test
    void percentileUsesNearestRank() {
        double[] values = {10, 9, 8, 7, 6, 5, 4, 3, 2, 1}; // unsorted on purpose
        assertEquals(5.0, EvalRunner.percentile(values, 50), EPS);
        assertEquals(10.0, EvalRunner.percentile(values, 95), EPS);
        assertEquals(0.0, EvalRunner.percentile(new double[0], 50), EPS);
    }
}