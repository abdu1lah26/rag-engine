package io.github.abdullah.rag.eval;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class MetricsTest {

    private static final double EPS = 1e-9;

    // relevant docs: a, b, c (3 total)
    private static final Map<String, Integer> REL = Map.of("a", 1, "b", 1, "c", 1);

    @Test
    void recallCountsFractionOfRelevantFoundInTopK() {
        List<String> ranked = List.of("x", "a", "y", "b", "z");
        // top 2 = x, a -> 1 of 3 relevant
        assertEquals(1.0 / 3, Metrics.recallAtK(ranked, REL, 2), EPS);
        // top 4 = x, a, y, b -> 2 of 3
        assertEquals(2.0 / 3, Metrics.recallAtK(ranked, REL, 4), EPS);
    }

    @Test
    void recallIsOneWhenAllRelevantFound() {
        List<String> ranked = List.of("c", "b", "a");
        assertEquals(1.0, Metrics.recallAtK(ranked, REL, 3), EPS);
    }

    @Test
    void recallIsZeroWhenNothingRelevantFound() {
        List<String> ranked = List.of("x", "y", "z");
        assertEquals(0.0, Metrics.recallAtK(ranked, REL, 3), EPS);
    }

    @Test
    void kLargerThanRankingIsFine() {
        List<String> ranked = List.of("a");
        assertEquals(1.0 / 3, Metrics.recallAtK(ranked, REL, 100), EPS);
    }

    @Test
    void duplicateDocsInRankingCountOnce() {
        List<String> ranked = List.of("a", "a", "a");
        assertEquals(1.0 / 3, Metrics.recallAtK(ranked, REL, 3), EPS);
    }

    @Test
    void badInputFailsLoudly() {
        assertThrows(IllegalArgumentException.class, () -> Metrics.recallAtK(List.of("a"), REL, 0));
        assertThrows(IllegalArgumentException.class, () -> Metrics.recallAtK(List.of("a"), Map.of(), 5));
    }

        @Test
    void reciprocalRankIsOneOverRankOfFirstRelevant() {
        assertEquals(1.0, Metrics.reciprocalRank(List.of("a", "x"), REL, 10), EPS);
        assertEquals(0.5, Metrics.reciprocalRank(List.of("x", "b", "a"), REL, 10), EPS);
        assertEquals(1.0 / 3, Metrics.reciprocalRank(List.of("x", "y", "c"), REL, 10), EPS);
    }

    @Test
    void reciprocalRankIgnoresResultsBeyondK() {
        // first relevant doc is at rank 3, but we only look at the top 2
        assertEquals(0.0, Metrics.reciprocalRank(List.of("x", "y", "a"), REL, 2), EPS);
    }

    @Test
    void reciprocalRankIsZeroWhenNothingRelevant() {
        assertEquals(0.0, Metrics.reciprocalRank(List.of("x", "y"), REL, 10), EPS);
    }

        private static double d(int rank) {
        return Math.log(2) / Math.log(rank + 1);
    }

    @Test
    void ndcgIsOneForPerfectRanking() {
        assertEquals(1.0, Metrics.ndcgAtK(List.of("a", "b", "c", "x"), REL, 10), EPS);
    }

    @Test
    void ndcgIsZeroWhenNothingRelevant() {
        assertEquals(0.0, Metrics.ndcgAtK(List.of("x", "y", "z"), REL, 10), EPS);
    }

    @Test
    void ndcgPenalisesLowerRanks() {
        // hits at ranks 1 and 3: DCG = d(1) + d(3); ideal (k=3) = d(1) + d(2) + d(3)
        double expected = (d(1) + d(3)) / (d(1) + d(2) + d(3));
        assertEquals(expected, Metrics.ndcgAtK(List.of("a", "x", "b"), REL, 3), EPS);
    }

    @Test
    void ndcgUsesGradedRelevance() {
        Map<String, Integer> graded = Map.of("a", 2, "b", 1);
        // ranked b then a: DCG = 1*d(1) + 2*d(2); ideal = 2*d(1) + 1*d(2)
        double expected = (1 * d(1) + 2 * d(2)) / (2 * d(1) + 1 * d(2));
        assertEquals(expected, Metrics.ndcgAtK(List.of("b", "a"), graded, 10), EPS);
    }

    @Test
    void ndcgCountsDuplicatesOnce() {
        // only the first "a" earns gain: DCG = d(1); ideal (k=3) = d(1) + d(2) + d(3)
        double expected = d(1) / (d(1) + d(2) + d(3));
        assertEquals(expected, Metrics.ndcgAtK(List.of("a", "a", "a"), REL, 3), EPS);
    }
}