package io.github.abdullah.rag.eval;

import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Retrieval metrics for ONE query. Averaging over all queries happens
 * elsewhere.
 * Inputs: the ranked doc ids a retriever returned (best first) and the
 * relevant docs for that query (doc id -> relevance score).
 */
public final class Metrics {

    private Metrics() {
    }

    /**
     * Recall@k = (relevant docs found in the top k) / (all relevant docs).
     * Each doc is counted once even if the ranking repeats it.
     */
    public static double recallAtK(List<String> ranked, Map<String, Integer> relevant, int k) {
        if (k <= 0) {
            throw new IllegalArgumentException("k must be positive, got " + k);
        }
        if (relevant.isEmpty()) {
            throw new IllegalArgumentException("query has no relevant docs; it cannot be scored");
        }
        Set<String> found = new HashSet<>();
        int limit = Math.min(k, ranked.size());
        for (int i = 0; i < limit; i++) {
            String docId = ranked.get(i);
            if (relevant.containsKey(docId)) {
                found.add(docId);
            }
        }
        return (double) found.size() / relevant.size();
    }

    /**
     * Reciprocal rank = 1 / (rank of the first relevant doc), ranks start at 1.
     * Only the top k results are examined; if none is relevant, returns 0.
     */
    public static double reciprocalRank(List<String> ranked, Map<String, Integer> relevant, int k) {
        if (k <= 0) {
            throw new IllegalArgumentException("k must be positive, got " + k);
        }
        if (relevant.isEmpty()) {
            throw new IllegalArgumentException("query has no relevant docs; it cannot be scored");
        }
        int limit = Math.min(k, ranked.size());
        for (int i = 0; i < limit; i++) {
            if (relevant.containsKey(ranked.get(i))) {
                return 1.0 / (i + 1);
            }
        }
        return 0.0;
    }

    /**
     * nDCG@k with linear gain (gain = relevance score), as in BEIR's evaluator.
     * A doc repeated in the ranking earns its gain only once.
     */
    public static double ndcgAtK(List<String> ranked, Map<String, Integer> relevant, int k) {
        if (k <= 0) {
            throw new IllegalArgumentException("k must be positive, got " + k);
        }
        if (relevant.isEmpty()) {
            throw new IllegalArgumentException("query has no relevant docs; it cannot be scored");
        }

        // DCG of what the retriever returned
        double dcg = 0.0;
        Set<String> seen = new HashSet<>();
        int limit = Math.min(k, ranked.size());
        for (int i = 0; i < limit; i++) {
            String docId = ranked.get(i);
            if (seen.add(docId)) {
                dcg += relevant.getOrDefault(docId, 0) * discount(i + 1);
            }
        }

        // DCG of the perfect ranking: best scores first
        List<Integer> ideal = relevant.values().stream()
                .sorted(java.util.Comparator.reverseOrder())
                .toList();
        double idcg = 0.0;
        for (int i = 0; i < Math.min(k, ideal.size()); i++) {
            idcg += ideal.get(i) * discount(i + 1);
        }

        return idcg == 0.0 ? 0.0 : dcg / idcg;
    }

    /** 1 / log2(rank + 1), with rank starting at 1. */
    private static double discount(int rank) {
        return Math.log(2) / Math.log(rank + 1);
    }
}