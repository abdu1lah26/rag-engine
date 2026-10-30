package io.github.abdullah.rag.model;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

/**
 * Relevance judgments: for each query id, which doc ids are relevant and how
 * much.
 * SciFact uses binary relevance (score 1), but BEIR's format allows graded
 * scores,
 * and nDCG can use them, so we store the score instead of throwing it away.
 */
public final class Qrels {

    private final Map<String, Map<String, Integer>> judgments;

    public Qrels(Map<String, Map<String, Integer>> judgments) {
        Map<String, Map<String, Integer>> copy = new HashMap<>();
        for (var entry : judgments.entrySet()) {
            copy.put(entry.getKey(), Collections.unmodifiableMap(new HashMap<>(entry.getValue())));
        }
        this.judgments = Collections.unmodifiableMap(copy);
    }

    /** All query ids that have at least one judgment. */
    public Set<String> queryIds() {
        return judgments.keySet();
    }

    /**
     * doc id -> relevance score for one query. Empty map if the query is unknown.
     */
    public Map<String, Integer> forQuery(String queryId) {
        return judgments.getOrDefault(queryId, Map.of());
    }

    /**
     * Relevance score of one doc for one query. 0 means not relevant / not judged.
     */
    public int score(String queryId, String docId) {
        return forQuery(queryId).getOrDefault(docId, 0);
    }

    public int size() {
        return judgments.size();
    }
}