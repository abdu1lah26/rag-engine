package io.github.abdullah.rag.fusion;

import io.github.abdullah.rag.retrieval.ScoredDoc;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Reciprocal Rank Fusion. Each document scores the sum, over every input list
 * that contains it, of 1 / (rankConstant + rank), with rank starting at 1.
 *
 * Only ranks are used, never the input scores, because BM25 scores and cosine
 * similarities are on different scales and cannot be mixed directly.
 */
public final class RrfFusion {

    private final int rankConstant;

    public RrfFusion(int rankConstant) {
        if (rankConstant <= 0) {
            throw new IllegalArgumentException("rankConstant must be > 0");
        }
        this.rankConstant = rankConstant;
    }

    public int rankConstant() {
        return rankConstant;
    }

    /** Merge ranked lists (each best first) into the top k by fused score. */
    public List<ScoredDoc> fuse(List<List<ScoredDoc>> rankedLists, int k) {
        if (k < 0) {
            throw new IllegalArgumentException("k must be >= 0");
        }
        Map<String, Double> fused = new HashMap<>();
        for (List<ScoredDoc> list : rankedLists) {
            Set<String> seen = new HashSet<>();
            int rank = 0;
            for (ScoredDoc d : list) {
                if (!seen.add(d.docId())) {
                    continue; // a duplicate in one list counts once, at its best rank
                }
                rank++;
                fused.merge(d.docId(), 1.0 / (rankConstant + rank), Double::sum);
            }
        }
        List<ScoredDoc> out = new ArrayList<>(fused.size());
        fused.forEach((id, score) -> out.add(new ScoredDoc(id, score)));
        // Ties are broken by docId so results are deterministic across runs.
        out.sort(Comparator.comparingDouble(ScoredDoc::score).reversed()
                .thenComparing(ScoredDoc::docId));
        return out.size() > k ? new ArrayList<>(out.subList(0, k)) : out;
    }
}