package io.github.abdullah.rag.retrieval;

import io.github.abdullah.rag.vector.VectorIndex;
import io.github.abdullah.rag.wal.WalPayloads.AddOp;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Comparator;
import java.util.List;
import java.util.function.Predicate;

/**
 * Dense search over a base vector index plus a small delta searched by brute force.
 * Base hits for replaced or deleted docs are dropped; the base is over-fetched by
 * hiddenCount so filtering cannot leave fewer than k visible results.
 * Score = dot product (= 1 - distance for unit vectors), same scale as HnswIndex.
 */
public final class SegmentedDense {

    private final String expectedModel;

    public SegmentedDense(String expectedModel) {
        this.expectedModel = expectedModel;
    }

    public List<ScoredDoc> search(VectorIndex base, Collection<AddOp> deltaDocs,
                                  Predicate<String> baseVisible, int hiddenCount,
                                  float[] query, int k) {
        if (k <= 0) {
            throw new IllegalArgumentException("k must be positive, got " + k);
        }
        List<ScoredDoc> all = new ArrayList<>();

        int fetch = (int) Math.min((long) base.size(), (long) k + hiddenCount);
        if (fetch > 0) {
            for (ScoredDoc d : base.search(query, fetch)) {
                if (baseVisible.test(d.docId())) {
                    all.add(d);
                }
            }
        }

        for (AddOp op : deltaDocs) {
            if (!op.model().equals(expectedModel)) {
                throw new IllegalStateException("doc " + op.docId() + " was embedded with '" + op.model()
                        + "' but the index expects '" + expectedModel + "'");
            }
            float[] v = op.vector();
            if (v.length != query.length) {
                throw new IllegalArgumentException("doc " + op.docId() + " has dimension " + v.length
                        + " but the query has " + query.length);
            }
            double dot = 0;
            for (int i = 0; i < v.length; i++) {
                dot += (double) v[i] * query[i];
            }
            all.add(new ScoredDoc(op.docId(), dot));
        }

        all.sort(Comparator.comparingDouble(ScoredDoc::score).reversed().thenComparing(ScoredDoc::docId));
        return all.size() <= k ? all : new ArrayList<>(all.subList(0, k));
    }
}