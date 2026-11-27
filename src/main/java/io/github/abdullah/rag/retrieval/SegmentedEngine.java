package io.github.abdullah.rag.retrieval;

import io.github.abdullah.rag.analysis.TextAnalyzer;
import io.github.abdullah.rag.bm25.SegmentedBm25;
import io.github.abdullah.rag.fusion.RrfFusion;
import io.github.abdullah.rag.vector.VectorIndex;
import io.github.abdullah.rag.index.InvertedIndex;
import io.github.abdullah.rag.model.Document;
import io.github.abdullah.rag.wal.DurableStore;
import io.github.abdullah.rag.wal.LiveDocs.Snapshot;
import io.github.abdullah.rag.wal.WalPayloads.AddOp;
import java.util.Comparator;
import java.util.List;
import java.util.function.Function;
import java.util.function.Predicate;

/**
 * Hybrid search over an immutable base (BM25 + vectors) plus the live delta from the WAL.
 * Each query takes one snapshot of the state, so BM25, dense and the tombstone filter
 * all see the same version of the data.
 */
public final class SegmentedEngine {

    private record CachedDelta(long seq, InvertedIndex index) {}

    private final InvertedIndex baseBm25;
    private final VectorIndex baseVectors;
    private final TextAnalyzer analyzer;
    private final Function<String, float[]> embedQuery;
    private final DurableStore store;
    private final RrfFusion fusion;
    private final int candidateDepth;
    private final SegmentedBm25 bm25;
    private final SegmentedDense dense;

    private volatile CachedDelta cachedDelta;

    public SegmentedEngine(InvertedIndex baseBm25, VectorIndex baseVectors, TextAnalyzer analyzer,
                           String model, Function<String, float[]> embedQuery,
                           DurableStore store, RrfFusion fusion, int candidateDepth) {
        this.baseBm25 = baseBm25;
        this.baseVectors = baseVectors;
        this.analyzer = analyzer;
        this.embedQuery = embedQuery;
        this.store = store;
        this.fusion = fusion;
        this.candidateDepth = candidateDepth;
        this.bm25 = new SegmentedBm25(analyzer);
        this.dense = new SegmentedDense(model);
    }

    public List<ScoredDoc> search(String queryText, int k) {
        Snapshot snap = store.state().snapshot();
        Predicate<String> baseVisible = id -> !snap.touched().contains(id);
        int hidden = snap.touched().size(); // upper bound on hidden base docs

        InvertedIndex delta = deltaIndex(snap);
        List<ScoredDoc> lexical = bm25.search(baseBm25, delta, baseVisible, queryText, candidateDepth);
        List<ScoredDoc> semantic = dense.search(baseVectors, snap.live().values(), baseVisible,
                hidden, embedQuery.apply(queryText), candidateDepth);
        return fusion.fuse(List.of(lexical, semantic), k);
    }

    /** The delta is small, so it is rebuilt whenever the state version changes. */
    private InvertedIndex deltaIndex(Snapshot snap) {
        CachedDelta c = cachedDelta;
        if (c != null && c.seq() == snap.seq()) {
            return c.index();
        }
        List<Document> docs = snap.live().values().stream()
                .sorted(Comparator.comparing(AddOp::docId))
                .map(op -> new Document(op.docId(), op.title(), op.text()))
                .toList();
        InvertedIndex idx = InvertedIndex.build(docs, analyzer);
        cachedDelta = new CachedDelta(snap.seq(), idx);
        return idx;
    }
}