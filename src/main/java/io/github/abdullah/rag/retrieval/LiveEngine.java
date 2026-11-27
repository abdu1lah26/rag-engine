package io.github.abdullah.rag.retrieval;

import io.github.abdullah.rag.analysis.TextAnalyzer;
import io.github.abdullah.rag.fusion.RrfFusion;
import io.github.abdullah.rag.vector.VectorIndex;
import io.github.abdullah.rag.index.InvertedIndex;
import io.github.abdullah.rag.model.Document;
import io.github.abdullah.rag.wal.DurableStore;
import io.github.abdullah.rag.wal.DurableStore.CompactionResult;
import io.github.abdullah.rag.wal.WalPayloads.AddOp;
import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.locks.ReentrantReadWriteLock;
import java.util.function.Function;
import java.util.function.Supplier;

/**
 * A searchable, writable index: DurableStore (WAL + generations) + base indexes + SegmentedEngine.
 * Searches share a read lock; compaction takes the write lock, so a search never sees
 * the new generation's tombstone state together with the old generation's base indexes.
 */
public final class LiveEngine implements AutoCloseable {

    private final DurableStore store;
    private final TextAnalyzer analyzer;
    private final String model;
    private final Function<String, float[]> embedQuery;
    private final Supplier<VectorIndex> vectorFactory;
    private final RrfFusion fusion;
    private final int candidateDepth;
    private final ReentrantReadWriteLock lock = new ReentrantReadWriteLock();
    private SegmentedEngine engine; // null only if a rebuild failed after a commit

    public LiveEngine(Path dir, TextAnalyzer analyzer, String model, Function<String, float[]> embedQuery,
                      Supplier<VectorIndex> vectorFactory, RrfFusion fusion, int candidateDepth)
            throws IOException {
        this.analyzer = analyzer;
        this.model = model;
        this.embedQuery = embedQuery;
        this.vectorFactory = vectorFactory;
        this.fusion = fusion;
        this.candidateDepth = candidateDepth;
        this.store = DurableStore.open(dir);
        try {
            this.engine = buildEngine();
        } catch (RuntimeException e) {
            store.close();
            throw e;
        }
    }

    /** Base indexes are built from the base docs in id order, so the build is deterministic. */
    private SegmentedEngine buildEngine() {
        List<AddOp> docs = store.baseDocs();
        List<Document> corpus = docs.stream()
                .map(op -> new Document(op.docId(), op.title(), op.text()))
                .toList();
        InvertedIndex bm25 = InvertedIndex.build(corpus, analyzer);
        VectorIndex vectors = vectorFactory.get();
        for (AddOp op : docs) {
            if (!op.model().equals(model)) {
                throw new IllegalStateException("base doc " + op.docId() + " was embedded with '"
                        + op.model() + "' but the engine expects '" + model + "'");
            }
            vectors.add(op.docId(), op.vector());
        }
        return new SegmentedEngine(bm25, vectors, analyzer, model, embedQuery, store, fusion, candidateDepth);
    }

    public void put(AddOp op) throws IOException {
        if (!op.model().equals(model)) {
            throw new IllegalArgumentException("doc " + op.docId() + " has model '" + op.model()
                    + "' but the engine expects '" + model + "'");
        }
        store.put(op);
    }

    public void delete(String docId) throws IOException {
        store.delete(docId);
    }

    public List<ScoredDoc> search(String queryText, int k) {
        lock.readLock().lock();
        try {
            if (engine == null) {
                throw new IllegalStateException("engine is unusable after a failed rebuild; reopen the store");
            }
            return engine.search(queryText, k);
        } finally {
            lock.readLock().unlock();
        }
    }

    /** Folds the delta into a new base and rebuilds the indexes. Searches wait while this runs. */
    public CompactionResult compact() throws IOException {
        lock.writeLock().lock();
        try {
            CompactionResult result = store.compact();
            try {
                engine = buildEngine();
            } catch (RuntimeException e) {
                engine = null; // committed on disk, but memory is stale: refuse to serve it
                throw e;
            }
            return result;
        } finally {
            lock.writeLock().unlock();
        }
    }

    public long generation() {
        return store.generation();
    }

    @Override
    public void close() throws IOException {
        store.close();
    }
}