package io.github.abdullah.rag.wal;

import io.github.abdullah.rag.wal.WalPayloads.AddOp;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

/**
 * In-memory result of applying WAL records in order.
 *  - live:    docs added or updated since the base segment, minus deleted ones
 *  - touched: every docId any record has mentioned; the base segment's copy of
 *             such a doc is hidden (it was replaced or deleted)
 * Used by live writes and by recovery replay, so both produce the same state.
 */
public final class LiveDocs {

    private final Map<String, AddOp> live = new HashMap<>();
    private final Set<String> touched = new HashSet<>();
    private long lastSeq;

    public LiveDocs(long lastSeq) {
        this.lastSeq = lastSeq;
    }

    public synchronized void apply(WalRecord r) {
        if (r.seq() <= lastSeq) {
            throw new IllegalStateException("record seq " + r.seq() + " is not after " + lastSeq);
        }
        switch (r.type()) {
            case ADD -> {
                AddOp op = WalPayloads.decodeAdd(r.payload());
                live.put(op.docId(), op);
                touched.add(op.docId());
            }
            case DELETE -> {
                String docId = WalPayloads.decodeDelete(r.payload());
                live.remove(docId);
                touched.add(docId);
            }
            case CHECKPOINT -> { }
        }
        lastSeq = r.seq();
    }

        public record Snapshot(long seq, Map<String, AddOp> live, Set<String> touched) {}

    /** One consistent view of the state, taken under the lock. */
    public synchronized Snapshot snapshot() {
        return new Snapshot(lastSeq, new HashMap<>(live), new HashSet<>(touched));
    }

    /** Latest live version of a doc from the delta, or null. */
    public synchronized AddOp get(String docId) {
        return live.get(docId);
    }

    /** Should a hit from the base segment be shown? False if replaced or deleted. */
    public synchronized boolean baseVisible(String docId) {
        return !touched.contains(docId);
    }

    public synchronized Map<String, AddOp> liveDocs() {
        return Collections.unmodifiableMap(new HashMap<>(live));
    }

    public synchronized Set<String> touchedIds() {
        return Collections.unmodifiableSet(new HashSet<>(touched));
    }

    public synchronized long lastSeq() {
        return lastSeq;
    }
}