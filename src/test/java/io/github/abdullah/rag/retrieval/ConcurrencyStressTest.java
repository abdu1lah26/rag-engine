package io.github.abdullah.rag.retrieval;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.abdullah.rag.analysis.SimpleAnalyzer;
import io.github.abdullah.rag.fusion.RrfFusion;
import io.github.abdullah.rag.retrieval.SegmentedDenseTest.FakeIndex;
import io.github.abdullah.rag.wal.WalPayloads.AddOp;
import java.nio.file.Path;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class ConcurrencyStressTest {

    private static final int DOCS = 600;

    @TempDir
    Path dir;

    private static LiveEngine open(Path d) throws Exception {
        return new LiveEngine(d, new SimpleAnalyzer(), "m", q -> new float[] {1f, 0f},
                FakeIndex::new, new RrfFusion(60), 5000);
    }

    private static List<String> ids(List<ScoredDoc> r) {
        return r.stream().map(ScoredDoc::docId).toList();
    }

    /** Docs with index % 3 == 1 are deleted by the writer; all others stay live. */
    private static Set<String> expectedLive() {
        Set<String> s = new HashSet<>();
        for (int i = 0; i < DOCS; i++) {
            if (i % 3 != 1) {
                s.add("doc-" + i);
            }
        }
        return s;
    }

    private static Set<String> found(LiveEngine e) {
        return new HashSet<>(ids(e.search("alpha", 5000)));
    }

    @Test
    void readersNeverSeeADeletedDocWhileWritesAndCompactionsRun() throws Exception {
        Path d = dir.resolve("live");
        Set<String> deletedDone = ConcurrentHashMap.newKeySet();
        AtomicBoolean writerDone = new AtomicBoolean(false);
        AtomicInteger searches = new AtomicInteger();
        AtomicInteger compactions = new AtomicInteger();
        ExecutorService pool = Executors.newFixedThreadPool(4);

        try (LiveEngine e = open(d)) {
            Callable<Void> writer = () -> {
                try {
                    for (int i = 0; i < DOCS; i++) {
                        e.put(new AddOp("doc-" + i, "", "alpha doc" + i, "m", new float[] {1f, i / (float) DOCS}));
                        if (i % 3 == 2) {
                            String victim = "doc-" + (i - 1);
                            e.delete(victim);
                            deletedDone.add(victim); // recorded only after delete returned
                        }
                        if (i >= 3 && i % 7 == 6) { // update a doc that is never deleted
                            int target = (i / 3) * 3 - 3;
                            e.put(new AddOp("doc-" + target, "", "alpha updated" + i, "m", new float[] {1f, 0f}));
                        }
                    }
                } finally {
                    writerDone.set(true);
                }
                return null;
            };
            Callable<Void> compactor = () -> {
                while (!writerDone.get()) {
                    e.compact();
                    compactions.incrementAndGet();
                    Thread.sleep(50);
                }
                return null;
            };
            Callable<Void> reader = () -> {
                while (!writerDone.get()) {
                    Set<String> before = new HashSet<>(deletedDone); // deletes already complete
                    List<String> r = ids(e.search("alpha", 5000));
                    assertEquals(r.size(), new HashSet<>(r).size(), "duplicate ids in one result");
                    for (String id : r) {
                        assertFalse(before.contains(id), "deleted doc returned: " + id);
                    }
                    searches.incrementAndGet();
                }
                return null;
            };

            List<Future<Void>> futures = List.of(pool.submit(writer), pool.submit(compactor),
                    pool.submit(reader), pool.submit(reader));
            for (Future<Void> f : futures) {
                f.get(180, TimeUnit.SECONDS); // rethrows any assertion or exception from a thread
            }
            pool.shutdown();

            System.out.printf("stress: searches=%d compactions=%d generation=%d%n",
                    searches.get(), compactions.get(), e.generation());
            assertTrue(searches.get() > 0 && compactions.get() > 0, "test did not overlap reads and compactions");

            assertEquals(expectedLive(), found(e), "final state before last compaction");
            e.compact();
            assertEquals(expectedLive(), found(e), "final state after last compaction");
        } finally {
            pool.shutdownNow();
        }

        try (LiveEngine e = open(d)) {
            assertEquals(expectedLive(), found(e), "final state after restart");
        }
    }
}