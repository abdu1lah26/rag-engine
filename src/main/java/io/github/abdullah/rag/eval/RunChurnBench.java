package io.github.abdullah.rag.eval;

import io.github.abdullah.rag.embedding.EmbeddingCache;
import io.github.abdullah.rag.model.Document;
import io.github.abdullah.rag.model.Query;
import io.github.abdullah.rag.retrieval.LiveEngine;
import io.github.abdullah.rag.retrieval.ScoredDoc;
import io.github.abdullah.rag.wal.WalPayloads.AddOp;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;
import java.util.Set;

/**
 * M6 benchmark, part 2: churn. Delete 10% of docs and re-put (update) another
 * 10% with identical
 * content, without compacting. Compare against a reference engine built from
 * scratch on the
 * survivors, then compact and compare again. Docs deleted from the corpus are
 * also removed from
 * each query's relevance judgments, so every engine is scored on the same
 * surviving ground truth.
 */
public final class RunChurnBench {

    private static final String MODEL = "onnx-all-MiniLM-L6-v2";
    private static final int K = 10;
    private static final long SEED = 42;
    private static final double DELETE_FRACTION = 0.10;
    private static final double UPDATE_FRACTION = 0.10;

    private RunChurnBench() {
    }

    private record Result(double ndcg, int scored, int deletedReturned, double p50, double p95,
            List<List<String>> rankings) {
    }

    private static AddOp op(Document d, Map<String, float[]> vectors) {
        float[] v = vectors.get(d.id());
        if (v == null) {
            throw new IllegalStateException("no cached vector for doc " + d.id());
        }
        return new AddOp(d.id(), d.title() == null ? "" : d.title(), d.text(), MODEL, v);
    }

    private static Result evaluate(LiveEngine engine, BeirDataset ds, Set<String> deleted) {
        double[] latencies = new double[ds.queries().size()];
        List<List<String>> rankings = new ArrayList<>();
        double sum = 0;
        int scored = 0;
        int deletedReturned = 0;
        int i = 0;
        for (Query q : ds.queries()) {
            long t = System.nanoTime();
            List<ScoredDoc> res = engine.search(q.text(), K);
            latencies[i++] = (System.nanoTime() - t) / 1e6;

            List<String> ranked = new ArrayList<>();
            for (ScoredDoc s : res) {
                ranked.add(s.docId());
                if (deleted.contains(s.docId())) {
                    deletedReturned++;
                }
            }
            rankings.add(ranked);

            Map<String, Integer> relevant = new HashMap<>(ds.qrels().forQuery(q.id()));
            relevant.keySet().removeAll(deleted);
            if (relevant.isEmpty()) {
                continue; // no surviving relevant doc: excluded for every engine alike
            }
            sum += Metrics.ndcgAtK(ranked, relevant, K);
            scored++;
        }
        double[] sorted = latencies.clone();
        Arrays.sort(sorted);
        return new Result(sum / scored, scored, deletedReturned,
                sorted[sorted.length / 2], sorted[(int) (sorted.length * 0.95)], rankings);
    }

    /** Warm-up pass first (JIT), then the measured pass. */
    private static Result measure(String label, LiveEngine engine, BeirDataset ds, Set<String> deleted) {
        evaluate(engine, ds, deleted);
        Result r = evaluate(engine, ds, deleted);
        System.out.println(String.format(Locale.ROOT,
                "%-26s nDCG@10=%.4f (over %d queries)  deleted docs returned=%d  p50=%.3f ms  p95=%.3f ms",
                label, r.ndcg(), r.scored(), r.deletedReturned(), r.p50(), r.p95()));
        return r;
    }

    private static int differing(Result a, Result b) {
        int n = 0;
        for (int i = 0; i < a.rankings().size(); i++) {
            if (!a.rankings().get(i).equals(b.rankings().get(i))) {
                n++;
            }
        }
        return n;
    }

    /**
     * Queries whose top-10 holds a different SET of docs (not just a different
     * order).
     */
    private static int differingSets(Result a, Result b) {
        int n = 0;
        for (int i = 0; i < a.rankings().size(); i++) {
            if (!new HashSet<>(a.rankings().get(i)).equals(new HashSet<>(b.rankings().get(i)))) {
                n++;
            }
        }
        return n;
    }

    public static void main(String[] args) throws Exception {
        Path dataDir = Path.of("data", "scifact");
        BeirDataset dataset = BeirLoader.load(dataDir, "test");
        Map<String, float[]> queryVectors = RunDenseBaseline.loadQueryVectors(dataDir, dataset);

        EmbeddingCache.Loaded cache = EmbeddingCache.loadFor(
                dataDir.resolve("embeddings-" + MODEL + "-docs.bin"), MODEL, 384);
        Map<String, float[]> vectors = new HashMap<>();
        for (int i = 0; i < cache.ids().size(); i++) {
            vectors.put(cache.ids().get(i), cache.vectors().get(i));
        }

        List<Document> corpus = dataset.corpus();
        List<String> shuffled = new ArrayList<>();
        for (Document d : corpus) {
            shuffled.add(d.id());
        }
        Collections.sort(shuffled);
        Collections.shuffle(shuffled, new Random(SEED));
        int nDelete = (int) (corpus.size() * DELETE_FRACTION);
        int nUpdate = (int) (corpus.size() * UPDATE_FRACTION);
        Set<String> deleted = new HashSet<>(shuffled.subList(0, nDelete));
        Set<String> updated = new HashSet<>(shuffled.subList(nDelete, nDelete + nUpdate));
        System.out.printf("corpus=%d  deleting=%d  updating=%d  surviving=%d%n",
                corpus.size(), deleted.size(), updated.size(), corpus.size() - deleted.size());

        Path liveDir = Path.of("results", "churn-live");
        Path refDir = Path.of("results", "churn-ref");
        RunLiveBench.deleteTree(liveDir);
        RunLiveBench.deleteTree(refDir);

        try (LiveEngine live = RunLiveBench.open(liveDir, queryVectors);
                LiveEngine ref = RunLiveBench.open(refDir, queryVectors)) {

            // Live engine: full corpus, compacted into a base, then churned without
            // compaction.
            for (Document d : corpus) {
                live.put(op(d, vectors));
            }
            live.compact();
            for (Document d : corpus) {
                if (deleted.contains(d.id())) {
                    live.delete(d.id());
                }
            }
            for (Document d : corpus) {
                if (updated.contains(d.id())) {
                    live.put(op(d, vectors)); // identical content: exercises the replace path
                }
            }

            // Reference: built from scratch on the survivors only.
            for (Document d : corpus) {
                if (!deleted.contains(d.id())) {
                    ref.put(op(d, vectors));
                }
            }
            ref.compact();

            System.out.println("--- scored against the surviving relevance judgments");
            Result churned = measure("live, churned (no compact)", live, dataset, deleted);
            Result fresh = measure("reference (from scratch)", ref, dataset, deleted);

            long t = System.nanoTime();
            live.compact();
            System.out.printf(Locale.ROOT, "compaction after churn: %.0f ms%n", (System.nanoTime() - t) / 1e6);
            Result compacted = measure("live, compacted", live, dataset, deleted);

            int diffAll = differing(churned, fresh);
            int diffSets = differingSets(churned, fresh);
            System.out.printf(Locale.ROOT,
                    "churned vs reference:   top-10 differs on %d of %d queries (%d different doc set, %d order only), nDCG diff %+.4f%n",
                    diffAll, dataset.queries().size(), diffSets, diffAll - diffSets,
                    churned.ndcg() - fresh.ndcg());
            System.out.printf("compacted vs reference: top-10 differs on %d of %d queries, nDCG diff %+.4f%n",
                    differing(compacted, fresh), dataset.queries().size(), compacted.ndcg() - fresh.ndcg());
        }
    }
}