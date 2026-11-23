package io.github.abdullah.rag.eval;

import io.github.abdullah.rag.longdoc.PairedBootstrap;
import io.github.abdullah.rag.model.Query;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Offline sweep on the TRAIN split: blends the hybrid rank with the cached
 * cross-encoder rank using weighted RRF, score = (1-w)/(c+hybridRank) + w/(c+ceRank).
 * w=0 is the hybrid ordering, w=1 is pure rerank. No model calls.
 */
public final class RunBlendSweep {

    private static final int K = 10;
    private static final int[] RRF_CS = { 1, 10, 60 };
    private static final int RESAMPLES = 10_000;
    private static final long SEED = 42;
    private static final double TOLERANCE = 0.002;

    record Cand(String docId, int hybridRank, float ce) {
    }

    private record Config(int c, double w, double ndcg, double recall, double mrr) {
    }

    private RunBlendSweep() {
    }

    public static void main(String[] args) throws Exception {
        Path dataDir = Path.of("data", "scifact");
        BeirDataset train = BeirLoader.load(dataDir, "train");
        Map<String, List<Cand>> byQuery = readScores(Path.of("results", "rerank-scores-train.tsv"));
        System.out.println("Loaded " + train + ", cached candidates for " + byQuery.size() + " queries");

        List<String> csv = new ArrayList<>();
        csv.add("c,w,ndcg_at_10,recall_at_10,mrr_at_10");
        List<Config> all = new ArrayList<>();
        System.out.printf("%-4s %-5s %-8s %-10s %-8s%n", "c", "w", "nDCG@10", "Recall@10", "MRR@10");
        for (int c : RRF_CS) {
            for (int i = 0; i <= 10; i++) {
                double w = i / 10.0;
                Config cfg = evaluate(train, byQuery, c, w);
                all.add(cfg);
                System.out.printf(Locale.ROOT, "%-4d %-5.1f %-8.4f %-10.4f %-8.4f%n",
                        c, w, cfg.ndcg, cfg.recall, cfg.mrr);
                csv.add(String.format(Locale.ROOT, "%d,%.1f,%.4f,%.4f,%.4f",
                        c, w, cfg.ndcg, cfg.recall, cfg.mrr));
            }
        }
        Path out = Path.of("results", "blend-train-sweep.csv");
        Files.write(out, csv);
        System.out.println("Wrote " + out);

        // Selection rule: best nDCG, then smallest w within TOLERANCE of it.
        double best = all.stream().mapToDouble(Config::ndcg).max().orElseThrow();
        Config chosen = all.stream()
                .filter(x -> x.ndcg >= best - TOLERANCE)
                .min(Comparator.comparingDouble(Config::w)
                        .thenComparing(Comparator.comparingDouble(Config::ndcg).reversed()))
                .orElseThrow();
        System.out.printf(Locale.ROOT, "%nBest train nDCG@10 = %.4f. Chosen (rule): c=%d w=%.1f nDCG@10=%.4f%n",
                best, chosen.c, chosen.w, chosen.ndcg);

        double[] hybrid = perQuery(train, byQuery, 60, 0.0);
        double[] blend = perQuery(train, byQuery, chosen.c, chosen.w);
        PairedBootstrap.Result r = PairedBootstrap.compare(blend, hybrid, RESAMPLES, SEED);
        System.out.println(String.format(Locale.ROOT,
                "Train paired bootstrap, chosen - hybrid: %+.4f  95%% CI [%+.4f, %+.4f]  p=%.4f",
                r.meanDiff(), r.lo(), r.hi(), r.pValue()));
    }

    private static Config evaluate(BeirDataset ds, Map<String, List<Cand>> byQuery, int c, double w) {
        double nd = 0, rc = 0, mr = 0;
        for (Query q : ds.queries()) {
            List<String> ranked = rank(byQuery.get(q.id()), c, w);
            var rel = ds.qrels().forQuery(q.id());
            nd += Metrics.ndcgAtK(ranked, rel, K);
            rc += Metrics.recallAtK(ranked, rel, K);
            mr += Metrics.reciprocalRank(ranked, rel, K);
        }
        int n = ds.queries().size();
        return new Config(c, w, nd / n, rc / n, mr / n);
    }

    private static double[] perQuery(BeirDataset ds, Map<String, List<Cand>> byQuery, int c, double w) {
        double[] out = new double[ds.queries().size()];
        int i = 0;
        for (Query q : ds.queries()) {
            out[i++] = Metrics.ndcgAtK(rank(byQuery.get(q.id()), c, w),
                    ds.qrels().forQuery(q.id()), K);
        }
        return out;
    }

    /** Top-K doc ids after blending the two ranks. */
    static List<String> rank(List<Cand> cands, int c, double w) {
        if (cands == null) {
            throw new IllegalStateException("query missing from the score cache");
        }
        List<Cand> byCe = new ArrayList<>(cands);
        byCe.sort(Comparator.comparingDouble((Cand x) -> x.ce).reversed()
                .thenComparing(Cand::docId));
        Map<String, Integer> ceRank = new HashMap<>();
        for (int i = 0; i < byCe.size(); i++) {
            ceRank.put(byCe.get(i).docId, i + 1);
        }
        List<Cand> sorted = new ArrayList<>(cands);
        sorted.sort(Comparator.comparingDouble((Cand x) ->
                -((1 - w) / (c + x.hybridRank) + w / (c + ceRank.get(x.docId))))
                .thenComparingInt(Cand::hybridRank));
        List<String> ids = new ArrayList<>(K);
        for (int i = 0; i < Math.min(K, sorted.size()); i++) {
            ids.add(sorted.get(i).docId);
        }
        return ids;
    }

    static Map<String, List<Cand>> readScores(Path file) throws Exception {
        Map<String, List<Cand>> map = new LinkedHashMap<>();
        for (String line : Files.readAllLines(file, StandardCharsets.UTF_8)) {
            if (line.isEmpty()) {
                continue;
            }
            String[] p = line.split("\t");
            map.computeIfAbsent(p[0], k -> new ArrayList<>())
                    .add(new Cand(p[1], Integer.parseInt(p[2]), Float.parseFloat(p[3])));
        }
        return map;
    }
}