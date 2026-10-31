package io.github.abdullah.rag.eval;

import io.github.abdullah.rag.model.Query;
import io.github.abdullah.rag.retrieval.Retriever;
import io.github.abdullah.rag.retrieval.ScoredDoc;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

/** Runs a retriever over every query in a dataset and averages the metrics. */
public final class EvalRunner {

    private EvalRunner() {
    }

    public static EvalReport run(Retriever retriever, BeirDataset dataset, int k) {
        long indexStart = System.nanoTime();
        retriever.index(dataset.corpus());
        double indexMs = (System.nanoTime() - indexStart) / 1_000_000.0;

        double sumRecall = 0, sumMrr = 0, sumNdcg = 0;
        double[] latenciesMs = new double[dataset.queries().size()];
        int i = 0;

        for (Query q : dataset.queries()) {
            long start = System.nanoTime();
            List<ScoredDoc> results = retriever.search(q.text(), k);
            latenciesMs[i++] = (System.nanoTime() - start) / 1_000_000.0;

            List<String> ranked = new ArrayList<>(results.size());
            for (ScoredDoc r : results) {
                ranked.add(r.docId());
            }
            Map<String, Integer> relevant = dataset.qrels().forQuery(q.id());

            sumRecall += Metrics.recallAtK(ranked, relevant, k);
            sumMrr += Metrics.reciprocalRank(ranked, relevant, k);
            sumNdcg += Metrics.ndcgAtK(ranked, relevant, k);
        }

        int n = dataset.queries().size();
        return new EvalReport(
                retriever.name(), dataset.name(), n, k,
                sumRecall / n, sumMrr / n, sumNdcg / n,
                indexMs, percentile(latenciesMs, 50), percentile(latenciesMs, 95));
    }

    /**
     * Nearest-rank percentile: the smallest value with at least p% of samples at or
     * below it.
     */
    static double percentile(double[] values, double p) {
        if (values.length == 0) {
            return 0.0;
        }
        double[] sorted = values.clone();
        Arrays.sort(sorted);
        int idx = (int) Math.ceil(p / 100.0 * sorted.length) - 1;
        return sorted[Math.max(0, Math.min(idx, sorted.length - 1))];
    }
}