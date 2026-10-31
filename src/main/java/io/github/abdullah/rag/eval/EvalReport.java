package io.github.abdullah.rag.eval;

import java.util.Locale;

/** Averaged scores and timings for one retriever on one dataset split. */
public record EvalReport(
        String retriever,
        String dataset,
        int queries,
        int k,
        double recall,
        double mrr,
        double ndcg,
        double indexTimeMs,
        double p50Ms,
        double p95Ms) {

    @Override
    public String toString() {
        return String.format(Locale.ROOT,
                "%s on %s (%d queries) | Recall@%d=%.4f  MRR@%d=%.4f  nDCG@%d=%.4f | "
                        + "index=%.0f ms  p50=%.3f ms  p95=%.3f ms",
                retriever, dataset, queries, k, recall, k, mrr, k, ndcg,
                indexTimeMs, p50Ms, p95Ms);
    }
}