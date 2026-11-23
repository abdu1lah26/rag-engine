package io.github.abdullah.rag.eval;

import io.github.abdullah.rag.analysis.StemmingAnalyzer;
import io.github.abdullah.rag.index.IndexedBm25Retriever;
import io.github.abdullah.rag.longdoc.PairedBootstrap;
import io.github.abdullah.rag.model.Query;
import io.github.abdullah.rag.retrieval.DenseRetriever;
import io.github.abdullah.rag.retrieval.Retriever;
import io.github.abdullah.rag.retrieval.ScoredDoc;
import io.github.abdullah.rag.vector.HnswConfig;
import io.github.abdullah.rag.vector.HnswIndex;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * One evaluation of the config frozen on train (weighted RRF, c=1, w=0.6) on the
 * TEST split, from the cached cross-encoder scores. Only three configs are printed
 * on purpose: hybrid (w=0, sanity), pure rerank (w=1, sanity), and the frozen blend.
 */
public final class RunBlendTest {

    private static final String MODEL = "onnx-all-MiniLM-L6-v2";
    private static final int DIM = 384;
    private static final int K = 10;
    private static final int EF_SEARCH = 40;
    private static final int FROZEN_C = 1;
    private static final double FROZEN_W = 0.6;
    private static final int RESAMPLES = 10_000;
    private static final long SEED = 42;

    private RunBlendTest() {
    }

    public static void main(String[] args) throws Exception {
        Path dataDir = Path.of("data", "scifact");
        BeirDataset test = BeirLoader.load(dataDir, "test");
        Map<String, List<RunBlendSweep.Cand>> cache =
                RunBlendSweep.readScores(Path.of("results", "rerank-scores-test.tsv"));
        System.out.println("Loaded " + test + ", cached candidates for " + cache.size() + " queries");
        if (cache.size() != test.queries().size()) {
            throw new IllegalStateException("score cache is incomplete");
        }

        Retriever bm25 = new IndexedBm25Retriever(new StemmingAnalyzer());
        HnswIndex hnsw = new HnswIndex(DIM, HnswConfig.defaults());
        hnsw.setEfSearch(EF_SEARCH);
        Retriever dense = new DenseRetriever(hnsw,
                dataDir.resolve("embeddings-" + MODEL + "-docs.bin"), MODEL,
                RunDenseBaseline.loadQueryVectors(dataDir, test));
        bm25.index(test.corpus());
        dense.index(test.corpus());

        double[] b = perQuery(bm25, test);
        double[] d = perQuery(dense, test);
        double[] h = perQuery(test, cache, 60, 0.0);
        double[] r = perQuery(test, cache, 60, 1.0);
        double[] f = perQuery(test, cache, FROZEN_C, FROZEN_W);

        System.out.println(String.format(Locale.ROOT, "nDCG@10  bm25=%.4f  dense=%.4f", mean(b), mean(d)));
        System.out.println(String.format(Locale.ROOT,
                "nDCG@10  hybrid(c=60,w=0)=%.4f  [must be 0.7166]", mean(h)));
        System.out.println(String.format(Locale.ROOT,
                "nDCG@10  pure rerank(w=1)=%.4f  [must be 0.6884]", mean(r)));
        System.out.println(String.format(Locale.ROOT,
                "nDCG@10  FROZEN blend(c=%d,w=%.1f)=%.4f", FROZEN_C, FROZEN_W, mean(f)));
        System.out.println(allMetrics(test, cache, FROZEN_C, FROZEN_W));

        System.out.println("Paired bootstrap, nDCG@10 (" + RESAMPLES + " resamples):");
        printDiff("frozen - hybrid", PairedBootstrap.compare(f, h, RESAMPLES, SEED));
        printDiff("frozen - bm25  ", PairedBootstrap.compare(f, b, RESAMPLES, SEED));
        printDiff("frozen - dense ", PairedBootstrap.compare(f, d, RESAMPLES, SEED));
    }

    private static double[] perQuery(Retriever r, BeirDataset ds) {
        double[] out = new double[ds.queries().size()];
        int i = 0;
        for (Query q : ds.queries()) {
            List<String> ranked = new ArrayList<>();
            for (ScoredDoc s : r.search(q.text(), K)) {
                ranked.add(s.docId());
            }
            out[i++] = Metrics.ndcgAtK(ranked, ds.qrels().forQuery(q.id()), K);
        }
        return out;
    }

    private static double[] perQuery(BeirDataset ds, Map<String, List<RunBlendSweep.Cand>> cache,
            int c, double w) {
        double[] out = new double[ds.queries().size()];
        int i = 0;
        for (Query q : ds.queries()) {
            out[i++] = Metrics.ndcgAtK(RunBlendSweep.rank(cache.get(q.id()), c, w),
                    ds.qrels().forQuery(q.id()), K);
        }
        return out;
    }

    private static String allMetrics(BeirDataset ds, Map<String, List<RunBlendSweep.Cand>> cache,
            int c, double w) {
        double nd = 0, rc = 0, mr = 0;
        for (Query q : ds.queries()) {
            List<String> ranked = RunBlendSweep.rank(cache.get(q.id()), c, w);
            var rel = ds.qrels().forQuery(q.id());
            nd += Metrics.ndcgAtK(ranked, rel, K);
            rc += Metrics.recallAtK(ranked, rel, K);
            mr += Metrics.reciprocalRank(ranked, rel, K);
        }
        int n = ds.queries().size();
        return String.format(Locale.ROOT, "frozen blend: nDCG@10=%.4f  Recall@10=%.4f  MRR@10=%.4f",
                nd / n, rc / n, mr / n);
    }

    private static double mean(double[] a) {
        double s = 0;
        for (double v : a) {
            s += v;
        }
        return s / a.length;
    }

    private static void printDiff(String label, PairedBootstrap.Result r) {
        System.out.println(String.format(Locale.ROOT,
                "  %s: %+.4f  95%% CI [%+.4f, %+.4f]  p=%.4f",
                label, r.meanDiff(), r.lo(), r.hi(), r.pValue()));
    }
}