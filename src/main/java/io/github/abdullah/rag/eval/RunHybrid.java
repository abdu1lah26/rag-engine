package io.github.abdullah.rag.eval;

import io.github.abdullah.rag.analysis.StemmingAnalyzer;
import io.github.abdullah.rag.fusion.RrfFusion;
import io.github.abdullah.rag.index.IndexedBm25Retriever;
import io.github.abdullah.rag.longdoc.PairedBootstrap;
import io.github.abdullah.rag.model.Query;
import io.github.abdullah.rag.retrieval.DenseRetriever;
import io.github.abdullah.rag.retrieval.HybridRetriever;
import io.github.abdullah.rag.retrieval.Retriever;
import io.github.abdullah.rag.retrieval.ScoredDoc;
import io.github.abdullah.rag.vector.HnswConfig;
import io.github.abdullah.rag.vector.HnswIndex;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * BM25 (stop+stem) vs dense HNSW vs hybrid RRF on SciFact test, with paired
 * bootstrap on per-query nDCG@10. Parameters are fixed before the run (RRF c=60,
 * candidate depth 100, efSearch 40): no tuning on test.
 */
public final class RunHybrid {

    private static final String MODEL = "onnx-all-MiniLM-L6-v2";
    private static final int DIM = 384;
    private static final int K = 10;
    private static final int DEPTH = 100;
    private static final int RRF_C = 60;
    private static final int EF_SEARCH = 40;
    private static final int RESAMPLES = 10_000;
    private static final long SEED = 42;

    private RunHybrid() {
    }

    public static void main(String[] args) throws Exception {
        Path dataDir = Path.of("data", "scifact");
        BeirDataset dataset = BeirLoader.load(dataDir, "test");
        System.out.println("Loaded " + dataset);

        Retriever bm25 = new IndexedBm25Retriever(new StemmingAnalyzer());
        HnswIndex hnsw = new HnswIndex(DIM, HnswConfig.defaults());
        hnsw.setEfSearch(EF_SEARCH);
        Retriever dense = new DenseRetriever(hnsw,
                dataDir.resolve("embeddings-" + MODEL + "-docs.bin"), MODEL,
                RunDenseBaseline.loadQueryVectors(dataDir, dataset));
        // Children are indexed by EvalRunner below; the hybrid must not index them again.
        Retriever hybrid = new HybridRetriever(List.of(bm25, dense),
                new RrfFusion(RRF_C), DEPTH, false);

        System.out.println(EvalRunner.run(bm25, dataset, K));
        System.out.println(EvalRunner.run(dense, dataset, K));
        System.out.println(EvalRunner.run(hybrid, dataset, K));

        double[] b = perQueryNdcg(bm25, dataset);
        double[] d = perQueryNdcg(dense, dataset);
        double[] h = perQueryNdcg(hybrid, dataset);
        System.out.println("Paired bootstrap, nDCG@10 (" + RESAMPLES + " resamples):");
        printDiff("hybrid - bm25 ", PairedBootstrap.compare(h, b, RESAMPLES, SEED));
        printDiff("hybrid - dense", PairedBootstrap.compare(h, d, RESAMPLES, SEED));
    }

    private static double[] perQueryNdcg(Retriever r, BeirDataset ds) {
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

    private static void printDiff(String label, PairedBootstrap.Result r) {
        System.out.println(String.format(Locale.ROOT,
                "  %s: %+.4f  95%% CI [%+.4f, %+.4f]  p=%.4f",
                label, r.meanDiff(), r.lo(), r.hi(), r.pValue()));
    }
}