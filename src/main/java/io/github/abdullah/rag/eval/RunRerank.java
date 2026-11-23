package io.github.abdullah.rag.eval;

import io.github.abdullah.rag.analysis.StemmingAnalyzer;
import io.github.abdullah.rag.embedding.OnnxCrossEncoder;
import io.github.abdullah.rag.fusion.RrfFusion;
import io.github.abdullah.rag.index.IndexedBm25Retriever;
import io.github.abdullah.rag.longdoc.PairedBootstrap;
import io.github.abdullah.rag.model.Query;
import io.github.abdullah.rag.retrieval.DenseRetriever;
import io.github.abdullah.rag.retrieval.HybridRetriever;
import io.github.abdullah.rag.retrieval.RerankingRetriever;
import io.github.abdullah.rag.retrieval.Retriever;
import io.github.abdullah.rag.retrieval.ScoredDoc;
import io.github.abdullah.rag.vector.HnswConfig;
import io.github.abdullah.rag.vector.HnswIndex;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * BM25 vs dense vs hybrid vs hybrid + cross-encoder rerank on SciFact test.
 * All parameters fixed before the run: RRF c=60, hybrid depth 100, rerank depth
 * 50,
 * efSearch 40, cross-encoder max length 512. No tuning on test.
 */
public final class RunRerank {

    private static final String MODEL = "onnx-all-MiniLM-L6-v2";
    private static final int DIM = 384;
    private static final int K = 10;
    private static final int HYBRID_DEPTH = 100;
    private static final int RERANK_DEPTH = 50;
    private static final int RRF_C = 60;
    private static final int EF_SEARCH = 40;
    private static final int RESAMPLES = 10_000;
    private static final long SEED = 42;

    private RunRerank() {
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
        Retriever hybrid = new HybridRetriever(List.of(bm25, dense),
                new RrfFusion(RRF_C), HYBRID_DEPTH, false);

        try (OnnxCrossEncoder ce = new OnnxCrossEncoder(
                Path.of("models", "ms-marco-MiniLM-L-6-v2"), 512)) {
            // Memoise scores so the second (per-query) pass does not repeat 15,000
            // inferences. The first pass, run by EvalRunner, is cold and gives the
            // honest latency.
            Map<String, Float> cache = new HashMap<>();
            RerankingRetriever.PairScorer scorer = (q, d) -> cache.computeIfAbsent(q + "\u0000" + d,
                    key -> ce.score(q, d));
            Retriever rerank = new RerankingRetriever(hybrid, scorer, ce.name(),
                    RERANK_DEPTH, false);

            // EvalRunner indexes each retriever; the wrappers do not index their children.
            System.out.println(EvalRunner.run(bm25, dataset, K));
            System.out.println(EvalRunner.run(dense, dataset, K));
            System.out.println(EvalRunner.run(hybrid, dataset, K));
            System.out.println("Reranking 300 queries x " + RERANK_DEPTH
                    + " candidates (this takes a few minutes)...");
            System.out.println(EvalRunner.run(rerank, dataset, K));

            double ceiling = 0;
            for (Query q : dataset.queries()) {
                List<String> ranked = new ArrayList<>();
                for (ScoredDoc s : hybrid.search(q.text(), RERANK_DEPTH)) {
                    ranked.add(s.docId());
                }
                ceiling += Metrics.recallAtK(ranked, dataset.qrels().forQuery(q.id()), RERANK_DEPTH);
            }
            System.out.println(String.format(Locale.ROOT,
                    "Hybrid Recall@%d (rerank ceiling) = %.4f",
                    RERANK_DEPTH, ceiling / dataset.queries().size()));

            double[] b = perQueryNdcg(bm25, dataset);
            double[] d = perQueryNdcg(dense, dataset);
            double[] h = perQueryNdcg(hybrid, dataset);
            double[] r = perQueryNdcg(rerank, dataset);
            System.out.println("Paired bootstrap, nDCG@10 (" + RESAMPLES + " resamples):");
            printDiff("rerank - hybrid", PairedBootstrap.compare(r, h, RESAMPLES, SEED));
            printDiff("rerank - bm25  ", PairedBootstrap.compare(r, b, RESAMPLES, SEED));
            printDiff("rerank - dense ", PairedBootstrap.compare(r, d, RESAMPLES, SEED));
        }
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