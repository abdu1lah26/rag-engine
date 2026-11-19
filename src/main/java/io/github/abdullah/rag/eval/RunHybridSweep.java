package io.github.abdullah.rag.eval;

import io.github.abdullah.rag.analysis.StemmingAnalyzer;
import io.github.abdullah.rag.fusion.RrfFusion;
import io.github.abdullah.rag.index.IndexedBm25Retriever;
import io.github.abdullah.rag.model.Query;
import io.github.abdullah.rag.retrieval.DenseRetriever;
import io.github.abdullah.rag.retrieval.HybridRetriever;
import io.github.abdullah.rag.retrieval.Retriever;
import io.github.abdullah.rag.retrieval.ScoredDoc;
import io.github.abdullah.rag.vector.HnswConfig;
import io.github.abdullah.rag.vector.HnswIndex;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * Sweeps RRF constant and candidate depth on the TRAIN split only.
 * Children are built once and shared by every hybrid config.
 */
public final class RunHybridSweep {

    private static final String MODEL = "onnx-all-MiniLM-L6-v2";
    private static final int DIM = 384;
    private static final int K = 10;
    private static final int EF_SEARCH = 40;
    private static final int[] RRF_CS = { 1, 10, 30, 60, 100 };
    private static final int[] DEPTHS = { 10, 20, 50, 100, 200 };

    private RunHybridSweep() {
    }

    public static void main(String[] args) throws Exception {
        Path dataDir = Path.of("data", "scifact");
        BeirDataset train = BeirLoader.load(dataDir, "train");
        System.out.println("Loaded " + train);

        Retriever bm25 = new IndexedBm25Retriever(new StemmingAnalyzer());
        HnswIndex hnsw = new HnswIndex(DIM, HnswConfig.defaults());
        hnsw.setEfSearch(EF_SEARCH);
        Retriever dense = new DenseRetriever(hnsw,
                dataDir.resolve("embeddings-" + MODEL + "-docs.bin"), MODEL,
                RunDenseBaseline.loadQueryVectors(dataDir, train,
                        "embeddings-" + MODEL + "-queries-train.bin"));
        bm25.index(train.corpus());
        dense.index(train.corpus());

        List<String> csv = new ArrayList<>();
        csv.add("config,ndcg_at_10,recall_at_10,mrr_at_10");

        row("bm25-only", bm25, train, csv);
        row("dense-only", dense, train, csv);
        for (int c : RRF_CS) {
            for (int depth : DEPTHS) {
                Retriever h = new HybridRetriever(List.of(bm25, dense),
                        new RrfFusion(c), depth, false);
                row("rrf c=" + c + " depth=" + depth, h, train, csv);
            }
        }

        Path out = Path.of("results", "hybrid-train-sweep.csv");
        Files.createDirectories(out.getParent());
        Files.write(out, csv);
        System.out.println("Wrote " + out);
    }

    private static void row(String label, Retriever r, BeirDataset ds, List<String> csv) {
        double nd = 0, rc = 0, mr = 0;
        for (Query q : ds.queries()) {
            List<String> ranked = new ArrayList<>();
            for (ScoredDoc s : r.search(q.text(), K)) {
                ranked.add(s.docId());
            }
            var rel = ds.qrels().forQuery(q.id());
            nd += Metrics.ndcgAtK(ranked, rel, K);
            rc += Metrics.recallAtK(ranked, rel, K);
            mr += Metrics.reciprocalRank(ranked, rel, K);
        }
        int n = ds.queries().size();
        String line = String.format(Locale.ROOT, "%-22s nDCG@10=%.4f  Recall@10=%.4f  MRR@10=%.4f",
                label, nd / n, rc / n, mr / n);
        System.out.println(line);
        csv.add(String.format(Locale.ROOT, "%s,%.4f,%.4f,%.4f", label, nd / n, rc / n, mr / n));
    }
}