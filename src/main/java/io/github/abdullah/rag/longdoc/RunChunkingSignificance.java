package io.github.abdullah.rag.longdoc;

import io.github.abdullah.rag.analysis.StemmingAnalyzer;
import io.github.abdullah.rag.analysis.TextAnalyzer;
import io.github.abdullah.rag.chunking.Chunker;
import io.github.abdullah.rag.chunking.Embedder;
import io.github.abdullah.rag.chunking.FixedSizeChunker;
import io.github.abdullah.rag.chunking.HashedTfEmbedder;
import io.github.abdullah.rag.chunking.RecursiveChunker;
import io.github.abdullah.rag.chunking.SemanticChunker;
import io.github.abdullah.rag.eval.BeirDataset;
import io.github.abdullah.rag.eval.BeirLoader;
import io.github.abdullah.rag.eval.Metrics;
import io.github.abdullah.rag.longdoc.LongDocBuilder.LongDocs;
import io.github.abdullah.rag.model.Query;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Paired bootstrap on per-query scores for the locked configs, on the TEST split.
 * Usage: RunChunkingSignificance [datasetDir] [split]   (defaults: data/scifact, test)
 */
public final class RunChunkingSignificance {

    private static final long SEED = 42;
    private static final int PER_DOC = 10;
    private static final int K = 10;
    private static final int CHUNK_DEPTH = 100;
    private static final int RESAMPLES = 10_000;

    private interface Metric {
        double score(List<String> ranked, Map<String, Integer> relevant);
    }

    public static void main(String[] args) throws Exception {
        Path dir = Path.of(args.length > 0 ? args[0] : "data/scifact");
        String split = args.length > 1 ? args[1] : "test";
        BeirDataset ds = BeirLoader.load(dir, split);

        TextAnalyzer analyzer = new StemmingAnalyzer();
        Embedder embedder = new HashedTfEmbedder(analyzer, 1024);
        Chunker semantic = new SemanticChunker(embedder, 3, 100, 300, 0.0);
        Chunker fixed50 = new FixedSizeChunker(50, 10);

        compareMode(ds, analyzer, "hard (space)", " ", semantic, fixed50, new RecursiveChunker(50, 10));
        compareMode(ds, analyzer, "paragraph", "\n\n", semantic, fixed50, new RecursiveChunker(300, 0));
    }

    private static void compareMode(BeirDataset ds, TextAnalyzer analyzer, String mode, String sep,
                                    Chunker semantic, Chunker fixed, Chunker recursive) {
        LongDocs longDocs = LongDocBuilder.build(ds.corpus(), PER_DOC, sep, SEED);
        Chunker oracle = new OracleChunker(longDocs.spans());

        Metric ndcg = (ranked, rel) -> Metrics.ndcgAtK(ranked, rel, K);
        Metric recall = (ranked, rel) -> Metrics.recallAtK(ranked, rel, K);

        double[] sN = perQuery(semantic, analyzer, longDocs, ds, ndcg);
        double[] fN = perQuery(fixed, analyzer, longDocs, ds, ndcg);
        double[] rN = perQuery(recursive, analyzer, longDocs, ds, ndcg);
        double[] oN = perQuery(oracle, analyzer, longDocs, ds, ndcg);
        double[] sR = perQuery(semantic, analyzer, longDocs, ds, recall);
        double[] rR = perQuery(recursive, analyzer, longDocs, ds, recall);

        System.out.println("\n=== SIGNIFICANCE on " + ds.name() + " | mode: " + mode
                + " | queries=" + sN.length + " | resamples=" + RESAMPLES + " ===");
        System.out.printf("%-58s %9s %9s %9s %8s %s%n", "A  vs  B (metric)", "A-B", "ci95 lo", "ci95 hi", "p", "verdict");

        print("semantic vs " + recursive.name() + " (nDCG@10)", sN, rN);
        print("semantic vs " + recursive.name() + " (Recall@10)", sR, rR);
        print("semantic vs " + fixed.name() + " (nDCG@10)", sN, fN);
        print(recursive.name() + " vs " + fixed.name() + " (nDCG@10)", rN, fN);
        print("oracle vs semantic (nDCG@10)", oN, sN);
        print("oracle vs " + recursive.name() + " (nDCG@10)", oN, rN);
    }

    private static void print(String label, double[] a, double[] b) {
        PairedBootstrap.Result r = PairedBootstrap.compare(a, b, RESAMPLES, SEED);
        String verdict = (r.lo() > 0) ? "A better" : (r.hi() < 0) ? "B better" : "no detectable difference";
        System.out.printf("%-58s %+9.4f %+9.4f %+9.4f %8.4f %s%n",
                label, r.meanDiff(), r.lo(), r.hi(), r.pValue(), verdict);
    }

    /** Per-query scores over judged queries, in dataset order, so arrays from different chunkers line up. */
    private static double[] perQuery(Chunker chunker, TextAnalyzer analyzer, LongDocs longDocs,
                                     BeirDataset ds, Metric metric) {
        ChunkedAbstractRetriever r = new ChunkedAbstractRetriever(chunker, analyzer, CHUNK_DEPTH);
        r.index(longDocs);
        List<Double> scores = new ArrayList<>();
        for (Query q : ds.queries()) {
            Map<String, Integer> relevant = ds.qrels().forQuery(q.id());
            if (relevant == null || relevant.isEmpty()) continue;
            scores.add(metric.score(r.search(q.text(), K), relevant));
        }
        double[] out = new double[scores.size()];
        for (int i = 0; i < out.length; i++) out[i] = scores.get(i);
        return out;
    }
}