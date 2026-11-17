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
import io.github.abdullah.rag.model.Document;
import io.github.abdullah.rag.model.Query;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Title ablation on the locked configs (TEST split): does re-attaching an abstract's title
 * to chunks that start mid-abstract help? Same chunks, same queries, only the indexed text changes.
 * Usage: RunChunkingTitleAblation [datasetDir] [split]   (defaults: data/scifact, test)
 */
public final class RunChunkingTitleAblation {

    private static final long SEED = 42;
    private static final int PER_DOC = 10;
    private static final int K = 10;
    private static final int CHUNK_DEPTH = 100;
    private static final int RESAMPLES = 10_000;

    public static void main(String[] args) throws Exception {
        Path dir = Path.of(args.length > 0 ? args[0] : "data/scifact");
        String split = args.length > 1 ? args[1] : "test";
        BeirDataset ds = BeirLoader.load(dir, split);

        Map<String, String> titles = new HashMap<>();
        for (Document d : ds.corpus()) titles.put(d.id(), d.title());

        TextAnalyzer analyzer = new StemmingAnalyzer();
        Embedder embedder = new HashedTfEmbedder(analyzer, 1024);
        Chunker semantic = new SemanticChunker(embedder, 3, 100, 300, 0.0);

        run(ds, analyzer, titles, "hard (space)", " ",
                List.of(new FixedSizeChunker(50, 10), new RecursiveChunker(50, 10), semantic));
        run(ds, analyzer, titles, "paragraph", "\n\n",
                List.of(new FixedSizeChunker(50, 10), new RecursiveChunker(300, 0), semantic));
    }

    private static void run(BeirDataset ds, TextAnalyzer analyzer, Map<String, String> titles,
                            String mode, String sep, List<Chunker> chunkers) {
        LongDocs longDocs = LongDocBuilder.build(ds.corpus(), PER_DOC, sep, SEED);
        System.out.println("\n=== TITLE ABLATION on " + ds.name() + " | mode: " + mode + " ===");
        System.out.printf("%-48s %9s %9s %9s %9s %9s %s%n",
                "chunker", "no title", "w/ title", "diff", "ci95 lo", "ci95 hi", "verdict");

        List<Chunker> all = new ArrayList<>(chunkers);
        all.add(new OracleChunker(longDocs.spans())); // sanity row: must show diff 0
        for (Chunker chunker : all) {
            double[] without = perQuery(chunker, analyzer, longDocs, ds, null);
            double[] with = perQuery(chunker, analyzer, longDocs, ds, titles);
            PairedBootstrap.Result r = PairedBootstrap.compare(with, without, RESAMPLES, SEED);
            String verdict = r.lo() > 0 ? "title helps" : r.hi() < 0 ? "title hurts" : "no detectable difference";
            System.out.printf("%-48s %9.4f %9.4f %+9.4f %+9.4f %+9.4f %s%n",
                    chunker.name(), mean(without), mean(with), r.meanDiff(), r.lo(), r.hi(), verdict);
        }
    }

    private static double[] perQuery(Chunker chunker, TextAnalyzer analyzer, LongDocs longDocs,
                                     BeirDataset ds, Map<String, String> titles) {
        ChunkedAbstractRetriever r = new ChunkedAbstractRetriever(chunker, analyzer, CHUNK_DEPTH);
        if (titles != null) r.withTitles(titles);
        r.index(longDocs);
        List<Double> scores = new ArrayList<>();
        for (Query q : ds.queries()) {
            Map<String, Integer> relevant = ds.qrels().forQuery(q.id());
            if (relevant == null || relevant.isEmpty()) continue;
            scores.add(Metrics.ndcgAtK(r.search(q.text(), K), relevant, K));
        }
        double[] out = new double[scores.size()];
        for (int i = 0; i < out.length; i++) out[i] = scores.get(i);
        return out;
    }

    private static double mean(double[] a) {
        double s = 0;
        for (double v : a) s += v;
        return s / a.length;
    }
}