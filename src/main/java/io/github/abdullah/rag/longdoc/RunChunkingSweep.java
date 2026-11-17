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
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Parameter sweep for all three chunkers, scored on the TRAIN split so the test
 * split stays
 * untouched for the final comparison.
 * Usage: RunChunkingSweep [datasetDir] [split] (defaults: data/scifact, train)
 */
public final class RunChunkingSweep {

    private static final long SEED = 42;
    private static final int PER_DOC = 10;
    private static final int K = 10;
    private static final int CHUNK_DEPTH = 100;
    private static final int TOP_PER_FAMILY = 5;
    private static final double MIN_INTACT = 0.5;

    private record Row(String name, int chunks, double ndcg, double recall, double mrr,
            double avgWords, double purity, double boundary, double intact) {
    }

    public static void main(String[] args) throws Exception {
        Path dir = Path.of(args.length > 0 ? args[0] : "data/scifact");
        String split = args.length > 1 ? args[1] : "train";
        BeirDataset ds = BeirLoader.load(dir, split);

        TextAnalyzer analyzer = new StemmingAnalyzer();
        Embedder embedder = new HashedTfEmbedder(analyzer, 1024);

        List<Chunker> grid = new ArrayList<>();
        for (int size : new int[] { 50, 100, 150, 200, 300 }) {
            for (int overlap : new int[] { 0, size / 5 }) {
                grid.add(new FixedSizeChunker(size, overlap));
                grid.add(new RecursiveChunker(size, overlap));
            }
        }
        for (int window : new int[] { 1, 2, 3 }) {
            for (double alpha : new double[] { 0.0, 0.5, 1.0 }) {
                for (int[] mm : new int[][] { { 40, 100 }, { 60, 200 }, { 100, 300 }, { 150, 300 } }) {
                    grid.add(new SemanticChunker(embedder, window, mm[0], mm[1], alpha));
                }
            }
        }

        for (String[] mode : new String[][] { { "hard (space)", " " }, { "paragraph", "\n\n" } }) {
            LongDocs longDocs = LongDocBuilder.build(ds.corpus(), PER_DOC, mode[1], SEED);
            System.out.println("\n=== SWEEP on " + split + " | mode: " + mode[0]
                    + " | queries=" + ds.queries().size() + " ===");

            Map<String, List<Row>> byFamily = new LinkedHashMap<>();
            for (Chunker chunker : grid) {
                ChunkedAbstractRetriever r = new ChunkedAbstractRetriever(chunker, analyzer, CHUNK_DEPTH);
                r.index(longDocs);
                Row row = evaluate(r, ds, ChunkQuality.measure(chunker, longDocs));
                String family = chunker.name().split("-")[0];
                byFamily.computeIfAbsent(family, f -> new ArrayList<>()).add(row);
            }

            System.out.printf("%-52s %7s %8s %7s %7s %7s %7s%n", "config (top " + TOP_PER_FAMILY + " per family)",
                    "chunks", "nDCG@10", "avgW", "purity", "bndry", "intact");
            for (var e : byFamily.entrySet()) {
                e.getValue().sort(Comparator.comparingDouble(Row::ndcg).reversed());
                for (Row row : e.getValue().subList(0, Math.min(TOP_PER_FAMILY, e.getValue().size()))) {
                    System.out.printf("%-52s %7d %8.4f %7.1f %7.3f %7.3f %7.3f%n",
                            row.name(), row.chunks(), row.ndcg(), row.avgWords(),
                            row.purity(), row.boundary(), row.intact());
                }
                Row pick = e.getValue().stream().filter(x -> x.intact() >= MIN_INTACT).findFirst().orElse(null);
                if (pick != null) {
                    System.out.printf("  SELECTED: %s  nDCG=%.4f intact=%.3f%n", pick.name(), pick.ndcg(),
                            pick.intact());
                } else {
                    Row best = e.getValue().get(0);
                    System.out.printf("  NO CONFIG MEETS intact>=%.1f. best nDCG: %s  nDCG=%.4f intact=%.3f%n",
                            MIN_INTACT, best.name(), best.ndcg(), best.intact());
                }
                System.out.println();
            }
        }
    }

    private static Row evaluate(ChunkedAbstractRetriever r, BeirDataset ds, ChunkQuality.Result quality) {
        double ndcg = 0, recall = 0, mrr = 0;
        int n = 0;
        for (Query q : ds.queries()) {
            Map<String, Integer> relevant = ds.qrels().forQuery(q.id());
            if (relevant == null || relevant.isEmpty())
                continue;
            List<String> ranked = r.search(q.text(), K);
            ndcg += Metrics.ndcgAtK(ranked, relevant, K);
            recall += Metrics.recallAtK(ranked, relevant, K);
            mrr += Metrics.reciprocalRank(ranked, relevant, K);
            n++;
        }
        return new Row(r.name(), r.chunkCount(), ndcg / n, recall / n, mrr / n,
                quality.avgWords(), quality.purity(), quality.boundaryIntegrity(), quality.intactRate());
    }
}