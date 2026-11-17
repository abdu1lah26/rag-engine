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
import java.util.List;
import java.util.Map;

/**
 * Final chunking comparison on the TEST split, using only configs locked from the train sweep.
 * Do not edit the configs below after seeing test numbers.
 * Usage: RunChunkingFinal [datasetDir] [split]   (defaults: data/scifact, test)
 */
public final class RunChunkingFinal {

    private static final long SEED = 42;
    private static final int PER_DOC = 10;
    private static final int K = 10;
    private static final int CHUNK_DEPTH = 100;

    public static void main(String[] args) throws Exception {
        Path dir = Path.of(args.length > 0 ? args[0] : "data/scifact");
        String split = args.length > 1 ? args[1] : "test";
        BeirDataset ds = BeirLoader.load(dir, split);

        TextAnalyzer analyzer = new StemmingAnalyzer();
        Embedder embedder = new HashedTfEmbedder(analyzer, 1024);
        Chunker semantic = new SemanticChunker(embedder, 3, 100, 300, 0.0);

        run(ds, analyzer, "hard (space)", " ", List.of(
                new FixedSizeChunker(50, 10),
                new RecursiveChunker(50, 10),
                semantic));
        run(ds, analyzer, "paragraph", "\n\n", List.of(
                new FixedSizeChunker(50, 10),
                new RecursiveChunker(300, 0),
                semantic));
    }

    private static void run(BeirDataset ds, TextAnalyzer analyzer, String mode, String sep, List<Chunker> chunkers) {
        LongDocs longDocs = LongDocBuilder.build(ds.corpus(), PER_DOC, sep, SEED);
        System.out.println("\n=== FINAL on " + ds.name() + " | mode: " + mode
                + " | queries=" + ds.queries().size() + " ===");
        System.out.printf("%-48s %7s %8s %8s %7s %7s %7s %7s %7s%n",
                "chunker", "chunks", "nDCG@10", "R@10", "MRR@10", "avgW", "purity", "bndry", "intact");

        List<Chunker> all = new java.util.ArrayList<>(chunkers);
        all.add(new OracleChunker(longDocs.spans()));
        for (Chunker chunker : all) {
            ChunkedAbstractRetriever r = new ChunkedAbstractRetriever(chunker, analyzer, CHUNK_DEPTH);
            r.index(longDocs);
            ChunkQuality.Result q = ChunkQuality.measure(chunker, longDocs);

            double ndcg = 0, recall = 0, mrr = 0;
            int n = 0;
            for (Query query : ds.queries()) {
                Map<String, Integer> relevant = ds.qrels().forQuery(query.id());
                if (relevant == null || relevant.isEmpty()) continue;
                List<String> ranked = r.search(query.text(), K);
                ndcg += Metrics.ndcgAtK(ranked, relevant, K);
                recall += Metrics.recallAtK(ranked, relevant, K);
                mrr += Metrics.reciprocalRank(ranked, relevant, K);
                n++;
            }
            String flag = q.intactRate() < 0.5 && !(chunker instanceof OracleChunker) ? "  *fails intact>=0.5" : "";
            System.out.printf("%-48s %7d %8.4f %8.4f %7.4f %7.1f %7.3f %7.3f %7.3f%s%n",
                    chunker.name(), r.chunkCount(), ndcg / n, recall / n, mrr / n,
                    q.avgWords(), q.purity(), q.boundaryIntegrity(), q.intactRate(), flag);
        }
    }
}