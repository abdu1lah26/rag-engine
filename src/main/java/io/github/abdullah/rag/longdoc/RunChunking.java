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
 * Compares chunkers on SciFact packed into long documents.
 * Usage: RunChunking [datasetDir] [split] (defaults: data/scifact, test)
 */
public final class RunChunking {

    private static final long SEED = 42;
    private static final int PER_DOC = 10;
    private static final int K = 10;
    private static final int CHUNK_DEPTH = 100;

    public static void main(String[] args) throws Exception {
        Path dir = Path.of(args.length > 0 ? args[0] : "data/scifact");
        String split = args.length > 1 ? args[1] : "test";
        BeirDataset ds = BeirLoader.load(dir, split);
        System.out.println("corpus=" + ds.corpus().size() + " queries=" + ds.queries().size()
                + " seed=" + SEED + " perDoc=" + PER_DOC);

        TextAnalyzer analyzer = new StemmingAnalyzer();
        Embedder embedder = new HashedTfEmbedder(analyzer, 1024);

        List<Chunker> chunkers = List.of(
                new FixedSizeChunker(100, 20),
                new FixedSizeChunker(200, 40),
                new RecursiveChunker(100, 20),
                new RecursiveChunker(200, 40),
                new SemanticChunker(embedder, 3, 60, 200, 0.5));

        for (String[] mode : new String[][] { { "hard (space)", " " }, { "paragraph", "\n\n" } }) {
            LongDocs longDocs = LongDocBuilder.build(ds.corpus(), PER_DOC, mode[1], SEED);
            System.out.println("\n=== mode: " + mode[0] + " | long docs: " + longDocs.docs().size() + " ===");
            System.out.printf("%-48s %7s %8s %8s %7s %9s%n",
                    "chunker", "chunks", "nDCG@10", "R@10", "MRR@10", "ms/query");

            List<Chunker> all = new java.util.ArrayList<>(chunkers);
            all.add(new OracleChunker(longDocs.spans()));
            for (Chunker chunker : all) {
                ChunkedAbstractRetriever r = new ChunkedAbstractRetriever(chunker, analyzer, CHUNK_DEPTH);
                r.index(longDocs);
                evaluate(r, ds);
            }
        }
    }

    private static void evaluate(ChunkedAbstractRetriever r, BeirDataset ds) {
        double ndcg = 0, recall = 0, mrr = 0;
        int n = 0;
        long nanos = 0;
        for (Query q : ds.queries()) {
            Map<String, Integer> relevant = ds.qrels().forQuery(q.id());
            if (relevant == null || relevant.isEmpty())
                continue;
            long t0 = System.nanoTime();
            List<String> ranked = r.search(q.text(), K);
            nanos += System.nanoTime() - t0;
            ndcg += Metrics.ndcgAtK(ranked, relevant, K);
            recall += Metrics.recallAtK(ranked, relevant, K);
            mrr += Metrics.reciprocalRank(ranked, relevant, K);
            n++;
        }
        System.out.printf("%-48s %7d %8.4f %8.4f %7.4f %9.2f%n",
                r.name(), r.chunkCount(), ndcg / n, recall / n, mrr / n, nanos / 1e6 / n);
    }
}