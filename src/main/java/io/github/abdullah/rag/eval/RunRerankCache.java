package io.github.abdullah.rag.eval;

import io.github.abdullah.rag.analysis.StemmingAnalyzer;
import io.github.abdullah.rag.embedding.OnnxCrossEncoder;
import io.github.abdullah.rag.fusion.RrfFusion;
import io.github.abdullah.rag.index.IndexedBm25Retriever;
import io.github.abdullah.rag.model.Document;
import io.github.abdullah.rag.model.Query;
import io.github.abdullah.rag.retrieval.DenseRetriever;
import io.github.abdullah.rag.retrieval.HybridRetriever;
import io.github.abdullah.rag.retrieval.Retriever;
import io.github.abdullah.rag.retrieval.ScoredDoc;
import io.github.abdullah.rag.vector.HnswConfig;
import io.github.abdullah.rag.vector.HnswIndex;
import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Scores the top RERANK_DEPTH hybrid candidates of every query with the cross-encoder
 * and appends "queryId, docId, hybridRank, ceScore" (tab-separated) to
 * results/rerank-scores-<split>.tsv. Resumable: complete queries are skipped.
 * Usage: RunRerankCache [train|test]   (default: train)
 */
public final class RunRerankCache {

    private static final String MODEL = "onnx-all-MiniLM-L6-v2";
    private static final int DIM = 384;
    private static final int HYBRID_DEPTH = 100;
    private static final int RERANK_DEPTH = 50;
    private static final int RRF_C = 60;
    private static final int EF_SEARCH = 40;

    private RunRerankCache() {
    }

    public static void main(String[] args) throws Exception {
        String split = args.length > 0 ? args[0] : "train";
        if (!split.equals("train") && !split.equals("test")) {
            throw new IllegalArgumentException("split must be train or test");
        }
        Path dataDir = Path.of("data", "scifact");
        BeirDataset dataset = BeirLoader.load(dataDir, split);
        System.out.println("Loaded " + dataset);

        Path out = Path.of("results", "rerank-scores-" + split + ".tsv");
        Files.createDirectories(out.getParent());
        Set<String> done = keepCompleteQueries(out);
        System.out.println(done.size() + " queries already complete in " + out);

        String queryCache = split.equals("test")
                ? "embeddings-" + MODEL + "-queries.bin"
                : "embeddings-" + MODEL + "-queries-train.bin";
        Retriever bm25 = new IndexedBm25Retriever(new StemmingAnalyzer());
        HnswIndex hnsw = new HnswIndex(DIM, HnswConfig.defaults());
        hnsw.setEfSearch(EF_SEARCH);
        Retriever dense = new DenseRetriever(hnsw,
                dataDir.resolve("embeddings-" + MODEL + "-docs.bin"), MODEL,
                RunDenseBaseline.loadQueryVectors(dataDir, dataset, queryCache));
        bm25.index(dataset.corpus());
        dense.index(dataset.corpus());
        Retriever hybrid = new HybridRetriever(List.of(bm25, dense),
                new RrfFusion(RRF_C), HYBRID_DEPTH, false);

        Map<String, String> text = new HashMap<>();
        for (Document d : dataset.corpus()) {
            text.put(d.id(), d.indexableText());
        }

        int total = dataset.queries().size();
        int finished = done.size();
        long start = System.nanoTime();
        int doneThisRun = 0;
        try (OnnxCrossEncoder ce = new OnnxCrossEncoder(
                Path.of("models", "ms-marco-MiniLM-L-6-v2"), 512);
                BufferedWriter w = Files.newBufferedWriter(out, StandardCharsets.UTF_8,
                        StandardOpenOption.CREATE, StandardOpenOption.APPEND)) {
            for (Query q : dataset.queries()) {
                if (done.contains(q.id())) {
                    continue;
                }
                StringBuilder sb = new StringBuilder();
                int rank = 0;
                for (ScoredDoc c : hybrid.search(q.text(), RERANK_DEPTH)) {
                    rank++;
                    float s = ce.score(q.text(), text.get(c.docId()));
                    sb.append(q.id()).append('\t').append(c.docId()).append('\t')
                            .append(rank).append('\t').append(s).append('\n');
                }
                w.write(sb.toString());
                w.flush();
                finished++;
                doneThisRun++;
                if (doneThisRun % 10 == 0) {
                    double secs = (System.nanoTime() - start) / 1e9;
                    double eta = secs / doneThisRun * (total - finished);
                    System.out.printf("  %d/%d queries  (%.0f s elapsed, ~%.0f s left)%n",
                            finished, total, secs, eta);
                }
            }
        }
        System.out.println("Done: " + out);
    }

    /** Rewrites the file keeping only queries with a full set of rows; returns their ids. */
    private static Set<String> keepCompleteQueries(Path file) throws IOException {
        if (!Files.exists(file)) {
            return Set.of();
        }
        Map<String, List<String>> byQuery = new LinkedHashMap<>();
        for (String line : Files.readAllLines(file, StandardCharsets.UTF_8)) {
            if (line.isEmpty()) {
                continue;
            }
            String qid = line.substring(0, line.indexOf('\t'));
            byQuery.computeIfAbsent(qid, k -> new ArrayList<>()).add(line);
        }
        List<String> kept = new ArrayList<>();
        Map<String, List<String>> complete = new LinkedHashMap<>();
        byQuery.forEach((qid, lines) -> {
            if (lines.size() == RERANK_DEPTH) {
                complete.put(qid, lines);
                kept.addAll(lines);
            }
        });
        Files.write(file, kept, StandardCharsets.UTF_8);
        return complete.keySet();
    }
}