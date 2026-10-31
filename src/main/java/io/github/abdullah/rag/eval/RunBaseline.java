package io.github.abdullah.rag.eval;

import io.github.abdullah.rag.bm25.NaiveBm25Retriever;

import java.io.IOException;
import java.nio.file.Path;

/**
 * Runs the naive BM25 baseline on SciFact (test split) and prints one report
 * line.
 */
public final class RunBaseline {

    private RunBaseline() {
    }

    public static void main(String[] args) throws IOException {
        Path dataDir = Path.of(args.length > 0 ? args[0] : "data/scifact");
        BeirDataset dataset = BeirLoader.load(dataDir, "test");
        System.out.println("Loaded " + dataset);

        EvalReport report = EvalRunner.run(new NaiveBm25Retriever(), dataset, 10);
        System.out.println(report);
    }
}