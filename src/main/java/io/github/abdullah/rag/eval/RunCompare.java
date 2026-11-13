package io.github.abdullah.rag.eval;

import io.github.abdullah.rag.bm25.NaiveBm25Retriever;
import io.github.abdullah.rag.index.IndexedBm25Retriever;

import java.io.IOException;
import java.nio.file.Path;

/**
 * Runs naive and indexed BM25 on SciFact (test split) and prints both reports.
 */
public final class RunCompare {

    private RunCompare() {
    }

    public static void main(String[] args) throws IOException {
        Path dataDir = Path.of(args.length > 0 ? args[0] : "data/scifact");
        BeirDataset dataset = BeirLoader.load(dataDir, "test");
        System.out.println("Loaded " + dataset);

        System.out.println(EvalRunner.run(new NaiveBm25Retriever(), dataset, 10));
        System.out.println(EvalRunner.run(new IndexedBm25Retriever(), dataset, 10));
    }
}