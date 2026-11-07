package io.github.abdullah.rag.eval;

import io.github.abdullah.rag.baseline.LuceneBm25Retriever;
import io.github.abdullah.rag.index.IndexedBm25Retriever;

import java.io.IOException;
import java.nio.file.Path;

/** Our indexed BM25 vs Lucene (standard and english analyzers) on SciFact test. */
public final class RunLuceneCompare {

    private RunLuceneCompare() {
    }

    public static void main(String[] args) throws IOException {
        Path dataDir = Path.of(args.length > 0 ? args[0] : "data/scifact");
        BeirDataset dataset = BeirLoader.load(dataDir, "test");
        System.out.println("Loaded " + dataset);

        System.out.println(EvalRunner.run(new IndexedBm25Retriever(), dataset, 10));
        try (LuceneBm25Retriever standard = LuceneBm25Retriever.standard()) {
            System.out.println(EvalRunner.run(standard, dataset, 10));
        }
        try (LuceneBm25Retriever english = LuceneBm25Retriever.english()) {
            System.out.println(EvalRunner.run(english, dataset, 10));
        }
    }
}