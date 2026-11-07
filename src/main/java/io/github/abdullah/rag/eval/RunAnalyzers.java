package io.github.abdullah.rag.eval;

import io.github.abdullah.rag.analysis.SimpleAnalyzer;
import io.github.abdullah.rag.analysis.StopwordAnalyzer;
import io.github.abdullah.rag.analysis.TextAnalyzer;
import io.github.abdullah.rag.index.IndexedBm25Retriever;
import io.github.abdullah.rag.analysis.StemmingAnalyzer;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;

/** Our indexed BM25 under each analyzer on SciFact test. Add new analyzers to the list. */
public final class RunAnalyzers {

    private RunAnalyzers() {
    }

    public static void main(String[] args) throws IOException {
        Path dataDir = Path.of(args.length > 0 ? args[0] : "data/scifact");
        BeirDataset dataset = BeirLoader.load(dataDir, "test");
        System.out.println("Loaded " + dataset);

        List<TextAnalyzer> analyzers = List.of(
            new SimpleAnalyzer(),
            new StopwordAnalyzer(),
            new StemmingAnalyzer());

        for (TextAnalyzer analyzer : analyzers) {
            System.out.println(EvalRunner.run(new IndexedBm25Retriever(analyzer), dataset, 10));
        }
    }
}