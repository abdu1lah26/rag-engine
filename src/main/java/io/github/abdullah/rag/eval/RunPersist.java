package io.github.abdullah.rag.eval;

import io.github.abdullah.rag.index.IndexedBm25Retriever;
import io.github.abdullah.rag.model.Document;
import io.github.abdullah.rag.retrieval.Retriever;
import io.github.abdullah.rag.retrieval.ScoredDoc;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Locale;
import java.util.stream.Stream;

/** Build -> save -> load -> evaluate the loaded index, with timings and disk size. */
public final class RunPersist {

    private RunPersist() {
    }

    /** Wraps an already-built retriever so EvalRunner does not rebuild it. */
    private record PreBuilt(Retriever inner) implements Retriever {
        @Override
        public String name() {
            return inner.name() + "-loaded";
        }

        @Override
        public void index(List<Document> corpus) {
            // no-op: the index was loaded from disk
        }

        @Override
        public List<ScoredDoc> search(String queryText, int k) {
            return inner.search(queryText, k);
        }
    }

    public static void main(String[] args) throws IOException {
        Path dataDir = Path.of(args.length > 0 ? args[0] : "data/scifact");
        Path indexDir = Path.of("target", "index-scifact");
        BeirDataset dataset = BeirLoader.load(dataDir, "test");

        IndexedBm25Retriever built = new IndexedBm25Retriever();
        long t = System.nanoTime();
        built.index(dataset.corpus());
        double buildMs = ms(t);

        t = System.nanoTime();
        built.save(indexDir);
        double saveMs = ms(t);

        t = System.nanoTime();
        IndexedBm25Retriever loaded = IndexedBm25Retriever.load(indexDir);
        double loadMs = ms(t);

        t = System.nanoTime();
        IndexedBm25Retriever.load(indexDir);
        double loadWarmMs = ms(t);

        System.out.println(String.format(Locale.ROOT,
                "build=%.0f ms  save=%.0f ms  load(cold)=%.0f ms  load(warm)=%.0f ms",
                buildMs, saveMs, loadMs, loadWarmMs));

        long total = 0;
        try (Stream<Path> files = Files.list(indexDir)) {
            for (Path f : (Iterable<Path>) files::iterator) {
                long size = Files.size(f);
                total += size;
                System.out.println(String.format(Locale.ROOT, "  %-14s %,10d bytes", f.getFileName(), size));
            }
        }
        System.out.println(String.format(Locale.ROOT, "  %-14s %,10d bytes", "TOTAL", total));

        System.out.println(EvalRunner.run(new PreBuilt(loaded), dataset, 10));
    }

    private static double ms(long startNanos) {
        return (System.nanoTime() - startNanos) / 1_000_000.0;
    }
}