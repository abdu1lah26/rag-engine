package io.github.abdullah.rag.eval;

import io.github.abdullah.rag.analysis.StemmingAnalyzer;
import io.github.abdullah.rag.index.IndexFormat;
import io.github.abdullah.rag.index.IndexWriter;
import io.github.abdullah.rag.index.InvertedIndex;
import io.github.abdullah.rag.index.Posting;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Locale;

/** How big is the saved postings data vs a plain fixed-width layout? */
public final class RunIndexSize {

    private RunIndexSize() {
    }

    public static void main(String[] args) throws IOException {
        Path dataDir = Path.of(args.length > 0 ? args[0] : "data/scifact");
        Path indexDir = Path.of("target", "index-size");
        BeirDataset dataset = BeirLoader.load(dataDir, "test");

        InvertedIndex index = InvertedIndex.build(dataset.corpus(), new StemmingAnalyzer());
        IndexWriter.write(index, indexDir);

        long postingCount = 0;
        for (List<Posting> list : index.allPostings().values()) {
            postingCount += list.size();
        }

        // plain layout: 4-byte docIndex + 4-byte tf per posting
        long fixedBytes = postingCount * 8;
        // file = 8-byte header + body + 8-byte footer; compare the body only
        long varintBytes = Files.size(indexDir.resolve(IndexFormat.POSTINGS_FILE)) - 16;

        System.out.println(String.format(Locale.ROOT, "terms=%,d  postings=%,d", index.numTerms(), postingCount));
        System.out.println(String.format(Locale.ROOT, "fixed-width postings : %,d bytes (8.00 bytes/posting)", fixedBytes));
        System.out.println(String.format(Locale.ROOT, "varint+delta postings: %,d bytes (%.2f bytes/posting)",
                varintBytes, (double) varintBytes / postingCount));
        System.out.println(String.format(Locale.ROOT, "saved: %.1f%%  (%.2fx smaller)",
                100.0 * (fixedBytes - varintBytes) / fixedBytes, (double) fixedBytes / varintBytes));
        System.out.println(String.format(Locale.ROOT, "terms.bin: %,d bytes   docs.bin: %,d bytes",
                Files.size(indexDir.resolve(IndexFormat.TERMS_FILE)),
                Files.size(indexDir.resolve(IndexFormat.DOCS_FILE))));
    }
}