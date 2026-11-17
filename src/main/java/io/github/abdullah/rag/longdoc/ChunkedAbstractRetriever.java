package io.github.abdullah.rag.longdoc;

import io.github.abdullah.rag.analysis.TextAnalyzer;
import io.github.abdullah.rag.chunking.Chunk;
import io.github.abdullah.rag.chunking.Chunker;
import io.github.abdullah.rag.index.IndexedBm25Retriever;
import io.github.abdullah.rag.longdoc.LongDocBuilder.LongDocs;
import io.github.abdullah.rag.longdoc.LongDocBuilder.Span;
import io.github.abdullah.rag.model.Document;
import io.github.abdullah.rag.retrieval.ScoredDoc;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Chunks long documents, indexes the chunks with our BM25, and answers queries
 * with a ranked
 * list of ORIGINAL abstract ids. A chunk credits every abstract whose char span
 * it overlaps,
 * so a chunk straddling two abstracts pushes both into the ranking.
 */
public final class ChunkedAbstractRetriever {

    private final Chunker chunker;
    private final IndexedBm25Retriever bm25;
    private final int chunkDepth;
    private final Map<String, List<String>> chunkToAbstracts = new HashMap<>();
    private int chunkCount;
    private Map<String, String> titles = Map.of();

    public ChunkedAbstractRetriever(Chunker chunker, TextAnalyzer analyzer, int chunkDepth) {
        if (chunkDepth < 1)
            throw new IllegalArgumentException("chunkDepth must be >= 1");
        this.chunker = chunker;
        this.bm25 = new IndexedBm25Retriever(analyzer);
        this.chunkDepth = chunkDepth;
    }

    /**
     * Optional: re-attach an abstract's title to chunks that overlap it but start
     * after its beginning.
     */
    public ChunkedAbstractRetriever withTitles(Map<String, String> titleById) {
        this.titles = titleById;
        return this;
    }

    public String name() {
        return chunker.name();
    }

    public int chunkCount() {
        return chunkCount;
    }

    public void index(LongDocs longDocs) {
        List<Document> chunkDocs = new ArrayList<>();
        chunkToAbstracts.clear();

        for (Document longDoc : longDocs.docs()) {
            List<Span> spans = longDocs.spans().get(longDoc.id());
            for (Chunk c : chunker.chunk(longDoc.id(), longDoc.text())) {
                List<String> covered = new ArrayList<>();
                StringBuilder prefix = new StringBuilder();
                for (Span s : spans) {
                    if (c.startChar() < s.end() && s.start() < c.endChar()) {
                        covered.add(s.abstractId());
                        String t = titles.get(s.abstractId());
                        // a chunk starting at the span start already contains the title
                        if (t != null && !t.isBlank() && c.startChar() > s.start()) {
                            prefix.append(t).append(". ");
                        }
                    }
                }
                chunkToAbstracts.put(c.chunkId(), covered);
                chunkDocs.add(new Document(c.chunkId(), "", prefix + c.text()));
            }
        }
        chunkCount = chunkDocs.size();
        bm25.index(chunkDocs);
    }

    /** Ranked original abstract ids, best first, no duplicates, at most k. */
    public List<String> search(String query, int k) {
        Set<String> ranked = new LinkedHashSet<>();
        for (ScoredDoc hit : bm25.search(query, chunkDepth)) {
            ranked.addAll(chunkToAbstracts.get(hit.docId()));
            if (ranked.size() >= k)
                break;
        }
        List<String> out = new ArrayList<>(ranked);
        return out.size() > k ? out.subList(0, k) : out;
    }
}