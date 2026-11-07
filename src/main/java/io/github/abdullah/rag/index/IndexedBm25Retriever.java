package io.github.abdullah.rag.index;

import io.github.abdullah.rag.analysis.SimpleAnalyzer;
import io.github.abdullah.rag.analysis.TextAnalyzer;
import io.github.abdullah.rag.model.Document;
import io.github.abdullah.rag.retrieval.Retriever;
import io.github.abdullah.rag.retrieval.ScoredDoc;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;

/**
 * BM25 over an inverted index. Same formula as NaiveBm25Retriever, but a query
 * only touches the posting lists of its own terms instead of scanning every
 * document. Documents and queries go through the SAME TextAnalyzer.
 */
public final class IndexedBm25Retriever implements Retriever {

    private final double k1;
    private final double b;
    private final TextAnalyzer analyzer;
    private InvertedIndex index;

    public IndexedBm25Retriever() {
        this(new SimpleAnalyzer(), 1.2, 0.75);
    }

    public IndexedBm25Retriever(TextAnalyzer analyzer) {
        this(analyzer, 1.2, 0.75);
    }

    public IndexedBm25Retriever(double k1, double b) {
        this(new SimpleAnalyzer(), k1, b);
    }

    public IndexedBm25Retriever(TextAnalyzer analyzer, double k1, double b) {
        this.analyzer = analyzer;
        this.k1 = k1;
        this.b = b;
    }

    /** Wrap an index that already exists. The analyzer must be the one that built it. */
    public IndexedBm25Retriever(InvertedIndex index, TextAnalyzer analyzer) {
        this(analyzer, 1.2, 0.75);
        if (!index.analyzerName().equals(analyzer.name())) {
            throw new IllegalArgumentException("index was built with analyzer '" + index.analyzerName()
                    + "' but the retriever was given '" + analyzer.name() + "'");
        }
        this.index = index;
    }

    /** Wrap an existing index built with the default (simple) analyzer. */
    public IndexedBm25Retriever(InvertedIndex index) {
        this(index, new SimpleAnalyzer());
    }

    @Override
    public String name() {
        return "bm25-indexed-" + analyzer.name();
    }

    @Override
    public void index(List<Document> corpus) {
        this.index = InvertedIndex.build(corpus, analyzer);
    }

    @Override
    public List<ScoredDoc> search(String queryText, int k) {
        if (k <= 0) {
            throw new IllegalArgumentException("k must be positive, got " + k);
        }
        int n = index.numDocs();
        double avgLen = index.avgDocLength();
        double[] scores = new double[n];

        // term-at-a-time: walk each query term's posting list, add into per-doc slots
        for (String term : analyzer.analyze(queryText)) {
            List<Posting> list = index.postings(term);
            if (list.isEmpty()) {
                continue;
            }
            double idf = idf(list.size(), n);
            for (Posting p : list) {
                double lengthNorm = 1.0 - b + b * index.docLength(p.docIndex()) / avgLen;
                scores[p.docIndex()] += idf * p.tf() * (k1 + 1.0) / (p.tf() + k1 * lengthNorm);
            }
        }

        // pick top-k with a min-heap, same as the naive version
        PriorityQueue<ScoredDoc> heap = new PriorityQueue<>(Comparator.comparingDouble(ScoredDoc::score));
        for (int i = 0; i < n; i++) {
            double s = scores[i];
            if (s <= 0.0) {
                continue;
            }
            if (heap.size() < k) {
                heap.add(new ScoredDoc(index.docId(i), s));
            } else if (s > heap.peek().score()) {
                heap.poll();
                heap.add(new ScoredDoc(index.docId(i), s));
            }
        }
        List<ScoredDoc> results = new ArrayList<>(heap);
        results.sort(Comparator.comparingDouble(ScoredDoc::score).reversed());
        return results;
    }

    /** Persist the built index to a directory (see IndexFormat). */
    public void save(Path dir) throws IOException {
        if (index == null) {
            throw new IllegalStateException("nothing to save: call index() first");
        }
        IndexWriter.write(index, dir);
    }

    /** Load with the default (simple) analyzer; fails if the saved index used another. */
    public static IndexedBm25Retriever load(Path dir) throws IOException {
        return load(dir, new SimpleAnalyzer());
    }

    /** Load a saved index; fails if its analyzer name differs from the one given. */
    public static IndexedBm25Retriever load(Path dir, TextAnalyzer analyzer) throws IOException {
        return new IndexedBm25Retriever(IndexReader.load(dir), analyzer);
    }

    private static double idf(int df, int n) {
        return Math.log(1.0 + (n - df + 0.5) / (df + 0.5));
    }
}