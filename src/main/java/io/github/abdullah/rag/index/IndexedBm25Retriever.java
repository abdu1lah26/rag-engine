package io.github.abdullah.rag.index;

import io.github.abdullah.rag.bm25.Tokenizer;
import io.github.abdullah.rag.model.Document;
import io.github.abdullah.rag.retrieval.Retriever;
import io.github.abdullah.rag.retrieval.ScoredDoc;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;

/**
 * BM25 over an inverted index. Same formula and same tokenizer as
 * NaiveBm25Retriever, but a query only touches the posting lists of its own
 * terms instead of scanning every document.
 */
public final class IndexedBm25Retriever implements Retriever {

    private final double k1;
    private final double b;
    private InvertedIndex index;

    public IndexedBm25Retriever() {
        this(1.2, 0.75);
    }

    public IndexedBm25Retriever(double k1, double b) {
        this.k1 = k1;
        this.b = b;
    }

    /** Wrap an index that already exists (the disk loader will use this). */
    public IndexedBm25Retriever(InvertedIndex index) {
        this(1.2, 0.75);
        this.index = index;
    }

    @Override
    public String name() {
        return "bm25-indexed";
    }

    @Override
    public void index(List<Document> corpus) {
        this.index = InvertedIndex.build(corpus);
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
        for (String term : Tokenizer.tokenize(queryText)) {
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

    private static double idf(int df, int n) {
        return Math.log(1.0 + (n - df + 0.5) / (df + 0.5));
    }
}