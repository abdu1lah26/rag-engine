package io.github.abdullah.rag.bm25;

import io.github.abdullah.rag.model.Document;
import io.github.abdullah.rag.retrieval.Retriever;
import io.github.abdullah.rag.retrieval.ScoredDoc;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;

/**
 * Naive BM25 baseline: no inverted index. Keeps per-document term counts in
 * memory
 * and scores every document for every query.
 */
public final class NaiveBm25Retriever implements Retriever {

    private final double k1;
    private final double b;

    private final List<String> docIds = new ArrayList<>();
    private final List<Map<String, Integer>> termCounts = new ArrayList<>();
    private final List<Integer> docLengths = new ArrayList<>();
    private final Map<String, Integer> docFreq = new HashMap<>();
    private double avgDocLength;

    public NaiveBm25Retriever() {
        this(1.2, 0.75);
    }

    public NaiveBm25Retriever(double k1, double b) {
        this.k1 = k1;
        this.b = b;
    }

    @Override
    public String name() {
        return "bm25-naive";
    }

    @Override
    public void index(List<Document> corpus) {
        docIds.clear();
        termCounts.clear();
        docLengths.clear();
        docFreq.clear();

        long totalLength = 0;
        for (Document doc : corpus) {
            List<String> tokens = Tokenizer.tokenize(doc.indexableText());
            Map<String, Integer> counts = new HashMap<>();
            for (String token : tokens) {
                counts.merge(token, 1, Integer::sum);
            }
            for (String term : counts.keySet()) {
                docFreq.merge(term, 1, Integer::sum);
            }
            docIds.add(doc.id());
            termCounts.add(counts);
            docLengths.add(tokens.size());
            totalLength += tokens.size();
        }
        avgDocLength = corpus.isEmpty() ? 0.0 : (double) totalLength / corpus.size();
    }

    @Override
    public List<ScoredDoc> search(String queryText, int k) {
        if (k <= 0) {
            throw new IllegalArgumentException("k must be positive, got " + k);
        }
        List<String> queryTerms = Tokenizer.tokenize(queryText);
        int n = docIds.size();

        // min-heap of size k: the weakest of the current top-k sits on top
        PriorityQueue<ScoredDoc> heap = new PriorityQueue<>(Comparator.comparingDouble(ScoredDoc::score));

        for (int i = 0; i < n; i++) {
            Map<String, Integer> counts = termCounts.get(i);
            double lengthNorm = 1.0 - b + b * docLengths.get(i) / avgDocLength;
            double score = 0.0;
            for (String term : queryTerms) {
                Integer tf = counts.get(term);
                if (tf == null) {
                    continue;
                }
                score += idf(term, n) * tf * (k1 + 1.0) / (tf + k1 * lengthNorm);
            }
            if (score <= 0.0) {
                continue;
            }
            if (heap.size() < k) {
                heap.add(new ScoredDoc(docIds.get(i), score));
            } else if (score > heap.peek().score()) {
                heap.poll();
                heap.add(new ScoredDoc(docIds.get(i), score));
            }
        }

        List<ScoredDoc> results = new ArrayList<>(heap);
        results.sort(Comparator.comparingDouble(ScoredDoc::score).reversed());
        return results;
    }

    private double idf(String term, int n) {
        int df = docFreq.getOrDefault(term, 0);
        return Math.log(1.0 + (n - df + 0.5) / (df + 0.5));
    }
}