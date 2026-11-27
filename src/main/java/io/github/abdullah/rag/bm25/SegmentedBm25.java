package io.github.abdullah.rag.bm25;

import io.github.abdullah.rag.analysis.TextAnalyzer;
import io.github.abdullah.rag.index.InvertedIndex;
import io.github.abdullah.rag.index.Posting;
import io.github.abdullah.rag.retrieval.ScoredDoc;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;
import java.util.function.Predicate;

/**
 * BM25 over a base segment plus a small delta segment, using combined statistics
 * (N, df, average length) so scores from both segments are comparable.
 * Base docs rejected by baseVisible (replaced or deleted) are never returned.
 *
 * Known limit: hidden base docs still count in N, df and avgLen until compaction
 * (same as Lucene before a merge). Quantified in the compaction step.
 */
public final class SegmentedBm25 {

    private final TextAnalyzer analyzer;
    private final double k1;
    private final double b;

    public SegmentedBm25(TextAnalyzer analyzer, double k1, double b) {
        this.analyzer = analyzer;
        this.k1 = k1;
        this.b = b;
    }

    public SegmentedBm25(TextAnalyzer analyzer) {
        this(analyzer, 1.2, 0.75);
    }

    public List<ScoredDoc> search(InvertedIndex base, InvertedIndex delta,
                                  Predicate<String> baseVisible, String queryText, int k) {
        if (k <= 0) {
            throw new IllegalArgumentException("k must be positive, got " + k);
        }
        int nBase = base.numDocs();
        int nDelta = delta.numDocs();
        int n = nBase + nDelta;
        if (n == 0) {
            return List.of();
        }
        double avgLen = (base.avgDocLength() * nBase + delta.avgDocLength() * nDelta) / n;

        double[] baseScores = new double[nBase];
        double[] deltaScores = new double[nDelta];

        for (String term : analyzer.analyze(queryText)) {
            List<Posting> bp = base.postings(term);
            List<Posting> dp = delta.postings(term);
            int df = bp.size() + dp.size();
            if (df == 0) {
                continue;
            }
            double idf = Math.log(1.0 + (n - df + 0.5) / (df + 0.5));
            for (Posting p : bp) {
                baseScores[p.docIndex()] += termScore(idf, p.tf(), base.docLength(p.docIndex()), avgLen);
            }
            for (Posting p : dp) {
                deltaScores[p.docIndex()] += termScore(idf, p.tf(), delta.docLength(p.docIndex()), avgLen);
            }
        }

        PriorityQueue<ScoredDoc> heap = new PriorityQueue<>(
                Comparator.comparingDouble(ScoredDoc::score).thenComparing(ScoredDoc::docId, Comparator.reverseOrder()));
        for (int i = 0; i < nBase; i++) {
            if (baseScores[i] > 0 && baseVisible.test(base.docId(i))) {
                offer(heap, k, new ScoredDoc(base.docId(i), baseScores[i]));
            }
        }
        for (int i = 0; i < nDelta; i++) {
            if (deltaScores[i] > 0) {
                offer(heap, k, new ScoredDoc(delta.docId(i), deltaScores[i]));
            }
        }
        List<ScoredDoc> out = new ArrayList<>(heap);
        out.sort(Comparator.comparingDouble(ScoredDoc::score).reversed().thenComparing(ScoredDoc::docId));
        return out;
    }

    private double termScore(double idf, int tf, int docLen, double avgLen) {
        double lengthNorm = 1.0 - b + b * docLen / avgLen;
        return idf * tf * (k1 + 1.0) / (tf + k1 * lengthNorm);
    }

    private static void offer(PriorityQueue<ScoredDoc> heap, int k, ScoredDoc d) {
        if (heap.size() < k) {
            heap.add(d);
        } else if (d.score() > heap.peek().score()) {
            heap.poll();
            heap.add(d);
        }
    }
}