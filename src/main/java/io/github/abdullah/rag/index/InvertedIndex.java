package io.github.abdullah.rag.index;

import io.github.abdullah.rag.bm25.Tokenizer;
import io.github.abdullah.rag.model.Document;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Immutable-after-build inverted index: term -> posting list, plus the
 * per-document statistics BM25 needs (lengths, average length).
 * Scoring lives elsewhere; this class only stores and exposes data.
 */
public final class InvertedIndex {

    private final List<String> docIds;
    private final int[] docLengths;
    private final double avgDocLength;
    private final Map<String, List<Posting>> postings;

    /** Used by the builder below and, later, by the disk loader. */
    public InvertedIndex(List<String> docIds, int[] docLengths, Map<String, List<Posting>> postings) {
        this.docIds = List.copyOf(docIds);
        this.docLengths = docLengths;
        this.postings = postings;
        long total = 0;
        for (int len : docLengths) {
            total += len;
        }
        this.avgDocLength = docIds.isEmpty() ? 0.0 : (double) total / docIds.size();
    }

    /** Build an index from a corpus in one pass. */
    public static InvertedIndex build(List<Document> corpus) {
        List<String> ids = new ArrayList<>(corpus.size());
        int[] lengths = new int[corpus.size()];
        Map<String, List<Posting>> postings = new HashMap<>();

        for (int docIndex = 0; docIndex < corpus.size(); docIndex++) {
            Document doc = corpus.get(docIndex);
            List<String> tokens = Tokenizer.tokenize(doc.indexableText());

            // count terms of this document; LinkedHashMap keeps output deterministic
            Map<String, Integer> counts = new LinkedHashMap<>();
            for (String token : tokens) {
                counts.merge(token, 1, Integer::sum);
            }
            for (Map.Entry<String, Integer> e : counts.entrySet()) {
                postings.computeIfAbsent(e.getKey(), t -> new ArrayList<>())
                        .add(new Posting(docIndex, e.getValue()));
            }
            ids.add(doc.id());
            lengths[docIndex] = tokens.size();
        }
        return new InvertedIndex(ids, lengths, postings);
    }

    public int numDocs() {
        return docIds.size();
    }

    public double avgDocLength() {
        return avgDocLength;
    }

    public int docLength(int docIndex) {
        return docLengths[docIndex];
    }

    public String docId(int docIndex) {
        return docIds.get(docIndex);
    }

    /** Posting list for a term, empty if the term is unknown. */
    public List<Posting> postings(String term) {
        return postings.getOrDefault(term, List.of());
    }

    /** Document frequency: how many documents contain the term. */
    public int docFreq(String term) {
        return postings(term).size();
    }

    public int numTerms() {
        return postings.size();
    }

    /** Read-only view of the whole map; the disk writer needs it. */
    public Map<String, List<Posting>> allPostings() {
        return java.util.Collections.unmodifiableMap(postings);
    }

    public List<String> allDocIds() {
        return docIds;
    }

    public int[] docLengthsCopy() {
        return docLengths.clone();
    }
}