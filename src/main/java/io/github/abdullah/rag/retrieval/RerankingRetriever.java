package io.github.abdullah.rag.retrieval;

import io.github.abdullah.rag.model.Document;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Two-stage retrieval: a cheap first stage proposes rerankDepth candidates, a
 * slower pair scorer (cross-encoder) re-scores each against the query, and the
 * top k by that score are returned. Reranked scores replace the first-stage
 * scores, and are only comparable within one query.
 */
public final class RerankingRetriever implements Retriever {

    /** Scores one (query, document text) pair. Higher = more relevant. */
    public interface PairScorer {
        float score(String query, String document);
    }

    private final Retriever firstStage;
    private final PairScorer scorer;
    private final String scorerName;
    private final int rerankDepth;
    private final boolean indexFirstStage;
    private final Map<String, String> textById = new HashMap<>();

    /**
     * @param indexFirstStage true: index() also builds the first stage. false: it is
     *                        already built, so index() only stores the document texts.
     */
    public RerankingRetriever(Retriever firstStage, PairScorer scorer, String scorerName,
            int rerankDepth, boolean indexFirstStage) {
        if (rerankDepth <= 0) {
            throw new IllegalArgumentException("rerankDepth must be > 0");
        }
        this.firstStage = firstStage;
        this.scorer = scorer;
        this.scorerName = scorerName;
        this.rerankDepth = rerankDepth;
        this.indexFirstStage = indexFirstStage;
    }

    @Override
    public String name() {
        return "rerank[" + firstStage.name() + " -> " + scorerName + ", depth=" + rerankDepth + "]";
    }

    @Override
    public void index(List<Document> corpus) {
        if (indexFirstStage) {
            firstStage.index(corpus);
        }
        textById.clear();
        for (Document d : corpus) {
            textById.put(d.id(), d.indexableText());
        }
    }

    @Override
    public List<ScoredDoc> search(String queryText, int k) {
        List<ScoredDoc> candidates = firstStage.search(queryText, rerankDepth);
        List<ScoredDoc> rescored = new ArrayList<>(candidates.size());
        for (ScoredDoc c : candidates) {
            String text = textById.get(c.docId());
            if (text == null) {
                throw new IllegalStateException("no text for doc " + c.docId());
            }
            rescored.add(new ScoredDoc(c.docId(), scorer.score(queryText, text)));
        }
        // Ties are broken by docId so results are deterministic.
        rescored.sort(Comparator.comparingDouble(ScoredDoc::score).reversed()
                .thenComparing(ScoredDoc::docId));
        return rescored.size() > k ? new ArrayList<>(rescored.subList(0, k)) : rescored;
    }
}