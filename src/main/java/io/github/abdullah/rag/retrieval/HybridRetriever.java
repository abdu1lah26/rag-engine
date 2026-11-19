package io.github.abdullah.rag.retrieval;

import io.github.abdullah.rag.fusion.RrfFusion;
import io.github.abdullah.rag.model.Document;

import java.util.ArrayList;
import java.util.List;

/**
 * Runs several retrievers on the same query and merges their ranked lists
 * with RRF. Each child is asked for candidateDepth results, and the fused
 * list is cut to k.
 *
 * Children run one after another for now. Parallel execution is a later,
 * measured change, so the first numbers are easy to reason about.
 */
public final class HybridRetriever implements Retriever {

    private final List<Retriever> children;
    private final RrfFusion fusion;
    private final int candidateDepth;
    private final boolean indexChildren;

    /**
     * @param children       retrievers to combine, e.g. BM25 and dense
     * @param fusion         the RRF fuser (holds the rank constant)
     * @param candidateDepth how many results to take from each child before fusing
     * @param indexChildren  true: index() builds the children. false: the children
     *                       are already built and index() does nothing, so one built
     *                       graph can be reused across many hybrid configurations.
     */
    public HybridRetriever(List<Retriever> children, RrfFusion fusion,
            int candidateDepth, boolean indexChildren) {
        if (children.isEmpty()) {
            throw new IllegalArgumentException("need at least one child retriever");
        }
        if (candidateDepth <= 0) {
            throw new IllegalArgumentException("candidateDepth must be > 0");
        }
        this.children = List.copyOf(children);
        this.fusion = fusion;
        this.candidateDepth = candidateDepth;
        this.indexChildren = indexChildren;
    }

    @Override
    public String name() {
        StringBuilder sb = new StringBuilder("hybrid-rrf[");
        for (int i = 0; i < children.size(); i++) {
            if (i > 0) {
                sb.append('+');
            }
            sb.append(children.get(i).name());
        }
        return sb.append(", c=").append(fusion.rankConstant())
                .append(", depth=").append(candidateDepth).append(']').toString();
    }

    @Override
    public void index(List<Document> corpus) {
        if (!indexChildren) {
            return;
        }
        for (Retriever r : children) {
            r.index(corpus);
        }
    }

    @Override
    public List<ScoredDoc> search(String queryText, int k) {
        List<List<ScoredDoc>> lists = new ArrayList<>(children.size());
        for (Retriever r : children) {
            lists.add(r.search(queryText, candidateDepth));
        }
        return fusion.fuse(lists, k);
    }
}