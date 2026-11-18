package io.github.abdullah.rag.vector;

import io.github.abdullah.rag.retrieval.ScoredDoc;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Set;

/**
 * Exact nearest-neighbour search: compares the query against every stored
 * vector. O(n * d) per query. Slow at scale, but always correct, so it is
 * the ground truth that HNSW recall is measured against.
 */
public class BruteForceIndex implements VectorIndex {

    private final int dimension;
    private final List<String> ids = new ArrayList<>();
    private final List<float[]> vectors = new ArrayList<>();
    private final Set<String> idSet = new HashSet<>();

    public BruteForceIndex(int dimension) {
        if (dimension <= 0) {
            throw new IllegalArgumentException("dimension must be positive");
        }
        this.dimension = dimension;
    }

    @Override
    public String name() {
        return "brute-force";
    }

    @Override
    public int dimension() {
        return dimension;
    }

    @Override
    public int size() {
        return ids.size();
    }

    @Override
    public void add(String id, float[] vector) {
        if (vector.length != dimension) {
            throw new IllegalArgumentException(
                    "expected dimension " + dimension + " but got " + vector.length);
        }
        if (!idSet.add(id)) {
            throw new IllegalArgumentException("duplicate id: " + id);
        }
        ids.add(id);
        vectors.add(vector);
    }

    @Override
    public List<ScoredDoc> search(float[] query, int k) {
        if (query.length != dimension) {
            throw new IllegalArgumentException(
                    "expected dimension " + dimension + " but got " + query.length);
        }
        if (k <= 0) {
            return List.of();
        }
        // Min-heap on score: the root is the worst of the current top-k, so it
        // is the one to evict when a better candidate arrives.
        PriorityQueue<ScoredDoc> heap =
                new PriorityQueue<>((a, b) -> Double.compare(a.score(), b.score()));
        for (int i = 0; i < ids.size(); i++) {
            double score = VectorMath.dot(query, vectors.get(i));
            if (heap.size() < k) {
                heap.add(new ScoredDoc(ids.get(i), score));
            } else if (score > heap.peek().score()) {
                heap.poll();
                heap.add(new ScoredDoc(ids.get(i), score));
            }
        }
        List<ScoredDoc> result = new ArrayList<>(heap);
        result.sort((a, b) -> Double.compare(b.score(), a.score()));
        return result;
    }
}