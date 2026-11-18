package io.github.abdullah.rag.vector;

import io.github.abdullah.rag.retrieval.ScoredDoc;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Comparator;
import java.util.HashSet;
import java.util.PriorityQueue;
import java.util.Set;

/**
 * Hierarchical Navigable Small World graph index (Malkov & Yashunin, 2016).
 * Approximate nearest-neighbour search over unit vectors using cosine distance.
 *
 * Internally every vector gets an int "slot" (its position in `nodes`);
 * graph edges are slot numbers, not string ids, because ints are cheaper to
 * store and compare.
 */
public class HnswIndex implements VectorIndex {

    /** One vector plus its edges on every layer it appears in. */
    static final class Node {
        final String id;
        final float[] vector;
        final int level; // highest layer this node appears in
        final List<Integer>[] neighbours; // neighbours[layer] = slots of linked nodes

        @SuppressWarnings("unchecked")
        Node(String id, float[] vector, int level) {
            this.id = id;
            this.vector = vector;
            this.level = level;
            this.neighbours = new List[level + 1];
            for (int l = 0; l <= level; l++) {
                this.neighbours[l] = new ArrayList<>();
            }
        }
    }

    private final int dimension;
    private final HnswConfig config;
    private final Random rng;

    private final List<Node> nodes = new ArrayList<>();
    private final Map<String, Integer> idToSlot = new HashMap<>();

    private int entryPoint = -1; // slot of the node on the top layer, -1 when empty
    private int maxLevel = -1; // top layer currently in the graph

    private int efSearch = 50; // query-time beam width, changeable without rebuilding

    private boolean useHeuristic = true; // false = plain "M nearest" (for the ablation)

    /** Must be called before the first add(). */
    public void setUseHeuristic(boolean useHeuristic) {
        if (!nodes.isEmpty()) {
            throw new IllegalStateException("set before adding vectors");
        }
        this.useHeuristic = useHeuristic;
    }

    public HnswIndex(int dimension, HnswConfig config) {
        if (dimension <= 0) {
            throw new IllegalArgumentException("dimension must be positive");
        }
        this.dimension = dimension;
        this.config = config;
        this.rng = new Random(config.seed());
    }

    @Override
    public String name() {
        return "hnsw-M" + config.m() + "-efC" + config.efConstruction();
    }

    @Override
    public int dimension() {
        return dimension;
    }

    @Override
    public int size() {
        return nodes.size();
    }

    public int getEfSearch() {
        return efSearch;
    }

    public void setEfSearch(int efSearch) {
        if (efSearch < 1) {
            throw new IllegalArgumentException("efSearch must be >= 1");
        }
        this.efSearch = efSearch;
    }

    /**
     * Picks the top layer for a new node: floor(-ln(U) * mL), U uniform in (0,1].
     * Layer 0 gets ~everyone, each higher layer ~1/M of the layer below.
     */
    int randomLevel() {
        double u = 1.0 - rng.nextDouble(); // in (0,1], avoids ln(0)
        return (int) Math.floor(-Math.log(u) * config.levelMultiplier());
    }

    /** A node slot paired with its distance to the query being processed. */
    record Candidate(int slot, float dist) {
    }

    private float distanceTo(float[] query, int slot) {
        return VectorMath.distance(query, nodes.get(slot).vector);
    }

    /** Test hook: appends an unlinked node and returns its slot. */
    int addRawNode(String id, float[] vector, int level) {
        nodes.add(new Node(id, vector, level));
        int slot = nodes.size() - 1;
        idToSlot.put(id, slot);
        return slot;
    }

    /** Test hook: adds an edge in both directions on the given layer. */
    void link(int a, int b, int layer) {
        nodes.get(a).neighbours[layer].add(b);
        nodes.get(b).neighbours[layer].add(a);
    }

    /**
     * Neighbour selection heuristic (Algorithm 4 in the paper, with
     * keepPrunedConnections). Picks up to maxCount slots from candidates,
     * preferring diverse directions over a tight cluster of near-duplicates.
     *
     * @param candidates must be sorted nearest-first, dist = distance to the base
     *                   node
     * @return chosen slots, in the order they were selected
     */
    List<Integer> selectNeighbours(List<Candidate> candidates, int maxCount) {
        if (!useHeuristic) {
            List<Integer> nearest = new ArrayList<>();
            for (Candidate c : candidates) {
                if (nearest.size() >= maxCount) {
                    break;
                }
                nearest.add(c.slot());
            }
            return nearest;
        }
        List<Integer> selected = new ArrayList<>();
        List<Candidate> discarded = new ArrayList<>();

        for (Candidate c : candidates) {
            if (selected.size() >= maxCount) {
                break;
            }
            float[] cv = nodes.get(c.slot()).vector;
            boolean covered = false;
            for (int s : selected) {
                // c is covered if some already-chosen neighbour is closer to c
                // than the base node is: that neighbour already points this way.
                if (VectorMath.distance(cv, nodes.get(s).vector) < c.dist()) {
                    covered = true;
                    break;
                }
            }
            if (covered) {
                discarded.add(c);
            } else {
                selected.add(c.slot());
            }
        }

        // keepPrunedConnections: top up with the nearest discarded candidates.
        for (Candidate c : discarded) {
            if (selected.size() >= maxCount) {
                break;
            }
            selected.add(c.slot());
        }
        return selected;
    }

    /**
     * Beam search on one layer (Algorithm 2 in the paper).
     * Returns up to ef nodes found closest to the query, nearest first.
     */
    List<Candidate> searchLayer(float[] query, int entrySlot, int ef, int layer) {
        Set<Integer> visited = new HashSet<>();
        // Min-heap by distance: next node to expand is the closest unexpanded one.
        PriorityQueue<Candidate> candidates = new PriorityQueue<>(Comparator.comparingDouble(Candidate::dist));
        // Max-heap by distance: root is the worst of the current best ef.
        PriorityQueue<Candidate> results = new PriorityQueue<>((a, b) -> Float.compare(b.dist(), a.dist()));

        Candidate start = new Candidate(entrySlot, distanceTo(query, entrySlot));
        visited.add(entrySlot);
        candidates.add(start);
        results.add(start);

        while (!candidates.isEmpty()) {
            Candidate current = candidates.poll();
            if (current.dist() > results.peek().dist()) {
                break; // everything left is farther than our worst result
            }
            for (int neighbour : nodes.get(current.slot()).neighbours[layer]) {
                if (!visited.add(neighbour)) {
                    continue;
                }
                float d = distanceTo(query, neighbour);
                if (results.size() < ef || d < results.peek().dist()) {
                    Candidate c = new Candidate(neighbour, d);
                    candidates.add(c);
                    results.add(c);
                    if (results.size() > ef) {
                        results.poll(); // drop the farthest
                    }
                }
            }
        }

        List<Candidate> out = new ArrayList<>(results);
        out.sort(Comparator.comparingDouble(Candidate::dist));
        return out;
    }

    @Override
    public void add(String id, float[] vector) {
        if (vector.length != dimension) {
            throw new IllegalArgumentException(
                    "expected dimension " + dimension + " but got " + vector.length);
        }
        if (idToSlot.containsKey(id)) {
            throw new IllegalArgumentException("duplicate id: " + id);
        }

        int level = randomLevel();
        int slot = addRawNode(id, vector, level);

        if (entryPoint == -1) { // first node: nothing to connect to
            entryPoint = slot;
            maxLevel = level;
            return;
        }

        // Phase 1: greedy descent through layers above the new node's top layer.
        int current = entryPoint;
        for (int layer = maxLevel; layer > level; layer--) {
            current = searchLayer(vector, current, 1, layer).get(0).slot();
        }

        // Phase 2: connect on every layer from min(level, maxLevel) down to 0.
        for (int layer = Math.min(level, maxLevel); layer >= 0; layer--) {
            List<Candidate> found = searchLayer(vector, current, config.efConstruction(), layer);
            List<Integer> chosen = selectNeighbours(found, config.m());
            int maxConn = (layer == 0) ? config.maxNeighboursLayer0() : config.m();

            for (int n : chosen) {
                nodes.get(slot).neighbours[layer].add(n);
                nodes.get(n).neighbours[layer].add(slot);
                if (nodes.get(n).neighbours[layer].size() > maxConn) {
                    shrinkConnections(n, layer, maxConn);
                }
            }
            current = found.get(0).slot(); // nearest hit seeds the next layer down
        }

        if (level > maxLevel) {
            entryPoint = slot;
            maxLevel = level;
        }
    }

    /**
     * Re-prunes an over-full neighbour list back to maxConn using the heuristic.
     */
    private void shrinkConnections(int slot, int layer, int maxConn) {
        Node node = nodes.get(slot);
        List<Candidate> cands = new ArrayList<>();
        for (int nb : node.neighbours[layer]) {
            cands.add(new Candidate(nb,
                    VectorMath.distance(node.vector, nodes.get(nb).vector)));
        }
        cands.sort(Comparator.comparingDouble(Candidate::dist));
        List<Integer> kept = selectNeighbours(cands, maxConn);
        node.neighbours[layer].clear();
        node.neighbours[layer].addAll(kept);
    }

    @Override
    public List<ScoredDoc> search(float[] query, int k) {
        if (query.length != dimension) {
            throw new IllegalArgumentException(
                    "expected dimension " + dimension + " but got " + query.length);
        }
        if (k <= 0 || entryPoint == -1) {
            return List.of();
        }

        int current = entryPoint;
        for (int layer = maxLevel; layer >= 1; layer--) {
            current = searchLayer(query, current, 1, layer).get(0).slot();
        }

        List<Candidate> found = searchLayer(query, current, Math.max(efSearch, k), 0);

        List<ScoredDoc> out = new ArrayList<>();
        for (int i = 0; i < Math.min(k, found.size()); i++) {
            Candidate c = found.get(i);
            out.add(new ScoredDoc(nodes.get(c.slot()).id, 1.0 - c.dist()));
        }
        return out;
    }
}