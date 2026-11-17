package io.github.abdullah.rag.chunking;

import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * TextTiling-style chunker. Embeds every sentence, then for each gap between sentences
 * compares the `window` sentences before it with the `window` sentences after it.
 * A gap whose similarity is a local minimum below (mean - alpha * std) is a topic shift
 * and becomes a chunk boundary, subject to minWords and a hard maxWords cap.
 */
public final class SemanticChunker implements Chunker {

    private static final Pattern SENTENCE_SPLIT = Pattern.compile("(?<=[.!?])\\s+|\\n\\s*\\n");

    private record Sentence(int start, int end, int words, float[] vec) {}

    private final Embedder embedder;
    private final int window;
    private final int minWords;
    private final int maxWords;
    private final double alpha;

    public SemanticChunker(Embedder embedder, int window, int minWords, int maxWords, double alpha) {
        if (embedder == null) throw new IllegalArgumentException("embedder required");
        if (window < 1) throw new IllegalArgumentException("window must be >= 1");
        if (minWords < 1 || maxWords < minWords) throw new IllegalArgumentException("need 1 <= minWords <= maxWords");
        if (alpha < 0) throw new IllegalArgumentException("alpha must be >= 0");
        this.embedder = embedder;
        this.window = window;
        this.minWords = minWords;
        this.maxWords = maxWords;
        this.alpha = alpha;
    }

    @Override
    public String name() {
        return "semantic-" + embedder.name() + "-k" + window + "-min" + minWords + "-max" + maxWords + "-a" + alpha;
    }

    @Override
    public List<Chunk> chunk(String docId, String text) {
        List<Sentence> sents = splitSentences(text);
        int n = sents.size();
        List<Chunk> out = new ArrayList<>();
        if (n == 0) return out;

        boolean[] cut = findBoundaries(sents);

        int index = 0;
        int start = 0;
        int wc = 0;
        for (int i = 0; i < n; i++) {
            int w = sents.get(i).words();
            // hard cap: close the current chunk before this sentence would overflow it
            if (i > start && wc + w > maxWords) {
                index = emit(out, docId, text, sents, start, i - 1, index);
                start = i;
                wc = 0;
            }
            wc += w;
            // topic shift after sentence i, and the chunk is big enough
            if (i < n - 1 && cut[i] && wc >= minWords) {
                index = emit(out, docId, text, sents, start, i, index);
                start = i + 1;
                wc = 0;
            }
        }
        if (start <= n - 1) emit(out, docId, text, sents, start, n - 1, index);
        return out;
    }

    /** cut[g] = true means "put a boundary after sentence g". */
    private boolean[] findBoundaries(List<Sentence> sents) {
        int n = sents.size();
        boolean[] cut = new boolean[n];
        if (n < 2) return cut;

        double[] score = new double[n - 1];
        for (int g = 0; g < n - 1; g++) {
            double[] left = sum(sents, Math.max(0, g - window + 1), g);
            double[] right = sum(sents, g + 1, Math.min(n - 1, g + window));
            score[g] = cosine(left, right);
        }

        double mean = 0;
        for (double s : score) mean += s;
        mean /= score.length;
        double var = 0;
        for (double s : score) var += (s - mean) * (s - mean);
        double std = Math.sqrt(var / score.length);
        double threshold = mean - alpha * std;

        for (int g = 0; g < score.length; g++) {
            boolean leftOk = g == 0 || score[g] <= score[g - 1];
            boolean rightOk = g == score.length - 1 || score[g] <= score[g + 1];
            cut[g] = score[g] < threshold && leftOk && rightOk;
        }
        return cut;
    }

    private double[] sum(List<Sentence> sents, int from, int to) {
        double[] acc = new double[embedder.dimension()];
        for (int i = from; i <= to; i++) {
            float[] v = sents.get(i).vec();
            for (int d = 0; d < acc.length; d++) acc[d] += v[d];
        }
        return acc;
    }

    private static double cosine(double[] a, double[] b) {
        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < a.length; i++) {
            dot += a[i] * b[i];
            na += a[i] * a[i];
            nb += b[i] * b[i];
        }
        if (na == 0 || nb == 0) return 0;
        return dot / (Math.sqrt(na) * Math.sqrt(nb));
    }

    private int emit(List<Chunk> out, String docId, String text, List<Sentence> sents,
                     int first, int last, int index) {
        int cs = sents.get(first).start();
        int ce = sents.get(last).end();
        out.add(Chunk.of(docId, index, text.substring(cs, ce), cs, ce));
        return index + 1;
    }

    private List<Sentence> splitSentences(String text) {
        List<Sentence> out = new ArrayList<>();
        Matcher m = SENTENCE_SPLIT.matcher(text);
        int pos = 0;
        while (m.find()) {
            addSentence(text, pos, m.start(), out);
            pos = m.end();
        }
        addSentence(text, pos, text.length(), out);
        return out;
    }

    private void addSentence(String text, int s, int e, List<Sentence> out) {
        while (s < e && Character.isWhitespace(text.charAt(s))) s++;
        while (e > s && Character.isWhitespace(text.charAt(e - 1))) e--;
        if (s >= e) return;
        String piece = text.substring(s, e);
        out.add(new Sentence(s, e, countWords(piece), embedder.embed(piece)));
    }

    private static int countWords(String s) {
        int count = 0;
        boolean inWord = false;
        for (int i = 0; i < s.length(); i++) {
            boolean ws = Character.isWhitespace(s.charAt(i));
            if (!ws && !inWord) count++;
            inWord = !ws;
        }
        return count;
    }
}