package io.github.abdullah.rag.chunking;

import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Sliding window over whitespace-separated words.
 * Each chunk holds up to `size` words; consecutive chunks share `overlap` words.
 */
public final class FixedSizeChunker implements Chunker {

    private static final Pattern WORD = Pattern.compile("\\S+");

    private final int size;
    private final int overlap;

    public FixedSizeChunker(int size, int overlap) {
        if (size <= 0) throw new IllegalArgumentException("size must be > 0");
        if (overlap < 0 || overlap >= size) throw new IllegalArgumentException("overlap must be in [0, size)");
        this.size = size;
        this.overlap = overlap;
    }

    @Override
    public String name() {
        return "fixed-" + size + "-o" + overlap;
    }

    @Override
    public List<Chunk> chunk(String docId, String text) {
        List<int[]> words = new ArrayList<>();
        Matcher m = WORD.matcher(text);
        while (m.find()) words.add(new int[]{m.start(), m.end()});

        List<Chunk> out = new ArrayList<>();
        int n = words.size();
        int step = size - overlap;
        int index = 0;

        for (int start = 0; start < n; start += step) {
            int end = Math.min(start + size, n);
            int charStart = words.get(start)[0];
            int charEnd = words.get(end - 1)[1];
            out.add(Chunk.of(docId, index++, text.substring(charStart, charEnd), charStart, charEnd));
            if (end == n) break; // last window reached, avoid a redundant tail chunk
        }
        return out;
    }
}