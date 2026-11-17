package io.github.abdullah.rag.chunking;

import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Splits on the largest natural boundary that fits: paragraph, then line, then sentence, then word.
 * Pieces are packed greedily into chunks of up to `size` words.
 * Consecutive chunks share trailing pieces totalling at most `overlap` words.
 */
public final class RecursiveChunker implements Chunker {

    private static final Pattern[] LEVELS = {
        Pattern.compile("\\n\\s*\\n"),             // paragraph
        Pattern.compile("\\n"),                    // line
        Pattern.compile("(?<=[.!?])\\s+"),         // sentence
        Pattern.compile("\\s+")                    // word
    };

    private record Piece(int start, int end, int words) {}

    private final int size;
    private final int overlap;

    public RecursiveChunker(int size, int overlap) {
        if (size <= 0) throw new IllegalArgumentException("size must be > 0");
        if (overlap < 0 || overlap >= size) throw new IllegalArgumentException("overlap must be in [0, size)");
        this.size = size;
        this.overlap = overlap;
    }

    @Override
    public String name() {
        return "recursive-" + size + "-o" + overlap;
    }

    @Override
    public List<Chunk> chunk(String docId, String text) {
        List<Piece> pieces = new ArrayList<>();
        collect(text, 0, text.length(), 0, pieces);

        List<Chunk> out = new ArrayList<>();
        int n = pieces.size();
        int i = 0;
        int index = 0;

        while (i < n) {
            int j = i;
            int wc = 0;
            while (j < n && (j == i || wc + pieces.get(j).words() <= size)) {
                wc += pieces.get(j).words();
                j++;
            }
            int cs = pieces.get(i).start();
            int ce = pieces.get(j - 1).end();
            out.add(Chunk.of(docId, index++, text.substring(cs, ce), cs, ce));
            if (j >= n) break;

            // back up over trailing pieces to create overlap, but always move forward
            int next = j;
            int ov = 0;
            while (next - 1 > i && ov + pieces.get(next - 1).words() <= overlap) {
                ov += pieces.get(next - 1).words();
                next--;
            }
            i = next;
        }
        return out;
    }

    /** Split [from,to) at this level; keep spans that fit, recurse into spans that do not. */
    private void collect(String text, int from, int to, int level, List<Piece> out) {
        Matcher m = LEVELS[level].matcher(text);
        m.region(from, to);
        m.useTransparentBounds(true);

        int pos = from;
        while (m.find()) {
            add(text, pos, m.start(), level, out);
            pos = m.end();
        }
        add(text, pos, to, level, out);
    }

    private void add(String text, int s, int e, int level, List<Piece> out) {
        while (s < e && Character.isWhitespace(text.charAt(s))) s++;
        while (e > s && Character.isWhitespace(text.charAt(e - 1))) e--;
        if (s >= e) return;

        int wc = countWords(text, s, e);
        if (wc <= size) {
            out.add(new Piece(s, e, wc));
        } else {
            collect(text, s, e, level + 1, out); // never exceeds last level: a single word has wc = 1 <= size
        }
    }

    private static int countWords(String text, int s, int e) {
        int count = 0;
        boolean inWord = false;
        for (int k = s; k < e; k++) {
            boolean ws = Character.isWhitespace(text.charAt(k));
            if (!ws && !inWord) count++;
            inWord = !ws;
        }
        return count;
    }
}