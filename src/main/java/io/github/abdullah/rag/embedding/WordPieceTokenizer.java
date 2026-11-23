package io.github.abdullah.rag.embedding;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.text.Normalizer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * BERT-style uncased WordPiece tokenizer, matching what all-MiniLM-L6-v2 was
 * trained with. Not handled: CJK character splitting (irrelevant for English
 * benchmarks).
 */
public final class WordPieceTokenizer {

    /** Model inputs. ONNX BERT models expect int64 tensors. */
    public record Encoding(long[] inputIds, long[] attentionMask, long[] tokenTypeIds) {
    }

    private static final int MAX_WORD_CHARS = 100;

    private final Map<String, Integer> vocab = new HashMap<>();
    private final int clsId;
    private final int sepId;
    private final int unkId;
    private final int maxLength;

    /** @param maxLength max tokens including [CLS] and [SEP]; MiniLM uses 256 */
    public WordPieceTokenizer(Path vocabFile, int maxLength) throws IOException {
        if (maxLength < 3) {
            throw new IllegalArgumentException("maxLength must be >= 3");
        }
        this.maxLength = maxLength;
        List<String> lines = Files.readAllLines(vocabFile, StandardCharsets.UTF_8);
        for (int i = 0; i < lines.size(); i++) {
            vocab.put(lines.get(i), i);
        }
        this.clsId = requireId("[CLS]");
        this.sepId = requireId("[SEP]");
        this.unkId = requireId("[UNK]");
    }

    private int requireId(String token) {
        Integer id = vocab.get(token);
        if (id == null) {
            throw new IllegalStateException("vocab is missing " + token);
        }
        return id;
    }

    public Encoding encode(String text) {
        List<Integer> ids = new ArrayList<>();
        for (String word : basicTokenize(text)) {
            wordPiece(word, ids);
            if (ids.size() >= maxLength - 2) {
                break;
            }
        }
        int n = Math.min(ids.size(), maxLength - 2);
        long[] input = new long[n + 2];
        input[0] = clsId;
        for (int i = 0; i < n; i++) {
            input[i + 1] = ids.get(i);
        }
        input[n + 1] = sepId;

        long[] attention = new long[n + 2];
        java.util.Arrays.fill(attention, 1L);
        long[] types = new long[n + 2]; // all zeros: single-sentence input
        return new Encoding(input, attention, types);
    }

    /**
     * Cross-encoder input: [CLS] first [SEP] second [SEP]. Segment ids are 0 for
     * [CLS], the first text and its [SEP], and 1 for the second text and the final
     * [SEP]. If the pair is too long, the first text (the query) is capped at half
     * the token budget and the second text (the document) is cut to what is left.
     */
    public Encoding encodePair(String first, String second) {
        int budget = maxLength - 3; // [CLS] + 2 x [SEP]
        List<Integer> a = pieces(first, budget / 2);
        List<Integer> b = pieces(second, budget - a.size());

        int total = a.size() + b.size() + 3;
        long[] input = new long[total];
        long[] types = new long[total];
        int p = 0;
        input[p++] = clsId;
        for (int id : a) {
            input[p++] = id;
        }
        input[p++] = sepId;
        int secondStart = p;
        for (int id : b) {
            input[p++] = id;
        }
        input[p] = sepId;
        for (int i = secondStart; i < total; i++) {
            types[i] = 1L;
        }
        long[] attention = new long[total];
        java.util.Arrays.fill(attention, 1L);
        return new Encoding(input, attention, types);
    }

    /** WordPiece ids for a text, cut to at most limit pieces. */
    private List<Integer> pieces(String text, int limit) {
        List<Integer> ids = new ArrayList<>();
        for (String word : basicTokenize(text)) {
            wordPiece(word, ids);
            if (ids.size() >= limit) {
                break;
            }
        }
        return ids.size() > limit ? new ArrayList<>(ids.subList(0, limit)) : ids;
    }

    /** Clean, lowercase, strip accents, split on whitespace and punctuation. */
    List<String> basicTokenize(String text) {
        StringBuilder cleaned = new StringBuilder();
        for (int i = 0; i < text.length(); i++) {
            char c = text.charAt(i);
            if (c == 0 || c == 0xFFFD) {
                continue;
            }
            if (Character.isWhitespace(c) || Character.isSpaceChar(c)) {
                cleaned.append(' ');
            } else if (!isControl(c)) {
                cleaned.append(c);
            }
        }

        String decomposed = Normalizer.normalize(
                cleaned.toString().toLowerCase(Locale.ROOT), Normalizer.Form.NFD);

        List<String> tokens = new ArrayList<>();
        StringBuilder cur = new StringBuilder();
        for (int i = 0; i < decomposed.length(); i++) {
            char c = decomposed.charAt(i);
            if (Character.getType(c) == Character.NON_SPACING_MARK) {
                continue; // accent marks left over after NFD
            }
            if (c == ' ') {
                flush(cur, tokens);
            } else if (isPunctuation(c)) {
                flush(cur, tokens);
                tokens.add(String.valueOf(c));
            } else {
                cur.append(c);
            }
        }
        flush(cur, tokens);
        return tokens;
    }

    private static void flush(StringBuilder cur, List<String> tokens) {
        if (cur.length() > 0) {
            tokens.add(cur.toString());
            cur.setLength(0);
        }
    }

    private static boolean isControl(char c) {
        int t = Character.getType(c);
        return t == Character.CONTROL || t == Character.FORMAT;
    }

    private static boolean isPunctuation(char c) {
        if ((c >= 33 && c <= 47) || (c >= 58 && c <= 64)
                || (c >= 91 && c <= 96) || (c >= 123 && c <= 126)) {
            return true;
        }
        int t = Character.getType(c);
        return t == Character.CONNECTOR_PUNCTUATION
                || t == Character.DASH_PUNCTUATION
                || t == Character.START_PUNCTUATION
                || t == Character.END_PUNCTUATION
                || t == Character.INITIAL_QUOTE_PUNCTUATION
                || t == Character.FINAL_QUOTE_PUNCTUATION
                || t == Character.OTHER_PUNCTUATION;
    }

    /** Greedy longest-match-first split of one word into vocab pieces. */
    private void wordPiece(String word, List<Integer> out) {
        if (word.length() > MAX_WORD_CHARS) {
            out.add(unkId);
            return;
        }
        List<Integer> pieces = new ArrayList<>();
        int start = 0;
        while (start < word.length()) {
            int end = word.length();
            Integer found = null;
            while (start < end) {
                String sub = word.substring(start, end);
                if (start > 0) {
                    sub = "##" + sub;
                }
                Integer id = vocab.get(sub);
                if (id != null) {
                    found = id;
                    break;
                }
                end--;
            }
            if (found == null) {
                out.add(unkId); // one unmatched piece makes the whole word [UNK]
                return;
            }
            pieces.add(found);
            start = end;
        }
        out.addAll(pieces);
    }
}