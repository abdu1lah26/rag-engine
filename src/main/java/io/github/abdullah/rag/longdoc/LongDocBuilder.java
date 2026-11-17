package io.github.abdullah.rag.longdoc;

import io.github.abdullah.rag.model.Document;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * Packs the original corpus into long documents: shuffle with a fixed seed, then glue
 * `perDoc` abstracts together. Every abstract lands in exactly one long doc, and we record
 * where it sits (char span), so chunks can be mapped back to the abstracts they cover.
 */
public final class LongDocBuilder {

    /** Where one original abstract lives inside a long document. [start, end) */
    public record Span(String abstractId, int start, int end) {}

    /** The long documents plus, per long doc id, the spans of the abstracts inside it. */
    public record LongDocs(List<Document> docs, Map<String, List<Span>> spans) {}

    private LongDocBuilder() {}

    public static LongDocs build(List<Document> corpus, int perDoc, String separator, long seed) {
        if (perDoc < 1) throw new IllegalArgumentException("perDoc must be >= 1");

        List<Document> shuffled = new ArrayList<>(corpus);
        Collections.shuffle(shuffled, new Random(seed));

        List<Document> docs = new ArrayList<>();
        Map<String, List<Span>> spans = new HashMap<>();

        for (int from = 0, n = 0; from < shuffled.size(); from += perDoc, n++) {
            int to = Math.min(from + perDoc, shuffled.size());
            String longId = "L" + n;
            StringBuilder sb = new StringBuilder();
            List<Span> list = new ArrayList<>();

            for (int i = from; i < to; i++) {
                if (i > from) sb.append(separator);
                Document d = shuffled.get(i);
                String body = (d.title() == null || d.title().isBlank())
                        ? d.text()
                        : d.title() + ". " + d.text();
                int start = sb.length();
                sb.append(body);
                list.add(new Span(d.id(), start, sb.length()));
            }
            docs.add(new Document(longId, "", sb.toString()));
            spans.put(longId, list);
        }
        return new LongDocs(docs, spans);
    }
}