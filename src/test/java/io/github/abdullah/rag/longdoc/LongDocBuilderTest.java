package io.github.abdullah.rag.longdoc;

import static org.junit.jupiter.api.Assertions.*;

import io.github.abdullah.rag.longdoc.LongDocBuilder.LongDocs;
import io.github.abdullah.rag.longdoc.LongDocBuilder.Span;
import io.github.abdullah.rag.model.Document;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import org.junit.jupiter.api.Test;

class LongDocBuilderTest {

    private static List<Document> corpus(int n) {
        List<Document> list = new ArrayList<>();
        for (int i = 0; i < n; i++) list.add(new Document("a" + i, "Title" + i, "body text number " + i));
        return list;
    }

    @Test
    void everyAbstractAppearsExactlyOnce() {
        LongDocs ld = LongDocBuilder.build(corpus(25), 10, " ", 42);
        Set<String> seen = new HashSet<>();
        for (List<Span> list : ld.spans().values()) {
            for (Span s : list) assertTrue(seen.add(s.abstractId()), "duplicate " + s.abstractId());
        }
        assertEquals(25, seen.size());
    }

    @Test
    void groupsHaveExpectedSizes() {
        LongDocs ld = LongDocBuilder.build(corpus(25), 10, " ", 42);
        assertEquals(3, ld.docs().size()); // 10 + 10 + 5
        assertEquals(10, ld.spans().get("L0").size());
        assertEquals(5, ld.spans().get("L2").size());
    }

    @Test
    void spansPointToTheRightText() {
        List<Document> c = corpus(12);
        LongDocs ld = LongDocBuilder.build(c, 5, "\n\n", 7);
        for (Document d : ld.docs()) {
            for (Span s : ld.spans().get(d.id())) {
                String slice = d.text().substring(s.start(), s.end());
                Document orig = c.stream().filter(x -> x.id().equals(s.abstractId())).findFirst().orElseThrow();
                assertEquals(orig.title() + ". " + orig.text(), slice);
            }
        }
    }

    @Test
    void separatorSitsBetweenSpans() {
        LongDocs ld = LongDocBuilder.build(corpus(4), 4, "\n\n", 1);
        List<Span> spans = ld.spans().get("L0");
        for (int i = 1; i < spans.size(); i++) {
            assertEquals(spans.get(i - 1).end() + 2, spans.get(i).start());
        }
    }

    @Test
    void sameSeedSameLayoutDifferentSeedDifferentLayout() {
        List<Document> c = corpus(30);
        String a = LongDocBuilder.build(c, 10, " ", 42).docs().get(0).text();
        String b = LongDocBuilder.build(c, 10, " ", 42).docs().get(0).text();
        String other = LongDocBuilder.build(c, 10, " ", 43).docs().get(0).text();
        assertEquals(a, b);
        assertNotEquals(a, other);
    }

    @Test
    void inputCorpusIsNotMutated() {
        List<Document> c = corpus(10);
        String firstBefore = c.get(0).id();
        LongDocBuilder.build(c, 5, " ", 99);
        assertEquals(firstBefore, c.get(0).id());
    }

    @Test
    void invalidPerDocRejected() {
        assertThrows(IllegalArgumentException.class, () -> LongDocBuilder.build(corpus(3), 0, " ", 1));
    }
}