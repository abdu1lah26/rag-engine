package io.github.abdullah.rag.analysis;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class PorterStemmerTest {

    private static void check(String[][] cases) {
        for (String[] c : cases) {
            assertEquals(c[1], PorterStemmer.stem(c[0]), "stem of '" + c[0] + "'");
        }
    }

    @Test
    void step1PluralsAndEdIng() {
        check(new String[][] {
                {"caresses", "caress"}, {"ponies", "poni"}, {"ties", "ti"}, {"caress", "caress"},
                {"cats", "cat"}, {"feed", "feed"}, {"agreed", "agre"}, {"plastered", "plaster"},
                {"bled", "bled"}, {"motoring", "motor"}, {"sing", "sing"},
                {"conflated", "conflat"}, {"troubled", "troubl"}, {"sized", "size"},
                {"hopping", "hop"}, {"falling", "fall"}, {"filing", "file"},
                {"running", "run"}, {"studies", "studi"}
        });
    }

    @Test
    void step1cYToI() {
        check(new String[][] {{"happy", "happi"}, {"sky", "sky"}});
    }

    @Test
    void laterSteps() {
        check(new String[][] {
                {"relational", "relat"}, {"conditional", "condit"}, {"organization", "organ"},
                {"generalizations", "gener"}, {"adjustment", "adjust"}, {"inhibition", "inhibit"},
                {"controlling", "control"}, {"probate", "probat"}, {"rate", "rate"}, {"cease", "ceas"}
        });
    }

    @Test
    void relatedWordsCollapseToOneTerm() {
        for (String w : new String[] {"mutation", "mutations", "mutated", "mutate"}) {
            assertEquals("mutat", PorterStemmer.stem(w), w);
        }
        for (String w : new String[] {"connect", "connected", "connecting", "connection", "connections"}) {
            assertEquals("connect", PorterStemmer.stem(w), w);
        }
    }

    @Test
    void shortWordsAndDigitsAreUntouched() {
        check(new String[][] {{"a", "a"}, {"as", "as"}, {"is", "is"}, {"2024", "2024"}, {"p53", "p53"}});
        assertEquals("", PorterStemmer.stem(""));
    }

    @Test
    void oddInputNeverThrowsOrBecomesEmpty() {
        for (String w : new String[] {"ies", "eed", "ing", "sss", "ses", "yyy", "ation", "ational", "ement", "ion"}) {
            String s = PorterStemmer.stem(w);
            assertTrue(!s.isEmpty(), "stem of '" + w + "' was empty");
        }
    }
}