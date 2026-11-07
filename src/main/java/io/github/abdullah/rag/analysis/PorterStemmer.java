package io.github.abdullah.rag.analysis;

/**
 * Porter stemming algorithm (M.F. Porter, 1980), the same algorithm Lucene's
 * PorterStemFilter uses. Strips common English suffixes so that
 * "mutations", "mutation" and "mutated" map to the same index term.
 * Input must be lowercase. Words of 1-2 letters are returned unchanged.
 */
public final class PorterStemmer {

    private final char[] b; // the word being stemmed
    private int k; // index of the last character of the current word
    private int j; // end of the stem, set by ends() when a suffix matches

    private PorterStemmer(String word) {
        b = new char[word.length() + 2]; // +2: a replacement can make the word one char longer
        word.getChars(0, word.length(), b, 0);
        k = word.length() - 1;
    }

    public static String stem(String word) {
        if (word.length() <= 2) {
            return word;
        }
        PorterStemmer s = new PorterStemmer(word);
        s.step1ab();
        s.step1c();
        s.step2();
        s.step3();
        s.step4();
        s.step5();
        return new String(s.b, 0, s.k + 1);
    }

    // ---------- helpers ----------

    /** True if b[i] is a consonant. 'y' is a consonant only if not after a consonant. */
    private boolean cons(int i) {
        switch (b[i]) {
            case 'a', 'e', 'i', 'o', 'u':
                return false;
            case 'y':
                return i == 0 || !cons(i - 1);
            default:
                return true;
        }
    }

    /** Measure: number of consonant-sequence -> vowel-sequence pairs in b[0..j]. */
    private int m() {
        int n = 0;
        int i = 0;
        while (true) { // skip leading consonants
            if (i > j) {
                return n;
            }
            if (!cons(i)) {
                break;
            }
            i++;
        }
        i++;
        while (true) {
            while (true) { // skip vowels
                if (i > j) {
                    return n;
                }
                if (cons(i)) {
                    break;
                }
                i++;
            }
            i++;
            n++;
            while (true) { // skip consonants
                if (i > j) {
                    return n;
                }
                if (!cons(i)) {
                    break;
                }
                i++;
            }
            i++;
        }
    }

    /** True if b[0..j] contains a vowel. */
    private boolean vowelInStem() {
        for (int i = 0; i <= j; i++) {
            if (!cons(i)) {
                return true;
            }
        }
        return false;
    }

    /** True if b[i-1..i] is a double consonant, e.g. "tt". */
    private boolean doubleC(int i) {
        return i >= 1 && b[i] == b[i - 1] && cons(i);
    }

    /** True if b[i-2..i] is consonant-vowel-consonant and the last is not w, x or y. */
    private boolean cvc(int i) {
        if (i < 2 || !cons(i) || cons(i - 1) || !cons(i - 2)) {
            return false;
        }
        char ch = b[i];
        return ch != 'w' && ch != 'x' && ch != 'y';
    }

    /** True if the word ends with s; if so, sets j to the end of the remaining stem. */
    private boolean ends(String s) {
        int len = s.length();
        int offset = k - len + 1;
        if (offset < 0) {
            return false;
        }
        for (int i = 0; i < len; i++) {
            if (b[offset + i] != s.charAt(i)) {
                return false;
            }
        }
        j = k - len;
        return true;
    }

    /** Replace everything after the stem (position j) with s. */
    private void setTo(String s) {
        int len = s.length();
        for (int i = 0; i < len; i++) {
            b[j + 1 + i] = s.charAt(i);
        }
        k = j + len;
    }

    /** setTo(s) only if the stem's measure is greater than 0. */
    private void replaceIfMeasured(String s) {
        if (m() > 0) {
            setTo(s);
        }
    }

    // ---------- step 1: plurals and -ed / -ing ----------

    private void step1ab() {
        if (b[k] == 's') {
            if (ends("sses")) {
                k -= 2; // caresses -> caress
            } else if (ends("ies")) {
                setTo("i"); // ponies -> poni
            } else if (b[k - 1] != 's') {
                k--; // cats -> cat
            }
        }
        if (ends("eed")) {
            if (m() > 0) {
                k--; // agreed -> agree
            }
        } else if ((ends("ed") || ends("ing")) && vowelInStem()) {
            k = j; // plastered -> plaster, motoring -> motor
            if (ends("at")) {
                setTo("ate"); // conflated -> conflate
            } else if (ends("bl")) {
                setTo("ble"); // troubled -> trouble
            } else if (ends("iz")) {
                setTo("ize"); // sized -> size
            } else if (doubleC(k)) {
                k--; // hopping -> hop
                char ch = b[k];
                if (ch == 'l' || ch == 's' || ch == 'z') {
                    k++; // but falling -> fall
                }
            } else if (m() == 1 && cvc(k)) {
                setTo("e"); // filing -> file
            }
        }
    }

    /** y -> i when the stem contains another vowel: happy -> happi. */
    private void step1c() {
        if (ends("y") && vowelInStem()) {
            b[k] = 'i';
        }
    }

        // ---------- step 2: double suffixes -> single suffix ----------

    // {suffix, replacement}. Longer suffixes come first, so "ational" wins over "tional".
    private static final String[][] STEP2 = {
            {"ational", "ate"}, {"tional", "tion"}, {"enci", "ence"}, {"anci", "ance"},
            {"izer", "ize"}, {"bli", "ble"}, {"alli", "al"}, {"entli", "ent"},
            {"eli", "e"}, {"ousli", "ous"}, {"ization", "ize"}, {"ation", "ate"},
            {"ator", "ate"}, {"alism", "al"}, {"iveness", "ive"}, {"fulness", "ful"},
            {"ousness", "ous"}, {"aliti", "al"}, {"iviti", "ive"}, {"biliti", "ble"},
            {"logi", "log"}
    };

    /** First matching suffix decides; it is replaced only if the stem has m() > 0. */
    private void step2() {
        for (String[] rule : STEP2) {
            if (ends(rule[0])) {
                replaceIfMeasured(rule[1]);
                return;
            }
        }
    }

    // ---------- step 3: -icate, -ative, -ful, -ness ... ----------

    private static final String[][] STEP3 = {
            {"icate", "ic"}, {"ative", ""}, {"alize", "al"}, {"iciti", "ic"},
            {"ical", "ic"}, {"ful", ""}, {"ness", ""}
    };

    private void step3() {
        for (String[] rule : STEP3) {
            if (ends(rule[0])) {
                replaceIfMeasured(rule[1]);
                return;
            }
        }
    }

        // ---------- step 4: strip suffixes from long stems ----------

    // Order matters where one suffix ends another: "ement" before "ment" before "ent".
    private static final String[] STEP4 = {
            "al", "ance", "ence", "er", "ic", "able", "ible", "ant", "ement", "ment", "ent",
            "ion", "ou", "ism", "ate", "iti", "ous", "ive", "ize"
    };

    /** First matching suffix decides; removed only if the stem has m() > 1. */
    private void step4() {
        for (String suffix : STEP4) {
            if (!ends(suffix)) {
                continue;
            }
            if (suffix.equals("ion") && !(j >= 0 && (b[j] == 's' || b[j] == 't'))) {
                continue; // -ion only comes off after s or t: adoption -> adopt, but not million
            }
            if (m() > 1) {
                k = j;
            }
            return;
        }
    }

    // ---------- step 5: tidy up a final -e and a double -l ----------

    private void step5() {
        j = k;
        if (b[k] == 'e') {
            int a = m();
            if (a > 1 || (a == 1 && !cvc(k - 1))) {
                k--; // probate -> probat, rate -> rate
            }
        }
        if (b[k] == 'l' && doubleC(k) && m() > 1) {
            k--; // controll -> control
        }
    }
}