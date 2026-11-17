package io.github.abdullah.rag.longdoc;

import static org.junit.jupiter.api.Assertions.*;

import io.github.abdullah.rag.longdoc.PairedBootstrap.Result;
import org.junit.jupiter.api.Test;

class PairedBootstrapTest {

    private static double[] filled(int n, double v) {
        double[] a = new double[n];
        java.util.Arrays.fill(a, v);
        return a;
    }

    @Test
    void identicalScoresGiveZeroDifferenceAndPOne() {
        Result r = PairedBootstrap.compare(filled(50, 0.6), filled(50, 0.6), 1000, 1);
        assertEquals(0.0, r.meanDiff(), 1e-12);
        assertEquals(0.0, r.lo(), 1e-12);
        assertEquals(0.0, r.hi(), 1e-12);
        assertEquals(1.0, r.pValue(), 1e-12);
    }

    @Test
    void constantAdvantageGivesPositiveIntervalAndTinyP() {
        Result r = PairedBootstrap.compare(filled(50, 0.7), filled(50, 0.6), 1000, 1);
        assertEquals(0.1, r.meanDiff(), 1e-9);
        assertEquals(0.1, r.lo(), 1e-9);
        assertEquals(0.1, r.hi(), 1e-9);
        assertEquals(0.0, r.pValue(), 1e-12);
    }

    @Test
    void symmetricNoiseAroundZeroIncludesZero() {
        // differences alternate +0.5 / -0.5: mean exactly 0, so the interval must straddle 0
        double[] a = new double[100];
        double[] b = new double[100];
        for (int i = 0; i < 100; i++) {
            a[i] = i % 2 == 0 ? 1.0 : 0.0;
            b[i] = i % 2 == 0 ? 0.5 : 0.5;
        }
        Result r = PairedBootstrap.compare(a, b, 5000, 7);
        assertEquals(0.0, r.meanDiff(), 1e-12);
        assertTrue(r.lo() < 0 && r.hi() > 0);
        assertTrue(r.pValue() > 0.5);
    }

    @Test
    void noisyButConsistentAdvantageExcludesZero() {
        // differences alternate 0.15 / 0.05: always positive, mean 0.10
        double[] a = new double[100];
        double[] b = new double[100];
        for (int i = 0; i < 100; i++) {
            a[i] = 0.5 + (i % 2 == 0 ? 0.15 : 0.05);
            b[i] = 0.5;
        }
        Result r = PairedBootstrap.compare(a, b, 5000, 7);
        assertEquals(0.10, r.meanDiff(), 1e-9);
        assertTrue(r.lo() > 0);
        assertTrue(r.pValue() < 0.05);
    }

    @Test
    void swappingArgumentsFlipsTheSign() {
        double[] a = filled(30, 0.8);
        double[] b = filled(30, 0.5);
        assertEquals(0.3, PairedBootstrap.compare(a, b, 500, 3).meanDiff(), 1e-9);
        assertEquals(-0.3, PairedBootstrap.compare(b, a, 500, 3).meanDiff(), 1e-9);
    }

    @Test
    void sameSeedIsReproducible() {
        double[] a = {0.9, 0.1, 0.5, 0.7, 0.3, 0.8, 0.2, 0.6};
        double[] b = {0.4, 0.2, 0.5, 0.1, 0.9, 0.3, 0.3, 0.2};
        Result x = PairedBootstrap.compare(a, b, 2000, 99);
        Result y = PairedBootstrap.compare(a, b, 2000, 99);
        assertEquals(x, y);
    }

    @Test
    void invalidInputsRejected() {
        assertThrows(IllegalArgumentException.class,
                () -> PairedBootstrap.compare(filled(3, 1), filled(4, 1), 1000, 1));
        assertThrows(IllegalArgumentException.class,
                () -> PairedBootstrap.compare(new double[0], new double[0], 1000, 1));
        assertThrows(IllegalArgumentException.class,
                () -> PairedBootstrap.compare(filled(3, 1), filled(3, 1), 10, 1));
    }
}