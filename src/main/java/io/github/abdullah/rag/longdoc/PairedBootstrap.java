package io.github.abdullah.rag.longdoc;

import java.util.Arrays;
import java.util.Random;

/**
 * Paired bootstrap over queries: resample the per-query score differences with replacement
 * and report the mean difference with a 95% percentile interval and a two-sided p-value.
 * "Paired" matters: both systems answer the same queries, so we resample query indices,
 * not scores independently.
 */
public final class PairedBootstrap {

    /** meanDiff = mean(a - b). If [lo, hi] excludes 0, the difference is unlikely to be noise. */
    public record Result(double meanDiff, double lo, double hi, double pValue) {}

    private PairedBootstrap() {}

    public static Result compare(double[] a, double[] b, int resamples, long seed) {
        if (a.length != b.length || a.length == 0) {
            throw new IllegalArgumentException("need equal-length, non-empty score arrays");
        }
        if (resamples < 100) throw new IllegalArgumentException("resamples must be >= 100");

        int n = a.length;
        double[] diff = new double[n];
        double mean = 0;
        for (int i = 0; i < n; i++) {
            diff[i] = a[i] - b[i];
            mean += diff[i];
        }
        mean /= n;

        Random rnd = new Random(seed);
        double[] means = new double[resamples];
        int atOrBelowZero = 0;
        int atOrAboveZero = 0;
        for (int r = 0; r < resamples; r++) {
            double sum = 0;
            for (int i = 0; i < n; i++) sum += diff[rnd.nextInt(n)];
            means[r] = sum / n;
            if (means[r] <= 0) atOrBelowZero++;
            if (means[r] >= 0) atOrAboveZero++;
        }
        Arrays.sort(means);

        double lo = means[(int) Math.floor(0.025 * (resamples - 1))];
        double hi = means[(int) Math.ceil(0.975 * (resamples - 1))];
        double p = 2.0 * Math.min(atOrBelowZero, atOrAboveZero) / resamples;
        return new Result(mean, lo, hi, Math.min(1.0, p));
    }
}