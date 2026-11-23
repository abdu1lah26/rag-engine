package io.github.abdullah.rag.eval;

import io.github.abdullah.rag.embedding.OnnxCrossEncoder;
import java.nio.file.Path;
import java.util.Locale;

/** Throwaway sanity check: does the cross-encoder rank obvious pairs the right way? */
public final class RunCrossEncoderSanity {

    private RunCrossEncoderSanity() {
    }

    public static void main(String[] args) throws Exception {
        try (OnnxCrossEncoder ce = new OnnxCrossEncoder(
                Path.of("models", "ms-marco-MiniLM-L-6-v2"), 512)) {
            String q1 = "How many people live in Berlin?";
            show(ce, q1, "Berlin has a population of 3,520,031 registered inhabitants "
                    + "in an area of 891.82 square kilometers.");
            show(ce, q1, "New York City is famous for the Metropolitan Museum of Art.");

            String q2 = "Does aspirin reduce the risk of colorectal cancer?";
            show(ce, q2, "Long-term aspirin use was associated with a lower incidence "
                    + "of colorectal cancer in a large cohort study.");
            show(ce, q2, "The mitochondrial genome encodes 13 proteins involved in "
                    + "oxidative phosphorylation.");
        }
    }

    private static void show(OnnxCrossEncoder ce, String q, String d) {
        System.out.println(String.format(Locale.ROOT, "%8.3f  | %s  ->  %s",
                ce.score(q, d), q, d.length() > 60 ? d.substring(0, 60) + "..." : d));
    }
}