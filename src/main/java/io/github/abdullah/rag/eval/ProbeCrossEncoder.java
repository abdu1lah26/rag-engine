package io.github.abdullah.rag.eval;

import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import java.nio.file.Path;

/** Throwaway: prints a model's input and output names and shapes. */
public final class ProbeCrossEncoder {

    private ProbeCrossEncoder() {
    }

    public static void main(String[] args) throws Exception {
        Path model = Path.of(args.length > 0 ? args[0]
                : "models/ms-marco-MiniLM-L-6-v2/model.onnx");
        OrtEnvironment env = OrtEnvironment.getEnvironment();
        try (OrtSession s = env.createSession(model.toString(), new OrtSession.SessionOptions())) {
            System.out.println("inputs:");
            s.getInputInfo().forEach((n, i) -> System.out.println("  " + n + " -> " + i.getInfo()));
            System.out.println("outputs:");
            s.getOutputInfo().forEach((n, i) -> System.out.println("  " + n + " -> " + i.getInfo()));
        }
    }
}