package io.github.abdullah.rag.embedding;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import io.github.abdullah.rag.chunking.Embedder;
import io.github.abdullah.rag.vector.VectorMath;
import java.io.IOException;
import java.nio.file.Path;
import java.util.Map;

/**
 * Local sentence embedder: WordPiece tokenizer + ONNX transformer + mean
 * pooling + L2 normalisation. Built for all-MiniLM-L6-v2 (384 dimensions).
 */
public final class OnnxEmbedder implements Embedder, AutoCloseable {

    private final OrtEnvironment env;
    private final OrtSession session;
    private final WordPieceTokenizer tokenizer;
    private final String name;
    private final int dimension;

    /** @param modelDir folder holding model.onnx and vocab.txt */
    public OnnxEmbedder(Path modelDir, int maxLength) throws IOException, OrtException {
        this.env = OrtEnvironment.getEnvironment();
        this.session = env.createSession(
                modelDir.resolve("model.onnx").toString(), new OrtSession.SessionOptions());
        this.tokenizer = new WordPieceTokenizer(modelDir.resolve("vocab.txt"), maxLength);
        this.name = "onnx-" + modelDir.getFileName();
        this.dimension = run(tokenizer.encode("dimension probe")).length;
    }

    @Override
    public String name() {
        return name;
    }

    @Override
    public int dimension() {
        return dimension;
    }

    @Override
    public float[] embed(String text) {
        return run(tokenizer.encode(text));
    }

    private float[] run(WordPieceTokenizer.Encoding enc) {
        try (OnnxTensor ids = OnnxTensor.createTensor(env, new long[][] {enc.inputIds()});
                OnnxTensor mask = OnnxTensor.createTensor(env, new long[][] {enc.attentionMask()});
                OnnxTensor types = OnnxTensor.createTensor(env, new long[][] {enc.tokenTypeIds()});
                OrtSession.Result out = session.run(Map.of(
                        "input_ids", ids,
                        "attention_mask", mask,
                        "token_type_ids", types))) {
            float[][][] hidden = (float[][][]) out.get(0).getValue(); // [batch=1][tokens][dim]
            return VectorMath.normalize(meanPool(hidden[0], enc.attentionMask()));
        } catch (OrtException e) {
            throw new IllegalStateException("ONNX inference failed", e);
        }
    }

    /** Average of token vectors, counting only positions where mask == 1. */
    static float[] meanPool(float[][] tokens, long[] mask) {
        int dim = tokens[0].length;
        float[] sum = new float[dim];
        int count = 0;
        for (int t = 0; t < tokens.length; t++) {
            if (mask[t] == 0) {
                continue;
            }
            for (int d = 0; d < dim; d++) {
                sum[d] += tokens[t][d];
            }
            count++;
        }
        for (int d = 0; d < dim; d++) {
            sum[d] /= Math.max(count, 1);
        }
        return sum;
    }

    @Override
    public void close() throws OrtException {
        session.close();
    }
}