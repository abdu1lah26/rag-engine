package io.github.abdullah.rag.embedding;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import java.io.IOException;
import java.nio.file.Path;
import java.util.Map;

/**
 * Local cross-encoder: reads the query and one document together as
 * [CLS] query [SEP] document [SEP] and returns a single relevance logit.
 * Scores are only comparable between documents for the same query.
 * Built for cross-encoder/ms-marco-MiniLM-L-6-v2 (inputs input_ids,
 * attention_mask, token_type_ids; output "logits" of shape [batch, 1]).
 */
public final class OnnxCrossEncoder implements AutoCloseable {

    private final OrtEnvironment env;
    private final OrtSession session;
    private final WordPieceTokenizer tokenizer;
    private final String name;

    /**
     * @param modelDir  folder holding model.onnx and vocab.txt
     * @param maxLength max tokens for query + document together, including
     *                  special tokens
     */
    public OnnxCrossEncoder(Path modelDir, int maxLength) throws IOException, OrtException {
        this.env = OrtEnvironment.getEnvironment();
        this.session = env.createSession(
                modelDir.resolve("model.onnx").toString(), new OrtSession.SessionOptions());
        this.tokenizer = new WordPieceTokenizer(modelDir.resolve("vocab.txt"), maxLength);
        this.name = "onnx-" + modelDir.getFileName();
    }

    public String name() {
        return name;
    }

    /** Relevance logit of one (query, document) pair. Higher = more relevant. */
    public float score(String query, String document) {
        WordPieceTokenizer.Encoding enc = tokenizer.encodePair(query, document);
        try (OnnxTensor ids = OnnxTensor.createTensor(env, new long[][] {enc.inputIds()});
                OnnxTensor mask = OnnxTensor.createTensor(env, new long[][] {enc.attentionMask()});
                OnnxTensor types = OnnxTensor.createTensor(env, new long[][] {enc.tokenTypeIds()});
                OrtSession.Result out = session.run(Map.of(
                        "input_ids", ids,
                        "attention_mask", mask,
                        "token_type_ids", types))) {
            float[][] logits = (float[][]) out.get(0).getValue(); // [batch=1][1]
            return logits[0][0];
        } catch (OrtException e) {
            throw new IllegalStateException("ONNX inference failed", e);
        }
    }

    @Override
    public void close() throws OrtException {
        session.close();
    }
}