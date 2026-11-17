package io.github.abdullah.rag.chunking;

/**
 * One passage cut from a document.
 *
 * @param docId     id of the source document
 * @param chunkId   unique id of this chunk, formatted "docId#index"
 * @param index     position of this chunk within its document (0-based)
 * @param text      the chunk text
 * @param startChar offset of the first character in the original document
 *                  (inclusive)
 * @param endChar   offset after the last character in the original document
 *                  (exclusive)
 */
public record Chunk(String docId, String chunkId, int index, String text, int startChar, int endChar) {

    public Chunk {
        if (docId == null || docId.isEmpty())
            throw new IllegalArgumentException("docId required");
        if (text == null)
            throw new IllegalArgumentException("text required");
        if (startChar < 0 || endChar < startChar)
            throw new IllegalArgumentException("bad offsets");
    }

    public static Chunk of(String docId, int index, String text, int startChar, int endChar) {
        return new Chunk(docId, docId + "#" + index, index, text, startChar, endChar);
    }
}