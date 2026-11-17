package io.github.abdullah.rag.chunking;

import java.util.List;

/**
 * Splits one document into ordered chunks.
 * Implementations must be stateless and deterministic:
 * same input always gives the same chunks.
 */
public interface Chunker {

    /** Short name used in logs, DESIGN.md and benchmark tables, e.g. "fixed-200-o40". */
    String name();

    List<Chunk> chunk(String docId, String text);
}