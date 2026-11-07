package io.github.abdullah.rag.index;

/**
 * One entry in a posting list: "this term occurs tf times in the document at
 * internal index docIndex".
 * We use a compact int docIndex (position in the corpus) instead of the String
 * doc id, so postings stay small in memory and on disk.
 */
public record Posting(int docIndex, int tf) {
}