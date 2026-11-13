package io.github.abdullah.rag.index;

/**
 * On-disk layout of one index directory. Three binary files, big-endian
 * (DataOutputStream default), each = header + body + CRC32 footer.
 *
 * Header (every file):  int MAGIC, int VERSION
 * Footer (every file):  long crc32 of every byte before the footer
 *
 * docs.bin      body: UTF analyzerName, int numDocs, then per doc: UTF docId, int length
 * terms.bin     body: int numTerms, then per term (sorted A-Z):
 *                     UTF term, int docFreq, long postingsOffset
 * postings.bin  body: per term, at its postingsOffset, docFreq entries of
 *                     varint (docIndex - previousDocIndex), varint tf
 *                     (the first entry's previous docIndex is 0)
 *
 * Why 3 files: the term dictionary is small and always loaded; postings are
 * the bulk of the data and can later be memory-mapped and read lazily.
 */
public final class IndexFormat {

    private IndexFormat() {
    }

    public static final int MAGIC = 0x52414701; // "RAG" + 0x01
    public static final int VERSION = 2;

    public static final String DOCS_FILE = "docs.bin";
    public static final String TERMS_FILE = "terms.bin";
    public static final String POSTINGS_FILE = "postings.bin";
}