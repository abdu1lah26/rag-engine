package io.github.abdullah.rag.baseline;

import io.github.abdullah.rag.model.Document;
import io.github.abdullah.rag.retrieval.Retriever;
import io.github.abdullah.rag.retrieval.ScoredDoc;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.en.EnglishAnalyzer;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.similarities.BM25Similarity;
import org.apache.lucene.store.ByteBuffersDirectory;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.ArrayList;
import java.util.List;

/**
 * BASELINE ONLY: Lucene's BM25 (k1=1.2, b=0.75), used as the external
 * reference for our own index. Nothing in the engine depends on this class.
 * The analyzer is a parameter so we can compare Lucene's default analysis with
 * English stemming.
 */
public final class LuceneBm25Retriever implements Retriever, AutoCloseable {

    private static final String ID = "id";
    private static final String TEXT = "text";

    private final String name;
    private final Analyzer analyzer;
    private ByteBuffersDirectory dir;
    private DirectoryReader reader;
    private IndexSearcher searcher;

    private LuceneBm25Retriever(String name, Analyzer analyzer) {
        this.name = name;
        this.analyzer = analyzer;
    }

    /** Lucene's default analysis: Unicode word split + lowercase, no stemming, no stopwords. */
    public static LuceneBm25Retriever standard() {
        return new LuceneBm25Retriever("lucene-standard", new StandardAnalyzer());
    }

    /** English analysis: possessive removal, lowercase, stopwords, Porter stemming. */
    public static LuceneBm25Retriever english() {
        return new LuceneBm25Retriever("lucene-english", new EnglishAnalyzer());
    }

    @Override
    public String name() {
        return name;
    }

    @Override
    public void index(List<Document> corpus) {
        try {
            close();
            dir = new ByteBuffersDirectory();
            IndexWriterConfig config = new IndexWriterConfig(analyzer)
                    .setSimilarity(new BM25Similarity(1.2f, 0.75f));
            try (IndexWriter writer = new IndexWriter(dir, config)) {
                for (Document doc : corpus) {
                    org.apache.lucene.document.Document luceneDoc = new org.apache.lucene.document.Document();
                    luceneDoc.add(new StringField(ID, doc.id(), Field.Store.YES));
                    luceneDoc.add(new TextField(TEXT, doc.indexableText(), Field.Store.NO));
                    writer.addDocument(luceneDoc);
                }
            }
            reader = DirectoryReader.open(dir);
            searcher = new IndexSearcher(reader);
            searcher.setSimilarity(new BM25Similarity(1.2f, 0.75f));
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    @Override
    public List<ScoredDoc> search(String queryText, int k) {
        if (k <= 0) {
            throw new IllegalArgumentException("k must be positive, got " + k);
        }
        try {
            // Build the query from analyzed tokens ourselves: no query-syntax parsing,
            // so characters like ":" or "-" in a question cannot break it.
            BooleanQuery.Builder builder = new BooleanQuery.Builder();
            try (TokenStream ts = analyzer.tokenStream(TEXT, queryText)) {
                CharTermAttribute termAttr = ts.addAttribute(CharTermAttribute.class);
                ts.reset();
                while (ts.incrementToken()) {
                    builder.add(new TermQuery(new Term(TEXT, termAttr.toString())), BooleanClause.Occur.SHOULD);
                }
                ts.end();
            }

            List<ScoredDoc> results = new ArrayList<>();
            for (ScoreDoc hit : searcher.search(builder.build(), k).scoreDocs) {
                String id = searcher.storedFields().document(hit.doc).get(ID);
                results.add(new ScoredDoc(id, hit.score));
            }
            return results;
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    @Override
    public void close() throws IOException {
        if (reader != null) {
            reader.close();
            reader = null;
        }
        if (dir != null) {
            dir.close();
            dir = null;
        }
    }
}