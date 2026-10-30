package io.github.abdullah.rag.eval;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIf;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Checks the loader against the real SciFact files.
 * Skipped automatically if the dataset has not been downloaded yet
 * (run scripts\download-scifact.ps1 first).
 */
class BeirLoaderTest {

    private static final Path SCIFACT = Path.of("data", "scifact");

    static boolean datasetPresent() {
        return Files.exists(SCIFACT.resolve("corpus.jsonl"));
    }

    @Test
    @EnabledIf("datasetPresent")
    void loadsSciFactTestSplit() throws IOException {
        BeirDataset ds = BeirLoader.load(SCIFACT, "test");

        assertEquals(5183, ds.corpus().size(), "corpus size");
        assertEquals(300, ds.queries().size(), "test queries");
        assertEquals(300, ds.qrels().size(), "judged queries");

        // every kept query must have at least one relevant doc
        for (var q : ds.queries()) {
            assertFalse(ds.qrels().forQuery(q.id()).isEmpty(), "no judgments for query " + q.id());
        }
    }

    @Test
    @EnabledIf("datasetPresent")
    void firstQrelMatchesRawFile() throws IOException {
        BeirDataset ds = BeirLoader.load(SCIFACT, "test");

        // from the first data line of test.tsv: query 1 -> doc 31715818, score 1
        assertEquals(1, ds.qrels().score("1", "31715818"));
        assertEquals(0, ds.qrels().score("1", "does-not-exist"));
    }
}