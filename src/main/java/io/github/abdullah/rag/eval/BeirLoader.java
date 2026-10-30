package io.github.abdullah.rag.eval;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.github.abdullah.rag.model.Document;
import io.github.abdullah.rag.model.Qrels;
import io.github.abdullah.rag.model.Query;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Loads a BEIR-format dataset folder:
 * corpus.jsonl, queries.jsonl, qrels/<split>.tsv
 * Only queries that have judgments in the chosen split are kept.
 */
public final class BeirLoader {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    private BeirLoader() {
    }

    public static BeirDataset load(Path datasetDir, String split) throws IOException {
        List<Document> corpus = loadCorpus(datasetDir.resolve("corpus.jsonl"));
        Qrels qrels = loadQrels(datasetDir.resolve("qrels").resolve(split + ".tsv"));
        List<Query> queries = loadQueries(datasetDir.resolve("queries.jsonl"), qrels);
        String name = datasetDir.getFileName().toString() + "/" + split;
        return new BeirDataset(name, corpus, queries, qrels);
    }

    static List<Document> loadCorpus(Path file) throws IOException {
        List<Document> docs = new ArrayList<>();
        try (BufferedReader reader = Files.newBufferedReader(file, StandardCharsets.UTF_8)) {
            String line;
            while ((line = reader.readLine()) != null) {
                if (line.isBlank())
                    continue;
                JsonNode node = MAPPER.readTree(line);
                docs.add(new Document(
                        node.get("_id").asText(),
                        node.path("title").asText(""),
                        node.path("text").asText("")));
            }
        }
        return docs;
    }

    static Qrels loadQrels(Path file) throws IOException {
        Map<String, Map<String, Integer>> judgments = new HashMap<>();
        try (BufferedReader reader = Files.newBufferedReader(file, StandardCharsets.UTF_8)) {
            String line = reader.readLine(); // header: query-id corpus-id score
            while ((line = reader.readLine()) != null) {
                if (line.isBlank())
                    continue;
                String[] parts = line.split("\t");
                if (parts.length < 3) {
                    throw new IOException("Bad qrels line in " + file + ": " + line);
                }
                int score = Integer.parseInt(parts[2].trim());
                if (score <= 0)
                    continue; // 0 means not relevant, same as unjudged
                judgments.computeIfAbsent(parts[0].trim(), k -> new HashMap<>())
                        .put(parts[1].trim(), score);
            }
        }
        return new Qrels(judgments);
    }

    static List<Query> loadQueries(Path file, Qrels qrels) throws IOException {
        List<Query> queries = new ArrayList<>();
        try (BufferedReader reader = Files.newBufferedReader(file, StandardCharsets.UTF_8)) {
            String line;
            while ((line = reader.readLine()) != null) {
                if (line.isBlank())
                    continue;
                JsonNode node = MAPPER.readTree(line);
                String id = node.get("_id").asText();
                if (qrels.queryIds().contains(id)) {
                    queries.add(new Query(id, node.get("text").asText()));
                }
            }
        }
        return queries;
    }
}