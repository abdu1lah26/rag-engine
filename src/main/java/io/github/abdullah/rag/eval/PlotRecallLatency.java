package io.github.abdullah.rag.eval;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/** Reads results/hnsw-sweep.csv and writes results/hnsw-recall-vs-latency.svg. */
public final class PlotRecallLatency {

    private static final int W = 760;
    private static final int H = 460;
    private static final int LEFT = 80;
    private static final int RIGHT = 30;
    private static final int TOP = 50;
    private static final int BOTTOM = 65;

    private record Row(String label, double recall, double p50) {
    }

    private PlotRecallLatency() {
    }

    public static void main(String[] args) throws IOException {
        Path csv = Path.of("results", "hnsw-sweep.csv");
        Path svg = Path.of("results", "hnsw-recall-vs-latency.svg");

        List<String> lines = Files.readAllLines(csv);
        List<Row> hnsw = new ArrayList<>();
        Row brute = null;
        for (int i = 1; i < lines.size(); i++) {
            String[] p = lines.get(i).split(",");
            Row r = new Row(p[0], Double.parseDouble(p[1]), Double.parseDouble(p[3]));
            if (p[0].equals("brute-force")) {
                brute = r;
            } else {
                hnsw.add(r);
            }
        }
        Files.writeString(svg, render(hnsw, brute));
        System.out.println("Wrote " + svg);
    }

    static String render(List<Row> hnsw, Row brute) {
        double minX = Double.MAX_VALUE;
        double maxX = 0;
        double minY = 1.0;
        for (Row r : hnsw) {
            minX = Math.min(minX, r.p50());
            maxX = Math.max(maxX, r.p50());
            minY = Math.min(minY, r.recall());
        }
        if (brute != null) {
            minX = Math.min(minX, brute.p50());
            maxX = Math.max(maxX, brute.p50());
        }
        double xLo = minX / 1.5;
        double xHi = maxX * 1.5;
        double yLo = Math.floor((minY - 0.01) * 50) / 50;
        double yHi = 1.005;

        StringBuilder s = new StringBuilder();
        s.append(String.format(Locale.ROOT,
                "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"%d\" height=\"%d\" "
                        + "viewBox=\"0 0 %d %d\" font-family=\"sans-serif\" font-size=\"12\">%n",
                W, H, W, H));
        s.append("<rect width=\"100%\" height=\"100%\" fill=\"white\"/>\n");
        s.append(String.format(Locale.ROOT,
                "<text x=\"%d\" y=\"28\" font-size=\"16\" font-weight=\"bold\" "
                        + "text-anchor=\"middle\">HNSW recall vs latency (SciFact, MiniLM, "
                        + "M=16, efC=200)</text>%n", W / 2));

        // y grid and ticks
        for (double t = Math.ceil(yLo * 50) / 50; t <= 1.0001; t += 0.02) {
            double y = py(t, yLo, yHi);
            s.append(String.format(Locale.ROOT,
                    "<line x1=\"%d\" y1=\"%.1f\" x2=\"%d\" y2=\"%.1f\" stroke=\"#ddd\"/>%n",
                    LEFT, y, W - RIGHT, y));
            s.append(String.format(Locale.ROOT,
                    "<text x=\"%d\" y=\"%.1f\" text-anchor=\"end\">%.2f</text>%n",
                    LEFT - 8, y + 4, t));
        }
        // x grid and ticks (log scale)
        double[] xTicks = {0.1, 0.2, 0.5, 1, 2, 5, 10};
        for (double t : xTicks) {
            if (t < xLo || t > xHi) {
                continue;
            }
            double x = px(t, xLo, xHi);
            s.append(String.format(Locale.ROOT,
                    "<line x1=\"%.1f\" y1=\"%d\" x2=\"%.1f\" y2=\"%d\" stroke=\"#ddd\"/>%n",
                    x, TOP, x, H - BOTTOM));
            s.append(String.format(Locale.ROOT,
                    "<text x=\"%.1f\" y=\"%d\" text-anchor=\"middle\">%s</text>%n",
                    x, H - BOTTOM + 18, tickLabel(t)));
        }
        // axes
        s.append(String.format(Locale.ROOT,
                "<line x1=\"%d\" y1=\"%d\" x2=\"%d\" y2=\"%d\" stroke=\"black\"/>%n",
                LEFT, TOP, LEFT, H - BOTTOM));
        s.append(String.format(Locale.ROOT,
                "<line x1=\"%d\" y1=\"%d\" x2=\"%d\" y2=\"%d\" stroke=\"black\"/>%n",
                LEFT, H - BOTTOM, W - RIGHT, H - BOTTOM));
        s.append(String.format(Locale.ROOT,
                "<text x=\"%d\" y=\"%d\" text-anchor=\"middle\">p50 query latency (ms, log "
                        + "scale)</text>%n", (LEFT + W - RIGHT) / 2, H - 15));
        s.append(String.format(Locale.ROOT,
                "<text transform=\"translate(22 %d) rotate(-90)\" text-anchor=\"middle\">"
                        + "ANN recall@10 vs exact search</text>%n", (TOP + H - BOTTOM) / 2));

        // brute-force reference line
        if (brute != null) {
            double x = px(brute.p50(), xLo, xHi);
            s.append(String.format(Locale.ROOT,
                    "<line x1=\"%.1f\" y1=\"%d\" x2=\"%.1f\" y2=\"%d\" stroke=\"#c0392b\" "
                            + "stroke-dasharray=\"6 4\"/>%n", x, TOP, x, H - BOTTOM));
            s.append(String.format(Locale.ROOT,
                    "<text x=\"%.1f\" y=\"%d\" fill=\"#c0392b\" text-anchor=\"middle\">"
                            + "brute force (%.2f ms)</text>%n", x, TOP - 8, brute.p50()));
        }

        // HNSW curve
        StringBuilder pts = new StringBuilder();
        for (Row r : hnsw) {
            pts.append(String.format(Locale.ROOT, "%.1f,%.1f ",
                    px(r.p50(), xLo, xHi), py(r.recall(), yLo, yHi)));
        }
        s.append("<polyline fill=\"none\" stroke=\"#2471a3\" stroke-width=\"2\" points=\"")
                .append(pts.toString().trim()).append("\"/>\n");
        for (Row r : hnsw) {
            double x = px(r.p50(), xLo, xHi);
            double y = py(r.recall(), yLo, yHi);
            s.append(String.format(Locale.ROOT,
                    "<circle cx=\"%.1f\" cy=\"%.1f\" r=\"4\" fill=\"#2471a3\"/>%n", x, y));
            s.append(String.format(Locale.ROOT,
                    "<text x=\"%.1f\" y=\"%.1f\">%s</text>%n",
                    x + 7, y + 16, r.label().replace("hnsw ", "")));
        }
        s.append("</svg>\n");
        return s.toString();
    }

    private static double px(double x, double lo, double hi) {
        double f = (Math.log10(x) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo));
        return LEFT + f * (W - LEFT - RIGHT);
    }

    private static double py(double y, double lo, double hi) {
        return TOP + (hi - y) / (hi - lo) * (H - TOP - BOTTOM);
    }

    private static String tickLabel(double t) {
        return t == Math.rint(t) ? String.valueOf((int) t) : String.valueOf(t);
    }
}