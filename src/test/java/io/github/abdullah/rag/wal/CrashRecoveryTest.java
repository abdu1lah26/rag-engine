package io.github.abdullah.rag.wal;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.abdullah.rag.wal.WalPayloads.AddOp;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.file.Path;
import java.util.Map;
import java.util.Random;
import java.util.TreeMap;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class CrashRecoveryTest {

    @TempDir
    Path dir;

    @Test
    void killedWriterLosesNoAcknowledgedOperation() throws Exception {
        Random rnd = new Random(42);
        String java = Path.of(System.getProperty("java.home"), "bin", "java").toString();

        for (int round = 0; round < 6; round++) {
            Path d = dir.resolve("round" + round);
            int killAfter = 20 + rnd.nextInt(150);
            Process p = new ProcessBuilder(java, "-cp", System.getProperty("java.class.path"),
                    CrashChild.class.getName(), d.toString())
                    .redirectError(ProcessBuilder.Redirect.INHERIT)
                    .start();

            int acks = 0;
            boolean killed = false;
            try (BufferedReader r = new BufferedReader(new InputStreamReader(p.getInputStream()))) {
                String line;
                while ((line = r.readLine()) != null) {
                    if (!line.startsWith("ACK ") || !line.endsWith(" ok")) {
                        continue; // partial line from the kill
                    }
                    int idx = Integer.parseInt(line.substring(4, line.length() - 3));
                    assertEquals(acks, idx, "acks must arrive in order");
                    acks++;
                    if (!killed && acks == killAfter) {
                        p.destroyForcibly(); // hard kill, no shutdown hooks, no flush
                        killed = true;
                    }
                }
            }
            p.waitFor();
            assertTrue(killed, "child exited before the kill point in round " + round);

            try (DurableStore s = DurableStore.open(d)) {
                long n = s.state().lastSeq(); // op i has seq i+1, so lastSeq = number of durable ops
                assertTrue(n >= acks, "round " + round + ": acknowledged op lost (acked " + acks + ", recovered " + n + ")");
                assertTrue(n <= acks + 1, "round " + round + ": more ops than could have been issued");

                Map<String, String> expected = new TreeMap<>();
                for (int i = 0; i < n; i++) {
                    if (i % 5 == 4) {
                        expected.remove("doc-" + (i - 3));
                    } else {
                        expected.put("doc-" + i, "text-" + i);
                    }
                }
                Map<String, String> actual = new TreeMap<>();
                for (AddOp op : s.state().liveDocs().values()) {
                    actual.put(op.docId(), op.text());
                }
                assertEquals(expected, actual, "round " + round + ": recovered state differs from the op history");

                System.out.printf("round %d: acked=%d recovered=%d truncatedBytes=%d%n",
                        round, acks, n, s.recovery().truncatedBytes());
            }
        }
    }
}