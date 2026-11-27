package io.github.abdullah.rag.wal;

import io.github.abdullah.rag.wal.WalPayloads.AddOp;
import java.nio.file.Path;

/** Child process for CrashRecoveryTest. Op i: delete doc-(i-3) if i%5==4, else put doc-i. */
public final class CrashChild {

    public static void main(String[] args) throws Exception {
        try (DurableStore s = DurableStore.open(Path.of(args[0]))) {
            for (int i = 0; i < 20000; i++) {
                if (i % 5 == 4) {
                    s.delete("doc-" + (i - 3));
                } else {
                    s.put(new AddOp("doc-" + i, "t", "text-" + i, "m", new float[] {i, 1f}));
                }
                System.out.println("ACK " + i + " ok"); // printed only after the op is on disk
                System.out.flush();
            }
        }
    }
}