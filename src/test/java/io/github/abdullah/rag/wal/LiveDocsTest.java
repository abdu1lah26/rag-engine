package io.github.abdullah.rag.wal;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.abdullah.rag.wal.WalPayloads.AddOp;
import io.github.abdullah.rag.wal.WalRecord.Type;
import java.util.List;
import org.junit.jupiter.api.Test;

class LiveDocsTest {

    private static WalRecord add(long seq, String id, String text) {
        AddOp op = new AddOp(id, "t", text, "m", new float[] {1f});
        return new WalRecord(seq, Type.ADD, WalPayloads.encodeAdd(op));
    }

    private static WalRecord del(long seq, String id) {
        return new WalRecord(seq, Type.DELETE, WalPayloads.encodeDelete(id));
    }

    @Test
    void addMakesDocLiveAndHidesBaseCopy() {
        LiveDocs d = new LiveDocs(0);
        d.apply(add(1, "a", "v1"));
        assertEquals("v1", d.get("a").text());
        assertFalse(d.baseVisible("a"));
        assertTrue(d.baseVisible("other"));
    }

    @Test
    void addForExistingIdReplacesIt() {
        LiveDocs d = new LiveDocs(0);
        d.apply(add(1, "a", "v1"));
        d.apply(add(2, "a", "v2"));
        assertEquals("v2", d.get("a").text());
        assertEquals(1, d.liveDocs().size());
    }

    @Test
    void deleteRemovesLiveDocAndStaysHiddenFromBase() {
        LiveDocs d = new LiveDocs(0);
        d.apply(add(1, "a", "v1"));
        d.apply(del(2, "a"));
        assertNull(d.get("a"));
        assertFalse(d.baseVisible("a"));
    }

    @Test
    void deleteOfDocNotInDeltaStillHidesBaseCopy() {
        LiveDocs d = new LiveDocs(0);
        d.apply(del(1, "base-doc"));
        assertNull(d.get("base-doc"));
        assertFalse(d.baseVisible("base-doc"));
    }

    @Test
    void reAddAfterDeleteIsLiveAgain() {
        LiveDocs d = new LiveDocs(0);
        d.apply(add(1, "a", "v1"));
        d.apply(del(2, "a"));
        d.apply(add(3, "a", "v3"));
        assertEquals("v3", d.get("a").text());
    }

    @Test
    void checkpointChangesNothingButAdvancesSeq() {
        LiveDocs d = new LiveDocs(0);
        d.apply(add(1, "a", "v1"));
        d.apply(new WalRecord(2, Type.CHECKPOINT, new byte[0]));
        assertEquals(2, d.lastSeq());
        assertEquals(1, d.liveDocs().size());
    }

    @Test
    void outOfOrderOrDuplicateSeqIsRejected() {
        LiveDocs d = new LiveDocs(0);
        d.apply(add(5, "a", "v1"));
        assertThrows(IllegalStateException.class, () -> d.apply(add(5, "b", "x")));
        assertThrows(IllegalStateException.class, () -> d.apply(add(3, "b", "x")));
    }

    @Test
    void replayGivesSameStateAsLiveApplication() {
        List<WalRecord> log = List.of(
                add(1, "a", "v1"), add(2, "b", "v1"), del(3, "a"),
                add(4, "b", "v2"), add(5, "c", "v1"), del(6, "zzz"));
        LiveDocs live = new LiveDocs(0);
        for (WalRecord r : log) {
            live.apply(r);
        }
        LiveDocs replayed = new LiveDocs(0);
        for (WalRecord r : log) {
            replayed.apply(WalRecord.decode(r.encode(), 0, r.encodedSize()).record());
        }
        assertEquals(live.lastSeq(), replayed.lastSeq());
        assertEquals(live.touchedIds(), replayed.touchedIds());
        assertEquals(live.liveDocs().keySet(), replayed.liveDocs().keySet());
        assertEquals("v2", replayed.get("b").text());
    }
}