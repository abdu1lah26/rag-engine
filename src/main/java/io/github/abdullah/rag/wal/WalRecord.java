package io.github.abdullah.rag.wal;

import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.Objects;
import java.util.zip.CRC32;

/**
 * One entry in the write-ahead log.
 *
 * On-disk layout (big-endian):
 *   int  bodyLen            = 8 (seq) + 1 (type) + payload.length
 *   long seq
 *   byte type
 *   byte[] payload
 *   int  crc32              = CRC32 over seq + type + payload
 */
public final class WalRecord {

    public enum Type {
        ADD((byte) 1),
        DELETE((byte) 2),
        CHECKPOINT((byte) 3);

        final byte code;

        Type(byte code) {
            this.code = code;
        }

        static Type fromCode(byte code) {
            for (Type t : values()) {
                if (t.code == code) {
                    return t;
                }
            }
            throw new IllegalArgumentException("unknown record type code " + code);
        }
    }

    /** Sanity cap: a bodyLen above this is garbage, not a real record. */
    public static final int MAX_BODY_LEN = 64 * 1024 * 1024;
    /** seq (8 bytes) + type (1 byte). */
    static final int BODY_FIXED = 9;
    /** bodyLen field (4) + crc field (4). */
    static final int FRAMING = 8;

    private final long seq;
    private final Type type;
    private final byte[] payload;

    public WalRecord(long seq, Type type, byte[] payload) {
        if (seq < 0) {
            throw new IllegalArgumentException("seq must be >= 0");
        }
        Objects.requireNonNull(type, "type");
        Objects.requireNonNull(payload, "payload");
        if (BODY_FIXED + payload.length > MAX_BODY_LEN) {
            throw new IllegalArgumentException("payload too large: " + payload.length);
        }
        this.seq = seq;
        this.type = type;
        this.payload = payload.clone();
    }

    public long seq() {
        return seq;
    }

    public Type type() {
        return type;
    }

    public byte[] payload() {
        return payload.clone();
    }

    /** Total bytes this record occupies on disk. */
    public int encodedSize() {
        return FRAMING + BODY_FIXED + payload.length;
    }

    public byte[] encode() {
        int bodyLen = BODY_FIXED + payload.length;
        ByteBuffer buf = ByteBuffer.allocate(4 + bodyLen + 4);
        buf.putInt(bodyLen);
        buf.putLong(seq);
        buf.put(type.code);
        buf.put(payload);
        CRC32 crc = new CRC32();
        crc.update(buf.array(), 4, bodyLen);
        buf.putInt((int) crc.getValue());
        return buf.array();
    }

    public enum Status {
        /** A complete, valid record was read. */
        OK,
        /** Clean end of data: nothing left to read. */
        END,
        /** Data ends in the middle of a record (torn write). */
        TRUNCATED,
        /** Bytes are present but wrong: bad length, bad type, or CRC mismatch. */
        CORRUPT
    }

    public record DecodeResult(Status status, WalRecord record, int size) {
        static DecodeResult of(Status status) {
            return new DecodeResult(status, null, 0);
        }
    }

    /** Decodes one record from data[offset, limit). Never throws on bad bytes. */
    public static DecodeResult decode(byte[] data, int offset, int limit) {
        int avail = limit - offset;
        if (avail == 0) {
            return DecodeResult.of(Status.END);
        }
        if (avail < 4) {
            return DecodeResult.of(Status.TRUNCATED);
        }
        ByteBuffer buf = ByteBuffer.wrap(data, offset, avail);
        int bodyLen = buf.getInt();
        if (bodyLen < BODY_FIXED || bodyLen > MAX_BODY_LEN) {
            return DecodeResult.of(Status.CORRUPT);
        }
        int total = 4 + bodyLen + 4;
        if (avail < total) {
            return DecodeResult.of(Status.TRUNCATED);
        }
        CRC32 crc = new CRC32();
        crc.update(data, offset + 4, bodyLen);
        int stored = buf.getInt(offset + 4 + bodyLen);
        if ((int) crc.getValue() != stored) {
            return DecodeResult.of(Status.CORRUPT);
        }
        try {
            long seq = buf.getLong();
            Type type = Type.fromCode(buf.get());
            byte[] payload = new byte[bodyLen - BODY_FIXED];
            buf.get(payload);
            return new DecodeResult(Status.OK, new WalRecord(seq, type, payload), total);
        } catch (IllegalArgumentException e) {
            return DecodeResult.of(Status.CORRUPT);
        }
    }

    @Override
    public boolean equals(Object o) {
        if (!(o instanceof WalRecord other)) {
            return false;
        }
        return seq == other.seq && type == other.type && Arrays.equals(payload, other.payload);
    }

    @Override
    public int hashCode() {
        return 31 * Objects.hash(seq, type) + Arrays.hashCode(payload);
    }

    @Override
    public String toString() {
        return "WalRecord[seq=" + seq + ", type=" + type + ", payload=" + payload.length + "B]";
    }
}