"""Streaming pcap/pcapng reader and bidirectional session extraction.

Implements the "session" traffic unit used throughout the review (Section 2.2): a
bidirectional 5-tuple flow. Packets are streamed with dpkt; L3/L4 headers are parsed
with `struct` at fixed offsets (network byte order), which is much faster than building
dpkt objects and gives exact byte positions for masking.

To stay memory-safe on multi-GB captures, extraction runs in two passes per class:
  1. count kept packets per session key (cheap: one int per session);
  2. sample at most `max_sessions_per_class` eligible sessions (seeded), then collect
     packets only for the sampled sessions.

Simplification (documented): a session is identified by its 5-tuple only, without an
idle timeout, so port reuse within one capture merges into the same session.
"""

from __future__ import annotations

import struct
import zlib
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import dpkt
import numpy as np

LINKTYPE_NULL = 0
LINKTYPE_ETHERNET = 1
LINKTYPE_RAW = 101
LINKTYPE_LOOP = 108
LINKTYPE_LINUX_SLL = 113
LINKTYPE_IPV4 = 228

PROTO_TCP = 6
PROTO_UDP = 17
DNS_PORT = 53

PCAPNG_MAGIC = b"\x0a\x0d\x0d\x0a"

SessionKey = tuple[bytes, int, bytes, int, int]


@dataclass(frozen=True, slots=True)
class PacketInfo:
    """Parsed IPv4 TCP/UDP packet. `ip_bytes` starts at the IP header (no link layer)."""

    src: bytes
    dst: bytes
    sport: int
    dport: int
    proto: int
    ip_len: int
    payload_len: int
    ip_bytes: bytes


@dataclass
class Session:
    """Bidirectional session: per-packet timestamp, IP length, direction and raw bytes."""

    key: SessionKey
    proto: int
    initiator: tuple[bytes, int]
    ts: list[float] = field(default_factory=list)
    lens: list[int] = field(default_factory=list)
    dirs: list[int] = field(default_factory=list)
    raw: list[bytes] = field(default_factory=list)


def iter_pcap(path: str | Path) -> Iterator[tuple[float, bytes, int]]:
    """Stream (timestamp, frame bytes, linktype) from a pcap or pcapng file."""
    with open(path, "rb") as f:
        magic = f.read(4)
        f.seek(0)
        reader = dpkt.pcapng.Reader(f) if magic == PCAPNG_MAGIC else dpkt.pcap.Reader(f)
        linktype = reader.datalink()
        for ts, buf in reader:
            yield float(ts), bytes(buf), linktype


def l3_offset(buf: bytes, linktype: int) -> int | None:
    """Byte offset of the IPv4 header in a frame, or None if the frame is not IPv4."""
    if linktype == LINKTYPE_ETHERNET:
        off = 12
        while len(buf) >= off + 2:
            ethertype = int.from_bytes(buf[off : off + 2], "big")
            if ethertype in (0x8100, 0x88A8):  # VLAN tags
                off += 4
                continue
            return off + 2 if ethertype == 0x0800 else None
        return None
    if linktype in (LINKTYPE_RAW, LINKTYPE_IPV4):
        return 0
    if linktype == LINKTYPE_LINUX_SLL:
        return 16 if buf[14:16] == b"\x08\x00" else None
    if linktype in (LINKTYPE_NULL, LINKTYPE_LOOP):
        return 4
    return None


def parse_packet(buf: bytes, linktype: int) -> tuple[PacketInfo | None, str]:
    """Parse one frame. Returns (info, status); status is 'ok' or the drop reason."""
    off = l3_offset(buf, linktype)
    if off is None or len(buf) <= off or buf[off] >> 4 != 4:
        return None, "non_ipv4"
    if len(buf) < off + 20:
        return None, "malformed"
    ihl = (buf[off] & 0x0F) * 4
    total_len = int.from_bytes(buf[off + 2 : off + 4], "big")
    if ihl < 20 or total_len < ihl or len(buf) < off + ihl:
        return None, "malformed"
    if int.from_bytes(buf[off + 6 : off + 8], "big") & 0x1FFF:
        return None, "fragment"  # non-first fragment: no L4 header
    proto = buf[off + 9]
    if proto not in (PROTO_TCP, PROTO_UDP):
        return None, "non_tcp_udp"
    l4 = off + ihl
    end = min(len(buf), off + total_len)
    if proto == PROTO_TCP:
        if end < l4 + 20:
            return None, "malformed"
        thl = (buf[l4 + 12] >> 4) * 4
        if thl < 20:
            return None, "malformed"
        payload_len = total_len - ihl - thl
    else:
        if end < l4 + 8:
            return None, "malformed"
        payload_len = total_len - ihl - 8
    if payload_len < 0:
        return None, "malformed"
    sport, dport = struct.unpack_from("!HH", buf, l4)
    return (
        PacketInfo(
            src=buf[off + 12 : off + 16],
            dst=buf[off + 16 : off + 20],
            sport=sport,
            dport=dport,
            proto=proto,
            ip_len=total_len,
            payload_len=payload_len,
            ip_bytes=buf[off:end],
        ),
        "ok",
    )


def session_key(p: PacketInfo) -> SessionKey:
    """Direction-independent key: (min(ip:port), max(ip:port), proto)."""
    a, b = (p.src, p.sport), (p.dst, p.dport)
    lo, hi = (a, b) if a <= b else (b, a)
    return (lo[0], lo[1], hi[0], hi[1], p.proto)


def drop_reason(p: PacketInfo, pcfg: dict[str, Any]) -> str | None:
    """Packet-level filters from the spec; returns the reason or None to keep."""
    if pcfg.get("drop_dns", True) and DNS_PORT in (p.sport, p.dport):
        return "dns"
    if pcfg.get("drop_tcp_no_payload", True) and p.proto == PROTO_TCP and p.payload_len == 0:
        return "tcp_no_payload"
    return None


def _iter_parsed(
    path: Path, counters: Counter[str], limit: int | None
) -> Iterator[tuple[float, PacketInfo]]:
    for i, (ts, buf, linktype) in enumerate(iter_pcap(path)):
        if limit is not None and i >= limit:
            break
        counters["packets_read"] += 1
        p, status = parse_packet(buf, linktype)
        if p is None:
            counters[f"drop_{status}"] += 1
            continue
        yield ts, p


def count_sessions(
    path: Path, pcfg: dict[str, Any], counters: Counter[str], limit: int | None = None
) -> dict[SessionKey, int]:
    """Pass 1: number of kept packets per session key, in first-seen order."""
    counts: dict[SessionKey, int] = {}
    for _ts, p in _iter_parsed(path, counters, limit):
        key = session_key(p)
        if key not in counts:
            counts[key] = 0
        reason = drop_reason(p, pcfg)
        if reason is not None:
            counters[f"drop_{reason}"] += 1
            continue
        counts[key] += 1
    return counts


def collect_sessions(
    path: Path,
    pcfg: dict[str, Any],
    wanted: set[SessionKey],
    limit: int | None = None,
) -> dict[SessionKey, Session]:
    """Pass 2: gather packets of the `wanted` sessions (first N kept packets each).

    The initiator is the sender of the first packet seen for the key (SYN for TCP,
    first datagram for UDP), even if that packet itself is filtered out.
    """
    max_pkts = int(pcfg["max_packets_per_session"])
    store_bytes = int(pcfg["bytes_per_packet"])
    sessions: dict[SessionKey, Session] = {}
    scratch: Counter[str] = Counter()
    for ts, p in _iter_parsed(path, scratch, limit):
        key = session_key(p)
        if key not in wanted:
            continue
        s = sessions.get(key)
        if s is None:
            s = sessions[key] = Session(key=key, proto=p.proto, initiator=(p.src, p.sport))
        if len(s.ts) >= max_pkts or drop_reason(p, pcfg) is not None:
            continue
        s.ts.append(ts)
        s.lens.append(p.ip_len)
        s.dirs.append(1 if (p.src, p.sport) == s.initiator else -1)
        s.raw.append(p.ip_bytes[:store_bytes])
    return sessions


def extract_class_sessions(
    files: list[Path],
    label: str,
    pcfg: dict[str, Any],
    seed: int,
    limit: int | None = None,
) -> tuple[list[tuple[str, Session]], dict[str, int]]:
    """Extract up to `max_sessions_per_class` sampled sessions of one class.

    Returns ([(file name, session), ...], stats). Sampling is seeded per class so the
    result does not depend on worker scheduling.
    """
    counters: Counter[str] = Counter()
    min_pkts = int(pcfg["min_packets_per_session"])
    candidates: list[tuple[int, SessionKey]] = []
    for fi, path in enumerate(files):
        counts = count_sessions(path, pcfg, counters, limit)
        counters["sessions_total"] += len(counts)
        candidates.extend((fi, k) for k, n in counts.items() if n >= min_pkts)
    counters["sessions_eligible"] = len(candidates)

    max_sessions = pcfg.get("max_sessions_per_class")
    if max_sessions is not None and len(candidates) > int(max_sessions):
        rng = np.random.default_rng(seed + zlib.crc32(label.encode()))
        idx = np.sort(rng.choice(len(candidates), size=int(max_sessions), replace=False))
        candidates = [candidates[i] for i in idx]
    counters["sessions_sampled"] = len(candidates)

    out: list[tuple[str, Session]] = []
    for fi, path in enumerate(files):
        wanted = [k for f, k in candidates if f == fi]
        if not wanted:
            continue
        got = collect_sessions(path, pcfg, set(wanted), limit)
        out.extend((path.name, got[k]) for k in wanted)
    return out, dict(counters)
