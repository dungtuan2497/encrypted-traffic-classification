"""Phase 1 tests: session extraction, directions, masking positions, padding, de-dup, split."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from etc.config import load_config
from etc.data.build_dataset import build, deduplicate, stratified_split
from etc.data.bytes_repr import flatten_packets, mask_packet, packets_to_array, session_bytes
from etc.data.features import FEATURE_NAMES, size_sequence, statistical_features
from etc.data.pcap_reader import collect_sessions, extract_class_sessions, iter_pcap, parse_packet
from tests.pcap_factory import arp_frame, ip, tcp_session, udp_frame, write_pcap

CLIENT, SERVER = "10.0.0.1", "10.0.0.2"


@pytest.fixture
def pcfg() -> dict:
    return dict(load_config()["preprocess"])


@pytest.fixture
def simple_pcap(tmp_path: Path) -> Path:
    frames = tcp_session(CLIENT, 40000, SERVER, 443, 1.0, [b"a" * 100, b"b" * 200, b"c" * 50])
    frames += [(2.0 + i, udp_frame(CLIENT, 5000, "8.8.8.8", 53, b"q" * 30)) for i in range(3)]  # DNS
    frames += [(3.0 + i, udp_frame(CLIENT, 6000, "1.2.3.4", 9999, b"u" * 20)) for i in range(2)]  # too short
    frames.append((4.0, arp_frame()))
    frames.sort(key=lambda x: x[0])
    return write_pcap(tmp_path / "simple.pcap", frames)


def test_bidirectional_session_and_directions(simple_pcap: Path, pcfg: dict) -> None:
    pcfg["min_packets_per_session"] = 3
    sessions, stats = extract_class_sessions([simple_pcap], "X", pcfg, seed=42)
    assert len(sessions) == 1, "only the TCP session has >= 3 kept packets"
    _, s = sessions[0]
    # SYN, SYN-ACK and trailing pure ACK are dropped (no payload); both directions merged.
    assert s.dirs == [1, -1, 1]
    assert s.lens == [20 + 20 + 100, 20 + 20 + 200, 20 + 20 + 50]
    assert s.initiator == (ip(CLIENT), 40000)
    assert stats["drop_dns"] == 3
    assert stats["drop_non_ipv4"] == 1
    assert stats["drop_tcp_no_payload"] == 3
    assert stats["sessions_total"] == 3  # TCP + DNS + short UDP
    pcfg["min_packets_per_session"] = 1
    sessions, _ = extract_class_sessions([simple_pcap], "X", pcfg, seed=42)
    assert len(sessions) == 2, "with min 1 the 2-packet UDP session is kept; DNS never is"


def test_udp_initiator_is_first_sender(tmp_path: Path, pcfg: dict) -> None:
    frames = [
        (1.0, udp_frame("10.0.0.9", 7000, "10.0.0.1", 8000, b"x" * 10)),
        (1.1, udp_frame("10.0.0.1", 8000, "10.0.0.9", 7000, b"y" * 10)),
        (1.2, udp_frame("10.0.0.9", 7000, "10.0.0.1", 8000, b"z" * 10)),
    ]
    path = write_pcap(tmp_path / "udp.pcap", frames)
    [(_, s)] = extract_class_sessions([path], "U", pcfg, seed=0)[0]
    assert s.initiator == (ip("10.0.0.9"), 7000)
    assert s.dirs == [1, -1, 1]


def test_max_packets_per_session(tmp_path: Path, pcfg: dict) -> None:
    payloads = [bytes([i]) * 10 for i in range(10)]
    path = write_pcap(tmp_path / "long.pcap", tcp_session(CLIENT, 1, SERVER, 2, 0.0, payloads))
    pcfg["max_packets_per_session"] = 4
    [(_, s)] = extract_class_sessions([path], "L", pcfg, seed=0)[0]
    assert len(s.ts) == 4


def test_mask_positions(simple_pcap: Path) -> None:
    frames = list(iter_pcap(simple_pcap))
    p, status = parse_packet(frames[2][1], frames[2][2])  # first data packet
    assert status == "ok" and p is not None
    raw = p.ip_bytes
    assert raw[12:16] == ip(CLIENT) and raw[16:20] == ip(SERVER)

    m = mask_packet(raw, mask_ip=True, mask_port=False)
    assert len(m) == len(raw)
    assert m[12:20] == bytes(8)            # src/dst IP
    assert m[10:12] == bytes(2)            # IP checksum
    assert m[36:38] == bytes(2)            # TCP checksum (IHL 20 + offset 16)
    assert m[20:24] == raw[20:24]          # ports untouched
    assert m[24:36] == raw[24:36] and m[40:] == raw[40:]  # rest unchanged

    mp = mask_packet(raw, mask_ip=True, mask_port=True)
    assert mp[20:24] == bytes(4)           # src/dst port
    assert mask_packet(raw, False, False) == raw


def test_mask_udp_checksum_and_truncated() -> None:
    from etc.data.pcap_reader import LINKTYPE_ETHERNET

    p, _ = parse_packet(udp_frame(CLIENT, 1234, SERVER, 4321, b"hello"), LINKTYPE_ETHERNET)
    m = mask_packet(p.ip_bytes, True, True)
    assert m[26:28] == bytes(2)            # UDP checksum (IHL 20 + offset 6)
    # Truncated packet: masking only touches the bytes that exist.
    t = mask_packet(p.ip_bytes[:22], True, True)
    assert len(t) == 22 and t[12:22] == bytes(10) and t[:10] == p.ip_bytes[:10]


def test_padding_and_flatten() -> None:
    raw = [bytes([1]) * 100, bytes([2]) * 10, bytes([3]) * 64]
    arr, lens = packets_to_array(raw, max_packets=32, bytes_per_packet=64)
    assert arr.shape == (32, 64) and lens.tolist()[:4] == [64, 10, 64, 0]
    flat = flatten_packets(arr, lens, max_bytes=784)
    assert flat.shape == (784,) and flat.dtype == np.uint8
    assert (flat[:64] == 1).all() and (flat[64:74] == 2).all() and (flat[74:138] == 3).all()
    assert (flat[138:] == 0).all()
    short = flatten_packets(arr, lens, max_bytes=100)
    assert short.shape == (100,)
    f, pa, pl = session_bytes(raw, True, False, 32, 64, 512)
    assert f.shape == (512,) and pa.shape == (32, 64) and pl.shape == (32,)


def test_features_and_sequence() -> None:
    ts = np.array([0.0, 0.1, 0.3])
    lens = np.array([100, 200, 50])
    dirs = np.array([1, -1, 1])
    f = statistical_features(ts, lens, dirs, proto=6)
    assert f.shape == (len(FEATURE_NAMES),)
    named = dict(zip(FEATURE_NAMES, f))
    assert named["all_n_pkts"] == 3 and named["fwd_n_pkts"] == 2 and named["bwd_n_pkts"] == 1
    assert named["all_n_bytes"] == 350 and named["ratio_bytes_fwd"] == pytest.approx(150 / 350)
    assert named["all_duration"] == pytest.approx(0.3) and named["is_tcp"] == 1.0
    seq = size_sequence(ts, lens, dirs, seq_len=32)
    assert seq.shape == (32, 2)
    assert seq[:3, 0].tolist() == [100, -200, 50] and (seq[3:] == 0).all()
    assert seq[0, 1] == 0 and seq[1, 1] == pytest.approx(np.log1p(100.0), rel=1e-5)


def test_deduplicate_and_split() -> None:
    keep, st = deduplicate(["a", "b", "a", "c", "c"], ["x", "x", "x", "x", "y"])
    assert keep.tolist() == [True, True, False, False, False]
    assert st == {"duplicates_removed": 1, "label_conflicts_removed": 2}

    labels = np.array(["a"] * 50 + ["b"] * 30)
    sp = stratified_split(labels, {"train": 0.8, "val": 0.1, "test": 0.1}, seed=42)
    all_idx = sp["train"] + sp["val"] + sp["test"]
    assert sorted(all_idx) == list(range(80)) and len(set(all_idx)) == 80
    assert sum(labels[sp["test"]] == "a") == 5 and sum(labels[sp["test"]] == "b") == 3
    assert sp == stratified_split(labels, {"train": 0.8, "val": 0.1, "test": 0.1}, seed=42)


def test_build_end_to_end(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    (raw / "C").mkdir(parents=True)  # tiny class -> must be excluded
    write_pcap(raw / "C" / "C.pcap", tcp_session(CLIENT, 1, SERVER, 2, 0.0, [b"q" * 9]))
    for c, (cls, base) in enumerate([("A", 0x41), ("B", 0x61)]):
        (raw / cls).mkdir(parents=True)
        frames = []
        for i in range(20):
            payloads = [bytes([base + i % 26]) * (30 + i), bytes([c]) * (60 + i), b"z" * (10 + i)]
            frames += tcp_session(CLIENT, 30000 + i, SERVER, 443, i * 1.0, payloads)
        # an exact content duplicate of session 0 on another port -> must be de-duplicated
        frames += tcp_session(CLIENT, 39999, SERVER, 443, 100.0, [bytes([base]) * 30, bytes([c]) * 60, b"z" * 10])
        write_pcap(raw / cls / f"{cls}.pcap", sorted(frames, key=lambda x: x[0]))

    cfg = load_config(
        overrides=[
            f"paths.processed_dir={tmp_path / 'proc'}",
            "preprocess.num_workers=1",
            "preprocess.min_sessions_per_class=5",
        ]
    )
    ds = {"name": "toy", "raw_dir": str(raw), "files": {"A/*.pcap": "A", "B/*.pcap": "B", "C/*.pcap": "C"}}
    summary = build(cfg, ds)
    assert summary["dropped_classes"] == ["C"] and summary["classes"] == ["A", "B"]

    out = tmp_path / "proc"
    sessions = pd.read_parquet(out / "sessions.parquet")
    assert summary["dedup"]["duplicates_removed"] == 2
    assert len(sessions) == 40
    splits = json.loads((out / "splits.json").read_text())
    ids = splits["train"] + splits["val"] + splits["test"]
    assert sorted(ids) == list(range(40))
    assert not set(splits["train"]) & set(splits["test"])
    for v in ("none", "ip", "ip_port"):
        b = np.load(out / "bytes" / v / "bytes.npy")
        assert b.shape == (40, 784) and b.dtype == np.uint8
    # unmasked variant keeps the client IP at bytes 12..15; masked variant zeroes it
    none_b = np.load(out / "bytes" / "none" / "bytes.npy")
    ip_b = np.load(out / "bytes" / "ip" / "bytes.npy")
    assert bytes(none_b[0, 12:16]) == ip(CLIENT) and not ip_b[0, 12:20].any()
    assert np.load(out / "seqs.npy").shape == (40, 32, 2)
    assert {f"f_{n}" for n in FEATURE_NAMES} <= set(sessions.columns)
    assert not {"src_ip", "dst_ip", "src_port", "dst_port"} & {c for c in sessions.columns if c.startswith("f_")}
