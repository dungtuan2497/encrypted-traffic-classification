"""Helpers that write tiny synthetic pcaps with dpkt (unit tests only)."""

from __future__ import annotations

import socket
from pathlib import Path

import dpkt

ACK = dpkt.tcp.TH_ACK
SYN = dpkt.tcp.TH_SYN
PSH = dpkt.tcp.TH_PUSH


def ip(addr: str) -> bytes:
    return socket.inet_aton(addr)


def tcp_frame(src: str, sport: int, dst: str, dport: int, payload: bytes = b"", flags: int = ACK) -> bytes:
    seg = dpkt.tcp.TCP(sport=sport, dport=dport, flags=flags, data=payload)
    pkt = dpkt.ip.IP(src=ip(src), dst=ip(dst), p=dpkt.ip.IP_PROTO_TCP, ttl=64, data=seg)
    pkt.len = len(bytes(pkt))
    eth = dpkt.ethernet.Ethernet(src=b"\x00" * 6, dst=b"\x11" * 6, type=dpkt.ethernet.ETH_TYPE_IP, data=pkt)
    return bytes(eth)


def udp_frame(src: str, sport: int, dst: str, dport: int, payload: bytes) -> bytes:
    dgram = dpkt.udp.UDP(sport=sport, dport=dport, data=payload)
    dgram.ulen = len(bytes(dgram))
    pkt = dpkt.ip.IP(src=ip(src), dst=ip(dst), p=dpkt.ip.IP_PROTO_UDP, ttl=64, data=dgram)
    pkt.len = len(bytes(pkt))
    eth = dpkt.ethernet.Ethernet(src=b"\x00" * 6, dst=b"\x11" * 6, type=dpkt.ethernet.ETH_TYPE_IP, data=pkt)
    return bytes(eth)


def arp_frame() -> bytes:
    return bytes(dpkt.ethernet.Ethernet(src=b"\x00" * 6, dst=b"\xff" * 6, type=dpkt.ethernet.ETH_TYPE_ARP, data=b"\x00" * 28))


def write_pcap(path: Path, frames: list[tuple[float, bytes]]) -> Path:
    with open(path, "wb") as f:
        w = dpkt.pcap.Writer(f)
        for ts, frame in frames:
            w.writepkt(frame, ts=ts)
    return path


def tcp_session(client: str, cport: int, server: str, sport: int, t0: float, payloads: list[bytes]) -> list[tuple[float, bytes]]:
    """Handshake (no payload) + alternating data packets + a trailing pure ACK."""
    frames = [
        (t0, tcp_frame(client, cport, server, sport, flags=SYN)),
        (t0 + 0.001, tcp_frame(server, sport, client, cport, flags=SYN | ACK)),
    ]
    for i, p in enumerate(payloads):
        src, sp, dst, dp = (client, cport, server, sport) if i % 2 == 0 else (server, sport, client, cport)
        frames.append((t0 + 0.01 * (i + 1), tcp_frame(src, sp, dst, dp, p, flags=PSH | ACK)))
    frames.append((t0 + 0.5, tcp_frame(client, cport, server, sport)))  # pure ACK
    return frames
