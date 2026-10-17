"""Raw-byte representation with IP/port anonymisation (review Sections 3.1, 4.2.2, 5.3).

Each session becomes the concatenation of its first packets' bytes (from the IP header,
at most `bytes_per_packet` per packet), cut/padded to `max_bytes`. Identifier fields can
be zeroed to prevent shortcut learning (RQ3):
  * mask_ip   -> IPv4 src/dst address (bytes 12..19 of the IP header)
  * mask_port -> TCP/UDP src/dst port (first 4 bytes of the L4 header)
Whenever any masking is on, the IP header checksum and the TCP/UDP checksum are zeroed
too, because they are computed over the masked fields and would leak them indirectly.
"""

from __future__ import annotations

import hashlib

import numpy as np

# name -> (mask_ip, mask_port); used for the RQ3 ablation
MASK_VARIANTS: dict[str, tuple[bool, bool]] = {
    "none": (False, False),
    "ip": (True, False),
    "ip_port": (True, True),
}

IP_CHECKSUM = (10, 12)
IP_ADDRS = (12, 20)
TCP_CHECKSUM_OFFSET = 16
UDP_CHECKSUM_OFFSET = 6


def variant_name(mask_ip: bool, mask_port: bool) -> str:
    for name, flags in MASK_VARIANTS.items():
        if flags == (bool(mask_ip), bool(mask_port)):
            return name
    raise ValueError(f"Unsupported mask combination: mask_ip={mask_ip}, mask_port={mask_port}")


def mask_packet(ip_bytes: bytes, mask_ip: bool, mask_port: bool) -> bytes:
    """Zero identifier fields (and dependent checksums) in a packet starting at the IP header.

    Works on truncated packets: fields beyond the available bytes are simply skipped.
    """
    if not (mask_ip or mask_port) or len(ip_bytes) < 20:
        return ip_bytes
    b = bytearray(ip_bytes)
    n = len(b)

    def zero(start: int, end: int) -> None:
        if start < n:
            end = min(end, n)
            b[start:end] = bytes(end - start)

    ihl = (b[0] & 0x0F) * 4
    proto = b[9]
    zero(*IP_CHECKSUM)
    if mask_ip:
        zero(*IP_ADDRS)
    if proto in (6, 17):
        if mask_port:
            zero(ihl, ihl + 4)
        csum = ihl + (TCP_CHECKSUM_OFFSET if proto == 6 else UDP_CHECKSUM_OFFSET)
        zero(csum, csum + 2)
    return bytes(b)


def packets_to_array(
    raw: list[bytes], max_packets: int, bytes_per_packet: int
) -> tuple[np.ndarray, np.ndarray]:
    """Stack per-packet bytes into (max_packets, bytes_per_packet) uint8 + true lengths."""
    arr = np.zeros((max_packets, bytes_per_packet), dtype=np.uint8)
    lens = np.zeros(max_packets, dtype=np.uint8 if bytes_per_packet < 256 else np.uint16)
    for i, pkt in enumerate(raw[:max_packets]):
        chunk = pkt[:bytes_per_packet]
        arr[i, : len(chunk)] = np.frombuffer(chunk, dtype=np.uint8)
        lens[i] = len(chunk)
    return arr, lens


def flatten_packets(pkt_arr: np.ndarray, lens: np.ndarray, max_bytes: int) -> np.ndarray:
    """Concatenate the valid bytes of each packet and cut/pad with zeros to `max_bytes`."""
    parts = [pkt_arr[i, : int(n)] for i, n in enumerate(lens) if n > 0]
    flat = np.concatenate(parts) if parts else np.zeros(0, dtype=np.uint8)
    out = np.zeros(max_bytes, dtype=np.uint8)
    out[: min(max_bytes, len(flat))] = flat[:max_bytes]
    return out


def session_bytes(
    raw: list[bytes],
    mask_ip: bool,
    mask_port: bool,
    max_packets: int,
    bytes_per_packet: int,
    max_bytes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mask a session's packets and return (flat bytes, per-packet bytes, per-packet lengths)."""
    masked = [mask_packet(p, mask_ip, mask_port) for p in raw]
    pkt_arr, lens = packets_to_array(masked, max_packets, bytes_per_packet)
    return flatten_packets(pkt_arr, lens, max_bytes), pkt_arr, lens


def content_hash(flat: np.ndarray) -> str:
    """Hash of a session's preprocessed flat byte vector (used for de-duplication)."""
    return hashlib.blake2b(flat.tobytes(), digest_size=16).hexdigest()
