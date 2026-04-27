#!/usr/bin/env python3
"""
Extract PKESK packets from an ASCII-armored OpenPGP file and print them
as raw bytes plus a structured data representation.

Usage:
    python3 extract_pkesk.py message.asc

Notes:
- This parses ASCII armor and OpenPGP packet framing directly.
- It extracts packet tag 1 (Public-Key Encrypted Session Key, PKESK).
- It parses the common PKESK top-level fields:
    * version
    * recipient identifier (for v3: key ID)
    * public-key algorithm
    * remaining encrypted session key field bytes
- It does not attempt full algorithm-specific decoding of the encrypted
  session key material.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------

@dataclass
class OpenPGPPacket:
    tag: int
    new_format: bool
    header_len: int
    body_len: int
    raw_header: bytes
    raw_body: bytes
    raw_packet: bytes


@dataclass
class PKESKPacket:
    packet_tag: int
    packet_offset: int
    version: int
    recipient_key_id: Optional[str]
    recipient_fingerprint: Optional[str]
    pk_algorithm_id: Optional[int]
    pk_algorithm_name: Optional[str]
    esk_bytes: bytes
    body_bytes: bytes
    packet_bytes: bytes

    def to_display_dict(self) -> dict:
        d = asdict(self)
        d["esk_bytes"] = self.esk_bytes.hex()
        d["body_bytes"] = self.body_bytes.hex()
        d["packet_bytes"] = self.packet_bytes.hex()
        return d


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

PK_ALGO_NAMES = {
    1: "RSA (Encrypt or Sign)",
    2: "RSA Encrypt-Only",
    3: "RSA Sign-Only",
    16: "ElGamal (Encrypt-Only)",
    17: "DSA",
    18: "ECDH",
    19: "ECDSA",
    22: "EdDSA",
}


def read_file_bytes(path: Path) -> bytes:
    return path.read_bytes()


def strip_ascii_armor_and_decode(data: bytes) -> bytes:
    """
    Decode ASCII-armored OpenPGP data into raw binary packet bytes.

    Handles:
    - armor header/footer
    - optional armor headers like Version:, Comment:
    - base64 body
    - CRC24 line beginning with '=' is ignored
    """
    text = data.decode("utf-8", errors="replace")

    begin_match = re.search(
        r"-----BEGIN PGP [A-Z0-9 ,_-]+-----", text, flags=re.MULTILINE
    )
    end_match = re.search(
        r"-----END PGP [A-Z0-9 ,_-]+-----", text, flags=re.MULTILINE
    )

    if not begin_match or not end_match:
        raise ValueError("Input does not appear to be ASCII-armored OpenPGP data.")

    armored = text[begin_match.end():end_match.start()]
    lines = [line.strip() for line in armored.splitlines() if line.strip()]

    # Skip armor headers until the first non-header-ish line.
    b64_lines: List[str] = []
    in_body = False
    for line in lines:
        if not in_body:
            if ":" in line:
                # Armor header such as "Version: GnuPG ..."
                continue
            # First non-header line begins the base64 body.
            in_body = True

        if line.startswith("="):
            # CRC24 checksum line
            continue

        b64_lines.append(line)

    if not b64_lines:
        raise ValueError("No Base64 payload found inside ASCII armor.")

    b64_data = "".join(b64_lines)

    try:
        return base64.b64decode(b64_data, validate=True)
    except binascii.Error as exc:
        raise ValueError(f"Failed to decode Base64 armor payload: {exc}") from exc


def read_old_length(data: bytes, pos: int, length_type: int) -> Tuple[int, int]:
    """
    Read an old-format OpenPGP packet length.

    Returns:
        (body_length, bytes_consumed_for_length)

    Old-format length types:
        0 => 1-octet length
        1 => 2-octet length
        2 => 4-octet length
        3 => indeterminate length (unsupported here)
    """
    if length_type == 0:
        if pos + 1 > len(data):
            raise ValueError("Truncated old-format 1-octet length.")
        return data[pos], 1

    if length_type == 1:
        if pos + 2 > len(data):
            raise ValueError("Truncated old-format 2-octet length.")
        return int.from_bytes(data[pos:pos + 2], "big"), 2

    if length_type == 2:
        if pos + 4 > len(data):
            raise ValueError("Truncated old-format 4-octet length.")
        return int.from_bytes(data[pos:pos + 4], "big"), 4

    raise ValueError("Indeterminate-length old-format packets are not supported.")


def read_new_length(data: bytes, pos: int) -> Tuple[int, int]:
    """
    Read a new-format OpenPGP packet length.

    Returns:
        (body_length, bytes_consumed_for_length)

    Partial Body Lengths are not supported in this minimal parser.
    """
    if pos >= len(data):
        raise ValueError("Missing new-format length octet.")

    first = data[pos]

    if first < 192:
        return first, 1

    if 192 <= first <= 223:
        if pos + 2 > len(data):
            raise ValueError("Truncated 2-octet new-format length.")
        second = data[pos + 1]
        length = ((first - 192) << 8) + second + 192
        return length, 2

    if first == 255:
        if pos + 5 > len(data):
            raise ValueError("Truncated 5-octet new-format length.")
        length = int.from_bytes(data[pos + 1:pos + 5], "big")
        return length, 5

    # 224..254 => partial body lengths
    raise ValueError("Partial body lengths are not supported by this script.")


def parse_openpgp_packets(data: bytes) -> List[Tuple[int, OpenPGPPacket]]:
    """
    Parse a raw OpenPGP packet stream.

    Returns:
        list of (packet_offset, OpenPGPPacket)
    """
    packets: List[Tuple[int, OpenPGPPacket]] = []
    pos = 0

    while pos < len(data):
        packet_offset = pos

        first = data[pos]
        if not (first & 0x80):
            raise ValueError(f"Invalid packet header at offset {pos}: top bit not set.")

        new_format = bool(first & 0x40)

        if new_format:
            tag = first & 0x3F
            length, length_len = read_new_length(data, pos + 1)
            header_len = 1 + length_len
            body_start = pos + header_len
        else:
            tag = (first >> 2) & 0x0F
            length_type = first & 0x03
            length, length_len = read_old_length(data, pos + 1, length_type)
            header_len = 1 + length_len
            body_start = pos + header_len

        body_end = body_start + length
        if body_end > len(data):
            raise ValueError(
                f"Packet at offset {packet_offset} extends beyond end of data."
            )

        raw_header = data[pos:body_start]
        raw_body = data[body_start:body_end]
        raw_packet = data[pos:body_end]

        packets.append(
            (
                packet_offset,
                OpenPGPPacket(
                    tag=tag,
                    new_format=new_format,
                    header_len=header_len,
                    body_len=length,
                    raw_header=raw_header,
                    raw_body=raw_body,
                    raw_packet=raw_packet,
                ),
            )
        )

        pos = body_end

    return packets


def parse_pkesk_body(packet_offset: int, packet: OpenPGPPacket) -> PKESKPacket:
    """
    Parse the top-level fields of a PKESK packet body.
    """
    body = packet.raw_body
    if not body:
        raise ValueError(f"Empty PKESK packet body at offset {packet_offset}.")

    version = body[0]

    # Conservative parsing:
    # v3 typically:
    #   1 byte version
    #   8 bytes key ID
    #   1 byte public-key algorithm
    #   rest = algorithm-specific encrypted session key fields
    #
    # newer versions vary more, so we keep the raw body and try to expose
    # what we safely can.
    recipient_key_id: Optional[str] = None
    recipient_fingerprint: Optional[str] = None
    pk_algorithm_id: Optional[int] = None
    esk_bytes = b""

    if version == 3:
        if len(body) < 10:
            raise ValueError(f"Truncated v3 PKESK packet at offset {packet_offset}.")
        recipient_key_id = body[1:9].hex().upper()
        pk_algorithm_id = body[9]
        esk_bytes = body[10:]
    else:
        # For newer packet versions, formats differ by RFC generation.
        # We preserve the whole body and try a best-effort guess:
        #
        # byte 0 = version
        # last reliable common field to expose here without overcommitting
        # is: keep opaque remainder.
        #
        # If there is at least one more byte, expose it as "maybe algorithm"
        # only if you want a heuristic. Here we avoid incorrect decoding.
        esk_bytes = body[1:]

    return PKESKPacket(
        packet_tag=packet.tag,
        packet_offset=packet_offset,
        version=version,
        recipient_key_id=recipient_key_id,
        recipient_fingerprint=recipient_fingerprint,
        pk_algorithm_id=pk_algorithm_id,
        pk_algorithm_name=PK_ALGO_NAMES.get(pk_algorithm_id),
        esk_bytes=esk_bytes,
        body_bytes=body,
        packet_bytes=packet.raw_packet,
    )


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract PKESK packets from an ASCII-armored OpenPGP file."
    )
    parser.add_argument("input_file", type=Path, help="ASCII-armored OpenPGP file")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print structured PKESK data as JSON",
    )
    parser.add_argument(
        "--dump-pkesk",
        action="store_true",
        help="Also dump each PKESK packet body to pkesk_<n>.bin",
    )
    args = parser.parse_args()

    raw_input = read_file_bytes(args.input_file)
    binary_packets = strip_ascii_armor_and_decode(raw_input)
    packets = parse_openpgp_packets(binary_packets)

    pkesk_packets: List[PKESKPacket] = []

    for packet_offset, packet in packets:
        if packet.tag == 1:
            pkesk_packets.append(parse_pkesk_body(packet_offset, packet))

    print(f"Input file: {args.input_file}")
    print(f"Decoded binary size: {len(binary_packets)} bytes")
    print(f"Total OpenPGP packets: {len(packets)}")
    print(f"PKESK packets found: {len(pkesk_packets)}")
    print()

    if not pkesk_packets:
        print("No PKESK packets were found.")
        return

    for i, pkesk in enumerate(pkesk_packets, start=1):
        print("=" * 72)
        print(f"PKESK #{i}")
        print("=" * 72)
        print(f"packet_offset      : {pkesk.packet_offset}")
        print(f"version            : {pkesk.version}")
        print(f"recipient_key_id   : {pkesk.recipient_key_id}")
        print(f"recipient_fp       : {pkesk.recipient_fingerprint}")
        print(f"pk_algorithm_id    : {pkesk.pk_algorithm_id}")
        print(f"pk_algorithm_name  : {pkesk.pk_algorithm_name}")
        print(f"body_len           : {len(pkesk.body_bytes)}")
        print(f"esk_len            : {len(pkesk.esk_bytes)}")
        print(f"packet_sha256      : {sha256_hex(pkesk.packet_bytes)}")
        print()

        print("body_bytes.hex()")
        print(pkesk.body_bytes.hex())
        print()

        print("esk_bytes.hex()")
        print(pkesk.esk_bytes.hex())
        print()

        print("data structure")
        if args.json:
            print(json.dumps(pkesk.to_display_dict(), indent=2))
        else:
            print(pkesk)
        print()

        if args.dump_pkesk:
            out_path = Path(f"pkesk_{i}.bin")
            out_path.write_bytes(pkesk.body_bytes)
            print(f"Wrote {out_path}")
            print()


if __name__ == "__main__":
    main()
