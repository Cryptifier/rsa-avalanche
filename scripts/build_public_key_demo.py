#!/usr/bin/env python3
"""
Prepare a small public-key demo run from an armored OpenPGP message.

This script extracts the first RSA PKESK ciphertext from a generated `.asc`
message, decrypts the matching PKCS#1 v1.5 block with the provided RSA private
key YAML, writes plaintext comparison artifacts beside the message, and emits a
small-batch-derived config wired to replay that ciphertext in public-key mode.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from extract_pkesk import (
    parse_openpgp_packets,
    parse_pkesk_body,
    read_file_bytes,
    strip_ascii_armor_and_decode,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a public-key demo config from a generated OpenPGP message."
    )
    parser.add_argument("--input-asc", type=Path, required=True)
    parser.add_argument("--private-key", type=Path, required=True)
    parser.add_argument("--public-key", type=Path, required=True)
    parser.add_argument("--template-config", type=Path, required=True)
    parser.add_argument("--output-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--analysis-batches", type=int, default=2)
    return parser.parse_args()


def load_top_level_yaml_scalars(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line[0].isspace():
            continue
        match = re.match(r"^([A-Za-z_]+):\s*(.*?)\s*$", line)
        if not match:
            continue
        key = match.group(1)
        value = match.group(2)
        if value.startswith(("'", '"')) and value.endswith(("'", '"')) and len(value) >= 2:
            value = value[1:-1]
        values[key] = value
    return values


def load_rsa_private_components(path: Path) -> tuple[int, int, int]:
    values = load_top_level_yaml_scalars(path)
    try:
        modulus = int(values["modulus"])
        private_exponent = int(values["private_exponent"])
        public_exponent = int(values["public_exponent"])
    except KeyError as exc:
        raise ValueError(f"missing {exc.args[0]} in {path}") from exc
    return modulus, private_exponent, public_exponent


def strip_json_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"//.*", "", text)
    return text


def load_template_config(path: Path) -> dict:
    return json.loads(strip_json_comments(path.read_text(encoding="utf-8")))


def extract_first_rsa_pkesk_ciphertext(input_path: Path) -> bytes:
    raw_input = read_file_bytes(input_path)
    decoded = strip_ascii_armor_and_decode(raw_input)
    packets = parse_openpgp_packets(decoded)

    for packet_offset, packet in packets:
        if packet.tag != 1:
            continue
        pkesk = parse_pkesk_body(packet_offset, packet)
        if pkesk.pk_algorithm_id not in (1, 2, 3):
            continue
        if len(pkesk.esk_bytes) < 2:
            raise ValueError(f"PKESK packet in {input_path} is missing the RSA MPI length")
        bit_length = int.from_bytes(pkesk.esk_bytes[:2], "big")
        byte_length = (bit_length + 7) // 8
        mpi_bytes = pkesk.esk_bytes[2 : 2 + byte_length]
        if len(mpi_bytes) != byte_length:
            raise ValueError(f"PKESK packet in {input_path} has a truncated RSA MPI")
        return mpi_bytes

    raise ValueError(f"no RSA PKESK packet found in {input_path}")


def decrypt_pkcs1_v1_5_block(ciphertext_bytes: bytes, modulus: int, private_exponent: int) -> bytes:
    modulus_len = max(1, (modulus.bit_length() + 7) // 8)
    ciphertext = int.from_bytes(ciphertext_bytes, "big")
    if ciphertext <= 0:
        raise ValueError("PKESK ciphertext must be non-zero")
    if ciphertext >= modulus:
        raise ValueError("PKESK ciphertext must be smaller than the RSA modulus")
    plaintext = pow(ciphertext, private_exponent, modulus)
    return plaintext.to_bytes(modulus_len, "big")


def parse_pkcs1_v1_5_payload(block: bytes) -> bytes:
    if len(block) < 11:
        raise ValueError("PKCS#1 v1.5 block is too short")
    if block[0:2] != b"\x00\x02":
        raise ValueError("decrypted PKESK block is not a PKCS#1 v1.5 encryption block")
    separator_index = block.find(b"\x00", 2)
    if separator_index < 10:
        raise ValueError("PKCS#1 v1.5 padding is shorter than 8 bytes")
    padding = block[2:separator_index]
    if any(byte == 0 for byte in padding):
        raise ValueError("PKCS#1 v1.5 padding contains a zero byte")
    payload = block[separator_index + 1 :]
    if not payload:
        raise ValueError("PKCS#1 v1.5 payload is empty")
    return payload


def relative_path(path: Path, start: Path) -> str:
    return Path(os.path.relpath(path, start=start)).as_posix()


def write_text_file(path: Path, contents: str) -> None:
    path.write_text(contents, encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    args.output_config.parent.mkdir(parents=True, exist_ok=True)

    modulus, private_exponent, public_exponent = load_rsa_private_components(args.private_key)
    ciphertext_bytes = extract_first_rsa_pkesk_ciphertext(args.input_asc)
    pkcs1_block = decrypt_pkcs1_v1_5_block(ciphertext_bytes, modulus, private_exponent)
    pkcs1_payload = parse_pkcs1_v1_5_payload(pkcs1_block)

    stem = args.input_asc.stem
    ciphertext_hex_path = output_dir / f"{stem}.pkesk_ciphertext.hex"
    block_hex_path = output_dir / f"{stem}.pkcs1_v1_5_block.hex"
    payload_hex_path = output_dir / f"{stem}.pkcs1_v1_5_payload.hex"

    write_text_file(ciphertext_hex_path, f"{ciphertext_bytes.hex()}\n")
    write_text_file(block_hex_path, f"{pkcs1_block.hex()}\n")
    write_text_file(payload_hex_path, f"{pkcs1_payload.hex()}\n")

    config = load_template_config(args.template_config)
    config.setdefault("rsa_keypair", {})
    config.setdefault("engine", {})
    config["engine"].setdefault("message", {})

    output_config_dir = args.output_config.resolve().parent
    config["rsa_keypair"]["generate"] = False
    config["rsa_keypair"]["e"] = public_exponent
    config["rsa_keypair"]["keyfile"] = relative_path(args.public_key.resolve(), output_config_dir)
    config["rsa_keypair"]["private_keyfile"] = relative_path(
        args.private_key.resolve(), output_config_dir
    )

    engine = config["engine"]
    engine["process_min_count"] = 8
    engine["process_count"] = 8
    engine["process_max_best_attempts"] = 8
    engine["min_message_trials"] = 8
    engine["combiner_k_oracles"] = 16
    engine["analysis_batch_enable"] = True
    engine["analysis_batch_messages"] = 8
    engine["analysis_batch_candidates"] = 128
    engine["analysis_batch_batches"] = max(1, args.analysis_batches)
    engine["avalanche_solver_global_log_enable"] = True
    engine["avalanche_beam_top_k"] = 4
    engine["avalanche_fitness_shift_bytes"] = 0
    engine["avalanche_fitness_r_candidate_limit"] = 32
    engine["avalanche_fitness_cx_candidate_limit"] = 4
    engine["avalanche_combination_mixed_r_candidates"] = 8
    engine["avalanche_combination_samples"] = 64
    engine["avalanche_combination_size"] = 8
    engine["avalanche_combination_pool_size"] = 256
    engine["avalanche_combination_recursion_depth"] = 1
    engine["avalanche_combination_recursive_group_size"] = [8]
    engine["avalanche_combination_recursive_resample_count"] = [0]
    engine["avalanche_combination_hamming_distance_prune"] = False
    engine["avalanche_combination_hamming_distance_keep_percentile"] = 100.0
    engine["avalanche_combination_hamming_distance_outlier_preference_pct"] = 0.0
    engine["avalanche_statistics_collection"] = False

    message = engine["message"]
    message["is_random"] = False
    message["bits"] = len(pkcs1_payload) * 8
    message["fixed_message"] = ""
    message["fixed_file"] = relative_path(ciphertext_hex_path, output_config_dir)
    message["use_file"] = True
    message["is_encrypted"] = True

    args.output_config.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    print(f"Prepared PKESK ciphertext hex: {ciphertext_hex_path}")
    print(f"Prepared PKCS#1 v1.5 block hex: {block_hex_path}")
    print(f"Prepared PKCS#1 v1.5 payload hex: {payload_hex_path}")
    print(f"Prepared demo config: {args.output_config}")
    print(f"Public exponent: {public_exponent}")
    print(f"Payload bits: {len(pkcs1_payload) * 8}")


if __name__ == "__main__":
    main()
