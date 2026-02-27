#!/usr/bin/env python3
import argparse
import sys

GREEN = "\033[32m"
RED = "\033[31m"
RESET = "\033[0m"


def normalize_hex(value: str) -> str:
    v = value.strip()
    if v.startswith("0x") or v.startswith("0X"):
        v = v[2:]
    if not v:
        raise ValueError("empty hex string")
    if any(c not in "0123456789abcdefABCDEF" for c in v):
        raise ValueError(f"invalid hex string: {value}")
    return v.lower()


def hex_to_bits(value: str) -> str:
    return "".join(f"{int(ch, 16):04b}" for ch in value)


def colorize_bits(a_bits: str, b_bits: str) -> str:
    out = []
    for a, b in zip(a_bits, b_bits):
        if a == b:
            out.append(f"{GREEN}{a}{RESET}")
        else:
            out.append(f"{RED}{a}{RESET}")
    return "".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description="Colorized bit diff for two hex strings")
    parser.add_argument("hex_a", help="First hex string (with or without 0x)")
    parser.add_argument("hex_b", help="Second hex string (with or without 0x)")
    args = parser.parse_args()

    try:
        a = normalize_hex(args.hex_a)
        b = normalize_hex(args.hex_b)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    width = max(len(a), len(b))
    a = a.rjust(width, "0")
    b = b.rjust(width, "0")

    a_bits = hex_to_bits(a)
    b_bits = hex_to_bits(b)

    total = len(a_bits)
    matches = sum(1 for x, y in zip(a_bits, b_bits) if x == y)
    match_pct = (matches / total) * 100.0 if total else 0.0

    print("A:", colorize_bits(a_bits, b_bits))
    print("B:", colorize_bits(b_bits, a_bits))
    print(f"Match: {match_pct:.2f}% ({matches}/{total})")
    print("Key: green = match, red = mismatch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
