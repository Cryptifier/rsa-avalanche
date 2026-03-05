#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
import sys


GRID_RE = re.compile(r"rgen_grid_(?:(small|medium)_)?pct_(\d+)", re.IGNORECASE)


def load_session(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def find_sessions(paths, sessions_dir):
    if paths:
        return paths
    if not sessions_dir or not os.path.isdir(sessions_dir):
        return []
    entries = []
    for name in os.listdir(sessions_dir):
        if not name.endswith(".json"):
            continue
        entries.append(os.path.join(sessions_dir, name))
    return sorted(entries)


def get_feature(session, name):
    for feature in session.get("features", []):
        if feature.get("name") == name:
            return feature
    return None


def extract_reuse_path(session):
    reuse_path = ""
    for batch in session.get("r_candidate_batches", []):
        candidate_path = batch.get("reuse_path") or ""
        if candidate_path:
            reuse_path = candidate_path
    return reuse_path


def parse_grid_info(reuse_path, config_path):
    pct = ""
    size = ""
    if reuse_path:
        match = GRID_RE.search(os.path.basename(reuse_path))
        if match:
            size = (match.group(1) or "").lower()
            pct = match.group(2)
    if not size and config_path:
        lowered = os.path.basename(config_path).lower()
        if "small" in lowered:
            size = "small"
        elif "medium" in lowered:
            size = "medium"
    return size, pct


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate rgen grid session JSON files into a CSV summary."
    )
    parser.add_argument(
        "sessions",
        nargs="*",
        help="Session JSON files to include (defaults to --sessions-dir).",
    )
    parser.add_argument(
        "--sessions-dir",
        default="logs",
        help="Directory to scan for session JSON files (default: logs).",
    )
    parser.add_argument(
        "--out",
        default="data/rgen_grid_summary.csv",
        help="Output CSV path (use '-' for stdout).",
    )
    parser.add_argument(
        "--include-unknown",
        action="store_true",
        help="Include sessions without a grid percent in the output.",
    )
    args = parser.parse_args()

    session_paths = find_sessions(args.sessions, args.sessions_dir)
    if not session_paths:
        print("No session JSON files found.", file=sys.stderr)
        return 1

    rows = []
    for path in session_paths:
        try:
            session = load_session(path)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Skipping {path}: {exc}", file=sys.stderr)
            continue

        cli = session.get("cli", {}) or {}
        config_path = cli.get("config_path", "")
        reuse_path = extract_reuse_path(session)
        size, pct = parse_grid_info(reuse_path, config_path)

        info = get_feature(session, "information_sufficiency") or {}
        stats = info.get("stats", {}) or {}

        row = {
            "size": size,
            "pct": pct,
            "config_path": config_path,
            "reuse_path": reuse_path,
            "match_pct_mean": stats.get("match_pct_mean", ""),
            "speculative_match_pct": stats.get("speculative_match_pct", ""),
            "oracle_accuracy_mean_pct": stats.get("oracle_accuracy_mean_pct", ""),
            "status": stats.get("status", ""),
            "session_path": path,
        }

        if not row["pct"] and not args.include_unknown:
            continue

        rows.append(row)

    rows.sort(key=lambda r: (r["size"], int(r["pct"] or 0), r["session_path"]))

    fieldnames = [
        "size",
        "pct",
        "config_path",
        "reuse_path",
        "match_pct_mean",
        "speculative_match_pct",
        "oracle_accuracy_mean_pct",
        "status",
        "session_path",
    ]

    if args.out == "-":
        writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        return 0

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
