#!/usr/bin/env python3
import argparse
import json
import os
import time
from typing import Any, Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge session JSON logs into a combined session payload."
    )
    parser.add_argument(
        "--input",
        default="logs",
        help="Directory containing session JSON logs.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Output path for combined session JSON.",
    )
    parser.add_argument(
        "--glob",
        default="*.json",
        help="Glob pattern for session files within the input directory.",
    )
    return parser.parse_args()


def list_session_paths(input_dir: str, pattern: str) -> List[str]:
    try:
        entries = os.listdir(input_dir)
    except FileNotFoundError:
        raise RuntimeError(f"input directory not found: {input_dir}")

    paths: List[str] = []
    for entry in sorted(entries):
        if not matches_pattern(entry, pattern):
            continue
        paths.append(os.path.join(input_dir, entry))
    return paths


def matches_pattern(name: str, pattern: str) -> bool:
    if pattern == "*.json":
        return name.endswith(".json")
    if pattern.startswith("*."):
        return name.endswith(pattern[1:])
    return name == pattern


def load_sessions(paths: List[str]) -> List[Tuple[str, Dict[str, Any]]]:
    sessions: List[Tuple[str, Dict[str, Any]]] = []
    for path in paths:
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        sessions.append((path, data))
    return sessions


def merge_list_field(sessions: List[Dict[str, Any]], key: str) -> List[Any]:
    merged: List[Any] = []
    for session in sessions:
        value = session.get(key)
        if isinstance(value, list):
            merged.extend(value)
    return merged


def merge_sessions(sessions: List[Tuple[str, Dict[str, Any]]]) -> Dict[str, Any]:
    if not sessions:
        raise RuntimeError("no valid session JSON files found")

    sources = [path for path, _ in sessions]
    payloads = [session for _, session in sessions]

    started_vals = [s.get("started_unix_ms") for s in payloads if isinstance(s.get("started_unix_ms"), int)]
    finished_vals = [s.get("finished_unix_ms") for s in payloads if isinstance(s.get("finished_unix_ms"), int)]

    combined: Dict[str, Any] = {}
    combined["started_unix_ms"] = min(started_vals) if started_vals else int(time.time() * 1000)
    combined["finished_unix_ms"] = max(finished_vals) if finished_vals else None
    combined["cli"] = payloads[0].get("cli", {})
    combined["steps"] = merge_list_field(payloads, "steps")
    combined["step_summaries"] = merge_list_field(payloads, "step_summaries")
    combined["features"] = merge_list_field(payloads, "features")
    combined["r_candidate_batches"] = merge_list_field(payloads, "r_candidate_batches")
    combined["r_candidate_accuracy_batches"] = merge_list_field(payloads, "r_candidate_accuracy_batches")
    combined["r_candidate_traces"] = merge_list_field(payloads, "r_candidate_traces")
    combined["errors"] = merge_list_field(payloads, "errors")
    combined["merge_info"] = {
        "source_files": sources,
        "source_count": len(sources),
        "created_unix_ms": int(time.time() * 1000),
    }

    return combined


def main() -> None:
    args = parse_args()
    input_dir = args.input
    output_path = args.output
    if not output_path:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(input_dir, f"combined_session_{stamp}.json")

    paths = list_session_paths(input_dir, args.glob)
    sessions = load_sessions(paths)
    combined = merge_sessions(sessions)

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(combined, handle, indent=2, sort_keys=False)

    print(f"Merged {len(sessions)} session logs into {output_path}")


if __name__ == "__main__":
    main()
