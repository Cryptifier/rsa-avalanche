#!/usr/bin/env python3
"""Plot final-tier center-bias bit-position frequencies from a session log."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the center-bias histogram tool."""

    parser = argparse.ArgumentParser(
        description=(
            "Read a session log and plot a histogram of final-tier center-bias bit positions."
        )
    )
    parser.add_argument(
        "--session",
        required=True,
        help="Session JSON/NDJSON file produced by the analysis pipeline.",
    )
    parser.add_argument(
        "--output",
        default="",
        help=(
            "Optional PNG output path. Defaults to <session stem>_center_biases.png next to the session file."
        ),
    )
    parser.add_argument(
        "--top",
        type=int,
        default=64,
        help="Maximum number of most-frequent bit positions to plot (default: 64).",
    )
    return parser.parse_args()


def load_session(path: Path) -> dict[str, Any]:
    """Load either a merged JSON session or a streaming NDJSON session log."""

    raw = path.read_text(encoding="utf-8")
    first_nonempty = next((line.strip() for line in raw.splitlines() if line.strip()), "")
    if first_nonempty.startswith('{"event"'):
        return load_ndjson_session(raw.splitlines())

    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError(f"session payload must be a JSON object, got {type(data).__name__}")
    return data


def load_ndjson_session(lines: list[str]) -> dict[str, Any]:
    """Build a minimal merged session shape from NDJSON analytics events."""

    session: dict[str, Any] = {
        "cli": {},
        "features": [],
        "r_candidate_accuracy_batches": [],
    }
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        event = json.loads(stripped)
        if not isinstance(event, dict):
            continue
        event_name = event.get("event")
        payload = event.get("payload")
        if event_name == "session_start" and isinstance(payload, dict):
            cli = payload.get("cli")
            if isinstance(cli, dict):
                session["cli"] = cli
        elif event_name == "feature" and isinstance(payload, dict):
            session["features"].append(payload)
        elif event_name == "r_candidate_accuracy_batch" and isinstance(payload, dict):
            session["r_candidate_accuracy_batches"].append(payload)
    return session


def extract_center_bias_entries_from_feature(session: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract the best-only center-bias entries from the run-level feature stats."""

    features = session.get("features", [])
    if not isinstance(features, list):
        return []

    for feature in features:
        if not isinstance(feature, dict):
            continue
        if feature.get("name") != "r_candidate_accuracy":
            continue
        stats = feature.get("stats")
        if not isinstance(stats, dict):
            continue
        report = stats.get("avalanche_best_center_bias_report")
        if not isinstance(report, dict):
            continue
        center_biases = report.get("center_biases", [])
        if isinstance(center_biases, list):
            return [entry for entry in center_biases if isinstance(entry, dict)]
    return []


def extract_bit_position_counts(session: dict[str, Any]) -> Counter[int]:
    """Count how often each bit position appears in final-tier center-bias reports."""

    counts: Counter[int] = Counter()
    batches = session.get("r_candidate_accuracy_batches", [])
    if not isinstance(batches, list):
        return counts

    for batch in batches:
        if not isinstance(batch, dict):
            continue
        reports = batch.get("avalanche_final_tier_bias_reports", [])
        if not isinstance(reports, list):
            continue
        for report in reports:
            if not isinstance(report, dict):
                continue
            center_biases = report.get("center_biases", [])
            if not isinstance(center_biases, list):
                continue
            for entry in center_biases:
                if not isinstance(entry, dict):
                    continue
                bit_index = entry.get("bit_index_lsb0")
                if isinstance(bit_index, int):
                    counts[bit_index] += 1

    for entry in extract_center_bias_entries_from_feature(session):
        bit_index = entry.get("bit_index_lsb0")
        if isinstance(bit_index, int):
            counts[bit_index] += 1
    return counts


def resolve_output_path(session_path: Path, explicit_output: str) -> Path:
    """Resolve the histogram output path."""

    if explicit_output:
        return Path(explicit_output)
    return session_path.with_name(f"{session_path.stem}_center_biases.png")


def plot_counts(
    counts: Counter[int],
    session_path: Path,
    output_path: Path,
    top_n: int,
) -> None:
    """Render and save the histogram chart."""

    most_common = counts.most_common(max(top_n, 1))
    if not most_common:
        raise ValueError("session does not contain any final-tier center-bias entries")

    bit_positions = [str(bit_index) for bit_index, _ in most_common]
    frequencies = [frequency for _, frequency in most_common]

    width = max(10.0, min(24.0, 0.35 * len(most_common) + 6.0))
    height = 6.0
    fig, ax = plt.subplots(figsize=(width, height))
    bars = ax.bar(bit_positions, frequencies, color="#2f6db0", edgecolor="#153a63")

    ax.set_title(f"Center-Bias Bit Frequencies\n{session_path.name}")
    ax.set_xlabel("Bit position (lsb0)")
    ax.set_ylabel("Occurrences in final-tier center-bias reports")
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.set_axisbelow(True)

    for bar, frequency in zip(bars, frequencies, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            str(frequency),
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> int:
    """Load the session, count bit positions, and save the histogram."""

    args = parse_args()
    session_path = Path(args.session)
    if not session_path.is_file():
        raise SystemExit(f"missing session file: {session_path}")

    session = load_session(session_path)
    counts = extract_bit_position_counts(session)
    output_path = resolve_output_path(session_path, args.output)
    plot_counts(counts, session_path, output_path, args.top)

    total_entries = sum(counts.values())
    unique_positions = len(counts)
    print(
        f"Wrote {output_path} using {total_entries} center-bias entries across {unique_positions} bit positions."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
