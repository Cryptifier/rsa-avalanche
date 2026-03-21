#!/usr/bin/env python3
import argparse
import json
import os
import sys
from datetime import datetime

from PySide6 import QtCore, QtGui, QtWidgets


def load_session(path):
    with open(path, "r", encoding="utf-8") as handle:
        raw = handle.read()
    raw_stripped = raw.lstrip()
    if not raw_stripped:
        return empty_session()
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            if "event" in data and "payload" in data:
                return build_session_from_events([data])
            return normalize_session(data)
        if isinstance(data, list):
            return build_session_from_events(data)
    except json.JSONDecodeError:
        pass
    events = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return build_session_from_events(events)


def format_unix_ms(value):
    if value is None:
        return "N/A"
    try:
        return datetime.fromtimestamp(value / 1000.0).isoformat(sep=" ", timespec="seconds")
    except (OSError, OverflowError, ValueError):
        return str(value)


def get_feature(session, name):
    for feature in session.get("features", []):
        if feature.get("name") == name:
            return feature
    return None


def flatten_candidate_batches(session):
    rows = []
    for batch in session.get("r_candidate_batches", []):
        context = batch.get("context", "")
        mode = batch.get("mode", "")
        for idx, entry in enumerate(batch.get("candidates", [])):
            factors = entry.get("factors", [])
            factor_str = "; ".join(
                f"{f.get('prime')}^{f.get('exponent')}" for f in factors
            )
            rows.append(
                {
                    "context": context,
                    "mode": mode,
                    "index": idx,
                    "r": entry.get("r", ""),
                    "r_bits": entry.get("r_bits", ""),
                    "factors": factor_str,
                }
            )
    return rows


def hex_to_bits_le(hex_str, bit_width):
    if not hex_str:
        value = 0
    else:
        text = hex_str.strip().lower()
        if text.startswith("0x"):
            text = text[2:]
        value = int(text, 16) if text else 0
    bits = []
    for idx in range(bit_width):
        bits.append(((value >> idx) & 1) == 1)
    return bits


def empty_session():
    return {
        "started_unix_ms": None,
        "finished_unix_ms": None,
        "cli": {},
        "steps": [],
        "step_summaries": [],
        "features": [],
        "r_candidate_batches": [],
        "r_candidate_accuracy_batches": [],
        "r_candidate_traces": [],
        "errors": [],
    }


def coerce_str(value):
    if value is None:
        return ""
    return str(value)


def coerce_int(value):
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def coerce_float(value):
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def coerce_bool(value):
    if value is None:
        return False
    return bool(value)


def coerce_list(value):
    return list(value) if isinstance(value, list) else []


def coerce_dict(value):
    return dict(value) if isinstance(value, dict) else {}


def coerce_optional(value):
    return value if value is not None else None


def normalize_cli(cli):
    cli = coerce_dict(cli)
    return {
        "bits": coerce_int(cli.get("bits")),
        "message_override": coerce_optional(cli.get("message_override")),
        "public_exponent": coerce_int(cli.get("public_exponent")),
        "seed": coerce_optional(cli.get("seed")),
        "crypto_rng": coerce_bool(cli.get("crypto_rng")),
        "config_path": coerce_str(cli.get("config_path")),
        "tests": coerce_bool(cli.get("tests")),
        "export": coerce_bool(cli.get("export")),
        "session_json": coerce_str(cli.get("session_json")),
        "shift": coerce_bool(cli.get("shift")),
        "ciphertext_modify": coerce_bool(cli.get("ciphertext_modify")),
        "use_hamming_distance": coerce_bool(cli.get("use_hamming_distance")),
        "mirror_invert_candidates": coerce_bool(cli.get("mirror_invert_candidates")),
        "bits_decrypt": coerce_optional(cli.get("bits_decrypt")),
    }


def normalize_feature(feature):
    feature = coerce_dict(feature)
    return {
        "name": coerce_str(feature.get("name")),
        "enabled": coerce_bool(feature.get("enabled")),
        "duration_ms": coerce_optional(feature.get("duration_ms")),
        "notes": coerce_list(feature.get("notes")),
        "stats": coerce_dict(feature.get("stats")),
    }


def normalize_step(step):
    step = coerce_dict(step)
    return {
        "name": coerce_str(step.get("name")),
        "duration_ms": coerce_int(step.get("duration_ms")),
    }


def normalize_step_summary(summary):
    summary = coerce_dict(summary)
    return {
        "name": coerce_str(summary.get("name")),
        "count": coerce_int(summary.get("count")),
        "total_ms": coerce_int(summary.get("total_ms")),
        "mean_ms": coerce_float(summary.get("mean_ms")),
    }


def normalize_r_candidate_factor(factor):
    factor = coerce_dict(factor)
    return {
        "prime": coerce_str(factor.get("prime")),
        "exponent": coerce_int(factor.get("exponent")),
        "prime_bits": coerce_int(factor.get("prime_bits")),
    }


def normalize_r_candidate_entry(entry):
    entry = coerce_dict(entry)
    factors = [normalize_r_candidate_factor(factor) for factor in coerce_list(entry.get("factors"))]
    return {
        "r": coerce_str(entry.get("r")),
        "r_bits": coerce_int(entry.get("r_bits")),
        "factors": factors,
    }


def normalize_r_candidate_batch(batch):
    batch = coerce_dict(batch)
    candidates = [
        normalize_r_candidate_entry(entry)
        for entry in coerce_list(batch.get("candidates"))
    ]
    return {
        "context": coerce_str(batch.get("context")),
        "mode": coerce_str(batch.get("mode")),
        "target_count": coerce_int(batch.get("target_count")),
        "generated_count": coerce_int(batch.get("generated_count")),
        "duration_ms": coerce_int(batch.get("duration_ms")),
        "reuse_path": coerce_str(batch.get("reuse_path")),
        "reuse_enabled": coerce_bool(batch.get("reuse_enabled")),
        "reuse_append_only": coerce_bool(batch.get("reuse_append_only")),
        "min_factor": coerce_str(batch.get("min_factor")),
        "process_scale": coerce_int(batch.get("process_scale")),
        "small_prime_factors": coerce_int(batch.get("small_prime_factors")),
        "max_factors": coerce_int(batch.get("max_factors")),
        "target_bit_length": coerce_optional(batch.get("target_bit_length")),
        "candidates": candidates,
    }


def normalize_r_candidate_accuracy_entry(entry):
    entry = coerce_dict(entry)
    factors = [normalize_r_candidate_factor(factor) for factor in coerce_list(entry.get("factors"))]
    return {
        "r": coerce_str(entry.get("r")),
        "r_bits": coerce_int(entry.get("r_bits")),
        "factors": factors,
        "accuracy_pct": coerce_float(entry.get("accuracy_pct")),
        "hbc_ciphertexts_r": coerce_list(entry.get("hbc_ciphertexts_r")),
        "candidate_decryptions": coerce_list(entry.get("candidate_decryptions")),
    }


def normalize_r_candidate_accuracy_batch(batch):
    batch = coerce_dict(batch)
    candidates = [
        normalize_r_candidate_accuracy_entry(entry)
        for entry in coerce_list(batch.get("candidates"))
    ]
    return {
        "context": coerce_str(batch.get("context")),
        "messages": coerce_list(batch.get("messages")),
        "ciphertexts": coerce_list(batch.get("ciphertexts")),
        "shifted_ciphertexts": coerce_list(batch.get("shifted_ciphertexts")),
        "rabin_exponent": coerce_int(batch.get("rabin_exponent")),
        "tonelli_shanks_modulus": coerce_str(batch.get("tonelli_shanks_modulus")),
        "tonelli_shanks_ciphertexts": coerce_list(batch.get("tonelli_shanks_ciphertexts")),
        "candidates": candidates,
        "beam_match_pct": coerce_optional(batch.get("beam_match_pct")),
        "beam_ones_match_pct": coerce_optional(batch.get("beam_ones_match_pct")),
        "beam_score": coerce_optional(batch.get("beam_score")),
        "beam_bit_width": coerce_optional(batch.get("beam_bit_width")),
    }


def normalize_r_candidate_trace_entry(entry):
    entry = coerce_dict(entry)
    return {
        "r": coerce_str(entry.get("r")),
        "r_bits": coerce_int(entry.get("r_bits")),
        "hbc_ciphertext_r": coerce_str(entry.get("hbc_ciphertext_r")),
        "candidate_decryption": coerce_str(entry.get("candidate_decryption")),
    }


def normalize_r_candidate_trace_batch(batch):
    batch = coerce_dict(batch)
    candidates = [
        normalize_r_candidate_trace_entry(entry)
        for entry in coerce_list(batch.get("candidates"))
    ]
    return {
        "context": coerce_str(batch.get("context")),
        "message": coerce_str(batch.get("message")),
        "ciphertext": coerce_str(batch.get("ciphertext")),
        "shifted_ciphertext": coerce_str(batch.get("shifted_ciphertext")),
        "rabin_exponent": coerce_int(batch.get("rabin_exponent")),
        "tonelli_shanks_modulus": coerce_str(batch.get("tonelli_shanks_modulus")),
        "tonelli_shanks_ciphertext": coerce_str(batch.get("tonelli_shanks_ciphertext")),
        "candidates": candidates,
    }


def normalize_session(session):
    session = coerce_dict(session)
    normalized = empty_session()
    normalized["started_unix_ms"] = coerce_optional(session.get("started_unix_ms"))
    normalized["finished_unix_ms"] = coerce_optional(session.get("finished_unix_ms"))
    normalized["cli"] = normalize_cli(session.get("cli", {}))
    normalized["steps"] = [
        normalize_step(step) for step in coerce_list(session.get("steps"))
    ]
    normalized["step_summaries"] = [
        normalize_step_summary(summary)
        for summary in coerce_list(session.get("step_summaries"))
    ]
    normalized["features"] = [
        normalize_feature(feature)
        for feature in coerce_list(session.get("features"))
    ]
    normalized["r_candidate_batches"] = [
        normalize_r_candidate_batch(batch)
        for batch in coerce_list(session.get("r_candidate_batches"))
    ]
    normalized["r_candidate_accuracy_batches"] = [
        normalize_r_candidate_accuracy_batch(batch)
        for batch in coerce_list(session.get("r_candidate_accuracy_batches"))
    ]
    normalized["r_candidate_traces"] = [
        normalize_r_candidate_trace_batch(batch)
        for batch in coerce_list(session.get("r_candidate_traces"))
    ]
    normalized["errors"] = coerce_list(session.get("errors"))
    return normalized


def build_session_from_events(events):
    session = empty_session()
    for event in events:
        if not isinstance(event, dict):
            continue
        event_name = event.get("event")
        payload = event.get("payload", {})
        if event_name == "session_start":
            payload = coerce_dict(payload)
            session["started_unix_ms"] = coerce_optional(payload.get("started_unix_ms"))
            session["cli"] = normalize_cli(payload.get("cli", {}))
        elif event_name == "session_finish":
            payload = coerce_dict(payload)
            session["finished_unix_ms"] = coerce_optional(payload.get("finished_unix_ms"))
            session["errors"] = coerce_list(payload.get("errors"))
        elif event_name == "step":
            session["steps"].append(normalize_step(payload))
        elif event_name == "step_summary":
            session["step_summaries"].append(normalize_step_summary(payload))
        elif event_name == "feature":
            normalized = normalize_feature(payload)
            existing = get_feature(session, normalized.get("name"))
            if existing is None:
                session["features"].append(normalized)
            else:
                existing.update(normalized)
        elif event_name == "r_candidate_batch":
            session["r_candidate_batches"].append(normalize_r_candidate_batch(payload))
        elif event_name == "r_candidate_accuracy_batch":
            session["r_candidate_accuracy_batches"].append(
                normalize_r_candidate_accuracy_batch(payload)
            )
        elif event_name == "r_candidate_trace_batch":
            session["r_candidate_traces"].append(
                normalize_r_candidate_trace_batch(payload)
            )
    return normalize_session(session)


def compute_basic_stats(values):
    if not values:
        return None, None, None, None
    count = len(values)
    mean = sum(values) / count
    variance = sum((value - mean) ** 2 for value in values) / count
    stddev = variance ** 0.5
    return mean, stddev, min(values), max(values)


def pearson_corr(pairs):
    if len(pairs) < 2:
        return None
    xs = [pair[0] for pair in pairs]
    ys = [pair[1] for pair in pairs]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in pairs)
    denom_x = sum((x - mean_x) ** 2 for x in xs)
    denom_y = sum((y - mean_y) ** 2 for y in ys)
    denom = (denom_x * denom_y) ** 0.5
    if denom == 0:
        return None
    return num / denom


def collect_log_paths(default_paths, log_dir):
    seen = set()
    results = []
    for path in default_paths:
        if not path:
            continue
        full = os.path.abspath(path)
        if os.path.exists(full) and full not in seen:
            results.append(full)
            seen.add(full)

    if log_dir and os.path.isdir(log_dir):
        entries = []
        for name in os.listdir(log_dir):
            if not (name.endswith(".json") or name.endswith(".log")):
                continue
            full = os.path.abspath(os.path.join(log_dir, name))
            try:
                mtime = os.path.getmtime(full)
            except OSError:
                mtime = 0
            entries.append((mtime, full))
        for _mtime, full in sorted(entries, key=lambda item: item[0], reverse=True):
            if full not in seen:
                results.append(full)
                seen.add(full)
    return results


class BitSimilarityCanvas(QtWidgets.QAbstractScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._rows = []
        self._bit_width = 0
        self._original_bits = []
        self._start_index = 0
        self._display_count = 0
        self._show_all = True
        self._bit_cache = {}
        self._match_counts = []
        self._row_tops = []
        self._content_height_px = 0
        self._max_shift = 0

        self._margin = 8
        self._label_width = 90
        self._bit_size = 10
        self._bit_spacing = 1
        self._row_spacing = 8
        self._header_height = 18
        self._row_padding = 6

        self._match_color = QtGui.QColor(46, 160, 67)
        self._mismatch_color = QtGui.QColor(220, 72, 72)
        self._multi_match_color = QtGui.QColor(242, 201, 76)
        self._masked_fill = QtGui.QColor(0, 0, 0)
        self._masked_text = QtGui.QColor(46, 160, 67)
        self._text_color = QtGui.QColor(40, 40, 40)

        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)

    def set_data(self, rows, bit_width, original_hex, match_counts=None):
        self._rows = list(rows)
        self._bit_width = int(bit_width or 0)
        self._original_bits = hex_to_bits_le(original_hex, self._bit_width)
        self._bit_cache.clear()
        self._match_counts = list(match_counts or [])
        self._max_shift = self._max_shift_for_rows(self._rows)
        self._start_index = 0
        self._display_count = len(self._rows)
        self._show_all = True
        self._rebuild_row_metrics()
        self._update_scrollbars()
        self.viewport().update()

    def set_view(self, start_index, count, show_all):
        self._show_all = bool(show_all)
        total = len(self._rows)
        if total == 0:
            self._start_index = 0
            self._display_count = 0
        elif self._show_all:
            self._start_index = 0
            self._display_count = total
        else:
            start_index = max(0, min(start_index, total - 1))
            count = max(1, min(count, total - start_index))
            self._start_index = start_index
            self._display_count = count
        self._rebuild_row_metrics()
        self._update_scrollbars()
        self.viewport().update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_scrollbars()

    def _row_height(self):
        return self._header_height + self._row_padding

    def _row_height_for(self, row):
        entries = row.get("entries", [])
        row_gap = self._bit_spacing + 8
        extra_rows = 1 if entries else 0
        return (
            self._header_height
            + self._bit_size
            + (len(entries) + extra_rows) * (self._bit_size + row_gap)
            + self._row_padding
        )

    def _content_width(self):
        bits_width = (self._bit_width + self._max_shift) * (self._bit_size + self._bit_spacing)
        return self._margin * 2 + self._label_width + bits_width

    def _content_height(self):
        return self._content_height_px

    def _max_shift_for_rows(self, rows):
        max_shift = 0
        for row in rows:
            for entry in row.get("entries", []):
                shift = int(entry.get("shift", 0) or 0)
                if shift > max_shift:
                    max_shift = shift
        return max_shift

    def _rebuild_row_metrics(self):
        self._row_tops = []
        if self._display_count == 0:
            self._content_height_px = 0
            return
        y = self._margin
        end_index = min(len(self._rows), self._start_index + self._display_count)
        for idx in range(self._start_index, end_index):
            row = self._rows[idx]
            self._row_tops.append(y)
            y += self._row_height_for(row) + self._row_spacing
        if self._row_tops:
            y -= self._row_spacing
        self._content_height_px = self._margin + y

    def _update_scrollbars(self):
        viewport = self.viewport().size()
        content_width = self._content_width()
        content_height = self._content_height()

        h_bar = self.horizontalScrollBar()
        v_bar = self.verticalScrollBar()

        h_max = max(0, content_width - viewport.width())
        v_max = max(0, content_height - viewport.height())

        h_bar.setRange(0, h_max)
        h_bar.setPageStep(viewport.width())
        v_bar.setRange(0, v_max)
        v_bar.setPageStep(viewport.height())

    def _candidate_bits(self, entry):
        cache_key = entry.get("_orig_index", entry.get("index", id(entry)))
        cached = self._bit_cache.get(cache_key)
        if cached is not None:
            return cached
        bits = hex_to_bits_le(entry.get("candidate_hex", ""), self._bit_width)
        self._bit_cache[cache_key] = bits
        return bits

    def _shifted_bits(self, bits, shift):
        if shift <= 0:
            return list(bits)
        width = self._bit_width
        shifted = [False] * width
        if shift >= width:
            return shifted
        for idx in range(shift, width):
            shifted[idx] = bits[idx - shift]
        return shifted

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QtGui.QPainter(self.viewport())
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, False)
        painter.fillRect(self.viewport().rect(), self.palette().base())
        base_font = painter.font()

        if self._display_count == 0:
            painter.setPen(self._text_color)
            painter.drawText(self._margin, self._margin + 16, "No bit similarity entries available.")
            painter.end()
            return

        x_offset = self.horizontalScrollBar().value()
        y_offset = self.verticalScrollBar().value()
        visible_top = y_offset
        visible_bottom = y_offset + self.viewport().height()
        start_offset = 0
        for idx, row_top in enumerate(self._row_tops):
            row = self._rows[self._start_index + idx]
            row_bottom = row_top + self._row_height_for(row)
            if row_bottom >= visible_top:
                start_offset = idx
                break

        for offset_idx in range(start_offset, len(self._row_tops)):
            row_idx = self._start_index + offset_idx
            if row_idx >= len(self._rows):
                break
            row = self._rows[row_idx]
            row_top_abs = self._row_tops[offset_idx]
            row_bottom_abs = row_top_abs + self._row_height_for(row)
            if row_top_abs > visible_bottom:
                break

            display_idx = row.get("index", row_idx)
            row_top = row_top_abs - y_offset
            header_y = row_top + self._header_height - 4
            bits_top = row_top + self._header_height

            painter.setPen(self._text_color)
            r_value = row.get("r", "")
            match_pct = row.get("base_match_pct", 0.0)
            matching_bits = row.get("base_matching_bits", 0)
            e_value = row.get("e")
            x_value = row.get("x")
            ex_suffix = (
                f" | e={e_value} | x={x_value}"
                if e_value is not None and x_value is not None
                else ""
            )
            painter.drawText(
                self._margin - x_offset,
                header_y,
                f"#{display_idx} | r={r_value} | match={match_pct:.2f}% | matching bits={matching_bits}{ex_suffix}",
            )

            label_x = self._margin - x_offset
            painter.drawText(label_x, bits_top + self._bit_size, "Original")

            original_bits = self._original_bits
            entries = row.get("entries", [])
            if not entries:
                continue
            base_bits = self._candidate_bits(entries[0])
            max_bits = self._bit_width
            small_font = QtGui.QFont(base_font)
            small_font.setPixelSize(max(4, int(self._bit_size * 0.5)))
            painter.setFont(small_font)
            row_gap = self._bit_spacing + 8
            y1 = bits_top
            for bit_idx in range(max_bits):
                orig_bit = original_bits[bit_idx] if bit_idx < len(original_bits) else False
                cand_bit = base_bits[bit_idx] if bit_idx < len(base_bits) else False
                matches = orig_bit == cand_bit
                base_original = self._match_color if matches else self._mismatch_color

                x = (
                    self._margin
                    + self._label_width
                    + bit_idx * (self._bit_size + self._bit_spacing)
                    - x_offset
                )

                color1 = base_original.lighter(130) if not orig_bit else base_original
                painter.fillRect(x, y1, self._bit_size, self._bit_size, color1)

                text1 = "1" if orig_bit else "0"
                text_color1 = QtGui.QColor(255, 255, 255) if orig_bit else QtGui.QColor(0, 0, 0)
                painter.setPen(text_color1)
                painter.drawText(
                    QtCore.QRectF(x, y1, self._bit_size, self._bit_size),
                    QtCore.Qt.AlignmentFlag.AlignCenter,
                    text1,
                )
            for entry_idx, entry in enumerate(entries):
                shift = int(entry.get("shift", 0) or 0)
                masked_bits = int(entry.get("masked_bits", max(0, shift)) or 0)
                row_label = "Candidate" if shift == 0 else f"Candidate << {shift}"
                entry_e = entry.get("e")
                entry_x = entry.get("x")
                if entry_e is not None and entry_x is not None:
                    row_label = f"{row_label} | e={entry_e} | x={entry_x}"
                adjusted_match_pct = entry.get("adjusted_match_pct", entry.get("match_pct", 0.0))
                adjusted_matching_bits = entry.get(
                    "adjusted_matching_bits", entry.get("matching_bits", 0)
                )
                adjusted_denom = max(1, self._bit_width - masked_bits)
                y2 = bits_top + self._bit_size + row_gap + entry_idx * (self._bit_size + row_gap)
                painter.drawText(
                    label_x,
                    y2 + self._bit_size,
                    f"{row_label} | adj={adjusted_match_pct:.2f}% ({adjusted_matching_bits}/{adjusted_denom})",
                )

                prev_bits = None
                if entry_idx > 0:
                    prev_bits = self._candidate_bits(entries[entry_idx - 1])
                candidate_bits = self._candidate_bits(entry)

                for bit_idx in range(max_bits):
                    cand_idx = bit_idx + shift
                    masked = cand_idx >= max_bits
                    matches_original = False
                    matches_prev = False
                    cand_bit = (
                        candidate_bits[cand_idx]
                        if cand_idx < len(candidate_bits)
                        else False
                    )
                    if not masked:
                        orig_bit = original_bits[bit_idx] if bit_idx < len(original_bits) else False
                        matches_original = orig_bit == cand_bit
                        if prev_bits is not None and cand_idx < len(prev_bits):
                            matches_prev = prev_bits[cand_idx] == cand_bit
                    if matches_original and matches_prev:
                        base_candidate = self._multi_match_color
                    elif matches_original:
                        base_candidate = self._match_color
                    else:
                        base_candidate = self._mismatch_color

                    x = (
                        self._margin
                        + self._label_width
                        + bit_idx * (self._bit_size + self._bit_spacing)
                        - x_offset
                    )
                    if masked:
                        color2 = self._masked_fill
                    else:
                        color2 = base_candidate.lighter(130) if not cand_bit else base_candidate
                    painter.fillRect(x, y2, self._bit_size, self._bit_size, color2)

                    if masked:
                        masked_bit = (
                            candidate_bits[bit_idx]
                            if bit_idx < len(candidate_bits)
                            else False
                        )
                        text2 = "1" if masked_bit else "0"
                        text_color2 = self._masked_text
                    else:
                        text2 = "1" if cand_bit else "0"
                        text_color2 = QtGui.QColor(255, 255, 255)
                    painter.setPen(text_color2)
                    painter.drawText(
                        QtCore.QRectF(x, y2, self._bit_size, self._bit_size),
                        QtCore.Qt.AlignmentFlag.AlignCenter,
                        text2,
                    )

            majority_bits = [False] * max_bits
            majority_votes = [0] * max_bits
            majority_ones = [0] * max_bits
            for entry in entries:
                shift = int(entry.get("shift", 0) or 0)
                candidate_bits = self._candidate_bits(entry)
                for bit_idx in range(max_bits):
                    cand_idx = bit_idx + shift
                    if cand_idx >= max_bits:
                        continue
                    bit_val = candidate_bits[cand_idx] if cand_idx < len(candidate_bits) else False
                    if bit_val:
                        majority_ones[bit_idx] += 1
                    majority_votes[bit_idx] += 1

            for bit_idx in range(max_bits):
                votes = majority_votes[bit_idx]
                if votes == 0:
                    continue
                ones = majority_ones[bit_idx]
                zeros = votes - ones
                majority_bits[bit_idx] = ones >= zeros

            majority_matches = 0
            majority_unmasked = 0
            for bit_idx in range(max_bits):
                if majority_votes[bit_idx] == 0:
                    continue
                majority_unmasked += 1
                if majority_bits[bit_idx] == original_bits[bit_idx]:
                    majority_matches += 1

            majority_denom = max(1, majority_unmasked)
            majority_pct = (majority_matches / float(majority_denom)) * 100.0
            majority_row_y = (
                bits_top
                + self._bit_size
                + row_gap
                + len(entries) * (self._bit_size + row_gap)
            )
            painter.drawText(
                label_x,
                majority_row_y + self._bit_size,
                f"Majority vote | adj={majority_pct:.2f}% ({majority_matches}/{majority_denom})",
            )

            for bit_idx in range(max_bits):
                votes = majority_votes[bit_idx]
                masked = votes == 0
                majority_bit = majority_bits[bit_idx]
                matches_original = (not masked) and (majority_bit == original_bits[bit_idx])
                if matches_original and votes > 1:
                    base_candidate = self._multi_match_color
                elif matches_original:
                    base_candidate = self._match_color
                else:
                    base_candidate = self._mismatch_color

                x = (
                    self._margin
                    + self._label_width
                    + bit_idx * (self._bit_size + self._bit_spacing)
                    - x_offset
                )
                if masked:
                    color2 = self._masked_fill
                else:
                    color2 = base_candidate.lighter(130) if not majority_bit else base_candidate
                painter.fillRect(x, majority_row_y, self._bit_size, self._bit_size, color2)

                if masked:
                    text2 = "1" if majority_bit else "0"
                    text_color2 = self._masked_text
                else:
                    text2 = "1" if majority_bit else "0"
                    text_color2 = QtGui.QColor(255, 255, 255)
                painter.setPen(text_color2)
                painter.drawText(
                    QtCore.QRectF(x, majority_row_y, self._bit_size, self._bit_size),
                    QtCore.Qt.AlignmentFlag.AlignCenter,
                    text2,
                )

            painter.setFont(base_font)

        painter.end()


class BitSimilarityTab(QtWidgets.QWidget):
    def __init__(self, bit_similarity=None, parent=None):
        super().__init__(parent)
        self._entries_raw = []
        self._entries = []
        self._bit_width = 0
        self._original_hex = ""
        self._bit_order = "lsb0"
        self._match_counts_raw = []
        self._pending_settings = None

        layout = QtWidgets.QVBoxLayout(self)

        control_row = QtWidgets.QHBoxLayout()
        self._sort_combo = QtWidgets.QComboBox()
        self._sort_combo.addItems(
            [
                "Original order",
                "Match % (Descending)",
                "Match % (Ascending)",
            ]
        )
        self._sort_combo.currentIndexChanged.connect(self._on_sort_changed)
        self._start_spin = QtWidgets.QSpinBox()
        self._start_spin.setRange(0, 0)
        self._start_spin.setValue(0)
        self._start_spin.valueChanged.connect(self._rebuild_rows)

        self._rows_spin = QtWidgets.QSpinBox()
        self._rows_spin.setRange(1, 1)
        self._rows_spin.setValue(1)
        self._rows_spin.valueChanged.connect(self._rebuild_rows)

        self._show_all = QtWidgets.QCheckBox("Show all rows")
        self._show_all.setChecked(True)
        self._hide_shifted = QtWidgets.QCheckBox("Hide shifted rows")
        self._hide_shifted.setChecked(False)

        control_row.addWidget(QtWidgets.QLabel("Sort:"))
        control_row.addWidget(self._sort_combo)
        control_row.addWidget(QtWidgets.QLabel("Start index:"))
        control_row.addWidget(self._start_spin)
        control_row.addWidget(QtWidgets.QLabel("Rows:"))
        control_row.addWidget(self._rows_spin)
        control_row.addWidget(self._show_all)
        control_row.addWidget(self._hide_shifted)
        control_row.addStretch(1)
        layout.addLayout(control_row)

        self._info_label = QtWidgets.QLabel("")
        layout.addWidget(self._info_label)

        self._canvas = BitSimilarityCanvas()
        layout.addWidget(self._canvas, 1)

        self._show_all.stateChanged.connect(self._toggle_show_all)
        self._hide_shifted.stateChanged.connect(self._on_hide_shifted)

        legend = QtWidgets.QLabel(
            "Green = matches original, Yellow = multi-match with original, Black = masked bits. LSB-first bit order."
        )
        legend.setStyleSheet("color: #555;")
        layout.addWidget(legend)

        if bit_similarity is not None:
            self.set_data(bit_similarity)

    def set_data(self, bit_similarity):
        bit_similarity = bit_similarity or {}
        self._entries_raw = [
            {**entry, "_orig_index": idx}
            for idx, entry in enumerate(bit_similarity.get("candidates", []))
        ]
        self._bit_width = int(bit_similarity.get("bit_width", 0) or 0)
        self._original_hex = bit_similarity.get("original_hex", "")
        self._bit_order = bit_similarity.get("bit_order", "lsb0")
        self._match_counts_raw = list(bit_similarity.get("match_counts_per_bit", []) or [])

        self._apply_sort()
        if self._pending_settings:
            pending = self._pending_settings
            self._pending_settings = None
            self.apply_settings(pending)

    def _apply_sort(self):
        sort_mode = self._sort_combo.currentText()
        entries = list(self._entries_raw)
        if self._hide_shifted.isChecked():
            entries = [entry for entry in entries if int(entry.get("shift", 0) or 0) == 0]
        base_by_index = {}
        entries_by_index = {}
        for entry in entries:
            idx = int(entry.get("index", entry.get("_orig_index", 0)))
            entries_by_index.setdefault(idx, []).append(entry)
            if idx not in base_by_index or int(entry.get("shift", 0) or 0) == 0:
                base_by_index[idx] = entry

        candidate_indices = list(entries_by_index.keys())
        if sort_mode == "Match % (Descending)":
            candidate_indices.sort(
                key=lambda idx: (
                    -float(
                        base_by_index[idx].get(
                            "base_match_pct",
                            base_by_index[idx].get("match_pct", 0.0),
                        )
                    ),
                    idx,
                )
            )
        elif sort_mode == "Match % (Ascending)":
            candidate_indices.sort(
                key=lambda idx: (
                    float(
                        base_by_index[idx].get(
                            "base_match_pct",
                            base_by_index[idx].get("match_pct", 0.0),
                        )
                    ),
                    idx,
                )
            )
        else:
            candidate_indices.sort()

        grouped = []
        for idx in candidate_indices:
            group_entries = entries_by_index.get(idx, [])
            group_entries.sort(key=lambda e: int(e.get("shift", 0) or 0))
            base_entry = base_by_index.get(idx, group_entries[0] if group_entries else {})
            grouped.append(
                {
                    "index": idx,
                    "r": base_entry.get("r", ""),
                    "e": base_entry.get("e"),
                    "x": base_entry.get("x"),
                    "base_match_pct": base_entry.get(
                        "base_match_pct", base_entry.get("match_pct", 0.0)
                    ),
                    "base_matching_bits": base_entry.get(
                        "base_matching_bits", base_entry.get("matching_bits", 0)
                    ),
                    "entries": group_entries,
                }
            )

        self._entries = grouped
        self._sync_ranges()
        match_counts = self._match_counts_raw
        if (
            not match_counts
            or len(match_counts) != self._bit_width
            or self._hide_shifted.isChecked()
        ):
            match_counts = self._build_match_counts(entries)
        self._canvas.set_data(self._entries, self._bit_width, self._original_hex, match_counts)
        self._rebuild_rows()

    def _on_sort_changed(self, _idx):
        self._apply_sort()

    def _on_hide_shifted(self, _state):
        self._apply_sort()

    def _build_match_counts(self, entries):
        if self._bit_width <= 0 or not entries:
            return []
        original_bits = hex_to_bits_le(self._original_hex, self._bit_width)
        counts = [0 for _ in range(self._bit_width)]
        for entry in entries:
            candidate_bits = hex_to_bits_le(entry.get("candidate_hex", ""), self._bit_width)
            shift = int(entry.get("shift", 0) or 0)
            for bit_idx in range(self._bit_width):
                cand_idx = bit_idx + shift
                if cand_idx >= self._bit_width:
                    continue
                if candidate_bits[cand_idx] == original_bits[bit_idx]:
                    counts[bit_idx] += 1
        return counts

    def _sync_ranges(self):
        total = len(self._entries)
        self._start_spin.blockSignals(True)
        self._rows_spin.blockSignals(True)
        self._start_spin.setRange(0, max(0, total - 1))
        default_rows = min(50, total) if total else 1
        self._rows_spin.setRange(1, max(1, total))
        if self._rows_spin.value() > total:
            self._rows_spin.setValue(default_rows)
        elif self._rows_spin.value() == 0:
            self._rows_spin.setValue(default_rows)
        if self._start_spin.value() >= total and total:
            self._start_spin.setValue(0)
        self._start_spin.blockSignals(False)
        self._rows_spin.blockSignals(False)

    def get_settings(self):
        return {
            "sort_index": self._sort_combo.currentIndex(),
            "show_all": self._show_all.isChecked(),
            "hide_shifted": self._hide_shifted.isChecked(),
            "start": self._start_spin.value(),
            "rows": self._rows_spin.value(),
        }

    def apply_settings(self, settings):
        if not settings:
            return
        if not self._entries_raw:
            self._pending_settings = settings
            return
        self._sort_combo.blockSignals(True)
        self._sort_combo.setCurrentIndex(settings.get("sort_index", 0))
        self._sort_combo.blockSignals(False)
        self._apply_sort()

        self._start_spin.blockSignals(True)
        self._rows_spin.blockSignals(True)
        self._start_spin.setValue(settings.get("start", self._start_spin.value()))
        self._rows_spin.setValue(settings.get("rows", self._rows_spin.value()))
        self._start_spin.blockSignals(False)
        self._rows_spin.blockSignals(False)

        self._show_all.blockSignals(True)
        self._show_all.setChecked(settings.get("show_all", True))
        self._show_all.blockSignals(False)
        self._toggle_show_all(self._show_all.isChecked())

        self._hide_shifted.blockSignals(True)
        self._hide_shifted.setChecked(settings.get("hide_shifted", False))
        self._hide_shifted.blockSignals(False)
        self._on_hide_shifted(self._hide_shifted.checkState())

    def _toggle_show_all(self, checked):
        show_all = bool(checked)
        self._start_spin.setEnabled(not show_all)
        self._rows_spin.setEnabled(not show_all)
        self._rebuild_rows()

    def _rebuild_rows(self):
        total = len(self._entries)
        if total == 0:
            self._info_label.setText("No bit similarity entries available.")
            self._canvas.set_view(0, 0, True)
            return

        if self._show_all.isChecked():
            start = 0
            count = total
        else:
            start = min(self._start_spin.value(), max(0, total - 1))
            count = min(self._rows_spin.value(), total - start)

        self._info_label.setText(
            f"Showing {start + 1}-{start + count} of {total} entries | bit order: {self._bit_order}"
        )
        self._canvas.set_view(start, count, self._show_all.isChecked())


class BitTrueTimelineCanvas(QtWidgets.QAbstractScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._frames = []
        self._bit_width = 0
        self._window = 0
        self._stride = 0

        self._margin = 12
        self._bar_width = 8
        self._bar_gap = 2
        self._depth_x = 4
        self._depth_y = 3
        self._max_height = 120

        self._base_color = QtGui.QColor(52, 128, 235)
        self._axis_color = QtGui.QColor(80, 80, 80)
        self._text_color = QtGui.QColor(40, 40, 40)

        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)

    def set_data(self, frames, bit_width, window, stride):
        self._frames = list(frames or [])
        self._bit_width = int(bit_width or 0)
        self._window = int(window or 0)
        self._stride = int(stride or 0)
        self._update_scrollbars()
        self.viewport().update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_scrollbars()

    def _content_width(self):
        frames = len(self._frames)
        return (
            self._margin * 2
            + frames * (self._bar_width + self._bar_gap)
            + self._bit_width * self._depth_x
            + self._bar_width
        )

    def _content_height(self):
        return (
            self._margin * 2
            + self._bit_width * self._depth_y
            + self._max_height
            + self._depth_y
        )

    def _update_scrollbars(self):
        viewport = self.viewport().size()
        content_width = self._content_width()
        content_height = self._content_height()

        h_bar = self.horizontalScrollBar()
        v_bar = self.verticalScrollBar()

        h_max = max(0, content_width - viewport.width())
        v_max = max(0, content_height - viewport.height())

        h_bar.setRange(0, h_max)
        h_bar.setPageStep(viewport.width())
        v_bar.setRange(0, v_max)
        v_bar.setPageStep(viewport.height())

    def _bar_color(self, prob):
        intensity = max(0.0, min(1.0, prob))
        base = QtGui.QColor(self._base_color)
        base.setAlphaF(0.25 + 0.75 * intensity)
        return base

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QtGui.QPainter(self.viewport())
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, False)
        painter.fillRect(self.viewport().rect(), self.palette().base())

        if not self._frames or self._bit_width == 0:
            painter.setPen(self._text_color)
            painter.drawText(self._margin, self._margin + 16, "No bit probability timeline data.")
            painter.end()
            return

        x_offset = self.horizontalScrollBar().value()
        y_offset = self.verticalScrollBar().value()

        origin_x = self._margin - x_offset
        origin_y = (
            self._margin
            + self._bit_width * self._depth_y
            + self._max_height
            - y_offset
        )

        painter.setPen(self._axis_color)
        painter.drawLine(origin_x, origin_y, origin_x + self._content_width(), origin_y)

        frames = len(self._frames)
        for frame_idx in range(frames):
            frame = self._frames[frame_idx]
            for bit_idx in range(min(self._bit_width, len(frame))):
                prob = frame[bit_idx]
                if prob <= 0.0:
                    continue
                height = max(1, int(prob * self._max_height))
                x = (
                    origin_x
                    + frame_idx * (self._bar_width + self._bar_gap)
                    + (self._bit_width - 1 - bit_idx) * self._depth_x
                )
                y = (
                    origin_y
                    - (self._bit_width - 1 - bit_idx) * self._depth_y
                    - height
                )

                base_color = self._bar_color(prob)
                top_color = base_color.lighter(130)
                side_color = base_color.darker(130)

                front = QtCore.QRectF(x, y, self._bar_width, height)
                painter.fillRect(front, base_color)

                top = QtGui.QPolygonF(
                    [
                        QtCore.QPointF(x, y),
                        QtCore.QPointF(x + self._depth_x, y - self._depth_y),
                        QtCore.QPointF(x + self._bar_width + self._depth_x, y - self._depth_y),
                        QtCore.QPointF(x + self._bar_width, y),
                    ]
                )
                painter.setBrush(top_color)
                painter.setPen(QtCore.Qt.PenStyle.NoPen)
                painter.drawPolygon(top)

                side = QtGui.QPolygonF(
                    [
                        QtCore.QPointF(x + self._bar_width, y),
                        QtCore.QPointF(x + self._bar_width + self._depth_x, y - self._depth_y),
                        QtCore.QPointF(
                            x + self._bar_width + self._depth_x,
                            y - self._depth_y + height,
                        ),
                        QtCore.QPointF(x + self._bar_width, y + height),
                    ]
                )
                painter.setBrush(side_color)
                painter.drawPolygon(side)

        painter.end()


class BitTrueTimelineTab(QtWidgets.QWidget):
    def __init__(self, timeline=None, parent=None):
        super().__init__(parent)
        self._bit_width = 0
        self._window = 0
        self._stride = 0
        self._frames = []

        layout = QtWidgets.QVBoxLayout(self)
        self._info_label = QtWidgets.QLabel("")
        layout.addWidget(self._info_label)

        self._canvas = BitTrueTimelineCanvas()
        layout.addWidget(self._canvas, 1)

        legend = QtWidgets.QLabel(
            "3D bars show per-bit P(1) over time (LSB-first)."
        )
        legend.setStyleSheet("color: #555;")
        layout.addWidget(legend)

        if timeline is not None:
            self.set_data(timeline)

    def set_data(self, timeline):
        timeline = timeline or {}
        self._bit_width = int(timeline.get("bit_width", 0) or 0)
        self._window = int(timeline.get("window", 0) or 0)
        self._stride = int(timeline.get("stride", 0) or 0)
        self._frames = list(timeline.get("frames", []) or [])
        frame_count = len(self._frames)
        self._info_label.setText(
            f"Frames: {frame_count} | Bits: {self._bit_width} | Window: {self._window} | Stride: {self._stride}"
        )
        self._canvas.set_data(self._frames, self._bit_width, self._window, self._stride)


class AvgTreeCanvas(QtWidgets.QAbstractScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._levels = [16, 8, 4, 2, 1]
        self._positions = []
        self._edges = []

        self._margin = 16
        self._box_size = 14
        self._box_gap = 6
        self._col_gap = 70
        self._text_color = QtGui.QColor(255, 255, 255)
        self._line_color = QtGui.QColor(40, 40, 40)

        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        self._rebuild_layout()

    def set_levels(self, levels):
        levels = [int(value) for value in (levels or []) if int(value) > 0]
        if not levels:
            levels = [16, 8, 4, 2, 1]
        self._levels = levels
        self._rebuild_layout()
        self._update_scrollbars()
        self.viewport().update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_scrollbars()

    def _content_width(self):
        if not self._levels:
            return 0
        cols = len(self._levels)
        return (
            self._margin * 2
            + cols * self._box_size
            + (cols - 1) * self._col_gap
        )

    def _content_height(self):
        if not self._levels:
            return 0
        max_count = max(self._levels)
        return (
            self._margin * 2
            + max_count * self._box_size
            + (max_count - 1) * self._box_gap
        )

    def _update_scrollbars(self):
        viewport = self.viewport().size()
        content_width = self._content_width()
        content_height = self._content_height()

        h_bar = self.horizontalScrollBar()
        v_bar = self.verticalScrollBar()

        h_max = max(0, content_width - viewport.width())
        v_max = max(0, content_height - viewport.height())

        h_bar.setRange(0, h_max)
        h_bar.setPageStep(viewport.width())
        v_bar.setRange(0, v_max)
        v_bar.setPageStep(viewport.height())

    def _color_for_column(self, idx):
        palette = [
            QtGui.QColor(255, 99, 71),
            QtGui.QColor(46, 204, 113),
            QtGui.QColor(52, 152, 219),
            QtGui.QColor(255, 193, 7),
            QtGui.QColor(155, 89, 182),
        ]
        color = palette[idx % len(palette)]
        if idx % 2 == 1:
            color = color.lighter(115)
        return color

    def _rebuild_layout(self):
        self._positions = []
        self._edges = []
        if not self._levels:
            return

        x_base = self._margin
        prev_positions = []
        prev_count = 0

        for col_idx, count in enumerate(self._levels):
            col_positions = []
            x = x_base + col_idx * (self._box_size + self._col_gap)
            if col_idx == 0:
                for idx in range(count):
                    y = self._margin + idx * (self._box_size + self._box_gap)
                    col_positions.append((x, y))
            else:
                for parent_idx in range(count):
                    start = int(round(parent_idx * prev_count / count))
                    end = int(round((parent_idx + 1) * prev_count / count))
                    if end <= start:
                        end = min(prev_count, start + 1)
                    group = prev_positions[start:end]
                    if not group:
                        continue
                    y_values = [pos[1] for pos in group]
                    y = sum(y_values) / len(y_values)
                    col_positions.append((x, y))
                    for child_idx in range(start, end):
                        child_x, child_y = prev_positions[child_idx]
                        self._edges.append(
                            (
                                child_x + self._box_size,
                                child_y + self._box_size * 0.5,
                                x,
                                y + self._box_size * 0.5,
                            )
                        )

            self._positions.append(col_positions)
            prev_positions = col_positions
            prev_count = len(col_positions)

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QtGui.QPainter(self.viewport())
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.viewport().rect(), self.palette().base())

        if not self._levels or not self._positions:
            painter.setPen(QtGui.QColor(60, 60, 60))
            painter.drawText(self._margin, self._margin + 16, "No Avg tree data.")
            painter.end()
            return

        x_offset = self.horizontalScrollBar().value()
        y_offset = self.verticalScrollBar().value()

        pen = QtGui.QPen(self._line_color)
        pen.setWidth(2)
        painter.setPen(pen)
        for x1, y1, x2, y2 in self._edges:
            painter.drawLine(
                QtCore.QPointF(x1 - x_offset, y1 - y_offset),
                QtCore.QPointF(x2 - x_offset, y2 - y_offset),
            )

        base_font = painter.font()
        font = QtGui.QFont(base_font)
        font.setPixelSize(max(8, int(self._box_size * 0.7)))
        painter.setFont(font)

        for col_idx, col_positions in enumerate(self._positions):
            fill_color = self._color_for_column(col_idx)
            border_color = fill_color.darker(140)
            painter.setPen(QtGui.QPen(border_color, 1))
            for x, y in col_positions:
                rect = QtCore.QRectF(
                    x - x_offset,
                    y - y_offset,
                    self._box_size,
                    self._box_size,
                )
                painter.fillRect(rect, fill_color)
                painter.drawRect(rect)
                painter.setPen(self._text_color)
                painter.drawText(
                    rect,
                    QtCore.Qt.AlignmentFlag.AlignCenter,
                    "0",
                )
                painter.setPen(QtGui.QPen(border_color, 1))

        painter.end()


class AvgTab(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._levels = [16, 8, 4, 2, 1]

        layout = QtWidgets.QVBoxLayout(self)
        header = QtWidgets.QLabel("Avg tree view (levels controlled by data structure).")
        header.setStyleSheet("color: #555;")
        layout.addWidget(header)

        self._canvas = AvgTreeCanvas()
        self._canvas.set_levels(self._levels)
        layout.addWidget(self._canvas, 1)

        levels_label = QtWidgets.QLabel(
            "Levels: " + " → ".join(str(value) for value in self._levels)
        )
        levels_label.setStyleSheet("color: #555;")
        layout.addWidget(levels_label)

class SessionFileWatcher(QtCore.QObject):
    session_updated = QtCore.Signal(dict)
    error = QtCore.Signal(str)

    def __init__(self, path, parent=None):
        super().__init__(parent)
        self._path = ""
        self._watcher = QtCore.QFileSystemWatcher(self)
        self._watcher.fileChanged.connect(self._schedule_reload)
        self._watcher.directoryChanged.connect(self._schedule_reload)

        self._debounce = QtCore.QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(300)
        self._debounce.timeout.connect(self._reload)

        self.set_path(path)

    def _watch_paths(self):
        if self._watcher.files():
            self._watcher.removePaths(self._watcher.files())
        if self._watcher.directories():
            self._watcher.removePaths(self._watcher.directories())

        if not self._path:
            return
        directory = os.path.dirname(self._path) or "."
        self._watcher.addPath(directory)
        if os.path.exists(self._path):
            self._watcher.addPath(self._path)

    def _schedule_reload(self, *_args):
        self._debounce.start()

    def _reload(self):
        self._watch_paths()
        if not self._path:
            return
        try:
            session = load_session(self._path)
        except (OSError, json.JSONDecodeError) as exc:
            self.error.emit(str(exc))
            return
        self.session_updated.emit(session)

    def set_path(self, path):
        self._path = os.path.abspath(path) if path else ""
        self._watch_paths()


class SessionViewer(QtWidgets.QMainWindow):
    def __init__(self, session, session_path, log_dir, default_paths, parent=None):
        super().__init__(parent)
        self._session_path = os.path.abspath(session_path) if session_path else ""
        self._default_paths = [p for p in (default_paths or []) if p]
        self._log_dir = os.path.abspath(log_dir) if log_dir else ""
        self._current_session_path = self._session_path
        self._bit_similarity_settings = None

        self._tabs = QtWidgets.QTabWidget()
        self._log_list = QtWidgets.QListWidget()
        self._log_list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self._log_list.currentItemChanged.connect(self._on_log_selected)

        splitter = QtWidgets.QSplitter()
        splitter.addWidget(self._log_list)
        splitter.addWidget(self._tabs)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setCollapsible(0, False)
        splitter.setSizes([240, 960])
        self.setCentralWidget(splitter)

        self.setStatusBar(QtWidgets.QStatusBar())
        self.setWindowTitle("RSA Session Viewer")
        self.resize(1280, 760)

        self._file_watcher = SessionFileWatcher(self._current_session_path, self)
        self._file_watcher.session_updated.connect(self.reload_session)
        self._file_watcher.error.connect(lambda msg: self.set_status(f"Reload failed: {msg}"))

        self._dir_watcher = QtCore.QFileSystemWatcher(self)
        self._dir_watcher.directoryChanged.connect(self.refresh_log_list)

        self.refresh_log_list(select_path=self._current_session_path, reload_selected=False)
        self.reload_session(session)

    def refresh_log_list(self, *_args, select_path=None, reload_selected=True):
        log_paths = collect_log_paths(self._default_paths, self._log_dir)
        if self._log_dir and os.path.isdir(self._log_dir):
            if self._log_dir not in self._dir_watcher.directories():
                self._dir_watcher.addPath(self._log_dir)

        current = select_path or self._current_session_path
        keep_current = (
            select_path is None
            and current
            and current == self._current_session_path
            and current in log_paths
        )
        self._log_list.blockSignals(True)
        self._log_list.clear()
        for path in log_paths:
            item = QtWidgets.QListWidgetItem(os.path.basename(path))
            item.setToolTip(path)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, path)
            self._log_list.addItem(item)
        selected_item = None
        if log_paths:
            selected_path = current if current in log_paths else log_paths[0]
            for idx in range(self._log_list.count()):
                item = self._log_list.item(idx)
                if item.data(QtCore.Qt.ItemDataRole.UserRole) == selected_path:
                    selected_item = item
                    self._log_list.setCurrentItem(item)
                    break
        else:
            self._current_session_path = ""
        self._log_list.blockSignals(False)

        if reload_selected and not keep_current and selected_item is not None:
            self._on_log_selected(selected_item, None)

    def _on_log_selected(self, current, _previous):
        if current is None:
            return
        path = current.data(QtCore.Qt.ItemDataRole.UserRole)
        if not path:
            return
        self._current_session_path = path
        self._file_watcher.set_path(path)
        try:
            session = load_session(path)
        except (OSError, json.JSONDecodeError) as exc:
            self.set_status(f"Failed to load {os.path.basename(path)}: {exc}")
            return
        self.reload_session(session)

    def reload_session(self, session):
        self._capture_bit_similarity_settings()
        current_index = self._tabs.currentIndex() if self._tabs.count() else 0
        self._tabs.clear()
        self._tabs.addTab(self._build_summary_tab(session), "Summary")
        self._tabs.addTab(self._build_candidates_tab(session), "Candidates")
        self._tabs.addTab(self._build_bit_similarity_tab(session), "Bit Similarity")
        self._tabs.addTab(self._build_bit_true_timeline_tab(session), "Bit True Timeline")
        self._tabs.addTab(self._build_avalanche_tab(session), "Avalanche")
        self._tabs.addTab(self._build_beam_vs_r_tab(session), "Beam vs R")
        if self._tabs.count():
            self._tabs.setCurrentIndex(min(current_index, self._tabs.count() - 1))
        if self._current_session_path:
            self.set_status(
                f"Loaded {os.path.basename(self._current_session_path)} at {datetime.now().isoformat(sep=' ', timespec='seconds')}"
            )

    def set_status(self, message):
        status = self.statusBar()
        if status:
            status.showMessage(message, 5000)

    def _build_summary_tab(self, session):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)

        rows = []
        cli = session.get("cli", {})
        rows.append(("Started", format_unix_ms(session.get("started_unix_ms"))))
        rows.append(("Finished", format_unix_ms(session.get("finished_unix_ms"))))
        if session.get("started_unix_ms") and session.get("finished_unix_ms"):
            duration = session["finished_unix_ms"] - session["started_unix_ms"]
            rows.append(("Duration (ms)", str(duration)))
        rows.append(("Bits", str(cli.get("bits", ""))))
        rows.append(("Config", cli.get("config_path", "")))
        rows.append(("Seed", str(cli.get("seed", ""))))
        rows.append(("Crypto RNG", str(cli.get("crypto_rng", ""))))
        rows.append(("Tests", str(cli.get("tests", ""))))
        rows.append(("Export", str(cli.get("export", ""))))
        rows.append(("Shift", str(cli.get("shift", ""))))
        rows.append(("Errors", str(len(session.get("errors", [])))))

        table = QtWidgets.QTableWidget(len(rows), 2)
        table.setHorizontalHeaderLabels(["Metric", "Value"])
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        for row_idx, (key, value) in enumerate(rows):
            table.setItem(row_idx, 0, QtWidgets.QTableWidgetItem(key))
            table.setItem(row_idx, 1, QtWidgets.QTableWidgetItem(value))
        table.resizeColumnsToContents()
        layout.addWidget(table)

        feature_table = QtWidgets.QTableWidget(0, 4)
        feature_table.setHorizontalHeaderLabels(
            ["Feature", "Enabled", "Duration (ms)", "Notes"]
        )
        feature_table.verticalHeader().setVisible(False)
        feature_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)

        for feature in session.get("features", []):
            row = feature_table.rowCount()
            feature_table.insertRow(row)
            feature_table.setItem(
                row, 0, QtWidgets.QTableWidgetItem(feature.get("name", ""))
            )
            feature_table.setItem(
                row, 1, QtWidgets.QTableWidgetItem(str(feature.get("enabled", "")))
            )
            feature_table.setItem(
                row, 2, QtWidgets.QTableWidgetItem(str(feature.get("duration_ms", "")))
            )
            notes = "; ".join(feature.get("notes", []) or [])
            feature_table.setItem(row, 3, QtWidgets.QTableWidgetItem(notes))

        feature_table.resizeColumnsToContents()
        layout.addWidget(QtWidgets.QLabel("Feature Summary"))
        layout.addWidget(feature_table)

        return widget

    def _build_candidates_tab(self, session):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        rows = flatten_candidate_batches(session)
        if not rows:
            layout.addWidget(QtWidgets.QLabel("No r-candidate batches recorded."))
            return widget

        table = QtWidgets.QTableWidget(len(rows), 6)
        table.setHorizontalHeaderLabels(
            ["Context", "Mode", "Index", "r", "Bits", "Factors"]
        )
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)

        for row_idx, row in enumerate(rows):
            table.setItem(row_idx, 0, QtWidgets.QTableWidgetItem(row["context"]))
            table.setItem(row_idx, 1, QtWidgets.QTableWidgetItem(row["mode"]))
            table.setItem(row_idx, 2, QtWidgets.QTableWidgetItem(str(row["index"])))
            table.setItem(row_idx, 3, QtWidgets.QTableWidgetItem(row["r"]))
            table.setItem(row_idx, 4, QtWidgets.QTableWidgetItem(str(row["r_bits"])))
            table.setItem(row_idx, 5, QtWidgets.QTableWidgetItem(row["factors"]))

        table.resizeColumnsToContents()
        layout.addWidget(table)
        return widget

    def _build_bit_similarity_tab(self, session):
        feature = get_feature(session, "information_sufficiency")
        bit_similarity = None
        if feature:
            stats = feature.get("stats", {})
            bit_similarity = stats.get("bit_similarity")

        if not bit_similarity:
            widget = QtWidgets.QWidget()
            layout = QtWidgets.QVBoxLayout(widget)
            layout.addWidget(
                QtWidgets.QLabel(
                    "Bit similarity data not found. Run analysis with --tests to populate it."
                )
            )
            return widget

        tab = BitSimilarityTab(bit_similarity)
        if self._bit_similarity_settings:
            tab.apply_settings(self._bit_similarity_settings)
        return tab

    def _build_bit_true_timeline_tab(self, session):
        feature = get_feature(session, "information_sufficiency")
        timeline = None
        if feature:
            stats = feature.get("stats", {})
            timeline = stats.get("bit_true_timeline")

        if not timeline:
            widget = QtWidgets.QWidget()
            layout = QtWidgets.QVBoxLayout(widget)
            layout.addWidget(
                QtWidgets.QLabel(
                    "Bit true timeline data not found. Run analysis with --tests to populate it."
                )
            )
            return widget

        return BitTrueTimelineTab(timeline)

    def _build_avalanche_tab(self, session):
        feature = get_feature(session, "information_sufficiency")
        avalanche = None
        if feature:
            stats = feature.get("stats", {})
            avalanche = stats.get("avalanche_tree")

        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)

        if not avalanche:
            layout.addWidget(
                QtWidgets.QLabel(
                    "Avalanche data not found. Run analysis with --tests to populate it."
                )
            )
            return widget

        biases = avalanche.get("biases") or []
        message_bits = avalanche.get("message_bits") or []
        bit_width = avalanche.get("bit_width", len(message_bits))
        unique_messages = avalanche.get("unique_messages", 0)
        ones_count = sum(1 for bit in message_bits if bit)

        if biases:
            bias_min = min(biases)
            bias_max = max(biases)
            bias_mean = sum(biases) / len(biases)
        else:
            bias_min = ""
            bias_max = ""
            bias_mean = ""

        rows = [
            ("Bit Width", str(bit_width)),
            ("Unique Messages", str(unique_messages)),
            ("Ones Count", str(ones_count)),
            ("Bias Min", f"{bias_min:.4f}" if isinstance(bias_min, float) else ""),
            ("Bias Mean", f"{bias_mean:.4f}" if isinstance(bias_mean, float) else ""),
            ("Bias Max", f"{bias_max:.4f}" if isinstance(bias_max, float) else ""),
        ]

        table = QtWidgets.QTableWidget(len(rows), 2)
        table.setHorizontalHeaderLabels(["Metric", "Value"])
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        for row_idx, (key, value) in enumerate(rows):
            table.setItem(row_idx, 0, QtWidgets.QTableWidgetItem(key))
            table.setItem(row_idx, 1, QtWidgets.QTableWidgetItem(value))
        table.resizeColumnsToContents()
        layout.addWidget(table)

        return widget

    def _build_beam_vs_r_tab(self, session):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)

        batches = session.get("r_candidate_accuracy_batches", []) or []
        if not batches:
            layout.addWidget(
                QtWidgets.QLabel(
                    "Beam vs r-candidate data not found. Run analysis batches to populate it."
                )
            )
            return widget

        rows = []
        mean_pairs = []
        max_pairs = []
        near_100 = 0
        for idx, batch in enumerate(batches, start=1):
            candidates = batch.get("candidates", []) or []
            accuracies = [
                entry.get("accuracy_pct")
                for entry in candidates
                if isinstance(entry.get("accuracy_pct"), (int, float))
            ]
            mean_acc, stddev_acc, min_acc, max_acc = compute_basic_stats(accuracies)
            beam_match = batch.get("beam_match_pct")
            beam_ones = batch.get("beam_ones_match_pct")
            beam_score = batch.get("beam_score")
            beam_bits = batch.get("beam_bit_width")
            if isinstance(beam_match, (int, float)) and isinstance(mean_acc, (int, float)):
                mean_pairs.append((float(beam_match), float(mean_acc)))
            if isinstance(beam_match, (int, float)) and isinstance(max_acc, (int, float)):
                max_pairs.append((float(beam_match), float(max_acc)))
            if isinstance(beam_match, (int, float)) and beam_match >= 99.0:
                near_100 += 1

            rows.append(
                {
                    "batch": batch.get("context", f"batch_{idx}"),
                    "beam_match": beam_match,
                    "beam_ones": beam_ones,
                    "beam_score": beam_score,
                    "beam_bits": beam_bits,
                    "r_mean": mean_acc,
                    "r_max": max_acc,
                    "r_min": min_acc,
                    "r_stddev": stddev_acc,
                    "candidate_count": len(candidates),
                }
            )

        corr_mean = pearson_corr(mean_pairs)
        corr_max = pearson_corr(max_pairs)
        summary_lines = [
            f"Batches with beam match >= 99%: {near_100} / {len(batches)}"
        ]
        if corr_mean is not None:
            summary_lines.append(f"Correlation (beam match vs r mean): {corr_mean:.3f}")
        else:
            summary_lines.append("Correlation (beam match vs r mean): N/A")
        if corr_max is not None:
            summary_lines.append(f"Correlation (beam match vs r max): {corr_max:.3f}")
        else:
            summary_lines.append("Correlation (beam match vs r max): N/A")

        layout.addWidget(QtWidgets.QLabel("\n".join(summary_lines)))

        table = QtWidgets.QTableWidget(len(rows), 10)
        table.setHorizontalHeaderLabels(
            [
                "Batch",
                "Beam Match %",
                "Beam Ones %",
                "Beam Score",
                "Beam Bits",
                "R Mean %",
                "R Max %",
                "R Min %",
                "R Stddev",
                "Candidates",
            ]
        )
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)

        for row_idx, row in enumerate(rows):
            table.setItem(row_idx, 0, QtWidgets.QTableWidgetItem(str(row["batch"])))
            table.setItem(
                row_idx,
                1,
                QtWidgets.QTableWidgetItem(
                    f"{row['beam_match']:.2f}" if isinstance(row["beam_match"], (int, float)) else ""
                ),
            )
            table.setItem(
                row_idx,
                2,
                QtWidgets.QTableWidgetItem(
                    f"{row['beam_ones']:.2f}" if isinstance(row["beam_ones"], (int, float)) else ""
                ),
            )
            table.setItem(
                row_idx,
                3,
                QtWidgets.QTableWidgetItem(
                    f"{row['beam_score']:.4f}" if isinstance(row["beam_score"], (int, float)) else ""
                ),
            )
            table.setItem(
                row_idx,
                4,
                QtWidgets.QTableWidgetItem(
                    str(row["beam_bits"]) if row["beam_bits"] is not None else ""
                ),
            )
            table.setItem(
                row_idx,
                5,
                QtWidgets.QTableWidgetItem(
                    f"{row['r_mean']:.2f}" if isinstance(row["r_mean"], (int, float)) else ""
                ),
            )
            table.setItem(
                row_idx,
                6,
                QtWidgets.QTableWidgetItem(
                    f"{row['r_max']:.2f}" if isinstance(row["r_max"], (int, float)) else ""
                ),
            )
            table.setItem(
                row_idx,
                7,
                QtWidgets.QTableWidgetItem(
                    f"{row['r_min']:.2f}" if isinstance(row["r_min"], (int, float)) else ""
                ),
            )
            table.setItem(
                row_idx,
                8,
                QtWidgets.QTableWidgetItem(
                    f"{row['r_stddev']:.2f}" if isinstance(row["r_stddev"], (int, float)) else ""
                ),
            )
            table.setItem(
                row_idx,
                9,
                QtWidgets.QTableWidgetItem(str(row["candidate_count"])),
            )

        table.resizeColumnsToContents()
        layout.addWidget(table)
        return widget

    def _capture_bit_similarity_settings(self):
        for idx in range(self._tabs.count()):
            if self._tabs.tabText(idx) != "Bit Similarity":
                continue
            tab = self._tabs.widget(idx)
            if hasattr(tab, "get_settings"):
                self._bit_similarity_settings = tab.get_settings()
            break


def main():
    parser = argparse.ArgumentParser(description="Qt6 viewer for session.json analytics")
    parser.add_argument(
        "session",
        nargs="?",
        default="session.log",
        help="Path to session log (default: session.log)",
    )
    parser.add_argument(
        "--log-dir",
        default="logs",
        help="Directory containing session logs (default: ./logs)",
    )
    args = parser.parse_args()
    session_path = args.session
    log_dir = args.log_dir

    default_candidates = [session_path]
    if session_path == "session.log":
        default_candidates.append("session.json")
    log_paths = collect_log_paths(default_candidates, log_dir)
    if log_paths:
        session_path = log_paths[0]

    try:
        session = load_session(session_path)
    except FileNotFoundError:
        session = empty_session()
    except json.JSONDecodeError as exc:
        print(f"failed to parse session file: {exc}", file=sys.stderr)
        session = empty_session()

    app = QtWidgets.QApplication(sys.argv)
    viewer = SessionViewer(session, session_path, log_dir, default_candidates)

    viewer.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
