#!/usr/bin/env python3
import argparse
import json
import os
import sys
from datetime import datetime

from PySide6 import QtCore, QtGui, QtWidgets


def load_session(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


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


class BitSimilarityCanvas(QtWidgets.QAbstractScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._entries = []
        self._bit_width = 0
        self._original_bits = []
        self._start_index = 0
        self._display_count = 0
        self._show_all = True
        self._bit_cache = {}

        self._margin = 8
        self._label_width = 90
        self._bit_size = 6
        self._bit_spacing = 1
        self._row_spacing = 8
        self._header_height = 18
        self._row_padding = 6

        self._match_color = QtGui.QColor(46, 160, 67)
        self._mismatch_color = QtGui.QColor(220, 72, 72)
        self._text_color = QtGui.QColor(40, 40, 40)

        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)

    def set_data(self, entries, bit_width, original_hex):
        self._entries = list(entries)
        self._bit_width = int(bit_width or 0)
        self._original_bits = hex_to_bits_le(original_hex, self._bit_width)
        self._bit_cache.clear()
        self._start_index = 0
        self._display_count = len(self._entries)
        self._show_all = True
        self._update_scrollbars()
        self.viewport().update()

    def set_view(self, start_index, count, show_all):
        self._show_all = bool(show_all)
        total = len(self._entries)
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
        self._update_scrollbars()
        self.viewport().update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_scrollbars()

    def _row_height(self):
        bit_rows_height = 2 * self._bit_size + self._bit_spacing + 8
        return self._header_height + bit_rows_height + self._row_padding

    def _content_width(self):
        bits_width = self._bit_width * (self._bit_size + self._bit_spacing)
        return self._margin * 2 + self._label_width + bits_width

    def _content_height(self):
        rows = self._display_count
        if rows == 0:
            return 0
        return self._margin * 2 + rows * (self._row_height() + self._row_spacing)

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

    def _candidate_bits(self, entry_idx):
        cached = self._bit_cache.get(entry_idx)
        if cached is not None:
            return cached
        entry = self._entries[entry_idx]
        bits = hex_to_bits_le(entry.get("candidate_hex", ""), self._bit_width)
        self._bit_cache[entry_idx] = bits
        return bits

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QtGui.QPainter(self.viewport())
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, False)
        painter.fillRect(self.viewport().rect(), self.palette().base())

        if self._display_count == 0:
            painter.setPen(self._text_color)
            painter.drawText(self._margin, self._margin + 16, "No bit similarity entries available.")
            painter.end()
            return

        x_offset = self.horizontalScrollBar().value()
        y_offset = self.verticalScrollBar().value()
        row_height = self._row_height() + self._row_spacing

        first_row = max(0, y_offset // row_height)
        last_row = min(
            self._display_count,
            (y_offset + self.viewport().height()) // row_height + 1,
        )

        for row in range(first_row, last_row):
            entry_idx = self._start_index + row
            if entry_idx >= len(self._entries):
                break

            entry = self._entries[entry_idx]
            row_top = self._margin + row * row_height - y_offset
            header_y = row_top + self._header_height - 4
            bits_top = row_top + self._header_height

            painter.setPen(self._text_color)
            r_value = entry.get("r", "")
            match_pct = entry.get("match_pct", 0.0)
            matching_bits = entry.get("matching_bits", 0)
            painter.drawText(
                self._margin - x_offset,
                header_y,
                f"#{entry_idx} | r={r_value} | match={match_pct:.2f}% | matching bits={matching_bits}",
            )

            label_x = self._margin - x_offset
            painter.drawText(label_x, bits_top + self._bit_size, "Original")
            painter.drawText(
                label_x, bits_top + self._bit_size + self._bit_spacing + 8 + self._bit_size, "Candidate"
            )

            original_bits = self._original_bits
            candidate_bits = self._candidate_bits(entry_idx)
            max_bits = self._bit_width

            for bit_idx in range(max_bits):
                orig_bit = original_bits[bit_idx] if bit_idx < len(original_bits) else False
                cand_bit = candidate_bits[bit_idx] if bit_idx < len(candidate_bits) else False
                matches = orig_bit == cand_bit
                base = self._match_color if matches else self._mismatch_color

                x = (
                    self._margin
                    + self._label_width
                    + bit_idx * (self._bit_size + self._bit_spacing)
                    - x_offset
                )
                y1 = bits_top
                y2 = bits_top + self._bit_size + self._bit_spacing + 8

                color1 = base.lighter(130) if not orig_bit else base
                color2 = base.lighter(130) if not cand_bit else base
                painter.fillRect(x, y1, self._bit_size, self._bit_size, color1)
                painter.fillRect(x, y2, self._bit_size, self._bit_size, color2)

                text1 = "1" if orig_bit else "0"
                text2 = "1" if cand_bit else "0"
                text_color1 = QtGui.QColor(255, 255, 255) if orig_bit else QtGui.QColor(0, 0, 0)
                text_color2 = QtGui.QColor(255, 255, 255) if cand_bit else QtGui.QColor(0, 0, 0)

                painter.setPen(text_color1)
                painter.drawText(
                    QtCore.QRectF(x, y1, self._bit_size, self._bit_size),
                    QtCore.Qt.AlignmentFlag.AlignCenter,
                    text1,
                )
                painter.setPen(text_color2)
                painter.drawText(
                    QtCore.QRectF(x, y2, self._bit_size, self._bit_size),
                    QtCore.Qt.AlignmentFlag.AlignCenter,
                    text2,
                )

        painter.end()


class BitSimilarityTab(QtWidgets.QWidget):
    def __init__(self, bit_similarity=None, parent=None):
        super().__init__(parent)
        self._entries = []
        self._bit_width = 0
        self._original_hex = ""
        self._bit_order = "lsb0"

        layout = QtWidgets.QVBoxLayout(self)

        control_row = QtWidgets.QHBoxLayout()
        self._start_spin = QtWidgets.QSpinBox()
        self._start_spin.setRange(0, 0)
        self._start_spin.setValue(0)
        self._start_spin.valueChanged.connect(self._rebuild_rows)

        self._rows_spin = QtWidgets.QSpinBox()
        self._rows_spin.setRange(1, 1)
        self._rows_spin.setValue(1)
        self._rows_spin.valueChanged.connect(self._rebuild_rows)

        self._show_all = QtWidgets.QCheckBox("Show all rows")
        self._show_all.stateChanged.connect(self._toggle_show_all)
        self._show_all.setChecked(True)

        control_row.addWidget(QtWidgets.QLabel("Start index:"))
        control_row.addWidget(self._start_spin)
        control_row.addWidget(QtWidgets.QLabel("Rows:"))
        control_row.addWidget(self._rows_spin)
        control_row.addWidget(self._show_all)
        control_row.addStretch(1)
        layout.addLayout(control_row)

        self._info_label = QtWidgets.QLabel("")
        layout.addWidget(self._info_label)

        self._canvas = BitSimilarityCanvas()
        layout.addWidget(self._canvas, 1)

        legend = QtWidgets.QLabel("Green = match, Red = mismatch. LSB-first bit order.")
        legend.setStyleSheet("color: #555;")
        layout.addWidget(legend)

        if bit_similarity is not None:
            self.set_data(bit_similarity)

    def set_data(self, bit_similarity):
        bit_similarity = bit_similarity or {}
        self._entries = list(bit_similarity.get("candidates", []))
        self._bit_width = int(bit_similarity.get("bit_width", 0) or 0)
        self._original_hex = bit_similarity.get("original_hex", "")
        self._bit_order = bit_similarity.get("bit_order", "lsb0")

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

        self._canvas.set_data(self._entries, self._bit_width, self._original_hex)
        self._toggle_show_all(self._show_all.isChecked())

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


class SessionFileWatcher(QtCore.QObject):
    session_updated = QtCore.Signal(dict)
    error = QtCore.Signal(str)

    def __init__(self, path, parent=None):
        super().__init__(parent)
        self._path = os.path.abspath(path)
        self._watcher = QtCore.QFileSystemWatcher(self)
        self._watcher.fileChanged.connect(self._schedule_reload)
        self._watcher.directoryChanged.connect(self._schedule_reload)

        self._debounce = QtCore.QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(300)
        self._debounce.timeout.connect(self._reload)

        self._watch_paths()

    def _watch_paths(self):
        if self._watcher.files():
            self._watcher.removePaths(self._watcher.files())
        if self._watcher.directories():
            self._watcher.removePaths(self._watcher.directories())

        directory = os.path.dirname(self._path) or "."
        self._watcher.addPath(directory)
        if os.path.exists(self._path):
            self._watcher.addPath(self._path)

    def _schedule_reload(self, *_args):
        self._debounce.start()

    def _reload(self):
        self._watch_paths()
        try:
            session = load_session(self._path)
        except (OSError, json.JSONDecodeError) as exc:
            self.error.emit(str(exc))
            return
        self.session_updated.emit(session)


class SessionViewer(QtWidgets.QMainWindow):
    def __init__(self, session, session_path, parent=None):
        super().__init__(parent)
        self._session_path = session_path
        self._tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(self._tabs)
        self.setStatusBar(QtWidgets.QStatusBar())
        self.setWindowTitle("RSA Session Viewer")
        self.resize(1200, 720)
        self.reload_session(session)

    def reload_session(self, session):
        current_index = self._tabs.currentIndex() if self._tabs.count() else 0
        self._tabs.clear()
        self._tabs.addTab(self._build_summary_tab(session), "Summary")
        self._tabs.addTab(self._build_candidates_tab(session), "Candidates")
        self._tabs.addTab(self._build_bit_similarity_tab(session), "Bit Similarity")
        if self._tabs.count():
            self._tabs.setCurrentIndex(min(current_index, self._tabs.count() - 1))
        self.set_status(
            f"Loaded {os.path.basename(self._session_path)} at {datetime.now().isoformat(sep=' ', timespec='seconds')}"
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

        return BitSimilarityTab(bit_similarity)


def main():
    parser = argparse.ArgumentParser(description="Qt6 viewer for session.json analytics")
    parser.add_argument(
        "session",
        nargs="?",
        default="session.json",
        help="Path to session.json (default: session.json)",
    )
    args = parser.parse_args()
    session_path = args.session

    try:
        session = load_session(session_path)
    except FileNotFoundError:
        print(f"session file not found: {session_path}", file=sys.stderr)
        return 1
    except json.JSONDecodeError as exc:
        print(f"failed to parse session file: {exc}", file=sys.stderr)
        return 1

    app = QtWidgets.QApplication(sys.argv)
    viewer = SessionViewer(session, session_path)

    watcher = SessionFileWatcher(session_path)
    watcher.session_updated.connect(viewer.reload_session)
    watcher.error.connect(lambda msg: viewer.set_status(f"Reload failed: {msg}"))

    viewer.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
