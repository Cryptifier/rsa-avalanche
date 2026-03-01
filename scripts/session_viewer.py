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


def empty_session():
    return {
        "started_unix_ms": None,
        "finished_unix_ms": None,
        "cli": {},
        "features": [],
        "r_candidate_batches": [],
        "errors": [],
    }


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
        self._entries = []
        self._bit_width = 0
        self._original_bits = []
        self._start_index = 0
        self._display_count = 0
        self._show_all = True
        self._bit_cache = {}

        self._margin = 8
        self._label_width = 90
        self._bit_size = 10
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

    def _candidate_bits(self, entry):
        cache_key = entry.get("_orig_index", id(entry))
        cached = self._bit_cache.get(cache_key)
        if cached is not None:
            return cached
        bits = hex_to_bits_le(entry.get("candidate_hex", ""), self._bit_width)
        self._bit_cache[cache_key] = bits
        return bits

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
            display_idx = entry.get("_orig_index", entry_idx)
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
                f"#{display_idx} | r={r_value} | match={match_pct:.2f}% | matching bits={matching_bits}",
            )

            label_x = self._margin - x_offset
            painter.drawText(label_x, bits_top + self._bit_size, "Original")
            painter.drawText(
                label_x, bits_top + self._bit_size + self._bit_spacing + 8 + self._bit_size, "Candidate"
            )

            original_bits = self._original_bits
            candidate_bits = self._candidate_bits(entry)
            max_bits = self._bit_width
            small_font = QtGui.QFont(base_font)
            small_font.setPixelSize(max(4, int(self._bit_size * 0.5)))
            painter.setFont(small_font)

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

        control_row.addWidget(QtWidgets.QLabel("Sort:"))
        control_row.addWidget(self._sort_combo)
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

        self._show_all.stateChanged.connect(self._toggle_show_all)

        legend = QtWidgets.QLabel("Green = match, Red = mismatch. LSB-first bit order.")
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

        self._apply_sort()
        if self._pending_settings:
            pending = self._pending_settings
            self._pending_settings = None
            self.apply_settings(pending)

    def _apply_sort(self):
        sort_mode = self._sort_combo.currentText()
        entries = list(self._entries_raw)
        if sort_mode == "Match % (Descending)":
            entries.sort(key=lambda e: e.get("match_pct", 0.0), reverse=True)
        elif sort_mode == "Match % (Ascending)":
            entries.sort(key=lambda e: e.get("match_pct", 0.0))
        self._entries = entries
        self._sync_ranges()
        self._canvas.set_data(self._entries, self._bit_width, self._original_hex)
        self._rebuild_rows()

    def _on_sort_changed(self, _idx):
        self._apply_sort()

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

        self.refresh_log_list(select_path=self._current_session_path)
        self.reload_session(session)

    def refresh_log_list(self, *_args, select_path=None):
        log_paths = collect_log_paths(self._default_paths, self._log_dir)
        if self._log_dir and os.path.isdir(self._log_dir):
            if self._log_dir not in self._dir_watcher.directories():
                self._dir_watcher.addPath(self._log_dir)

        current = select_path or self._current_session_path
        self._log_list.blockSignals(True)
        self._log_list.clear()
        for path in log_paths:
            item = QtWidgets.QListWidgetItem(os.path.basename(path))
            item.setToolTip(path)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, path)
            self._log_list.addItem(item)
        self._log_list.blockSignals(False)

        if log_paths:
            selected_path = current if current in log_paths else log_paths[0]
            for idx in range(self._log_list.count()):
                item = self._log_list.item(idx)
                if item.data(QtCore.Qt.ItemDataRole.UserRole) == selected_path:
                    self._log_list.setCurrentItem(item)
                    break
        else:
            self._current_session_path = ""

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
