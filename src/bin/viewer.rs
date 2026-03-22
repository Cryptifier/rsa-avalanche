use std::cell::{Cell, RefCell};
use std::collections::HashMap;
#[cfg(not(target_arch = "wasm32"))]
use std::fs::File;
#[cfg(not(target_arch = "wasm32"))]
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::rc::Rc;
#[cfg(not(target_arch = "wasm32"))]
use std::time::{Duration, Instant};
#[cfg(target_arch = "wasm32")]
use web_time::{Duration, Instant};

use eframe::egui;
use egui_extras::{Column, TableBuilder};
use egui_plot::{Plot, PlotPoints, Points};
use serde::Deserialize;
use serde_json::{Map, Value};

#[cfg(target_arch = "wasm32")]
use js_sys::encode_uri_component;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::JsCast;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_futures::{spawn_local, JsFuture};
#[cfg(target_arch = "wasm32")]
use web_sys::{window, Response};

/// Entry point for the egui-based session viewer.
#[cfg(not(target_arch = "wasm32"))]
fn main() -> eframe::Result<()> {
    let args = ViewerArgs::parse();
    let native_options = eframe::NativeOptions::default();
    eframe::run_native(
        "RSA Session Viewer (egui)",
        native_options,
        Box::new(|_cc| Box::new(ViewerApp::new(args))),
    )
}

/// Stub entry point for wasm builds.
#[cfg(target_arch = "wasm32")]
fn main() {}

/// Starts the egui viewer in a web canvas.
///
/// # Parameters
/// - `canvas_id`: DOM id for the canvas element.
///
/// # Returns
/// - `Result<(), JsValue>`: `Ok(())` on success.
///
/// # Expected Output
/// - Attaches the viewer to the target canvas.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub async fn start(canvas_id: &str) -> Result<(), JsValue> {
    let args = ViewerArgs::web_default();
    let app = ViewerApp::new(args);
    let web_options = eframe::WebOptions::default();
    eframe::WebRunner::new()
        .start(canvas_id, web_options, Box::new(|_cc| Box::new(app)))
        .await
        .map_err(|err| JsValue::from_str(&format!("{err:?}")))
}

#[derive(Debug)]
struct ViewerArgs {
    session_path: PathBuf,
    log_dir: PathBuf,
}

impl ViewerArgs {
    fn parse() -> Self {
        let mut session_path = PathBuf::from("session.log");
        let mut log_dir = PathBuf::from("logs");
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--log-dir" => {
                    if let Some(value) = args.next() {
                        log_dir = PathBuf::from(value);
                    }
                }
                "--session" => {
                    if let Some(value) = args.next() {
                        session_path = PathBuf::from(value);
                    }
                }
                _ => {
                    if !arg.starts_with("--") {
                        session_path = PathBuf::from(arg);
                    }
                }
            }
        }
        Self {
            session_path,
            log_dir,
        }
    }

    #[cfg(target_arch = "wasm32")]
    fn web_default() -> Self {
        Self {
            session_path: PathBuf::from("session.log"),
            log_dir: PathBuf::from("logs"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Tab {
    Summary,
    Candidates,
    BitSimilarity,
    BitTrueTimeline,
    Avalanche,
    BeamVsR,
    Bitflow,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BitSimilaritySort {
    Original,
    MatchDesc,
    MatchAsc,
}

impl BitSimilaritySort {
    fn label(&self) -> &'static str {
        match self {
            BitSimilaritySort::Original => "Original order",
            BitSimilaritySort::MatchDesc => "Match % (Descending)",
            BitSimilaritySort::MatchAsc => "Match % (Ascending)",
        }
    }

    fn all() -> [BitSimilaritySort; 3] {
        [
            BitSimilaritySort::Original,
            BitSimilaritySort::MatchDesc,
            BitSimilaritySort::MatchAsc,
        ]
    }
}

#[derive(Debug, Clone, Deserialize)]
struct LogEntry {
    name: String,
    #[serde(default)]
    size: u64,
    #[serde(default)]
    modified_ms: Option<u64>,
}

#[derive(Debug, Default)]
struct PendingUpdates {
    log_entries: Option<Vec<LogEntry>>,
    session_text: Option<String>,
    session_name: Option<String>,
    select_log: Option<String>,
    status: Option<String>,
}

#[derive(Debug)]
struct ViewerApp {
    session: Session,
    session_path: PathBuf,
    log_dir: PathBuf,
    log_entries: Vec<LogEntry>,
    selected_log: Option<String>,
    status: String,
    last_poll: Instant,
    last_scan: Instant,
    #[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
    last_log_fetch: Instant,
    #[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
    last_session_fetch: Instant,
    ndjson_mode: bool,
    offset: u64,
    buffer: String,
    tab: Tab,
    bit_true_bit_idx: usize,
    bitflow_selected: Option<String>,
    bit_similarity_sort: BitSimilaritySort,
    bit_similarity_show_all: bool,
    bit_similarity_hide_shifted: bool,
    bit_similarity_start: usize,
    bit_similarity_rows: usize,
    pending: Rc<RefCell<PendingUpdates>>,
    #[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
    log_request_in_flight: Rc<Cell<bool>>,
    #[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
    session_request_in_flight: Rc<Cell<bool>>,
}

impl ViewerApp {
    fn new(args: ViewerArgs) -> Self {
        let pending = Rc::new(RefCell::new(PendingUpdates::default()));
        let log_request_in_flight = Rc::new(Cell::new(false));
        let session_request_in_flight = Rc::new(Cell::new(false));
        let mut app = Self {
            session: Session::default(),
            session_path: args.session_path,
            log_dir: args.log_dir,
            log_entries: Vec::new(),
            selected_log: None,
            status: String::new(),
            last_poll: Instant::now(),
            last_scan: Instant::now(),
            last_log_fetch: Instant::now(),
            last_session_fetch: Instant::now(),
            ndjson_mode: false,
            offset: 0,
            buffer: String::new(),
            tab: Tab::Summary,
            bit_true_bit_idx: 0,
            bitflow_selected: None,
            bit_similarity_sort: BitSimilaritySort::Original,
            bit_similarity_show_all: true,
            bit_similarity_hide_shifted: true,
            bit_similarity_start: 0,
            bit_similarity_rows: 50,
            pending,
            log_request_in_flight,
            session_request_in_flight,
        };
        app.refresh_logs(true);
        app
    }

    fn refresh_logs(&mut self, select_default: bool) {
        #[cfg(target_arch = "wasm32")]
        {
            self.request_log_list(select_default);
            return;
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.log_entries = collect_log_entries(&self.session_path, &self.log_dir);
            if let Some(selected) = self.selected_log.clone() {
                if !self.log_entries.iter().any(|entry| entry.name == selected) {
                    self.selected_log = None;
                }
            }
            if select_default && self.selected_log.is_none() {
                if let Some(entry) = self.log_entries.first() {
                    let name = entry.name.clone();
                    let _ = self.load_session(&name);
                }
            }
        }
    }

    fn load_session(&mut self, path: &str) -> Result<(), String> {
        #[cfg(target_arch = "wasm32")]
        {
            self.selected_log = Some(path.to_string());
            self.request_session(path);
            return Ok(());
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let path_obj = Path::new(path);
            let (session, ndjson) = load_session_from_path(path_obj)?;
            self.session = session;
            self.ndjson_mode = ndjson;
            self.offset = file_size(path_obj).unwrap_or(0);
            self.buffer.clear();
            self.selected_log = Some(path.to_string());
            self.status = format!("Loaded {}", path_obj.display());
            Ok(())
        }
    }

    fn poll_updates(&mut self) {
        self.apply_pending();
        let now = Instant::now();
        #[cfg(target_arch = "wasm32")]
        {
            if now.duration_since(self.last_log_fetch) > Duration::from_secs(2) {
                self.request_log_list(false);
                self.last_log_fetch = now;
            }
            if let Some(selected) = self.selected_log.clone() {
                if now.duration_since(self.last_session_fetch) > Duration::from_secs(2) {
                    self.request_session(&selected);
                    self.last_session_fetch = now;
                }
            }
            return;
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
        if now.duration_since(self.last_scan) > Duration::from_secs(2) {
            self.refresh_logs(false);
            self.last_scan = now;
        }
        if now.duration_since(self.last_poll) < Duration::from_millis(400) {
            return;
        }
        self.last_poll = now;
        if !self.ndjson_mode {
            return;
        }
        let Some(path) = self.selected_log.clone() else {
            return;
        };
        let updated = self.ingest_tail(&path);
        if updated {
            self.status = format!("Updated {}", path);
        }
        }
    }

    fn apply_pending(&mut self) {
        let mut pending = self.pending.borrow_mut();
        let status = pending.status.take();
        if let Some(entries) = pending.log_entries.take() {
            self.log_entries = entries;
            if let Some(selected) = self.selected_log.clone() {
                if !self.log_entries.iter().any(|entry| entry.name == selected) {
                    self.selected_log = None;
                }
            }
            if self.selected_log.is_none() {
                if let Some(entry) = self.log_entries.first() {
                    pending.select_log = Some(entry.name.clone());
                }
            }
        }
        let select = pending.select_log.take();
        let session_text = pending.session_text.take();
        let session_name = pending.session_name.take();
        drop(pending);

        if let Some(status) = status {
            self.status = status;
        }
        if let Some(select) = select {
            let _ = self.load_session(&select);
        }
        if let Some(text) = session_text {
            match parse_session_from_str(&text) {
                Ok((session, ndjson)) => {
                    self.session = session;
                    self.ndjson_mode = ndjson;
                    if let Some(name) = session_name {
                        self.selected_log = Some(name.clone());
                        self.status = format!("Loaded {}", name);
                    }
                }
                Err(err) => {
                    self.status = format!("Failed to parse session: {err}");
                }
            }
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn ingest_tail(&mut self, path: &str) -> bool {
        let path_obj = Path::new(path);
        let Ok(mut file) = File::open(path_obj) else {
            return false;
        };
        let Ok(size) = file.metadata().map(|meta| meta.len()) else {
            return false;
        };
        if size < self.offset {
            self.offset = 0;
            self.buffer.clear();
            self.session = Session::default();
        }
        if file.seek(SeekFrom::Start(self.offset)).is_err() {
            return false;
        }
        let mut chunk = String::new();
        if file.read_to_string(&mut chunk).is_err() {
            return false;
        }
        if chunk.is_empty() {
            return false;
        }
        self.offset = self.offset.saturating_add(chunk.len() as u64);
        let mut data = String::new();
        data.push_str(&self.buffer);
        data.push_str(&chunk);
        let mut lines = data.lines().collect::<Vec<_>>();
        if !data.ends_with('\n') {
            self.buffer = lines.pop().unwrap_or_default().to_string();
        } else {
            self.buffer.clear();
        }
        let mut updated = false;
        for line in lines {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let Ok(value) = serde_json::from_str::<Value>(trimmed) else {
                continue;
            };
            if let Some(event) = value.as_object() {
                if event.contains_key("event") {
                    apply_event_to_session(&mut self.session, event);
                    updated = true;
                }
            }
        }
        updated
    }

    #[cfg(target_arch = "wasm32")]
    fn request_log_list(&self, select_default: bool) {
        if self.log_request_in_flight.get() {
            return;
        }
        self.log_request_in_flight.set(true);
        let pending = Rc::clone(&self.pending);
        let in_flight = Rc::clone(&self.log_request_in_flight);
        let current = self.selected_log.clone();
        spawn_local(async move {
            let result = fetch_text("/api/logs").await;
            in_flight.set(false);
            let mut pending = pending.borrow_mut();
            match result {
                Ok(text) => match serde_json::from_str::<Vec<LogEntry>>(&text) {
                    Ok(mut entries) => {
                        entries.sort_by(|a, b| {
                            b.modified_ms
                                .unwrap_or(0)
                                .cmp(&a.modified_ms.unwrap_or(0))
                                .then_with(|| a.name.cmp(&b.name))
                        });
                        pending.log_entries = Some(entries.clone());
                        if select_default && current.is_none() {
                            pending.select_log =
                                entries.first().map(|entry| entry.name.clone());
                        } else if let Some(current) = current {
                            if !entries.iter().any(|entry| entry.name == current) {
                                pending.select_log =
                                    entries.first().map(|entry| entry.name.clone());
                            }
                        }
                    }
                    Err(err) => {
                        pending.status = Some(format!("Failed to decode log list: {err}"));
                    }
                },
                Err(err) => {
                    pending.status = Some(format!("Failed to fetch log list: {err:?}"));
                }
            }
        });
    }

    #[cfg(target_arch = "wasm32")]
    fn request_session(&self, name: &str) {
        if self.session_request_in_flight.get() {
            return;
        }
        self.session_request_in_flight.set(true);
        let pending = Rc::clone(&self.pending);
        let in_flight = Rc::clone(&self.session_request_in_flight);
        let name = name.to_string();
        spawn_local(async move {
            let encoded = encode_uri_component(&name)
                .as_string()
                .unwrap_or_else(|| name.clone());
            let url = format!("/api/logs/{encoded}");
            let result = fetch_text(&url).await;
            in_flight.set(false);
            let mut pending = pending.borrow_mut();
            match result {
                Ok(text) => {
                    pending.session_text = Some(text);
                    pending.session_name = Some(name);
                }
                Err(err) => {
                    pending.status = Some(format!("Failed to fetch session: {err:?}"));
                }
            }
        });
    }

    fn draw_summary(&self, ui: &mut egui::Ui) {
        ui.heading("Summary");
        egui::ScrollArea::vertical()
            .id_source("summary_scroll")
            .auto_shrink([false, false])
            .show(ui, |ui| {
                let mut rows = Vec::new();
                rows.push(("Started", format_unix_ms(self.session.started_unix_ms)));
                rows.push(("Finished", format_unix_ms(self.session.finished_unix_ms)));
                if let (Some(start), Some(end)) =
                    (self.session.started_unix_ms, self.session.finished_unix_ms)
                {
                    rows.push(("Duration (ms)", (end - start).to_string()));
                }
                rows.push(("Bits", self.session.cli.bits.to_string()));
                rows.push(("Config", self.session.cli.config_path.clone()));
                rows.push(("Seed", opt_to_string(self.session.cli.seed.map(|v| v as u128))));
                rows.push(("Crypto RNG", self.session.cli.crypto_rng.to_string()));
                rows.push(("Tests", self.session.cli.tests.to_string()));
                rows.push(("Export", self.session.cli.export.to_string()));
                rows.push(("Shift", self.session.cli.shift.to_string()));
                rows.push(("Errors", self.session.errors.len().to_string()));

                ui.push_id("summary_table", |ui| {
                    TableBuilder::new(ui)
                        .striped(true)
                        .column(Column::initial(160.0).resizable(true))
                        .column(Column::remainder())
                        .header(22.0, |mut header| {
                            header.col(|ui| {
                                ui.label("Metric");
                            });
                            header.col(|ui| {
                                ui.label("Value");
                            });
                        })
                        .body(|mut body| {
                            for (metric, value) in rows {
                                body.row(22.0, |mut row| {
                                    row.col(|ui| {
                                        ui.label(metric);
                                    });
                                    row.col(|ui| {
                                        ui.label(value);
                                    });
                                });
                            }
                        });
                });

                ui.add_space(12.0);
                ui.heading("Feature Summary");
                ui.push_id("feature_table", |ui| {
                    TableBuilder::new(ui)
                        .striped(true)
                        .column(Column::initial(180.0).resizable(true))
                        .column(Column::initial(80.0).resizable(true))
                        .column(Column::initial(130.0).resizable(true))
                        .column(Column::remainder())
                        .header(22.0, |mut header| {
                            header.col(|ui| {
                                ui.label("Feature");
                            });
                            header.col(|ui| {
                                ui.label("Enabled");
                            });
                            header.col(|ui| {
                                ui.label("Duration (ms)");
                            });
                            header.col(|ui| {
                                ui.label("Notes");
                            });
                        })
                        .body(|mut body| {
                            for feature in &self.session.features {
                                body.row(22.0, |mut row| {
                                    row.col(|ui| {
                                        ui.label(&feature.name);
                                    });
                                    row.col(|ui| {
                                        ui.label(feature.enabled.to_string());
                                    });
                                    row.col(|ui| {
                                        ui.label(opt_to_string(feature.duration_ms.map(|v| v as u128)));
                                    });
                                    row.col(|ui| {
                                        ui.label(feature.notes.join("; "));
                                    });
                                });
                            }
                        });
                });
            });
    }

    fn draw_candidates(&self, ui: &mut egui::Ui) {
        ui.heading("r Candidate Batches");
        let rows = flatten_candidate_batches(&self.session);
        if rows.is_empty() {
            ui.label("No r-candidate batches recorded.");
            return;
        }
        ui.scope(|ui| {
            let mut style = ui.style().as_ref().clone();
            if let Some(text_style) = style.text_styles.get_mut(&egui::TextStyle::Body) {
                text_style.size = 10.0;
            }
            if let Some(text_style) = style.text_styles.get_mut(&egui::TextStyle::Monospace) {
                text_style.size = 9.0;
            }
            ui.set_style(style);
            ui.push_id("candidates_table", |ui| {
            TableBuilder::new(ui)
                    .striped(true)
                    .column(Column::initial(200.0).resizable(true))
                    .column(Column::initial(110.0).resizable(true))
                    .column(Column::initial(70.0).resizable(true))
                    .column(Column::initial(320.0).resizable(true))
                    .column(Column::initial(90.0).resizable(true))
                    .column(Column::remainder())
                    .header(14.0, |mut header| {
                        header.col(|ui| {
                            ui.label("Context");
                        });
                        header.col(|ui| {
                            ui.label("Mode");
                        });
                        header.col(|ui| {
                            ui.label("Index");
                        });
                        header.col(|ui| {
                            ui.label("r");
                        });
                        header.col(|ui| {
                            ui.label("Bits");
                        });
                        header.col(|ui| {
                            ui.label("Factors");
                        });
                    })
                    .body(|mut body| {
                        for row in rows {
                            body.row(16.0, |mut row_ui| {
                                row_ui.col(|ui| {
                                    ui.label(row.context);
                                });
                                row_ui.col(|ui| {
                                    ui.label(row.mode);
                                });
                                row_ui.col(|ui| {
                                    ui.label(row.index.to_string());
                                });
                                row_ui.col(|ui| {
                                    ui.label(row.r);
                                });
                                row_ui.col(|ui| {
                                    ui.label(row.r_bits.to_string());
                                });
                                row_ui.col(|ui| {
                                    ui.label(row.factors);
                                });
                            });
                        }
                    });
            });
        });
    }

    fn draw_beam_vs_r(&self, ui: &mut egui::Ui) {
        ui.heading("Beam vs R");
        let batches = &self.session.r_candidate_accuracy_batches;
        if batches.is_empty() {
            ui.label("Beam vs r-candidate data not found.");
            return;
        }
        let mut rows = Vec::new();
        let mut mean_pairs = Vec::new();
        let mut max_pairs = Vec::new();
        let mut near_100 = 0usize;
        for (idx, batch) in batches.iter().enumerate() {
            let accuracies = batch
                .candidates
                .iter()
                .map(|entry| entry.accuracy_pct)
                .collect::<Vec<_>>();
            let stats = compute_basic_stats(&accuracies);
            if let (Some(beam_match), Some(mean)) = (batch.beam_match_pct, stats.mean) {
                mean_pairs.push((beam_match, mean));
            }
            if let (Some(beam_match), Some(max_acc)) = (batch.beam_match_pct, stats.max) {
                max_pairs.push((beam_match, max_acc));
            }
            if let Some(beam_match) = batch.beam_match_pct {
                if beam_match >= 99.0 {
                    near_100 += 1;
                }
            }
            rows.push(BeamRow {
                batch: batch.context.clone().unwrap_or_else(|| format!("batch_{}", idx + 1)),
                beam_match: batch.beam_match_pct,
                beam_ones: batch.beam_ones_match_pct,
                beam_score: batch.beam_score,
                beam_bits: batch.beam_bit_width,
                r_mean: stats.mean,
                r_max: stats.max,
                r_min: stats.min,
                r_stddev: stats.stddev,
                candidate_count: batch.candidates.len(),
            });
        }

        let corr_mean = pearson_corr(&mean_pairs);
        let corr_max = pearson_corr(&max_pairs);
        ui.label(format!(
            "Batches with beam match >= 99%: {} / {}",
            near_100,
            batches.len()
        ));
        ui.label(format!(
            "Correlation (beam match vs r mean): {}",
            corr_mean
                .map(|v| format!("{v:.3}"))
                .unwrap_or_else(|| "N/A".to_string())
        ));
        ui.label(format!(
            "Correlation (beam match vs r max): {}",
            corr_max
                .map(|v| format!("{v:.3}"))
                .unwrap_or_else(|| "N/A".to_string())
        ));

        let points_mean = mean_pairs
            .iter()
            .map(|(x, y)| [*x, *y])
            .collect::<PlotPoints>();
        let points_max = max_pairs
            .iter()
            .map(|(x, y)| [*x, *y])
            .collect::<PlotPoints>();
        Plot::new("beam_vs_r_plot")
            .legend(egui_plot::Legend::default())
            .show(ui, |plot_ui| {
                plot_ui.points(Points::new(points_mean).name("Beam vs R mean"));
                plot_ui.points(Points::new(points_max).name("Beam vs R max"));
            });

        ui.push_id("beam_vs_r_table", |ui| {
            TableBuilder::new(ui)
                .columns(Column::auto(), 10)
                .striped(true)
                .header(20.0, |mut header| {
                    header.col(|ui| { ui.label("Batch"); });
                    header.col(|ui| { ui.label("Beam Match %"); });
                    header.col(|ui| { ui.label("Beam Ones %"); });
                    header.col(|ui| { ui.label("Beam Score"); });
                    header.col(|ui| { ui.label("Beam Bits"); });
                    header.col(|ui| { ui.label("R Mean %"); });
                    header.col(|ui| { ui.label("R Max %"); });
                    header.col(|ui| { ui.label("R Min %"); });
                    header.col(|ui| { ui.label("R Stddev"); });
                    header.col(|ui| { ui.label("Candidates"); });
                })
                .body(|mut body| {
                    for row in rows {
                        body.row(20.0, |mut row_ui| {
                            row_ui.col(|ui| { ui.label(row.batch); });
                            row_ui.col(|ui| { ui.label(format_opt_f64(row.beam_match)); });
                            row_ui.col(|ui| { ui.label(format_opt_f64(row.beam_ones)); });
                            row_ui.col(|ui| { ui.label(format_opt_f64(row.beam_score)); });
                            row_ui.col(|ui| { ui.label(opt_to_string(row.beam_bits.map(|v| v as u128))); });
                            row_ui.col(|ui| { ui.label(format_opt_f64(row.r_mean)); });
                            row_ui.col(|ui| { ui.label(format_opt_f64(row.r_max)); });
                            row_ui.col(|ui| { ui.label(format_opt_f64(row.r_min)); });
                            row_ui.col(|ui| { ui.label(format_opt_f64(row.r_stddev)); });
                            row_ui.col(|ui| { ui.label(row.candidate_count.to_string()); });
                        });
                    }
                });
        });
    }

    fn draw_bit_similarity(&mut self, ui: &mut egui::Ui) {
        ui.heading("Bit Similarity");
        let Some(feature) = self.session.feature("information_sufficiency") else {
            ui.label("Bit similarity data not found.");
            return;
        };
        let Some(bit_similarity) = feature.stats.get("bit_similarity") else {
            ui.label("Bit similarity data not found.");
            return;
        };
        let Some(map) = bit_similarity.as_object() else {
            ui.label("Bit similarity data not found.");
            return;
        };
        let data = parse_bit_similarity_data(map);
        if data.entries.is_empty() {
            ui.label("No bit similarity candidates recorded.");
            return;
        }

        ui.horizontal(|ui| {
            ui.label("Sort:");
            egui::ComboBox::from_id_source("bit_similarity_sort")
                .selected_text(self.bit_similarity_sort.label())
                .show_ui(ui, |ui| {
                    for option in BitSimilaritySort::all() {
                        ui.selectable_value(
                            &mut self.bit_similarity_sort,
                            option,
                            option.label(),
                        );
                    }
                });
            ui.checkbox(&mut self.bit_similarity_show_all, "Show all rows");
            ui.checkbox(&mut self.bit_similarity_hide_shifted, "Hide shifted rows");
        });

        let grouped = build_bit_similarity_rows(
            &data.entries,
            self.bit_similarity_hide_shifted,
            self.bit_similarity_sort,
        );
        let total = grouped.len();
        if total == 0 {
            ui.label("No bit similarity entries available.");
            return;
        }
        let default_rows = total.min(50).max(1);
        if self.bit_similarity_rows == 0 {
            self.bit_similarity_rows = default_rows;
        }
        if self.bit_similarity_start >= total {
            self.bit_similarity_start = 0;
        }
        if !self.bit_similarity_show_all && self.bit_similarity_rows > total {
            self.bit_similarity_rows = default_rows;
        }

        ui.horizontal(|ui| {
            ui.label("Start index:");
            ui.add_enabled(
                !self.bit_similarity_show_all,
                egui::DragValue::new(&mut self.bit_similarity_start)
                    .clamp_range(0..=total.saturating_sub(1)),
            );
            ui.label("Rows:");
            ui.add_enabled(
                !self.bit_similarity_show_all,
                egui::DragValue::new(&mut self.bit_similarity_rows)
                    .clamp_range(1..=total),
            );
            ui.add_space(12.0);
            ui.label(format!(
                "Bit width: {} | Shift levels: configured {}, used {}",
                data.bit_width, data.shift_levels_configured, data.shift_levels_used
            ));
        });

        let (start, count) = if self.bit_similarity_show_all {
            (0, total)
        } else {
            let start = self.bit_similarity_start.min(total.saturating_sub(1));
            let count = self.bit_similarity_rows.min(total - start).max(1);
            (start, count)
        };
        let end = (start + count).min(total);
        ui.label(format!(
            "Showing {}-{} of {} entries | bit order: {}",
            start + 1,
            end,
            total,
            data.bit_order
        ));

        let rows = &grouped[start..end];
        let max_shift = rows
            .iter()
            .flat_map(|row| row.entries.iter().map(|entry| entry.shift))
            .max()
            .unwrap_or(0);
        let original_bits = hex_to_bits_le(&data.original_hex, data.bit_width);
        let match_counts = if data.match_counts.len() == data.bit_width
            && !self.bit_similarity_hide_shifted
        {
            data.match_counts.clone()
        } else {
            build_match_counts(&data.entries, &original_bits, data.bit_width)
        };

        ui.add_space(8.0);
        let palette = bit_similarity_palette(ui);
        draw_bit_similarity_canvas(
            ui,
            rows,
            data.bit_width,
            &original_bits,
            &match_counts,
            max_shift,
            &palette,
        );

        ui.add_space(8.0);
        ui.label(
            "Green = matches original, Yellow = matches original + previous, Black = masked bits. LSB-first bit order.",
        );
    }

    fn draw_bit_true_timeline(&mut self, ui: &mut egui::Ui) {
        ui.heading("Bit True Timeline");
        let Some(feature) = self.session.feature("information_sufficiency") else {
            ui.label("Bit true timeline data not found.");
            return;
        };
        let Some(timeline) = feature.stats.get("bit_true_timeline") else {
            ui.label("Bit true timeline data not found.");
            return;
        };
        let Some(map) = timeline.as_object() else {
            ui.label("Bit true timeline data not found.");
            return;
        };
        let bit_width = value_as_usize(map.get("bit_width"));
        let window = value_as_u64(map.get("window"));
        let stride = value_as_u64(map.get("stride"));
        let frames = map
            .get("frames")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();
        ui.label(format!("Bit width: {}", bit_width));
        ui.label(format!("Window: {} | Stride: {}", window, stride));
        if frames.is_empty() || bit_width == 0 {
            ui.label("No bit true timeline frames recorded.");
            return;
        }
        if self.bit_true_bit_idx >= bit_width {
            self.bit_true_bit_idx = 0;
        }
        ui.add_space(8.0);
        ui.horizontal(|ui| {
            ui.label("Selected bit:");
            ui.add(egui::Slider::new(&mut self.bit_true_bit_idx, 0..=bit_width - 1));
        });
        let mut points = Vec::new();
        for (idx, frame) in frames.iter().enumerate() {
            let value = frame
                .as_array()
                .and_then(|row| row.get(self.bit_true_bit_idx))
                .map(|v| value_as_f64(Some(v)))
                .unwrap_or(0.0);
            points.push([idx as f64, value]);
        }
        Plot::new("bit_true_timeline_plot")
            .view_aspect(2.0)
            .show(ui, |plot_ui| {
                plot_ui.line(egui_plot::Line::new(PlotPoints::from(points)));
            });
    }

    fn draw_avalanche(&self, ui: &mut egui::Ui) {
        ui.heading("Avalanche");
        let Some(feature) = self.session.feature("information_sufficiency") else {
            ui.label("Avalanche data not found.");
            return;
        };
        let Some(avalanche) = feature.stats.get("avalanche_tree") else {
            ui.label("Avalanche data not found.");
            return;
        };
        let Some(map) = avalanche.as_object() else {
            ui.label("Avalanche data not found.");
            return;
        };
        let bit_width = value_as_usize(map.get("bit_width"));
        let unique_messages = value_as_u64(map.get("unique_messages"));
        let biases = map
            .get("biases")
            .and_then(|v| v.as_array())
            .map(|list| list.iter().map(|v| value_as_f64(Some(v))).collect::<Vec<_>>())
            .unwrap_or_default();
        let message_bits = map
            .get("message_bits")
            .and_then(|v| v.as_array())
            .map(|list| list.iter().map(|v| value_as_u64(Some(v)) as u8).collect::<Vec<_>>())
            .unwrap_or_default();
        ui.label(format!("Bit width: {}", bit_width));
        ui.label(format!("Unique messages: {}", unique_messages));
        if !message_bits.is_empty() {
            ui.label(format!("Message bits: {}", bits_preview(&message_bits, 96)));
        }
        if !biases.is_empty() {
            let points = biases
                .iter()
                .enumerate()
                .map(|(idx, value)| [idx as f64, *value])
                .collect::<PlotPoints>();
            Plot::new("avalanche_biases").show(ui, |plot_ui| {
                plot_ui.line(egui_plot::Line::new(points));
            });
        }
    }

    fn draw_bitflow(&mut self, ui: &mut egui::Ui) {
        ui.heading("Bitflow");
        if self.session.bitflow_runs.is_empty() && self.session.bitflow_candidates.is_empty() {
            ui.label("No bitflow events recorded.");
            return;
        }
        let mut run_ids = self
            .session
            .bitflow_runs
            .iter()
            .map(|run| run.run_id.clone())
            .collect::<Vec<_>>();
        run_ids.sort();
        run_ids.dedup();
        let mut selected = self
            .bitflow_selected
            .clone()
            .unwrap_or_else(|| "All runs".to_string());
        egui::ComboBox::from_label("Run")
            .selected_text(&selected)
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut selected, "All runs".to_string(), "All runs");
                for run_id in &run_ids {
                    ui.selectable_value(&mut selected, run_id.clone(), run_id);
                }
            });
        self.bitflow_selected = Some(selected.clone());
        let filtered_candidates = if selected == "All runs" {
            self.session.bitflow_candidates.clone()
        } else {
            self.session
                .bitflow_candidates
                .iter()
                .cloned()
                .filter(|candidate| candidate.run_id == selected)
                .collect::<Vec<_>>()
        };
        ui.label(format!(
            "Runs: {} | Candidates: {}",
            self.session.bitflow_runs.len(),
            filtered_candidates.len()
        ));

        ui.push_id("bitflow_table", |ui| {
            TableBuilder::new(ui)
                .columns(Column::auto(), 7)
                .striped(true)
                .header(20.0, |mut header| {
                    header.col(|ui| { ui.label("Run"); });
                    header.col(|ui| { ui.label("Iter"); });
                    header.col(|ui| { ui.label("Trial"); });
                    header.col(|ui| { ui.label("Partition"); });
                    header.col(|ui| { ui.label("Inverted"); });
                    header.col(|ui| { ui.label("Ones %"); });
                    header.col(|ui| { ui.label("Bits"); });
                })
                .body(|mut body| {
                    for candidate in filtered_candidates {
                        let ones_pct = bits_ones_pct(&candidate.bits);
                        body.row(20.0, |mut row| {
                            row.col(|ui| { ui.label(candidate.run_id); });
                            row.col(|ui| { ui.label(candidate.iteration.to_string()); });
                            row.col(|ui| { ui.label(candidate.trial.to_string()); });
                            row.col(|ui| { ui.label(candidate.partition_size.to_string()); });
                            row.col(|ui| {
                                ui.label(
                                    candidate
                                        .inverted_partitions
                                        .iter()
                                        .map(|v| v.to_string())
                                        .collect::<Vec<_>>()
                                        .join(","),
                                );
                            });
                            row.col(|ui| {
                                ui.label(format!("{ones_pct:.1}"));
                            });
                            row.col(|ui| { ui.label(bits_preview(&candidate.bits, 96)); });
                        });
                    }
                });
        });
    }
}

impl eframe::App for ViewerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll_updates();
        ctx.request_repaint_after(Duration::from_millis(250));

        egui::TopBottomPanel::top("top_bar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label("Session:");
                if let Some(name) = &self.selected_log {
                    ui.monospace(name);
                } else {
                    ui.monospace("none");
                }
                if ui.button("Reload").clicked() {
                    if let Some(name) = self.selected_log.clone() {
                        let _ = self.load_session(&name);
                    }
                }
                ui.label(&self.status);
            });
            ui.separator();
            ui.horizontal(|ui| {
                tab_button(ui, "Summary", Tab::Summary, &mut self.tab);
                tab_button(ui, "Candidates", Tab::Candidates, &mut self.tab);
                tab_button(ui, "Bit Similarity", Tab::BitSimilarity, &mut self.tab);
                tab_button(ui, "Bit True Timeline", Tab::BitTrueTimeline, &mut self.tab);
                tab_button(ui, "Avalanche", Tab::Avalanche, &mut self.tab);
                tab_button(ui, "Beam vs R", Tab::BeamVsR, &mut self.tab);
                tab_button(ui, "Bitflow", Tab::Bitflow, &mut self.tab);
            });
        });

        let mut selected_log = None;
        egui::SidePanel::left("log_list")
            .default_width(220.0)
            .show(ctx, |ui| {
                ui.heading("Logs");
                egui::ScrollArea::vertical()
                    .id_source("log_list_scroll")
                    .show(ui, |ui| {
                    for entry in &self.log_entries {
                        let label = log_label(entry);
                        let selected = self
                            .selected_log
                            .as_ref()
                            .map(|name| name == &entry.name)
                            .unwrap_or(false);
                        let response = ui.selectable_label(selected, label);
                        let clicked = response.clicked();
                        let _response = response.on_hover_text(format!("{} bytes", entry.size));
                        if clicked {
                            selected_log = Some(entry.name.clone());
                        }
                    }
                });
            });
        if let Some(name) = selected_log {
            let _ = self.load_session(&name);
        }

        egui::CentralPanel::default().show(ctx, |ui| match self.tab {
            Tab::Summary => self.draw_summary(ui),
            Tab::Candidates => self.draw_candidates(ui),
            Tab::BitSimilarity => self.draw_bit_similarity(ui),
            Tab::BitTrueTimeline => self.draw_bit_true_timeline(ui),
            Tab::Avalanche => self.draw_avalanche(ui),
            Tab::BeamVsR => self.draw_beam_vs_r(ui),
            Tab::Bitflow => self.draw_bitflow(ui),
        });
    }
}

fn tab_button(ui: &mut egui::Ui, label: &str, tab: Tab, selected: &mut Tab) {
    let active = *selected == tab;
    if ui.selectable_label(active, label).clicked() {
        *selected = tab;
    }
}

#[derive(Debug, Default, Clone)]
struct Session {
    started_unix_ms: Option<u128>,
    finished_unix_ms: Option<u128>,
    cli: CliInfo,
    steps: Vec<StepTiming>,
    step_summaries: Vec<StepSummary>,
    features: Vec<Feature>,
    r_candidate_batches: Vec<RCandidateBatch>,
    r_candidate_accuracy_batches: Vec<RCandidateAccuracyBatch>,
    r_candidate_traces: Vec<RCandidateTraceBatch>,
    bitflow_runs: Vec<BitflowRun>,
    bitflow_candidates: Vec<BitflowCandidate>,
    errors: Vec<String>,
}

impl Session {
    fn feature(&self, name: &str) -> Option<&Feature> {
        self.features.iter().find(|feature| feature.name == name)
    }
}

#[derive(Debug, Default, Clone)]
#[allow(dead_code)]
struct CliInfo {
    bits: u64,
    message_override: Option<String>,
    public_exponent: u64,
    seed: Option<u64>,
    crypto_rng: bool,
    config_path: String,
    tests: bool,
    export: bool,
    session_json: String,
    shift: bool,
    ciphertext_modify: bool,
    use_hamming_distance: bool,
    mirror_invert_candidates: bool,
    bits_decrypt: Option<u64>,
}

#[derive(Debug, Default, Clone)]
#[allow(dead_code)]
struct StepTiming {
    name: String,
    duration_ms: u128,
}

#[derive(Debug, Default, Clone)]
#[allow(dead_code)]
struct StepSummary {
    name: String,
    count: u64,
    total_ms: u128,
    mean_ms: f64,
}

#[derive(Debug, Default, Clone)]
struct Feature {
    name: String,
    enabled: bool,
    duration_ms: Option<u128>,
    notes: Vec<String>,
    stats: Map<String, Value>,
}

#[derive(Debug, Default, Clone)]
#[allow(dead_code)]
struct RCandidateFactor {
    prime: String,
    exponent: u64,
    prime_bits: u64,
}

#[derive(Debug, Default, Clone)]
struct RCandidateEntry {
    r: String,
    r_bits: u64,
    factors: Vec<RCandidateFactor>,
}

#[derive(Debug, Default, Clone)]
#[allow(dead_code)]
struct RCandidateBatch {
    context: Option<String>,
    mode: Option<String>,
    target_count: u64,
    generated_count: u64,
    duration_ms: u128,
    reuse_path: String,
    reuse_enabled: bool,
    reuse_append_only: bool,
    min_factor: String,
    process_scale: u64,
    small_prime_factors: u64,
    max_factors: u64,
    target_bit_length: Option<u64>,
    candidates: Vec<RCandidateEntry>,
}

#[derive(Debug, Default, Clone)]
#[allow(dead_code)]
struct RCandidateAccuracyEntry {
    r: String,
    r_bits: u64,
    factors: Vec<RCandidateFactor>,
    accuracy_pct: f64,
    hbc_ciphertexts_r: Vec<String>,
    candidate_decryptions: Vec<String>,
}

#[derive(Debug, Default, Clone)]
#[allow(dead_code)]
struct RCandidateAccuracyBatch {
    context: Option<String>,
    messages: Vec<String>,
    ciphertexts: Vec<String>,
    shifted_ciphertexts: Vec<String>,
    rabin_exponent: u64,
    tonelli_shanks_modulus: String,
    tonelli_shanks_ciphertexts: Vec<String>,
    candidates: Vec<RCandidateAccuracyEntry>,
    beam_match_pct: Option<f64>,
    beam_ones_match_pct: Option<f64>,
    beam_score: Option<f64>,
    beam_bit_width: Option<u64>,
}

#[derive(Debug, Default, Clone)]
#[allow(dead_code)]
struct RCandidateTraceEntry {
    r: String,
    r_bits: u64,
    hbc_ciphertext_r: String,
    candidate_decryption: String,
}

#[derive(Debug, Default, Clone)]
#[allow(dead_code)]
struct RCandidateTraceBatch {
    context: Option<String>,
    message: String,
    ciphertext: String,
    shifted_ciphertext: String,
    rabin_exponent: u64,
    tonelli_shanks_modulus: String,
    tonelli_shanks_ciphertext: String,
    candidates: Vec<RCandidateTraceEntry>,
}

#[derive(Debug, Default, Clone)]
#[allow(dead_code)]
struct BitflowRun {
    run_id: String,
    bit_width: u64,
    min_partition_size: u64,
    max_partition_size: u64,
    progression: String,
    max_iterations: u64,
    max_partitions_to_flip: u64,
    per_candidate_trials: u64,
    seed: u64,
    pow_mod_base: u64,
    pow_mod_modulus: u64,
    message_bits: Vec<u8>,
}

#[derive(Debug, Default, Clone)]
struct BitflowCandidate {
    run_id: String,
    iteration: u64,
    trial: u64,
    partition_size: u64,
    inverted_partitions: Vec<u64>,
    bits: Vec<u8>,
}

#[derive(Debug)]
struct CandidateRow {
    context: String,
    mode: String,
    index: usize,
    r: String,
    r_bits: u64,
    factors: String,
}

#[derive(Debug)]
struct BeamRow {
    batch: String,
    beam_match: Option<f64>,
    beam_ones: Option<f64>,
    beam_score: Option<f64>,
    beam_bits: Option<u64>,
    r_mean: Option<f64>,
    r_max: Option<f64>,
    r_min: Option<f64>,
    r_stddev: Option<f64>,
    candidate_count: usize,
}

#[derive(Debug)]
struct BasicStats {
    mean: Option<f64>,
    stddev: Option<f64>,
    min: Option<f64>,
    max: Option<f64>,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct BitSimilarityEntry {
    orig_index: usize,
    index: usize,
    shift: usize,
    r: String,
    e: Option<String>,
    x: Option<String>,
    candidate_hex: String,
    match_pct: f64,
    matching_bits: u64,
    adjusted_match_pct: f64,
    adjusted_matching_bits: u64,
    masked_bits: usize,
    base_match_pct: f64,
    base_matching_bits: u64,
}

#[derive(Debug, Clone)]
struct BitSimilarityRow {
    index: usize,
    r: String,
    e: Option<String>,
    x: Option<String>,
    base_match_pct: f64,
    base_matching_bits: u64,
    entries: Vec<BitSimilarityEntry>,
}

#[derive(Debug, Clone)]
struct BitSimilarityData {
    entries: Vec<BitSimilarityEntry>,
    bit_width: usize,
    original_hex: String,
    bit_order: String,
    match_counts: Vec<u64>,
    shift_levels_configured: u64,
    shift_levels_used: u64,
}

#[cfg(not(target_arch = "wasm32"))]
fn collect_log_entries(session_path: &Path, log_dir: &Path) -> Vec<LogEntry> {
    let mut results = Vec::new();
    let mut seen = HashMap::new();
    let candidates = vec![
        session_path.to_path_buf(),
        PathBuf::from("session.json"),
        PathBuf::from("session.log"),
    ];
    for path in candidates {
        if path.exists() {
            if let Some(entry) = log_entry_from_path(&path) {
                if seen.insert(entry.name.clone(), ()).is_none() {
                    results.push(entry);
                }
            }
        }
    }
    if log_dir.is_dir() {
        let mut entries = Vec::new();
        if let Ok(read_dir) = std::fs::read_dir(log_dir) {
            for entry in read_dir.flatten() {
                let path = entry.path();
                if !path.is_file() {
                    continue;
                }
                if let Some(ext) = path.extension() {
                    if ext != "json" && ext != "log" {
                        continue;
                    }
                }
                if let Some(item) = log_entry_from_path(&path) {
                    entries.push(item);
                }
            }
        }
        entries.sort_by(|a, b| {
            b.modified_ms
                .unwrap_or(0)
                .cmp(&a.modified_ms.unwrap_or(0))
                .then_with(|| a.name.cmp(&b.name))
        });
        for entry in entries {
            if seen.insert(entry.name.clone(), ()).is_none() {
                results.push(entry);
            }
        }
    }
    results
}

#[cfg(not(target_arch = "wasm32"))]
fn log_entry_from_path(path: &Path) -> Option<LogEntry> {
    let meta = std::fs::metadata(path).ok()?;
    let modified_ms = meta
        .modified()
        .ok()
        .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|duration| duration.as_millis() as u64);
    Some(LogEntry {
        name: path.to_string_lossy().to_string(),
        size: meta.len(),
        modified_ms,
    })
}

fn log_label(entry: &LogEntry) -> String {
    if cfg!(target_arch = "wasm32") {
        entry.name.clone()
    } else {
        Path::new(&entry.name)
            .file_name()
            .map(|name| name.to_string_lossy().to_string())
            .unwrap_or_else(|| entry.name.clone())
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn file_size(path: &Path) -> Option<u64> {
    std::fs::metadata(path).ok().map(|meta| meta.len())
}

#[cfg(target_arch = "wasm32")]
/// Fetches text content over HTTP from the viewer server.
///
/// # Parameters
/// - `url`: Relative URL to fetch.
///
/// # Returns
/// - `Result<String, JsValue>`: The response body as a string on success.
///
/// # Expected Output
/// - Performs an HTTP GET request; no other side effects.
async fn fetch_text(url: &str) -> Result<String, JsValue> {
    let window = window().ok_or_else(|| JsValue::from_str("Missing window"))?;
    let response_value = JsFuture::from(window.fetch_with_str(url)).await?;
    let response: Response = response_value.dyn_into()?;
    if !response.ok() {
        return Err(JsValue::from_str(&format!(
            "HTTP {} for {}",
            response.status(),
            url
        )));
    }
    let text = JsFuture::from(response.text()?).await?;
    Ok(text.as_string().unwrap_or_default())
}

#[cfg(not(target_arch = "wasm32"))]
fn load_session_from_path(path: &Path) -> Result<(Session, bool), String> {
    let mut file = File::open(path).map_err(|err| err.to_string())?;
    let mut raw = String::new();
    file.read_to_string(&mut raw)
        .map_err(|err| err.to_string())?;
    parse_session_from_str(&raw)
}

fn parse_session_from_str(raw: &str) -> Result<(Session, bool), String> {
    if raw.trim().is_empty() {
        return Ok((Session::default(), false));
    }
    if let Ok(value) = serde_json::from_str::<Value>(raw) {
        if let Some(obj) = value.as_object() {
            if obj.contains_key("event") && obj.contains_key("payload") {
                return Ok((build_session_from_events(&[obj.clone()]), true));
            }
            return Ok((normalize_session(obj), false));
        }
        if let Some(arr) = value.as_array() {
            let is_events = arr
                .iter()
                .all(|item| item.get("event").is_some() && item.get("payload").is_some());
            if is_events {
                let events = arr
                    .iter()
                    .filter_map(|item| item.as_object().cloned())
                    .collect::<Vec<_>>();
                return Ok((build_session_from_events(&events), true));
            }
            return Ok((normalize_session(&Map::new()), false));
        }
    }
    let mut events = Vec::new();
    for line in raw.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Ok(value) = serde_json::from_str::<Value>(trimmed) {
            if let Some(obj) = value.as_object() {
                if obj.contains_key("event") {
                    events.push(obj.clone());
                }
            }
        }
    }
    Ok((build_session_from_events(&events), true))
}

fn build_session_from_events(events: &[Map<String, Value>]) -> Session {
    let mut session = Session::default();
    for event in events {
        apply_event_to_session(&mut session, event);
    }
    normalize_session_values(&mut session);
    session
}

fn apply_event_to_session(session: &mut Session, event: &Map<String, Value>) {
    let Some(event_name) = event.get("event").and_then(|v| v.as_str()) else {
        return;
    };
    let payload = event.get("payload").and_then(|v| v.as_object());
    match event_name {
        "session_start" => {
            if let Some(payload) = payload {
                session.started_unix_ms = value_as_opt_u128(payload.get("started_unix_ms"));
                session.cli = parse_cli(payload.get("cli").and_then(|v| v.as_object()));
            }
        }
        "session_finish" => {
            if let Some(payload) = payload {
                session.finished_unix_ms = value_as_opt_u128(payload.get("finished_unix_ms"));
                session.errors = value_as_vec_string(payload.get("errors"));
            }
        }
        "step" => {
            if let Some(payload) = payload {
                session.steps.push(StepTiming {
                    name: value_as_string(payload.get("name")),
                    duration_ms: value_as_u128(payload.get("duration_ms")),
                });
            }
        }
        "step_summary" => {
            if let Some(payload) = payload {
                session.step_summaries.push(StepSummary {
                    name: value_as_string(payload.get("name")),
                    count: value_as_u64(payload.get("count")),
                    total_ms: value_as_u128(payload.get("total_ms")),
                    mean_ms: value_as_f64(payload.get("mean_ms")),
                });
            }
        }
        "feature" => {
            if let Some(payload) = payload {
                let feature = parse_feature(payload);
                upsert_feature(&mut session.features, feature);
            }
        }
        "r_candidate_batch" => {
            if let Some(payload) = payload {
                session
                    .r_candidate_batches
                    .push(parse_r_candidate_batch(payload));
            }
        }
        "r_candidate_accuracy_batch" => {
            if let Some(payload) = payload {
                session
                    .r_candidate_accuracy_batches
                    .push(parse_r_candidate_accuracy_batch(payload));
            }
        }
        "r_candidate_trace_batch" => {
            if let Some(payload) = payload {
                session
                    .r_candidate_traces
                    .push(parse_r_candidate_trace_batch(payload));
            }
        }
        "bitflow_run" => {
            if let Some(payload) = payload {
                session.bitflow_runs.push(parse_bitflow_run(payload));
            }
        }
        "bitflow_candidate" => {
            if let Some(payload) = payload {
                session
                    .bitflow_candidates
                    .push(parse_bitflow_candidate(payload));
            }
        }
        _ => {}
    }
}

fn normalize_session(map: &Map<String, Value>) -> Session {
    let mut session = Session::default();
    session.started_unix_ms = value_as_opt_u128(map.get("started_unix_ms"));
    session.finished_unix_ms = value_as_opt_u128(map.get("finished_unix_ms"));
    session.cli = parse_cli(map.get("cli").and_then(|v| v.as_object()));
    if let Some(steps) = map.get("steps").and_then(|v| v.as_array()) {
        for step in steps {
            if let Some(step) = step.as_object() {
                session.steps.push(StepTiming {
                    name: value_as_string(step.get("name")),
                    duration_ms: value_as_u128(step.get("duration_ms")),
                });
            }
        }
    }
    if let Some(summaries) = map.get("step_summaries").and_then(|v| v.as_array()) {
        for summary in summaries {
            if let Some(summary) = summary.as_object() {
                session.step_summaries.push(StepSummary {
                    name: value_as_string(summary.get("name")),
                    count: value_as_u64(summary.get("count")),
                    total_ms: value_as_u128(summary.get("total_ms")),
                    mean_ms: value_as_f64(summary.get("mean_ms")),
                });
            }
        }
    }
    if let Some(features) = map.get("features").and_then(|v| v.as_array()) {
        for feature in features {
            if let Some(feature) = feature.as_object() {
                session.features.push(parse_feature(feature));
            }
        }
    }
    if let Some(batches) = map.get("r_candidate_batches").and_then(|v| v.as_array()) {
        for batch in batches {
            if let Some(batch) = batch.as_object() {
                session.r_candidate_batches.push(parse_r_candidate_batch(batch));
            }
        }
    }
    if let Some(batches) = map
        .get("r_candidate_accuracy_batches")
        .and_then(|v| v.as_array())
    {
        for batch in batches {
            if let Some(batch) = batch.as_object() {
                session
                    .r_candidate_accuracy_batches
                    .push(parse_r_candidate_accuracy_batch(batch));
            }
        }
    }
    if let Some(batches) = map
        .get("r_candidate_traces")
        .and_then(|v| v.as_array())
    {
        for batch in batches {
            if let Some(batch) = batch.as_object() {
                session
                    .r_candidate_traces
                    .push(parse_r_candidate_trace_batch(batch));
            }
        }
    }
    if let Some(runs) = map.get("bitflow_runs").and_then(|v| v.as_array()) {
        for run in runs {
            if let Some(run) = run.as_object() {
                session.bitflow_runs.push(parse_bitflow_run(run));
            }
        }
    }
    if let Some(candidates) = map.get("bitflow_candidates").and_then(|v| v.as_array()) {
        for candidate in candidates {
            if let Some(candidate) = candidate.as_object() {
                session
                    .bitflow_candidates
                    .push(parse_bitflow_candidate(candidate));
            }
        }
    }
    session.errors = value_as_vec_string(map.get("errors"));
    normalize_session_values(&mut session);
    session
}

fn normalize_session_values(session: &mut Session) {
    for (idx, run) in session.bitflow_runs.iter_mut().enumerate() {
        if run.run_id.is_empty() {
            run.run_id = format!("run-{}", idx + 1);
        }
    }
    for candidate in &mut session.bitflow_candidates {
        if candidate.run_id.is_empty() {
            candidate.run_id = "run-unknown".to_string();
        }
    }
}

fn parse_cli(map: Option<&Map<String, Value>>) -> CliInfo {
    let map = map.cloned().unwrap_or_default();
    CliInfo {
        bits: value_as_u64(map.get("bits")),
        message_override: value_as_opt_string(map.get("message_override")),
        public_exponent: value_as_u64(map.get("public_exponent")),
        seed: value_as_opt_u64(map.get("seed")),
        crypto_rng: value_as_bool(map.get("crypto_rng")),
        config_path: value_as_string(map.get("config_path")),
        tests: value_as_bool(map.get("tests")),
        export: value_as_bool(map.get("export")),
        session_json: value_as_string(map.get("session_json")),
        shift: value_as_bool(map.get("shift")),
        ciphertext_modify: value_as_bool(map.get("ciphertext_modify")),
        use_hamming_distance: value_as_bool(map.get("use_hamming_distance")),
        mirror_invert_candidates: value_as_bool(map.get("mirror_invert_candidates")),
        bits_decrypt: value_as_opt_u64(map.get("bits_decrypt")),
    }
}

fn parse_feature(map: &Map<String, Value>) -> Feature {
    Feature {
        name: value_as_string(map.get("name")),
        enabled: value_as_bool(map.get("enabled")),
        duration_ms: value_as_opt_u128(map.get("duration_ms")),
        notes: value_as_vec_string(map.get("notes")),
        stats: map
            .get("stats")
            .and_then(|v| v.as_object())
            .cloned()
            .unwrap_or_default(),
    }
}

fn upsert_feature(features: &mut Vec<Feature>, feature: Feature) {
    if let Some(existing) = features.iter_mut().find(|f| f.name == feature.name) {
        *existing = feature;
    } else {
        features.push(feature);
    }
}

fn parse_r_candidate_batch(map: &Map<String, Value>) -> RCandidateBatch {
    let mut candidates = Vec::new();
    if let Some(list) = map.get("candidates").and_then(|v| v.as_array()) {
        for entry in list {
            if let Some(entry) = entry.as_object() {
                candidates.push(parse_r_candidate_entry(entry));
            }
        }
    }
    RCandidateBatch {
        context: value_as_opt_string(map.get("context")),
        mode: value_as_opt_string(map.get("mode")),
        target_count: value_as_u64(map.get("target_count")),
        generated_count: value_as_u64(map.get("generated_count")),
        duration_ms: value_as_u128(map.get("duration_ms")),
        reuse_path: value_as_string(map.get("reuse_path")),
        reuse_enabled: value_as_bool(map.get("reuse_enabled")),
        reuse_append_only: value_as_bool(map.get("reuse_append_only")),
        min_factor: value_as_string(map.get("min_factor")),
        process_scale: value_as_u64(map.get("process_scale")),
        small_prime_factors: value_as_u64(map.get("small_prime_factors")),
        max_factors: value_as_u64(map.get("max_factors")),
        target_bit_length: value_as_opt_u64(map.get("target_bit_length")),
        candidates,
    }
}

fn parse_r_candidate_entry(map: &Map<String, Value>) -> RCandidateEntry {
    let mut factors = Vec::new();
    if let Some(list) = map.get("factors").and_then(|v| v.as_array()) {
        for factor in list {
            if let Some(factor) = factor.as_object() {
                factors.push(RCandidateFactor {
                    prime: value_as_string(factor.get("prime")),
                    exponent: value_as_u64(factor.get("exponent")),
                    prime_bits: value_as_u64(factor.get("prime_bits")),
                });
            }
        }
    }
    RCandidateEntry {
        r: value_as_string(map.get("r")),
        r_bits: value_as_u64(map.get("r_bits")),
        factors,
    }
}

fn parse_r_candidate_accuracy_batch(map: &Map<String, Value>) -> RCandidateAccuracyBatch {
    let mut candidates = Vec::new();
    if let Some(list) = map.get("candidates").and_then(|v| v.as_array()) {
        for entry in list {
            if let Some(entry) = entry.as_object() {
                candidates.push(parse_r_candidate_accuracy_entry(entry));
            }
        }
    }
    RCandidateAccuracyBatch {
        context: value_as_opt_string(map.get("context")),
        messages: value_as_vec_string(map.get("messages")),
        ciphertexts: value_as_vec_string(map.get("ciphertexts")),
        shifted_ciphertexts: value_as_vec_string(map.get("shifted_ciphertexts")),
        rabin_exponent: value_as_u64(map.get("rabin_exponent")),
        tonelli_shanks_modulus: value_as_string(map.get("tonelli_shanks_modulus")),
        tonelli_shanks_ciphertexts: value_as_vec_string(map.get("tonelli_shanks_ciphertexts")),
        candidates,
        beam_match_pct: value_as_opt_f64(map.get("beam_match_pct")),
        beam_ones_match_pct: value_as_opt_f64(map.get("beam_ones_match_pct")),
        beam_score: value_as_opt_f64(map.get("beam_score")),
        beam_bit_width: value_as_opt_u64(map.get("beam_bit_width")),
    }
}

fn parse_r_candidate_accuracy_entry(map: &Map<String, Value>) -> RCandidateAccuracyEntry {
    let mut factors = Vec::new();
    if let Some(list) = map.get("factors").and_then(|v| v.as_array()) {
        for factor in list {
            if let Some(factor) = factor.as_object() {
                factors.push(RCandidateFactor {
                    prime: value_as_string(factor.get("prime")),
                    exponent: value_as_u64(factor.get("exponent")),
                    prime_bits: value_as_u64(factor.get("prime_bits")),
                });
            }
        }
    }
    RCandidateAccuracyEntry {
        r: value_as_string(map.get("r")),
        r_bits: value_as_u64(map.get("r_bits")),
        factors,
        accuracy_pct: value_as_f64(map.get("accuracy_pct")),
        hbc_ciphertexts_r: value_as_vec_string(map.get("hbc_ciphertexts_r")),
        candidate_decryptions: value_as_vec_string(map.get("candidate_decryptions")),
    }
}

fn parse_r_candidate_trace_batch(map: &Map<String, Value>) -> RCandidateTraceBatch {
    let mut candidates = Vec::new();
    if let Some(list) = map.get("candidates").and_then(|v| v.as_array()) {
        for entry in list {
            if let Some(entry) = entry.as_object() {
                candidates.push(RCandidateTraceEntry {
                    r: value_as_string(entry.get("r")),
                    r_bits: value_as_u64(entry.get("r_bits")),
                    hbc_ciphertext_r: value_as_string(entry.get("hbc_ciphertext_r")),
                    candidate_decryption: value_as_string(entry.get("candidate_decryption")),
                });
            }
        }
    }
    RCandidateTraceBatch {
        context: value_as_opt_string(map.get("context")),
        message: value_as_string(map.get("message")),
        ciphertext: value_as_string(map.get("ciphertext")),
        shifted_ciphertext: value_as_string(map.get("shifted_ciphertext")),
        rabin_exponent: value_as_u64(map.get("rabin_exponent")),
        tonelli_shanks_modulus: value_as_string(map.get("tonelli_shanks_modulus")),
        tonelli_shanks_ciphertext: value_as_string(map.get("tonelli_shanks_ciphertext")),
        candidates,
    }
}

fn parse_bitflow_run(map: &Map<String, Value>) -> BitflowRun {
    BitflowRun {
        run_id: value_as_string(map.get("run_id")),
        bit_width: value_as_u64(map.get("bit_width")),
        min_partition_size: value_as_u64(map.get("min_partition_size")),
        max_partition_size: value_as_u64(map.get("max_partition_size")),
        progression: value_as_string(map.get("progression")),
        max_iterations: value_as_u64(map.get("max_iterations")),
        max_partitions_to_flip: value_as_u64(map.get("max_partitions_to_flip")),
        per_candidate_trials: value_as_u64(map.get("per_candidate_trials")),
        seed: value_as_u64(map.get("seed")),
        pow_mod_base: value_as_u64(map.get("pow_mod_base")),
        pow_mod_modulus: value_as_u64(map.get("pow_mod_modulus")),
        message_bits: value_as_vec_u8(map.get("message_bits")),
    }
}

fn parse_bitflow_candidate(map: &Map<String, Value>) -> BitflowCandidate {
    BitflowCandidate {
        run_id: value_as_string(map.get("run_id")),
        iteration: value_as_u64(map.get("iteration")),
        trial: value_as_u64(map.get("trial")),
        partition_size: value_as_u64(map.get("partition_size")),
        inverted_partitions: value_as_vec_u64(map.get("inverted_partitions")),
        bits: value_as_vec_u8(map.get("bits")),
    }
}

fn value_as_string(value: Option<&Value>) -> String {
    match value {
        Some(Value::String(val)) => val.clone(),
        Some(Value::Number(num)) => num.to_string(),
        Some(Value::Bool(val)) => val.to_string(),
        Some(_) => String::new(),
        None => String::new(),
    }
}

fn value_as_opt_string(value: Option<&Value>) -> Option<String> {
    match value {
        Some(Value::String(val)) => Some(val.clone()),
        Some(Value::Number(num)) => Some(num.to_string()),
        Some(Value::Bool(val)) => Some(val.to_string()),
        _ => None,
    }
}

fn value_as_bool(value: Option<&Value>) -> bool {
    match value {
        Some(Value::Bool(val)) => *val,
        Some(Value::Number(num)) => num.as_u64().unwrap_or(0) != 0,
        Some(Value::String(val)) => val == "true" || val == "1",
        _ => false,
    }
}

fn value_as_u64(value: Option<&Value>) -> u64 {
    match value {
        Some(Value::Number(num)) => num.as_u64().unwrap_or(0),
        Some(Value::String(val)) => val.parse::<u64>().unwrap_or(0),
        Some(Value::Bool(val)) => if *val { 1 } else { 0 },
        _ => 0,
    }
}

fn value_as_usize(value: Option<&Value>) -> usize {
    value_as_u64(value) as usize
}

fn value_as_u128(value: Option<&Value>) -> u128 {
    match value {
        Some(Value::Number(num)) => num.as_u64().unwrap_or(0) as u128,
        Some(Value::String(val)) => val.parse::<u128>().unwrap_or(0),
        Some(Value::Bool(val)) => if *val { 1 } else { 0 },
        _ => 0,
    }
}

fn value_as_opt_u64(value: Option<&Value>) -> Option<u64> {
    match value {
        Some(Value::Number(num)) => num.as_u64(),
        Some(Value::String(val)) => val.parse::<u64>().ok(),
        _ => None,
    }
}

fn value_as_opt_u128(value: Option<&Value>) -> Option<u128> {
    match value {
        Some(Value::Number(num)) => num.as_u64().map(|v| v as u128),
        Some(Value::String(val)) => val.parse::<u128>().ok(),
        _ => None,
    }
}

fn value_as_f64(value: Option<&Value>) -> f64 {
    match value {
        Some(Value::Number(num)) => num.as_f64().unwrap_or(0.0),
        Some(Value::String(val)) => val.parse::<f64>().unwrap_or(0.0),
        Some(Value::Bool(val)) => if *val { 1.0 } else { 0.0 },
        _ => 0.0,
    }
}

fn value_as_opt_f64(value: Option<&Value>) -> Option<f64> {
    match value {
        Some(Value::Number(num)) => num.as_f64(),
        Some(Value::String(val)) => val.parse::<f64>().ok(),
        _ => None,
    }
}

fn value_as_vec_string(value: Option<&Value>) -> Vec<String> {
    match value {
        Some(Value::Array(list)) => list.iter().map(|v| value_as_string(Some(v))).collect(),
        _ => Vec::new(),
    }
}

fn value_as_vec_u64(value: Option<&Value>) -> Vec<u64> {
    match value {
        Some(Value::Array(list)) => list.iter().map(|v| value_as_u64(Some(v))).collect(),
        _ => Vec::new(),
    }
}

fn value_as_vec_u8(value: Option<&Value>) -> Vec<u8> {
    match value {
        Some(Value::Array(list)) => list
            .iter()
            .map(|v| value_as_u64(Some(v)).min(u8::MAX as u64) as u8)
            .collect(),
        _ => Vec::new(),
    }
}

fn format_unix_ms(value: Option<u128>) -> String {
    value.map_or_else(|| "N/A".to_string(), |val| val.to_string())
}

fn opt_to_string(value: Option<u128>) -> String {
    value.map_or_else(|| "".to_string(), |val| val.to_string())
}

fn compute_basic_stats(values: &[f64]) -> BasicStats {
    if values.is_empty() {
        return BasicStats {
            mean: None,
            stddev: None,
            min: None,
            max: None,
        };
    }
    let count = values.len() as f64;
    let mean = values.iter().sum::<f64>() / count;
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / count;
    let stddev = variance.sqrt();
    let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    BasicStats {
        mean: Some(mean),
        stddev: Some(stddev),
        min: Some(min),
        max: Some(max),
    }
}

fn pearson_corr(pairs: &[(f64, f64)]) -> Option<f64> {
    if pairs.len() < 2 {
        return None;
    }
    let mean_x = pairs.iter().map(|pair| pair.0).sum::<f64>() / pairs.len() as f64;
    let mean_y = pairs.iter().map(|pair| pair.1).sum::<f64>() / pairs.len() as f64;
    let num = pairs
        .iter()
        .map(|pair| (pair.0 - mean_x) * (pair.1 - mean_y))
        .sum::<f64>();
    let denom_x = pairs
        .iter()
        .map(|pair| (pair.0 - mean_x).powi(2))
        .sum::<f64>();
    let denom_y = pairs
        .iter()
        .map(|pair| (pair.1 - mean_y).powi(2))
        .sum::<f64>();
    let denom = (denom_x * denom_y).sqrt();
    if denom == 0.0 {
        return None;
    }
    Some(num / denom)
}

fn flatten_candidate_batches(session: &Session) -> Vec<CandidateRow> {
    let mut rows = Vec::new();
    for batch in &session.r_candidate_batches {
        let context = batch.context.clone().unwrap_or_default();
        let mode = batch.mode.clone().unwrap_or_default();
        for (idx, entry) in batch.candidates.iter().enumerate() {
            let factor_str = entry
                .factors
                .iter()
                .map(|factor| format!("{}^{}", factor.prime, factor.exponent))
                .collect::<Vec<_>>()
                .join("; ");
            rows.push(CandidateRow {
                context: context.clone(),
                mode: mode.clone(),
                index: idx,
                r: entry.r.clone(),
                r_bits: entry.r_bits,
                factors: factor_str,
            });
        }
    }
    rows
}

fn format_opt_f64(value: Option<f64>) -> String {
    value.map_or_else(|| "".to_string(), |val| format!("{val:.2}"))
}

fn bits_preview(bits: &[u8], max_len: usize) -> String {
    if bits.is_empty() {
        return String::new();
    }
    let mut text = bits
        .iter()
        .map(|bit| if *bit == 0 { '0' } else { '1' })
        .collect::<String>();
    if text.len() > max_len {
        text.truncate(max_len);
        text.push_str("...");
    }
    text
}

fn bits_ones_pct(bits: &[u8]) -> f64 {
    if bits.is_empty() {
        return 0.0;
    }
    let ones = bits.iter().filter(|bit| **bit != 0).count();
    100.0 * ones as f64 / bits.len() as f64
}

fn parse_bit_similarity_data(map: &Map<String, Value>) -> BitSimilarityData {
    let bit_width = value_as_usize(map.get("bit_width"));
    let original_hex = value_as_string(map.get("original_hex"));
    let bit_order = value_as_string(map.get("bit_order"));
    let match_counts = value_as_vec_u64(map.get("match_counts_per_bit"));
    let shift_levels_configured = value_as_u64(map.get("shift_levels_configured"));
    let shift_levels_used = value_as_u64(map.get("shift_levels_used"));
    let mut entries = Vec::new();
    if let Some(list) = map.get("candidates").and_then(|v| v.as_array()) {
        for (idx, entry) in list.iter().enumerate() {
            let Some(entry) = entry.as_object() else {
                continue;
            };
            let match_pct = value_as_f64(entry.get("match_pct"));
            let base_match_pct = if entry.get("base_match_pct").is_some() {
                value_as_f64(entry.get("base_match_pct"))
            } else {
                match_pct
            };
            let matching_bits = value_as_u64(entry.get("matching_bits"));
            let base_matching_bits = if entry.get("base_matching_bits").is_some() {
                value_as_u64(entry.get("base_matching_bits"))
            } else {
                matching_bits
            };
            entries.push(BitSimilarityEntry {
                orig_index: idx,
                index: value_as_usize(entry.get("index")),
                shift: value_as_usize(entry.get("shift")),
                r: value_as_string(entry.get("r")),
                e: value_as_opt_string(entry.get("e")),
                x: value_as_opt_string(entry.get("x")),
                candidate_hex: value_as_string(entry.get("candidate_hex")),
                match_pct,
                matching_bits,
                adjusted_match_pct: value_as_f64(entry.get("adjusted_match_pct")),
                adjusted_matching_bits: value_as_u64(entry.get("adjusted_matching_bits")),
                masked_bits: value_as_usize(entry.get("masked_bits")),
                base_match_pct,
                base_matching_bits,
            });
        }
    }
    BitSimilarityData {
        entries,
        bit_width,
        original_hex,
        bit_order: if bit_order.is_empty() {
            "lsb0".to_string()
        } else {
            bit_order
        },
        match_counts,
        shift_levels_configured,
        shift_levels_used,
    }
}

fn build_bit_similarity_rows(
    entries: &[BitSimilarityEntry],
    hide_shifted: bool,
    sort_mode: BitSimilaritySort,
) -> Vec<BitSimilarityRow> {
    let filtered = if hide_shifted {
        entries
            .iter()
            .cloned()
            .filter(|entry| entry.shift == 0)
            .collect::<Vec<_>>()
    } else {
        entries.to_vec()
    };
    let mut by_index: HashMap<usize, Vec<BitSimilarityEntry>> = HashMap::new();
    for entry in filtered {
        by_index.entry(entry.index).or_default().push(entry);
    }
    let mut rows = Vec::new();
    for (idx, mut entries) in by_index {
        entries.sort_by_key(|entry| entry.shift);
        let base_entry = entries
            .iter()
            .find(|entry| entry.shift == 0)
            .unwrap_or_else(|| &entries[0]);
        rows.push(BitSimilarityRow {
            index: idx,
            r: base_entry.r.clone(),
            e: base_entry.e.clone(),
            x: base_entry.x.clone(),
            base_match_pct: base_entry.base_match_pct,
            base_matching_bits: base_entry.base_matching_bits,
            entries,
        });
    }

    match sort_mode {
        BitSimilaritySort::MatchDesc => {
            rows.sort_by(|a, b| {
                b.base_match_pct
                    .total_cmp(&a.base_match_pct)
                    .then_with(|| a.index.cmp(&b.index))
            });
        }
        BitSimilaritySort::MatchAsc => {
            rows.sort_by(|a, b| {
                a.base_match_pct
                    .total_cmp(&b.base_match_pct)
                    .then_with(|| a.index.cmp(&b.index))
            });
        }
        BitSimilaritySort::Original => {
            rows.sort_by_key(|row| row.index);
        }
    }
    rows
}

fn build_match_counts(
    entries: &[BitSimilarityEntry],
    original_bits: &[bool],
    bit_width: usize,
) -> Vec<u64> {
    if bit_width == 0 {
        return Vec::new();
    }
    let mut counts = vec![0u64; bit_width];
    for entry in entries {
        let candidate_bits = hex_to_bits_le(&entry.candidate_hex, bit_width);
        for bit_idx in 0..bit_width {
            let cand_idx = bit_idx + entry.shift;
            if cand_idx >= bit_width {
                continue;
            }
            if candidate_bits
                .get(cand_idx)
                .copied()
                .unwrap_or(false)
                == original_bits.get(bit_idx).copied().unwrap_or(false)
            {
                counts[bit_idx] += 1;
            }
        }
    }
    counts
}

fn draw_bit_similarity_canvas(
    ui: &mut egui::Ui,
    rows: &[BitSimilarityRow],
    bit_width: usize,
    original_bits: &[bool],
    match_counts: &[u64],
    max_shift: usize,
    palette: &BitSimilarityPalette,
) {
    let _ = match_counts;
    let margin = 8.0;
    let bit_size = 10.0;
    let bit_spacing = 1.0;
    let row_spacing = 8.0;
    let header_height = 26.0;
    let header_gap = header_height * 0.75;
    let row_padding = 22.0;
    let row_gap = bit_spacing + 8.0;
    let box_offset = 0.0;
    let label_width = 320.0;

    let content_width =
        margin * 2.0
            + label_width
            + (bit_width + max_shift) as f32 * (bit_size + bit_spacing)
            + label_width;
    let mut content_height = margin * 2.0;
    for row in rows {
        content_height +=
            row_height_for(
                row,
                bit_size,
                row_gap,
                header_height,
                header_gap,
                row_padding,
                box_offset,
            );
        content_height += row_spacing;
    }
    if !rows.is_empty() {
        content_height -= row_spacing;
    }

    egui::ScrollArea::both()
        .id_source("bit_similarity_canvas")
        .auto_shrink([false, false])
        .show(ui, |ui| {
            let (rect, _) =
                ui.allocate_exact_size(egui::vec2(content_width, content_height), egui::Sense::hover());
            let painter = ui.painter_at(rect);
            let mut y = rect.min.y + margin;
            for row in rows {
                let row_height = row_height_for(
                    row,
                    bit_size,
                    row_gap,
                    header_height,
                    header_gap,
                    row_padding,
                    box_offset,
                );
                draw_bit_similarity_row(
                    &painter,
                    rect.min.x + margin,
                    y,
                    label_width,
                    bit_width,
                    original_bits,
                    row,
                    bit_size,
                    bit_spacing,
                    row_gap,
                    header_height,
                    header_gap,
                    box_offset,
                    palette,
                );
                y += row_height + row_spacing;
            }
        });
}

fn draw_bit_similarity_row(
    painter: &egui::Painter,
    origin_x: f32,
    origin_y: f32,
    label_width: f32,
    bit_width: usize,
    original_bits: &[bool],
    row: &BitSimilarityRow,
    bit_size: f32,
    bit_spacing: f32,
    row_gap: f32,
    header_height: f32,
    header_gap: f32,
    box_offset: f32,
    palette: &BitSimilarityPalette,
) {
    if row.entries.is_empty() || bit_width == 0 {
        painter.text(
            egui::pos2(origin_x, origin_y),
            egui::Align2::LEFT_TOP,
            "No entries",
            egui::FontId::proportional(12.0),
            palette.label_color,
        );
        return;
    }

    let header_y = origin_y + header_height - 4.0;
    let bits_top = origin_y + header_height + header_gap;
    let boxes_top = bits_top + box_offset;
    let boxes_start = origin_x + label_width;
    let label_x = origin_x;
    let suffix = match (&row.e, &row.x) {
        (Some(e), Some(x)) if !e.is_empty() && !x.is_empty() => format!(" | e={e} | x={x}"),
        _ => String::new(),
    };
    let header_text = format!(
        "#{} | r={} | match={:.2}% | matching bits={}{}",
        row.index, row.r, row.base_match_pct, row.base_matching_bits, suffix
    );
    painter.text(
        egui::pos2(origin_x, header_y),
        egui::Align2::LEFT_TOP,
        header_text,
        egui::FontId::proportional(12.0),
        palette.label_color,
    );

    painter.text(
        egui::pos2(label_x, bits_top),
        egui::Align2::LEFT_TOP,
        "Original",
        egui::FontId::proportional(11.0),
        palette.label_color,
    );

    let base_bits = hex_to_bits_le(&row.entries[0].candidate_hex, bit_width);
    let bit_font = egui::FontId::proportional(7.0);
    for bit_idx in 0..bit_width {
        let orig_bit = original_bits.get(bit_idx).copied().unwrap_or(false);
        let cand_bit = base_bits.get(bit_idx).copied().unwrap_or(false);
        let matches = orig_bit == cand_bit;
        let base_color = if matches {
            palette.match_color
        } else {
            palette.mismatch_color
        };
        let color = if orig_bit {
            base_color
        } else {
            lighten_color(base_color, 0.45)
        };
        let x = boxes_start + bit_idx as f32 * (bit_size + bit_spacing);
        let rect = egui::Rect::from_min_size(
            egui::pos2(x, boxes_top),
            egui::vec2(bit_size, bit_size),
        );
        painter.rect_filled(rect, 0.0, color);
        painter.rect_stroke(rect, 0.0, palette.stroke);
        let text_color = text_color_for(color);
        painter.text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            if orig_bit { "1" } else { "0" },
            bit_font.clone(),
            text_color,
        );
    }

    let mut prev_bits: Option<Vec<bool>> = None;
    for (entry_idx, entry) in row.entries.iter().enumerate() {
        let shift = entry.shift;
        let masked_bits = if entry.masked_bits == 0 { shift } else { entry.masked_bits };
        let mut label = if shift == 0 {
            "Candidate".to_string()
        } else {
            format!("Candidate << {shift}")
        };
        if let (Some(e), Some(x)) = (&entry.e, &entry.x) {
            if !e.is_empty() && !x.is_empty() {
                label.push_str(&format!(" | e={e} | x={x}"));
            }
        }
        let adjusted_denom = bit_width.saturating_sub(masked_bits).max(1) as u64;
        let line = format!(
            "{label} | adj={:.2}% ({}/{})",
            entry.adjusted_match_pct, entry.adjusted_matching_bits, adjusted_denom
        );
        let y = bits_top + bit_size + row_gap + entry_idx as f32 * (bit_size + row_gap);
        let y_boxes = y + box_offset;
        painter.text(
            egui::pos2(label_x, y),
            egui::Align2::LEFT_TOP,
            line,
            egui::FontId::proportional(11.0),
            palette.label_color,
        );

        let candidate_bits = hex_to_bits_le(&entry.candidate_hex, bit_width);
        for bit_idx in 0..bit_width {
            let cand_idx = bit_idx + shift;
            let masked = cand_idx >= bit_width;
            let cand_bit = if !masked {
                candidate_bits.get(cand_idx).copied().unwrap_or(false)
            } else {
                false
            };
            let matches_original = !masked
                && cand_bit == original_bits.get(bit_idx).copied().unwrap_or(false);
            let matches_prev = if let (false, Some(prev_bits)) = (masked, &prev_bits) {
                prev_bits
                    .get(cand_idx)
                    .copied()
                    .unwrap_or(false)
                    == cand_bit
            } else {
                false
            };
            let base_candidate = if matches_original && matches_prev {
                palette.multi_match_color
            } else if matches_original {
                palette.match_color
            } else {
                palette.mismatch_color
            };
            let color = if masked {
                palette.masked_fill
            } else if !cand_bit {
                lighten_color(base_candidate, 0.45)
            } else {
                base_candidate
            };
            let x = boxes_start + bit_idx as f32 * (bit_size + bit_spacing);
            let rect = egui::Rect::from_min_size(
                egui::pos2(x, y_boxes),
                egui::vec2(bit_size, bit_size),
            );
            painter.rect_filled(rect, 0.0, color);
            painter.rect_stroke(rect, 0.0, palette.stroke);
            let (text, text_color) = if masked {
                let masked_bit = candidate_bits.get(bit_idx).copied().unwrap_or(false);
                (
                    if masked_bit { "1" } else { "0" },
                    palette.masked_text,
                )
            } else {
                (
                    if cand_bit { "1" } else { "0" },
                    text_color_for(color),
                )
            };
            painter.text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                text,
                bit_font.clone(),
                text_color,
            );
        }
        prev_bits = Some(candidate_bits);
    }

    let mut majority_votes = vec![0u64; bit_width];
    let mut majority_ones = vec![0u64; bit_width];
    for entry in &row.entries {
        let candidate_bits = hex_to_bits_le(&entry.candidate_hex, bit_width);
        for bit_idx in 0..bit_width {
            let cand_idx = bit_idx + entry.shift;
            if cand_idx >= bit_width {
                continue;
            }
            if candidate_bits.get(cand_idx).copied().unwrap_or(false) {
                majority_ones[bit_idx] += 1;
            }
            majority_votes[bit_idx] += 1;
        }
    }
    let mut majority_bits = vec![false; bit_width];
    for bit_idx in 0..bit_width {
        let votes = majority_votes[bit_idx];
        if votes == 0 {
            continue;
        }
        let ones = majority_ones[bit_idx];
        let zeros = votes - ones;
        majority_bits[bit_idx] = ones >= zeros;
    }
    let mut majority_matches = 0u64;
    let mut majority_unmasked = 0u64;
    for bit_idx in 0..bit_width {
        if majority_votes[bit_idx] == 0 {
            continue;
        }
        majority_unmasked += 1;
        if majority_bits[bit_idx] == original_bits.get(bit_idx).copied().unwrap_or(false) {
            majority_matches += 1;
        }
    }
    let majority_denom = majority_unmasked.max(1);
    let majority_pct = majority_matches as f64 / majority_denom as f64 * 100.0;
    let majority_y =
        bits_top + bit_size + row_gap + row.entries.len() as f32 * (bit_size + row_gap);
    let majority_boxes_y = majority_y + box_offset;
    painter.text(
        egui::pos2(label_x, majority_y),
        egui::Align2::LEFT_TOP,
        format!(
            "Majority vote | adj={majority_pct:.2}% ({majority_matches}/{majority_denom})"
        ),
        egui::FontId::proportional(11.0),
        palette.label_color,
    );
    for bit_idx in 0..bit_width {
        let votes = majority_votes[bit_idx];
        let masked = votes == 0;
        let majority_bit = majority_bits[bit_idx];
        let matches_original = !masked
            && majority_bit == original_bits.get(bit_idx).copied().unwrap_or(false);
        let base_candidate = if matches_original && votes > 1 {
            palette.multi_match_color
        } else if matches_original {
            palette.match_color
        } else {
            palette.mismatch_color
        };
        let color = if masked {
            palette.masked_fill
        } else if !majority_bit {
            lighten_color(base_candidate, 0.45)
        } else {
            base_candidate
        };
        let x = boxes_start + bit_idx as f32 * (bit_size + bit_spacing);
        let rect = egui::Rect::from_min_size(
            egui::pos2(x, majority_boxes_y),
            egui::vec2(bit_size, bit_size),
        );
        painter.rect_filled(rect, 0.0, color);
        painter.rect_stroke(rect, 0.0, palette.stroke);
        let text_color = if masked {
            palette.masked_text
        } else {
            text_color_for(color)
        };
        painter.text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            if majority_bit { "1" } else { "0" },
            bit_font.clone(),
            text_color,
        );
    }
}

fn row_height_for(
    row: &BitSimilarityRow,
    bit_size: f32,
    row_gap: f32,
    header_height: f32,
    header_gap: f32,
    row_padding: f32,
    box_offset: f32,
) -> f32 {
    let entries = row.entries.len();
    header_height
        + header_gap
        + bit_size
        + box_offset
        + (entries as f32 + 1.0) * (bit_size + row_gap)
        + row_padding
}

fn hex_to_bits_le(hex_str: &str, bit_width: usize) -> Vec<bool> {
    if bit_width == 0 {
        return Vec::new();
    }
    let mut cleaned = hex_str.trim().to_string();
    if cleaned.starts_with("0x") || cleaned.starts_with("0X") {
        cleaned = cleaned[2..].to_string();
    }
    if cleaned.len() % 2 == 1 {
        cleaned = format!("0{cleaned}");
    }
    let bytes = match hex::decode(&cleaned) {
        Ok(bytes) => bytes,
        Err(_) => Vec::new(),
    };
    let mut bits = Vec::with_capacity(bit_width);
    for bit_idx in 0..bit_width {
        let byte_pos = bit_idx / 8;
        let bit_in_byte = bit_idx % 8;
        let idx_from_end = bytes.len().saturating_sub(1 + byte_pos);
        let bit = if idx_from_end < bytes.len() {
            (bytes[idx_from_end] >> bit_in_byte) & 1 == 1
        } else {
            false
        };
        bits.push(bit);
    }
    bits
}

fn lighten_color(color: egui::Color32, factor: f32) -> egui::Color32 {
    let factor = factor.clamp(0.0, 1.0);
    let r = color.r() as f32 + (255.0 - color.r() as f32) * factor;
    let g = color.g() as f32 + (255.0 - color.g() as f32) * factor;
    let b = color.b() as f32 + (255.0 - color.b() as f32) * factor;
    egui::Color32::from_rgb(r as u8, g as u8, b as u8)
}

struct BitSimilarityPalette {
    match_color: egui::Color32,
    mismatch_color: egui::Color32,
    multi_match_color: egui::Color32,
    masked_fill: egui::Color32,
    masked_text: egui::Color32,
    label_color: egui::Color32,
    stroke: egui::Stroke,
}

fn bit_similarity_palette(ui: &egui::Ui) -> BitSimilarityPalette {
    if ui.visuals().dark_mode {
        BitSimilarityPalette {
            match_color: egui::Color32::from_rgb(72, 196, 118),
            mismatch_color: egui::Color32::from_rgb(232, 96, 96),
            multi_match_color: egui::Color32::from_rgb(255, 214, 102),
            masked_fill: egui::Color32::from_rgb(28, 28, 28),
            masked_text: egui::Color32::from_rgb(146, 230, 176),
            label_color: egui::Color32::from_rgb(220, 220, 220),
            stroke: egui::Stroke::new(1.0, egui::Color32::from_rgb(90, 90, 90)),
        }
    } else {
        BitSimilarityPalette {
            match_color: egui::Color32::from_rgb(46, 160, 67),
            mismatch_color: egui::Color32::from_rgb(220, 72, 72),
            multi_match_color: egui::Color32::from_rgb(242, 201, 76),
            masked_fill: egui::Color32::from_rgb(0, 0, 0),
            masked_text: egui::Color32::from_rgb(46, 160, 67),
            label_color: egui::Color32::from_rgb(40, 40, 40),
            stroke: egui::Stroke::new(1.0, egui::Color32::from_rgb(160, 160, 160)),
        }
    }
}

fn text_color_for(color: egui::Color32) -> egui::Color32 {
    let luminance =
        0.2126 * color.r() as f32 + 0.7152 * color.g() as f32 + 0.0722 * color.b() as f32;
    if luminance > 140.0 {
        egui::Color32::BLACK
    } else {
        egui::Color32::WHITE
    }
}
