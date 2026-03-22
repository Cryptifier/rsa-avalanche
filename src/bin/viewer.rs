use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use eframe::egui;
use egui_extras::{Column, TableBuilder};
use egui_plot::{Plot, PlotPoints, Points};
use serde_json::{Map, Value};

/// Entry point for the egui-based session viewer.
fn main() -> eframe::Result<()> {
    let args = ViewerArgs::parse();
    let native_options = eframe::NativeOptions::default();
    eframe::run_native(
        "RSA Session Viewer (egui)",
        native_options,
        Box::new(|_cc| Box::new(ViewerApp::new(args))),
    )
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

#[derive(Debug)]
struct ViewerApp {
    session: Session,
    session_path: PathBuf,
    log_dir: PathBuf,
    log_paths: Vec<PathBuf>,
    selected_path: Option<PathBuf>,
    status: String,
    last_poll: Instant,
    last_scan: Instant,
    ndjson_mode: bool,
    offset: u64,
    buffer: String,
    tab: Tab,
    bit_true_bit_idx: usize,
    bitflow_selected: Option<String>,
}

impl ViewerApp {
    fn new(args: ViewerArgs) -> Self {
        let mut app = Self {
            session: Session::default(),
            session_path: args.session_path,
            log_dir: args.log_dir,
            log_paths: Vec::new(),
            selected_path: None,
            status: String::new(),
            last_poll: Instant::now(),
            last_scan: Instant::now(),
            ndjson_mode: false,
            offset: 0,
            buffer: String::new(),
            tab: Tab::Summary,
            bit_true_bit_idx: 0,
            bitflow_selected: None,
        };
        app.refresh_logs(true);
        app
    }

    fn refresh_logs(&mut self, select_default: bool) {
        self.log_paths = collect_log_paths(&self.session_path, &self.log_dir);
        if select_default {
            let selected = self
                .log_paths
                .first()
                .cloned()
                .or_else(|| Some(self.session_path.clone()));
            if let Some(path) = selected {
                let _ = self.load_session(&path);
            }
        }
    }

    fn load_session(&mut self, path: &Path) -> Result<(), String> {
        let (session, ndjson) = load_session_with_mode(path)?;
        self.session = session;
        self.ndjson_mode = ndjson;
        self.offset = file_size(path).unwrap_or(0);
        self.buffer.clear();
        self.selected_path = Some(path.to_path_buf());
        self.status = format!("Loaded {}", path.display());
        Ok(())
    }

    fn poll_updates(&mut self) {
        let now = Instant::now();
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
        let Some(path) = self.selected_path.clone() else {
            return;
        };
        let updated = self.ingest_tail(&path);
        if updated {
            self.status = format!("Updated {}", path.display());
        }
    }

    fn ingest_tail(&mut self, path: &Path) -> bool {
        let Ok(mut file) = File::open(path) else {
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

    fn draw_summary(&self, ui: &mut egui::Ui) {
        ui.heading("Summary");
        let mut rows = Vec::new();
        rows.push(("Started", format_unix_ms(self.session.started_unix_ms)));
        rows.push(("Finished", format_unix_ms(self.session.finished_unix_ms)));
        if let (Some(start), Some(end)) = (self.session.started_unix_ms, self.session.finished_unix_ms)
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

        TableBuilder::new(ui)
            .columns(Column::auto(), 2)
            .striped(true)
            .header(20.0, |mut header| {
                header.col(|ui| {
                    ui.label("Metric");
                });
                header.col(|ui| {
                    ui.label("Value");
                });
            })
            .body(|mut body| {
                for (metric, value) in rows {
                    body.row(20.0, |mut row| {
                        row.col(|ui| {
                            ui.label(metric);
                        });
                        row.col(|ui| {
                            ui.label(value);
                        });
                    });
                }
            });

        ui.add_space(12.0);
        ui.heading("Feature Summary");
        TableBuilder::new(ui)
            .columns(Column::auto(), 4)
            .striped(true)
            .header(20.0, |mut header| {
                header.col(|ui| ui.label("Feature"));
                header.col(|ui| ui.label("Enabled"));
                header.col(|ui| ui.label("Duration (ms)"));
                header.col(|ui| ui.label("Notes"));
            })
            .body(|mut body| {
                for feature in &self.session.features {
                    body.row(20.0, |mut row| {
                        row.col(|ui| ui.label(&feature.name));
                        row.col(|ui| ui.label(feature.enabled.to_string()));
                        row.col(|ui| {
                            ui.label(opt_to_string(feature.duration_ms.map(|v| v as u128)))
                        });
                        row.col(|ui| ui.label(feature.notes.join("; ")));
                    });
                }
            });
    }

    fn draw_candidates(&self, ui: &mut egui::Ui) {
        ui.heading("r Candidate Batches");
        let rows = flatten_candidate_batches(&self.session);
        if rows.is_empty() {
            ui.label("No r-candidate batches recorded.");
            return;
        }
        TableBuilder::new(ui)
            .columns(Column::auto(), 6)
            .striped(true)
            .header(20.0, |mut header| {
                header.col(|ui| ui.label("Context"));
                header.col(|ui| ui.label("Mode"));
                header.col(|ui| ui.label("Index"));
                header.col(|ui| ui.label("r"));
                header.col(|ui| ui.label("Bits"));
                header.col(|ui| ui.label("Factors"));
            })
            .body(|mut body| {
                for row in rows {
                    body.row(20.0, |mut row_ui| {
                        row_ui.col(|ui| ui.label(row.context));
                        row_ui.col(|ui| ui.label(row.mode));
                        row_ui.col(|ui| ui.label(row.index.to_string()));
                        row_ui.col(|ui| ui.label(row.r));
                        row_ui.col(|ui| ui.label(row.r_bits.to_string()));
                        row_ui.col(|ui| ui.label(row.factors));
                    });
                }
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

        TableBuilder::new(ui)
            .columns(Column::auto(), 10)
            .striped(true)
            .header(20.0, |mut header| {
                header.col(|ui| ui.label("Batch"));
                header.col(|ui| ui.label("Beam Match %"));
                header.col(|ui| ui.label("Beam Ones %"));
                header.col(|ui| ui.label("Beam Score"));
                header.col(|ui| ui.label("Beam Bits"));
                header.col(|ui| ui.label("R Mean %"));
                header.col(|ui| ui.label("R Max %"));
                header.col(|ui| ui.label("R Min %"));
                header.col(|ui| ui.label("R Stddev"));
                header.col(|ui| ui.label("Candidates"));
            })
            .body(|mut body| {
                for row in rows {
                    body.row(20.0, |mut row_ui| {
                        row_ui.col(|ui| ui.label(row.batch));
                        row_ui.col(|ui| ui.label(format_opt_f64(row.beam_match)));
                        row_ui.col(|ui| ui.label(format_opt_f64(row.beam_ones)));
                        row_ui.col(|ui| ui.label(format_opt_f64(row.beam_score)));
                        row_ui.col(|ui| ui.label(opt_to_string(row.beam_bits.map(|v| v as u128))));
                        row_ui.col(|ui| ui.label(format_opt_f64(row.r_mean)));
                        row_ui.col(|ui| ui.label(format_opt_f64(row.r_max)));
                        row_ui.col(|ui| ui.label(format_opt_f64(row.r_min)));
                        row_ui.col(|ui| ui.label(format_opt_f64(row.r_stddev)));
                        row_ui.col(|ui| ui.label(row.candidate_count.to_string()));
                    });
                }
            });
    }

    fn draw_bit_similarity(&self, ui: &mut egui::Ui) {
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
        let bit_width = value_as_usize(map.get("bit_width"));
        let shift_configured = value_as_u64(map.get("shift_levels_configured"));
        let shift_used = value_as_u64(map.get("shift_levels_used"));
        let original_hex = value_as_string(map.get("original_hex"));
        let match_counts = value_as_vec_u64(map.get("match_counts_per_bit"));
        let candidates = map
            .get("candidates")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();

        ui.label(format!("Bit width: {}", bit_width));
        ui.label(format!(
            "Shift levels: configured {}, used {}",
            shift_configured, shift_used
        ));
        ui.label(format!("Original hex: {}", original_hex));
        ui.add_space(8.0);

        if !match_counts.is_empty() {
            let points = match_counts
                .iter()
                .enumerate()
                .map(|(idx, count)| [idx as f64, *count as f64])
                .collect::<PlotPoints>();
            Plot::new("bit_similarity_counts").show(ui, |plot_ui| {
                plot_ui.points(Points::new(points).name("Match counts per bit"));
            });
        }

        ui.add_space(8.0);
        if candidates.is_empty() {
            ui.label("No bit similarity candidates recorded.");
            return;
        }
        TableBuilder::new(ui)
            .columns(Column::auto(), 9)
            .striped(true)
            .header(20.0, |mut header| {
                header.col(|ui| ui.label("Index"));
                header.col(|ui| ui.label("Shift"));
                header.col(|ui| ui.label("r"));
                header.col(|ui| ui.label("x"));
                header.col(|ui| ui.label("Match %"));
                header.col(|ui| ui.label("Adj Match %"));
                header.col(|ui| ui.label("Masked Bits"));
                header.col(|ui| ui.label("Base Match %"));
                header.col(|ui| ui.label("Candidate Hex"));
            })
            .body(|mut body| {
                for candidate in candidates {
                    let Some(obj) = candidate.as_object() else {
                        continue;
                    };
                    body.row(20.0, |mut row| {
                        row.col(|ui| ui.label(value_as_u64(obj.get("index")).to_string()));
                        row.col(|ui| ui.label(value_as_u64(obj.get("shift")).to_string()));
                        row.col(|ui| ui.label(value_as_string(obj.get("r"))));
                        row.col(|ui| ui.label(value_as_string(obj.get("x"))));
                        row.col(|ui| ui.label(format_opt_f64(value_as_opt_f64(obj.get("match_pct")))));
                        row.col(|ui| ui.label(format_opt_f64(value_as_opt_f64(obj.get("adjusted_match_pct")))));
                        row.col(|ui| ui.label(value_as_u64(obj.get("masked_bits")).to_string()));
                        row.col(|ui| ui.label(format_opt_f64(value_as_opt_f64(obj.get("base_match_pct")))));
                        row.col(|ui| ui.label(value_as_string(obj.get("candidate_hex"))));
                    });
                }
            });
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

        TableBuilder::new(ui)
            .columns(Column::auto(), 7)
            .striped(true)
            .header(20.0, |mut header| {
                header.col(|ui| ui.label("Run"));
                header.col(|ui| ui.label("Iter"));
                header.col(|ui| ui.label("Trial"));
                header.col(|ui| ui.label("Partition"));
                header.col(|ui| ui.label("Inverted"));
                header.col(|ui| ui.label("Ones %"));
                header.col(|ui| ui.label("Bits"));
            })
            .body(|mut body| {
                for candidate in filtered_candidates {
                    let ones_pct = bits_ones_pct(&candidate.bits);
                    body.row(20.0, |mut row| {
                        row.col(|ui| ui.label(candidate.run_id));
                        row.col(|ui| ui.label(candidate.iteration.to_string()));
                        row.col(|ui| ui.label(candidate.trial.to_string()));
                        row.col(|ui| ui.label(candidate.partition_size.to_string()));
                        row.col(|ui| {
                            ui.label(
                                candidate
                                    .inverted_partitions
                                    .iter()
                                    .map(|v| v.to_string())
                                    .collect::<Vec<_>>()
                                    .join(","),
                            )
                        });
                        row.col(|ui| ui.label(format!("{ones_pct:.1}")));
                        row.col(|ui| ui.label(bits_preview(&candidate.bits, 96)));
                    });
                }
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
                if let Some(path) = &self.selected_path {
                    ui.monospace(path.display().to_string());
                } else {
                    ui.monospace("none");
                }
                if ui.button("Reload").clicked() {
                    if let Some(path) = self.selected_path.clone() {
                        let _ = self.load_session(&path);
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

        egui::SidePanel::left("log_list")
            .default_width(220.0)
            .show(ctx, |ui| {
                ui.heading("Logs");
                egui::ScrollArea::vertical().show(ui, |ui| {
                    for path in &self.log_paths {
                        let label = path
                            .file_name()
                            .map(|name| name.to_string_lossy().to_string())
                            .unwrap_or_else(|| path.display().to_string());
                        let selected = self
                            .selected_path
                            .as_ref()
                            .map(|p| p == path)
                            .unwrap_or(false);
                        if ui.selectable_label(selected, label).clicked() {
                            let _ = self.load_session(path);
                        }
                    }
                });
            });

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
struct StepTiming {
    name: String,
    duration_ms: u128,
}

#[derive(Debug, Default, Clone)]
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
struct RCandidateAccuracyEntry {
    r: String,
    r_bits: u64,
    factors: Vec<RCandidateFactor>,
    accuracy_pct: f64,
    hbc_ciphertexts_r: Vec<String>,
    candidate_decryptions: Vec<String>,
}

#[derive(Debug, Default, Clone)]
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
struct RCandidateTraceEntry {
    r: String,
    r_bits: u64,
    hbc_ciphertext_r: String,
    candidate_decryption: String,
}

#[derive(Debug, Default, Clone)]
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

fn collect_log_paths(session_path: &Path, log_dir: &Path) -> Vec<PathBuf> {
    let mut results = Vec::new();
    let mut seen = HashMap::new();
    let candidates = vec![
        session_path.to_path_buf(),
        PathBuf::from("session.json"),
        PathBuf::from("session.log"),
    ];
    for path in candidates {
        if path.exists() && seen.insert(path.clone(), ()).is_none() {
            results.push(path);
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
                let modified = entry
                    .metadata()
                    .and_then(|meta| meta.modified())
                    .ok();
                entries.push((modified, path));
            }
        }
        entries.sort_by_key(|(mtime, _)| *mtime);
        entries.reverse();
        for (_, path) in entries {
            if seen.insert(path.clone(), ()).is_none() {
                results.push(path);
            }
        }
    }
    results
}

fn file_size(path: &Path) -> Option<u64> {
    std::fs::metadata(path).ok().map(|meta| meta.len())
}

fn load_session_with_mode(path: &Path) -> Result<(Session, bool), String> {
    let mut file = File::open(path).map_err(|err| err.to_string())?;
    let mut raw = String::new();
    file.read_to_string(&mut raw)
        .map_err(|err| err.to_string())?;
    if raw.trim().is_empty() {
        return Ok((Session::default(), false));
    }
    if let Ok(value) = serde_json::from_str::<Value>(&raw) {
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
