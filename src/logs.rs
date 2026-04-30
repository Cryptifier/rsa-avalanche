/// Eclipse Public License 2.0
/// SPDX-License-Identifier: EPL-2.0
/// Copyright (c) 2025 Nicholas LaRoche <nlaroche@cryptifier.dev>
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};

use serde::Serialize;

use crate::analytics::{AnalyticsCliInfo, SessionAnalytics};
use crate::bitflow::{BitflowCandidate, BitflowConfig};

/// Errors returned by JSON log writers.
#[derive(Debug)]
pub enum LogError {
    Io(std::io::Error),
    Serialize(serde_json::Error),
}

impl std::fmt::Display for LogError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LogError::Io(err) => write!(f, "log IO error: {err}"),
            LogError::Serialize(err) => write!(f, "log serialization error: {err}"),
        }
    }
}

impl std::error::Error for LogError {}

impl From<std::io::Error> for LogError {
    fn from(err: std::io::Error) -> Self {
        LogError::Io(err)
    }
}

impl From<serde_json::Error> for LogError {
    fn from(err: serde_json::Error) -> Self {
        LogError::Serialize(err)
    }
}

/// Streaming JSON (NDJSON) writer for analytics events.
#[derive(Debug)]
pub struct LogWriter<W: Write> {
    writer: W,
}

impl LogWriter<BufWriter<File>> {
    /// Creates a log writer that writes NDJSON to `path`.
    ///
    /// # Parameters
    /// - `path`: Output file path for the log stream.
    ///
    /// # Returns
    /// - `Result<LogWriter<BufWriter<File>>, LogError>`: Writer or error.
    ///
    /// # Expected Output
    /// - Creates/overwrites the file at `path`.
    pub fn create(path: &str) -> Result<Self, LogError> {
        let file = File::create(path)?;
        Ok(Self {
            writer: BufWriter::new(file),
        })
    }

    /// Creates a log writer that appends NDJSON to `path`.
    ///
    /// # Parameters
    /// - `path`: Output file path for the log stream.
    ///
    /// # Returns
    /// - `Result<LogWriter<BufWriter<File>>, LogError>`: Writer or error.
    ///
    /// # Expected Output
    /// - Opens the file at `path` in append mode, creating it when missing.
    pub fn create_append(path: &str) -> Result<Self, LogError> {
        let file = OpenOptions::new().create(true).append(true).open(path)?;
        Ok(Self {
            writer: BufWriter::new(file),
        })
    }
}

impl<W: Write> LogWriter<W> {
    /// Writes a single NDJSON event line.
    ///
    /// # Parameters
    /// - `event`: Event name identifier.
    /// - `payload`: Serializable payload for the event.
    ///
    /// # Returns
    /// - `Result<(), LogError>`: `Ok(())` on success.
    ///
    /// # Expected Output
    /// - Appends one JSON line to the output stream.
    pub fn write_event<T: Serialize>(&mut self, event: &str, payload: &T) -> Result<(), LogError> {
        let envelope = LogEnvelope { event, payload };
        serde_json::to_writer(&mut self.writer, &envelope)?;
        self.writer.write_all(b"\n")?;
        Ok(())
    }

    /// Flushes buffered output to the underlying writer.
    ///
    /// # Parameters
    /// - None.
    ///
    /// # Returns
    /// - `Result<(), LogError>`: `Ok(())` on success.
    ///
    /// # Expected Output
    /// - Flushes output; no other side effects.
    pub fn flush(&mut self) -> Result<(), LogError> {
        self.writer.flush()?;
        Ok(())
    }
}

/// Writes the session-start event for an analytics stream.
///
/// # Parameters
/// - `writer`: Log writer for NDJSON output.
/// - `started_unix_ms`: Session start timestamp in milliseconds since UNIX epoch.
/// - `cli`: CLI metadata captured for the session.
///
/// # Returns
/// - `Result<(), LogError>`: `Ok(())` on success.
///
/// # Expected Output
/// - Appends one `session_start` event to the output stream.
pub(crate) fn write_session_start<W: Write>(
    writer: &mut LogWriter<W>,
    started_unix_ms: u128,
    cli: &AnalyticsCliInfo,
) -> Result<(), LogError> {
    writer.write_event(
        "session_start",
        &SessionStart {
            started_unix_ms,
            cli,
        },
    )
}

/// Writes the session-finish event for an analytics stream.
///
/// # Parameters
/// - `writer`: Log writer for NDJSON output.
/// - `finished_unix_ms`: Optional session-finish timestamp in milliseconds since UNIX epoch.
/// - `errors`: Terminal error list captured for the session.
///
/// # Returns
/// - `Result<(), LogError>`: `Ok(())` on success.
///
/// # Expected Output
/// - Appends one `session_finish` event to the output stream.
pub(crate) fn write_session_finish<W: Write>(
    writer: &mut LogWriter<W>,
    finished_unix_ms: Option<u128>,
    errors: &[String],
) -> Result<(), LogError> {
    writer.write_event(
        "session_finish",
        &SessionFinish {
            finished_unix_ms,
            errors: errors.to_vec(),
        },
    )
}

/// Writes a full session analytics log as NDJSON.
///
/// # Parameters
/// - `session`: Analytics session to serialize.
///
/// # Returns
/// - `Result<(), LogError>`: `Ok(())` on success.
///
/// # Expected Output
/// - Writes pending NDJSON analytics events to the configured session path.
pub fn write_session_log(session: &mut SessionAnalytics) -> Result<(), LogError> {
    let mut writer = if let Some(writer) = session.stream_writer.take() {
        writer
    } else if session.stream_started {
        LogWriter::create_append(session.session_json_path())?
    } else {
        let mut writer = LogWriter::create(session.session_json_path())?;
        write_session_start(&mut writer, session.started_unix_ms, &session.cli)?;
        writer
    };
    for step in &session.steps {
        writer.write_event("step", step)?;
    }
    for summary in &session.step_summaries {
        writer.write_event("step_summary", summary)?;
    }
    for feature in &session.features {
        writer.write_event("feature", feature)?;
    }
    for batch in &session.r_candidate_batches {
        writer.write_event("r_candidate_batch", batch)?;
    }
    for batch in &session.r_candidate_accuracy_batches {
        writer.write_event("r_candidate_accuracy_batch", batch)?;
    }
    for batch in &session.r_candidate_traces {
        writer.write_event("r_candidate_trace_batch", batch)?;
    }
    write_session_finish(&mut writer, session.finished_unix_ms, &session.errors)?;
    writer.flush()?;
    Ok(())
}

/// Writes a bitflow run event and its candidates as NDJSON.
///
/// # Parameters
/// - `writer`: Log writer for NDJSON output.
/// - `run_id`: Unique identifier for the bitflow run.
/// - `config`: Bitflow configuration for the run.
/// - `message_bits`: Source message bits used for candidate generation.
/// - `candidates`: Candidate list generated by the run.
///
/// # Returns
/// - `Result<(), LogError>`: `Ok(())` on success.
///
/// # Expected Output
/// - Appends bitflow run metadata and candidates to the log stream.
pub fn write_bitflow_log<W: Write>(
    writer: &mut LogWriter<W>,
    run_id: &str,
    config: &BitflowConfig,
    message_bits: &[bool],
    candidates: &[BitflowCandidate],
) -> Result<(), LogError> {
    writer.write_event(
        "bitflow_run",
        &BitflowRun {
            run_id: run_id.to_string(),
            bit_width: config.bit_width,
            min_partition_size: config.min_partition_size,
            max_partition_size: config.max_partition_size,
            progression: format_progression(&config.progression),
            max_iterations: config.max_iterations,
            max_partitions_to_flip: config.max_partitions_to_flip,
            per_candidate_trials: config.per_candidate_trials,
            seed: config.seed,
            pow_mod_base: config.pow_mod_base,
            pow_mod_modulus: config.pow_mod_modulus,
            message_bits: message_bits.iter().map(|bit| *bit as u8).collect(),
        },
    )?;
    for candidate in candidates {
        writer.write_event(
            "bitflow_candidate",
            &BitflowCandidateLog {
                run_id: run_id.to_string(),
                iteration: candidate.iteration,
                trial: candidate.trial,
                partition_size: candidate.partition_size,
                inverted_partitions: candidate.inverted_partitions.clone(),
                bits: candidate.bits.iter().map(|bit| *bit as u8).collect(),
            },
        )?;
    }
    Ok(())
}

#[derive(Serialize)]
struct LogEnvelope<'a, T> {
    event: &'a str,
    payload: &'a T,
}

#[derive(Serialize)]
struct SessionStart<'a> {
    started_unix_ms: u128,
    cli: &'a AnalyticsCliInfo,
}

#[derive(Serialize)]
struct SessionFinish {
    finished_unix_ms: Option<u128>,
    errors: Vec<String>,
}

#[derive(Serialize)]
struct BitflowRun {
    run_id: String,
    bit_width: usize,
    min_partition_size: usize,
    max_partition_size: usize,
    progression: String,
    max_iterations: usize,
    max_partitions_to_flip: usize,
    per_candidate_trials: usize,
    seed: u64,
    pow_mod_base: u64,
    pow_mod_modulus: u64,
    message_bits: Vec<u8>,
}

#[derive(Serialize)]
struct BitflowCandidateLog {
    run_id: String,
    iteration: usize,
    trial: usize,
    partition_size: usize,
    inverted_partitions: Vec<usize>,
    bits: Vec<u8>,
}

fn format_progression(progression: &crate::bitflow::PartitionProgression) -> String {
    match progression {
        crate::bitflow::PartitionProgression::Fixed { size } => format!("fixed:{size}"),
        crate::bitflow::PartitionProgression::Linear { start, step } => {
            format!("linear:{start}:{step}")
        }
        crate::bitflow::PartitionProgression::Geometric { start, factor } => {
            format!("geometric:{start}:{factor}")
        }
        crate::bitflow::PartitionProgression::Sequence { sizes } => format!(
            "sequence:{}",
            sizes
                .iter()
                .map(|size| size.to_string())
                .collect::<Vec<_>>()
                .join(",")
        ),
    }
}
