/// Eclipse Public License 2.0
/// SPDX-License-Identifier: EPL-2.0
/// Copyright (c) 2025 Nicholas LaRoche <nlaroche@cryptifier.dev>
use bigdecimal::BigDecimal;
use diesel::dsl::count_star;
use diesel::prelude::*;
use diesel::r2d2::{ConnectionManager, CustomizeConnection, Error as R2D2Error, Pool};
use diesel::sql_query;
use diesel::sql_types::Text;
use num_bigint::BigUint;
use std::{
    collections::HashMap,
    error::Error,
    fs,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use crate::analytics::{
    AvalancheCenterBiasEntry, AvalancheCombinationBeamResult, AvalancheTierSampleStat,
    AvalancheTierStatistics,
};
use crate::avalanche::AvalancheNode;
use crate::config::EngineConfig;
use crate::fitness::RankedScoredAvalancheInput;
use crate::helpers::PackedBits;
use crate::methods::{
    ScoredAvalancheInput, ScoredAvalancheInputDetail, SelectedAvalancheSample, recursive_tier_bits,
};

type AvalancheSqlitePool = Pool<ConnectionManager<SqliteConnection>>;

diesel::table! {
    avalanche_cache_inputs (id) {
        id -> BigInt,
        batch_number -> Integer,
        batch_candidate_index -> Integer,
        message_index -> Integer,
        r_text -> Text,
        x_text -> Text,
        score_match_pct -> Double,
        contents_have_been_inverted -> Bool,
        fitness_score -> BigInt,
        fitness_total_score -> BigInt,
        fitness_message_count -> BigInt,
        message_bits -> Binary,
        message_bit_len -> Integer,
        target_exponent_text -> Nullable<Text>,
        hbc_ciphertext_r_text -> Nullable<Text>,
        candidate_decryption_text -> Nullable<Text>,
    }
}

diesel::table! {
    avalanche_cache_samples (id) {
        id -> BigInt,
        batch_number -> Integer,
        tier_index -> Integer,
        sample_index -> Integer,
        input_count -> Integer,
        average_score_pct -> Double,
        top_beam_score -> Double,
        top_beam_match_pct -> Nullable<Double>,
        best_match_pct -> Double,
        majority_vote_match_pct -> Double,
        majority_vote_ones_match_pct -> Double,
        best_bits -> Binary,
        best_bits_bit_len -> Integer,
        majority_vote_bits -> Binary,
        majority_vote_bits_bit_len -> Integer,
        recursive_bits -> Binary,
        recursive_bits_bit_len -> Integer,
        beam_results_json -> Text,
        center_biases_json -> Text,
    }
}

diesel::allow_tables_to_appear_in_same_query!(avalanche_cache_inputs, avalanche_cache_samples);

#[derive(Debug, Insertable)]
#[diesel(table_name = avalanche_cache_inputs)]
struct NewCachedAvalancheInput {
    batch_number: i32,
    batch_candidate_index: i32,
    message_index: i32,
    r_text: String,
    x_text: String,
    score_match_pct: f64,
    contents_have_been_inverted: bool,
    fitness_score: i64,
    fitness_total_score: i64,
    fitness_message_count: i64,
    message_bits: Vec<u8>,
    message_bit_len: i32,
    target_exponent_text: Option<String>,
    hbc_ciphertext_r_text: Option<String>,
    candidate_decryption_text: Option<String>,
}

#[derive(Debug, Queryable, Selectable)]
#[diesel(table_name = avalanche_cache_inputs)]
pub(crate) struct CachedAvalancheInputRow {
    pub(crate) id: i64,
    pub(crate) batch_candidate_index: i32,
    pub(crate) message_index: i32,
    pub(crate) r_text: String,
    pub(crate) x_text: String,
    pub(crate) score_match_pct: f64,
    pub(crate) contents_have_been_inverted: bool,
    pub(crate) fitness_score: i64,
    pub(crate) fitness_total_score: i64,
    pub(crate) fitness_message_count: i64,
    pub(crate) message_bits: Vec<u8>,
    pub(crate) message_bit_len: i32,
    pub(crate) target_exponent_text: Option<String>,
    pub(crate) hbc_ciphertext_r_text: Option<String>,
    pub(crate) candidate_decryption_text: Option<String>,
}

#[derive(Debug, Insertable)]
#[diesel(table_name = avalanche_cache_samples)]
struct NewCachedAvalancheSample {
    batch_number: i32,
    tier_index: i32,
    sample_index: i32,
    input_count: i32,
    average_score_pct: f64,
    top_beam_score: f64,
    top_beam_match_pct: Option<f64>,
    best_match_pct: f64,
    majority_vote_match_pct: f64,
    majority_vote_ones_match_pct: f64,
    best_bits: Vec<u8>,
    best_bits_bit_len: i32,
    majority_vote_bits: Vec<u8>,
    majority_vote_bits_bit_len: i32,
    recursive_bits: Vec<u8>,
    recursive_bits_bit_len: i32,
    beam_results_json: String,
    center_biases_json: String,
}

#[derive(Debug, Queryable, Selectable)]
#[diesel(table_name = avalanche_cache_samples)]
pub(crate) struct CachedAvalancheSampleRow {
    pub(crate) id: i64,
    pub(crate) tier_index: i32,
    pub(crate) sample_index: i32,
    pub(crate) input_count: i32,
    pub(crate) average_score_pct: f64,
    pub(crate) top_beam_score: f64,
    pub(crate) top_beam_match_pct: Option<f64>,
    pub(crate) best_match_pct: f64,
    pub(crate) majority_vote_match_pct: f64,
    pub(crate) majority_vote_ones_match_pct: f64,
    pub(crate) best_bits: Vec<u8>,
    pub(crate) best_bits_bit_len: i32,
    pub(crate) majority_vote_bits: Vec<u8>,
    pub(crate) majority_vote_bits_bit_len: i32,
    pub(crate) recursive_bits: Vec<u8>,
    pub(crate) recursive_bits_bit_len: i32,
    pub(crate) beam_results_json: String,
    pub(crate) center_biases_json: String,
}

#[derive(Debug, Clone)]
pub(crate) struct CachedRecursiveSampleSummary {
    pub(crate) id: i64,
}

#[derive(Debug, Clone, Copy)]
struct AvalancheSqliteSettings {
    soft_heap_limit_bytes: i64,
    hard_heap_limit_bytes: i64,
    mmap_size_bytes: i64,
}

#[derive(Debug, QueryableByName)]
struct SqliteTableInfoRow {
    #[diesel(sql_type = Text)]
    name: String,
}

#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct CachedPageLoadTiming {
    pub(crate) connection_wait: Duration,
    pub(crate) sql_query: Duration,
    pub(crate) row_decode: Duration,
}

impl CachedPageLoadTiming {
    /// Returns the total elapsed time captured by the timing breakdown.
    ///
    /// # Parameters
    /// - None.
    ///
    /// # Returns
    /// - `Duration`: Sum of connection wait, SQL execution, and row decoding.
    ///
    /// # Expected Output
    /// - Returns the aggregated timing without side effects.
    pub(crate) fn total(self) -> Duration {
        self.connection_wait + self.sql_query + self.row_decode
    }
}

impl std::ops::AddAssign for CachedPageLoadTiming {
    fn add_assign(&mut self, rhs: Self) {
        self.connection_wait += rhs.connection_wait;
        self.sql_query += rhs.sql_query;
        self.row_decode += rhs.row_decode;
    }
}

#[derive(Debug)]
pub(crate) struct CachedKeysetPage<T> {
    pub(crate) items: Vec<T>,
    pub(crate) last_row_id: Option<i64>,
    pub(crate) timing: CachedPageLoadTiming,
}

#[derive(Debug, Clone, Copy)]
struct AvalancheSqliteConnectionCustomizer {
    settings: AvalancheSqliteSettings,
}

/// Resolves the SQLite PRAGMA settings used by the Avalanche cache.
///
/// # Parameters
/// - `engine`: Engine configuration containing the configured SQLite byte limits.
///
/// # Returns
/// - `Result<AvalancheSqliteSettings, Box<dyn Error>>`: Validated SQLite settings ready for the connection pool.
///
/// # Expected Output
/// - Returns validated SQLite settings; no stdout/stderr output.
fn resolve_avalanche_sqlite_settings(
    engine: &EngineConfig,
) -> Result<AvalancheSqliteSettings, Box<dyn Error>> {
    let soft_heap_limit_bytes = i64::try_from(engine.sqlite_soft_heap)
        .map_err(|_| "engine.sqlite_soft_heap exceeds SQLite i64 range")?;
    let hard_heap_limit_bytes = i64::try_from(engine.sqlite_hard_heap)
        .map_err(|_| "engine.sqlite_hard_heap exceeds SQLite i64 range")?;
    let mmap_size_bytes = i64::try_from(engine.sqlite_mmap_size)
        .map_err(|_| "engine.sqlite_mmap_size exceeds SQLite i64 range")?;

    Ok(AvalancheSqliteSettings {
        soft_heap_limit_bytes,
        hard_heap_limit_bytes,
        mmap_size_bytes,
    })
}

/// Applies the Avalanche cache SQLite PRAGMA settings to one pooled connection.
///
/// # Parameters
/// - `connection`: SQLite connection that will back Avalanche cache reads and writes.
/// - `settings`: Validated SQLite PRAGMA settings derived from the engine config.
///
/// # Returns
/// - `Result<(), R2D2Error>`: `Ok(())` when all PRAGMAs are accepted by SQLite.
///
/// # Expected Output
/// - Updates SQLite connection settings in place; no stdout/stderr output.
fn configure_avalanche_cache_sqlite_connection(
    connection: &mut SqliteConnection,
    settings: AvalancheSqliteSettings,
) -> Result<(), R2D2Error> {
    sql_query(format!(
        "PRAGMA soft_heap_limit = {}",
        settings.soft_heap_limit_bytes
    ))
    .execute(connection)?;
    sql_query(format!(
        "PRAGMA hard_heap_limit = {}",
        settings.hard_heap_limit_bytes
    ))
    .execute(connection)?;
    sql_query(format!("PRAGMA mmap_size = {}", settings.mmap_size_bytes)).execute(connection)?;
    Ok(())
}

impl CustomizeConnection<SqliteConnection, R2D2Error> for AvalancheSqliteConnectionCustomizer {
    fn on_acquire(&self, connection: &mut SqliteConnection) -> Result<(), R2D2Error> {
        configure_avalanche_cache_sqlite_connection(connection, self.settings)
    }
}

/// Normalizes the configured SQLite cache folder for temporary Avalanche databases.
///
/// # Parameters
/// - `db_folder`: Configured filesystem folder for the SQLite cache database.
///
/// # Returns
/// - `PathBuf`: Normalized cache folder path.
///
/// # Expected Output
/// - Returns the configured folder or `/tmp` when the configured value is blank.
fn normalize_avalanche_cache_db_folder(db_folder: &str) -> PathBuf {
    let trimmed = db_folder.trim();
    if trimmed.is_empty() {
        PathBuf::from("/tmp")
    } else {
        PathBuf::from(trimmed)
    }
}

/// Resolves the on-disk SQLite path used for the temporary Avalanche cache.
///
/// # Parameters
/// - `seed`: Optional deterministic analysis seed used to key the cache filename.
/// - `db_folder`: Configured filesystem folder for the SQLite cache database.
///
/// # Returns
/// - `PathBuf`: Cache path in the configured folder using the configured or fallback seed.
///
/// # Expected Output
/// - Returns a deterministic filesystem path; no stdout/stderr output.
pub fn resolve_avalanche_cache_db_path(seed: Option<u64>, db_folder: &str) -> PathBuf {
    normalize_avalanche_cache_db_folder(db_folder)
        .join(format!("rsa_avalanche_{}.db", seed.unwrap_or(0)))
}

/// Resolves the SQLite connection target used for the temporary Avalanche cache.
///
/// # Parameters
/// - `seed`: Optional deterministic analysis seed used to key the cache identifier.
/// - `engine`: Engine configuration describing the cache storage mode.
///
/// # Returns
/// - `(String, PathBuf, Option<PathBuf>)`: `(connection_target, display_path, cleanup_path)`.
///
/// # Expected Output
/// - Returns either a shared in-memory SQLite URI or an on-disk path plus the optional cleanup
///   path; no stdout/stderr output.
fn resolve_avalanche_cache_connection_target(
    seed: Option<u64>,
    engine: &EngineConfig,
) -> (String, PathBuf, Option<PathBuf>) {
    if engine.sqlite_in_memory {
        let name = format!(
            "file:rsa_avalanche_{}?mode=memory&cache=shared",
            seed.unwrap_or(0)
        );
        let display_path = PathBuf::from(&name);
        (name, display_path, None)
    } else {
        let path = resolve_avalanche_cache_db_path(seed, &engine.sqlite_db_folder);
        let connection_target = path.to_string_lossy().to_string();
        (connection_target, path.clone(), Some(path))
    }
}

/// Creates the parent directory tree for the Avalanche cache database path.
///
/// # Parameters
/// - `path`: Full SQLite database path whose parent directory must exist.
///
/// # Returns
/// - `Result<(), Box<dyn Error>>`: `Ok(())` when the parent directory tree exists.
///
/// # Expected Output
/// - Creates intermediate directories for the database parent path when needed.
fn ensure_avalanche_cache_db_parent_dir(path: &Path) -> Result<(), Box<dyn Error>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    Ok(())
}

/// Removes the temporary Avalanche cache file if it exists.
///
/// # Parameters
/// - `seed`: Optional deterministic analysis seed used to key the cache filename.
/// - `db_folder`: Configured filesystem folder for the SQLite cache database.
///
/// # Returns
/// - `()`: This function returns nothing.
///
/// # Expected Output
/// - Deletes the cache file when present; no stdout/stderr output.
pub fn cleanup_avalanche_cache_db(seed: Option<u64>, db_folder: &str, in_memory: bool) {
    if in_memory {
        return;
    }
    let path = resolve_avalanche_cache_db_path(seed, db_folder);
    if path.exists() {
        let _ = fs::remove_file(path);
    }
}

/// Runtime SQLite cache used to spill Avalanche tier inputs and outputs to disk.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `AvalancheCacheGuard`: Pool-backed cache wrapper with automatic cleanup on drop.
///
/// # Expected Output
/// - Owns a SQLite connection pool and deletes the backing file on drop.
pub(crate) struct AvalancheCacheGuard {
    pub(crate) path: PathBuf,
    cleanup_path: Option<PathBuf>,
    pool: Option<AvalancheSqlitePool>,
    page_rows: i64,
}

/// Adds one column to a SQLite table when the schema is missing it.
///
/// # Parameters
/// - `connection`: SQLite connection to mutate.
/// - `table_name`: Table whose schema should be checked.
/// - `column_name`: Column that must exist.
/// - `column_definition`: Full SQLite column definition used for `ALTER TABLE`.
///
/// # Returns
/// - `Result<(), Box<dyn Error>>`: `Ok(())` when the column exists or is added successfully.
///
/// # Expected Output
/// - Reads SQLite table metadata and may mutate the table schema; no stdout/stderr output.
fn ensure_sqlite_column(
    connection: &mut SqliteConnection,
    table_name: &str,
    column_name: &str,
    column_definition: &str,
) -> Result<(), Box<dyn Error>> {
    let pragma = format!("PRAGMA table_info({table_name})");
    let rows = sql_query(pragma).load::<SqliteTableInfoRow>(connection)?;
    if rows.iter().any(|row| row.name == column_name) {
        return Ok(());
    }

    let alter = format!("ALTER TABLE {table_name} ADD COLUMN {column_definition}");
    sql_query(alter).execute(connection)?;
    Ok(())
}

impl AvalancheCacheGuard {
    /// Creates a new temporary Avalanche cache database and initializes its schema.
    ///
    /// # Parameters
    /// - `seed`: Optional deterministic analysis seed used to key the cache filename.
    /// - `engine`: Engine configuration that supplies the SQLite cache PRAGMA settings.
    ///
    /// # Returns
    /// - `Result<AvalancheCacheGuard, Box<dyn Error>>`: Ready-to-use cache wrapper.
    ///
    /// # Expected Output
    /// - Creates the SQLite database under the configured folder, creating parent directories as needed.
    pub(crate) fn new(seed: Option<u64>, engine: &EngineConfig) -> Result<Self, Box<dyn Error>> {
        let (connection_target, path, cleanup_path) =
            resolve_avalanche_cache_connection_target(seed, engine);
        if let Some(cleanup_path) = cleanup_path.as_ref() {
            ensure_avalanche_cache_db_parent_dir(cleanup_path)?;
            if cleanup_path.exists() {
                let _ = fs::remove_file(cleanup_path);
            }
        }
        let page_rows = i64::try_from(engine.sqlite_avalanche_page_size.max(1))
            .map_err(|_| "engine.sqlite_avalanche_page_size exceeds i64 range")?;
        let settings = resolve_avalanche_sqlite_settings(engine)?;
        let manager = ConnectionManager::<SqliteConnection>::new(connection_target);
        let pool = Pool::builder()
            .max_size(engine.sqlite_worker_count)
            .connection_customizer(Box::new(AvalancheSqliteConnectionCustomizer { settings }))
            .build(manager)?;
        let mut connection = pool.get()?;
        sql_query(
            "CREATE TABLE IF NOT EXISTS avalanche_cache_inputs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                batch_number INTEGER NOT NULL,
                batch_candidate_index INTEGER NOT NULL,
                message_index INTEGER NOT NULL,
                r_text TEXT NOT NULL,
                x_text TEXT NOT NULL,
                score_match_pct DOUBLE NOT NULL,
                contents_have_been_inverted BOOLEAN NOT NULL DEFAULT 0,
                fitness_score BIGINT NOT NULL DEFAULT 0,
                fitness_total_score BIGINT NOT NULL DEFAULT 0,
                fitness_message_count BIGINT NOT NULL DEFAULT 1,
                message_bits BLOB NOT NULL,
                message_bit_len INTEGER NOT NULL,
                target_exponent_text TEXT NULL,
                hbc_ciphertext_r_text TEXT NULL,
                candidate_decryption_text TEXT NULL
            )",
        )
        .execute(&mut connection)?;
        ensure_sqlite_column(
            &mut connection,
            "avalanche_cache_inputs",
            "contents_have_been_inverted",
            "contents_have_been_inverted BOOLEAN NOT NULL DEFAULT 0",
        )?;
        ensure_sqlite_column(
            &mut connection,
            "avalanche_cache_inputs",
            "fitness_score",
            "fitness_score BIGINT NOT NULL DEFAULT 0",
        )?;
        ensure_sqlite_column(
            &mut connection,
            "avalanche_cache_inputs",
            "fitness_total_score",
            "fitness_total_score BIGINT NOT NULL DEFAULT 0",
        )?;
        ensure_sqlite_column(
            &mut connection,
            "avalanche_cache_inputs",
            "fitness_message_count",
            "fitness_message_count BIGINT NOT NULL DEFAULT 1",
        )?;
        sql_query(
            "CREATE INDEX IF NOT EXISTS avalanche_cache_inputs_batch_idx
                ON avalanche_cache_inputs (batch_number, batch_candidate_index, message_index)",
        )
        .execute(&mut connection)?;
        sql_query(
            "CREATE INDEX IF NOT EXISTS avalanche_cache_inputs_batch_id_idx
                ON avalanche_cache_inputs (batch_number, id)",
        )
        .execute(&mut connection)?;
        sql_query(
            "CREATE TABLE IF NOT EXISTS avalanche_cache_samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                batch_number INTEGER NOT NULL,
                tier_index INTEGER NOT NULL,
                sample_index INTEGER NOT NULL,
                input_count INTEGER NOT NULL,
                average_score_pct DOUBLE NOT NULL,
                top_beam_score DOUBLE NOT NULL,
                top_beam_match_pct DOUBLE NULL,
                best_match_pct DOUBLE NOT NULL,
                majority_vote_match_pct DOUBLE NOT NULL,
                majority_vote_ones_match_pct DOUBLE NOT NULL,
                best_bits BLOB NOT NULL,
                best_bits_bit_len INTEGER NOT NULL,
                majority_vote_bits BLOB NOT NULL,
                majority_vote_bits_bit_len INTEGER NOT NULL,
                recursive_bits BLOB NOT NULL,
                recursive_bits_bit_len INTEGER NOT NULL,
                beam_results_json TEXT NOT NULL,
                center_biases_json TEXT NOT NULL DEFAULT '[]'
            )",
        )
        .execute(&mut connection)?;
        ensure_sqlite_column(
            &mut connection,
            "avalanche_cache_samples",
            "center_biases_json",
            "center_biases_json TEXT NOT NULL DEFAULT '[]'",
        )?;
        sql_query(
            "CREATE INDEX IF NOT EXISTS avalanche_cache_samples_batch_tier_idx
                ON avalanche_cache_samples (batch_number, tier_index, sample_index)",
        )
        .execute(&mut connection)?;
        Ok(Self {
            path,
            cleanup_path,
            pool: Some(pool),
            page_rows,
        })
    }

    /// Returns the pooled SQLite connections used by the cache.
    ///
    /// # Parameters
    /// - None.
    ///
    /// # Returns
    /// - `Result<&AvalancheSqlitePool, Box<dyn Error>>`: Shared SQLite pool.
    ///
    /// # Expected Output
    /// - Returns the initialized connection pool; no stdout/stderr output.
    pub(crate) fn pool(&self) -> Result<&AvalancheSqlitePool, Box<dyn Error>> {
        self.pool
            .as_ref()
            .ok_or_else(|| "avalanche cache pool is unavailable".into())
    }

    /// Returns the configured SQLite Avalanche cache page size as a row count.
    ///
    /// # Parameters
    /// - None.
    ///
    /// # Returns
    /// - `usize`: Positive page size in rows.
    ///
    /// # Expected Output
    /// - Returns the normalized cached page size; no stdout/stderr output.
    pub(crate) fn page_rows_usize(&self) -> usize {
        usize::try_from(self.page_rows).unwrap_or(1)
    }

    /// Returns the configured SQLite Avalanche cache page size as an `i64`.
    ///
    /// # Parameters
    /// - None.
    ///
    /// # Returns
    /// - `i64`: Positive page size in rows.
    ///
    /// # Expected Output
    /// - Returns the normalized cached page size; no stdout/stderr output.
    pub(crate) fn page_rows_i64(&self) -> i64 {
        self.page_rows
    }

    /// Deletes cached inputs and samples for one analysis batch.
    ///
    /// # Parameters
    /// - `batch_number`: One-based batch number whose rows should be removed.
    ///
    /// # Returns
    /// - `Result<(), Box<dyn Error>>`: `Ok(())` on success.
    ///
    /// # Expected Output
    /// - Removes persisted batch rows from the SQLite cache.
    pub(crate) fn clear_batch(&self, batch_number: usize) -> Result<(), Box<dyn Error>> {
        let batch_number =
            i32::try_from(batch_number).map_err(|_| "batch number exceeds i32 range")?;
        let mut connection = self.pool()?.get()?;
        diesel::delete(
            avalanche_cache_inputs::table
                .filter(avalanche_cache_inputs::batch_number.eq(batch_number)),
        )
        .execute(&mut connection)?;
        diesel::delete(
            avalanche_cache_samples::table
                .filter(avalanche_cache_samples::batch_number.eq(batch_number)),
        )
        .execute(&mut connection)?;
        Ok(())
    }
}

impl Drop for AvalancheCacheGuard {
    fn drop(&mut self) {
        let _ = self.pool.take();
        if let Some(path) = self.cleanup_path.as_ref() {
            let _ = fs::remove_file(path);
        }
    }
}

/// Estimates the serialized footprint of one cached scored-input row.
///
/// # Parameters
/// - `input`: Scored Avalanche input pending cache insertion.
///
/// # Returns
/// - `usize`: Approximate in-memory byte footprint used for chunked cache flushing.
///
/// # Expected Output
/// - Returns an approximate byte count; no stdout/stderr output.
pub(crate) fn approximate_scored_avalanche_input_bytes(input: &ScoredAvalancheInput) -> usize {
    let base = input.message_bits.bytes_le().len()
        + input.r.to_bytes_le().len()
        + input.x.to_bytes_le().len()
        + std::mem::size_of::<ScoredAvalancheInput>();
    let detail = input.detail.as_ref().map_or(0usize, |detail| {
        detail.hbc_ciphertext_r.to_bytes_le().len()
            + detail.candidate_decryption.to_bytes_le().len()
            + detail.target_exponent.normalized().to_string().len()
            + std::mem::size_of::<ScoredAvalancheInputDetail>()
    });
    base + detail
}

/// Converts a scored Avalanche input into one SQLite insert row.
///
/// # Parameters
/// - `batch_number`: One-based analysis batch number.
/// - `input`: Scored Avalanche input to persist.
///
/// # Returns
/// - `Result<NewCachedAvalancheInput, Box<dyn Error>>`: Insert row ready for Diesel.
///
/// # Expected Output
/// - Returns an owned insert payload; no stdout/stderr output.
fn serialize_scored_avalanche_input_for_cache(
    batch_number: usize,
    input: &RankedScoredAvalancheInput,
) -> Result<NewCachedAvalancheInput, Box<dyn Error>> {
    Ok(NewCachedAvalancheInput {
        batch_number: i32::try_from(batch_number).map_err(|_| "batch number exceeds i32 range")?,
        batch_candidate_index: i32::try_from(input.input.batch_candidate_index)
            .map_err(|_| "batch candidate index exceeds i32 range")?,
        message_index: i32::try_from(input.input.message_index)
            .map_err(|_| "message index exceeds i32 range")?,
        r_text: input.input.r.to_string(),
        x_text: input.input.x.to_string(),
        score_match_pct: input.input.score_match_pct,
        contents_have_been_inverted: input.input.contents_have_been_inverted,
        fitness_score: i64::try_from(input.fitness.fitness_score)
            .map_err(|_| "fitness score exceeds i64 range")?,
        fitness_total_score: i64::try_from(input.fitness.fitness_total_score)
            .map_err(|_| "fitness total score exceeds i64 range")?,
        fitness_message_count: i64::try_from(input.fitness.fitness_message_count)
            .map_err(|_| "fitness message count exceeds i64 range")?,
        message_bits: input.input.message_bits.bytes_le().to_vec(),
        message_bit_len: i32::try_from(input.input.message_bits.len())
            .map_err(|_| "message bit length exceeds i32 range")?,
        target_exponent_text: input
            .input
            .detail
            .as_ref()
            .map(|detail| detail.target_exponent.normalized().to_string()),
        hbc_ciphertext_r_text: input
            .input
            .detail
            .as_ref()
            .map(|detail| detail.hbc_ciphertext_r.to_string()),
        candidate_decryption_text: input
            .input
            .detail
            .as_ref()
            .map(|detail| detail.candidate_decryption.to_string()),
    })
}

/// Rebuilds a scored Avalanche input from one cached SQLite row.
///
/// # Parameters
/// - `row`: Cached SQLite row to deserialize.
///
/// # Returns
/// - `Result<ScoredAvalancheInput, Box<dyn Error>>`: Fully reconstructed scored input.
///
/// # Expected Output
/// - Returns the decoded scored input; no stdout/stderr output.
fn deserialize_scored_avalanche_input_row(
    row: CachedAvalancheInputRow,
) -> Result<ScoredAvalancheInput, Box<dyn Error>> {
    let target_exponent = match row.target_exponent_text {
        Some(raw) => Some(raw.parse::<BigDecimal>()?),
        None => None,
    };
    let hbc_ciphertext_r = match row.hbc_ciphertext_r_text {
        Some(raw) => Some(raw.parse::<BigUint>()?),
        None => None,
    };
    let candidate_decryption = match row.candidate_decryption_text {
        Some(raw) => Some(raw.parse::<BigUint>()?),
        None => None,
    };
    let detail = match (target_exponent, hbc_ciphertext_r, candidate_decryption) {
        (Some(target_exponent), Some(hbc_ciphertext_r), Some(candidate_decryption)) => {
            Some(ScoredAvalancheInputDetail {
                target_exponent,
                hbc_ciphertext_r,
                candidate_decryption,
            })
        }
        _ => None,
    };
    Ok(ScoredAvalancheInput {
        batch_candidate_index: usize::try_from(row.batch_candidate_index)
            .map_err(|_| "cached batch candidate index exceeds usize range")?,
        message_index: usize::try_from(row.message_index)
            .map_err(|_| "cached message index exceeds usize range")?,
        r: row.r_text.parse::<BigUint>()?,
        x: row.x_text.parse::<BigUint>()?,
        score_match_pct: row.score_match_pct,
        contents_have_been_inverted: row.contents_have_been_inverted,
        message_bits: PackedBits::from_bytes_le(
            &row.message_bits,
            usize::try_from(row.message_bit_len)
                .map_err(|_| "cached message bit length exceeds usize range")?,
        ),
        detail,
    })
}

/// Inserts scored Avalanche inputs into the SQLite cache in page-sized batches.
///
/// # Parameters
/// - `cache`: Shared SQLite cache wrapper.
/// - `batch_number`: One-based analysis batch number.
/// - `inputs`: Scored inputs to persist.
///
/// # Returns
/// - `Result<(), Box<dyn Error>>`: `Ok(())` on success.
///
/// # Expected Output
/// - Writes input rows to SQLite using page-sized Diesel insert batches.
pub(crate) fn insert_cached_scored_inputs(
    cache: &AvalancheCacheGuard,
    batch_number: usize,
    inputs: &[RankedScoredAvalancheInput],
) -> Result<(), Box<dyn Error>> {
    if inputs.is_empty() {
        return Ok(());
    }
    let rows = inputs
        .iter()
        .map(|input| serialize_scored_avalanche_input_for_cache(batch_number, input))
        .collect::<Result<Vec<_>, _>>()?;
    let mut connection = cache.pool()?.get()?;
    for row_chunk in rows.chunks(cache.page_rows_usize().max(1)) {
        diesel::insert_into(avalanche_cache_inputs::table)
            .values(row_chunk)
            .execute(&mut connection)?;
    }
    Ok(())
}

/// Loads one keyset page of cached scored-input rows for a batch.
///
/// # Parameters
/// - `cache`: Shared SQLite cache wrapper.
/// - `batch_number`: One-based analysis batch number.
/// - `after_row_id`: Optional last-seen row id from the prior page.
/// - `limit_rows`: Maximum rows to return.
///
/// # Returns
/// - `Result<(Vec<CachedAvalancheInputRow>, CachedPageLoadTiming), Box<dyn Error>>`: One page of
///   cached input rows plus connection/sql timing.
///
/// # Expected Output
/// - Reads a bounded page of cached rows from SQLite ordered by increasing row id.
pub(crate) fn load_cached_scored_input_rows_after_id_page(
    cache: &AvalancheCacheGuard,
    batch_number: usize,
    after_row_id: Option<i64>,
    limit_rows: i64,
) -> Result<(Vec<CachedAvalancheInputRow>, CachedPageLoadTiming), Box<dyn Error>> {
    let batch_number = i32::try_from(batch_number).map_err(|_| "batch number exceeds i32 range")?;
    let connection_wait_start = Instant::now();
    let mut connection = cache.pool()?.get()?;
    let connection_wait = connection_wait_start.elapsed();
    let sql_query_start = Instant::now();
    let rows = avalanche_cache_inputs::table
        .filter(avalanche_cache_inputs::batch_number.eq(batch_number))
        .filter(avalanche_cache_inputs::id.gt(after_row_id.unwrap_or(0)))
        .order(avalanche_cache_inputs::id.asc())
        .limit(limit_rows)
        .select(CachedAvalancheInputRow::as_select())
        .load::<CachedAvalancheInputRow>(&mut connection)?;
    let sql_query = sql_query_start.elapsed();
    Ok((
        rows,
        CachedPageLoadTiming {
            connection_wait,
            sql_query,
            row_decode: Duration::ZERO,
        },
    ))
}

/// Loads raw cached scored-input rows for a selected id set.
///
/// # Parameters
/// - `cache`: Shared SQLite cache wrapper.
/// - `ids`: Cache row ids to load.
///
/// # Returns
/// - `Result<Vec<CachedAvalancheInputRow>, Box<dyn Error>>`: Loaded cached rows.
///
/// # Expected Output
/// - Reads the selected row set from SQLite without preserving caller order.
pub(crate) fn load_cached_scored_input_rows_by_ids(
    cache: &AvalancheCacheGuard,
    ids: &[i64],
) -> Result<Vec<CachedAvalancheInputRow>, Box<dyn Error>> {
    if ids.is_empty() {
        return Ok(Vec::new());
    }
    let mut connection = cache.pool()?.get()?;
    let rows = avalanche_cache_inputs::table
        .filter(avalanche_cache_inputs::id.eq_any(ids))
        .select(CachedAvalancheInputRow::as_select())
        .load::<CachedAvalancheInputRow>(&mut connection)?;
    Ok(rows)
}

/// Loads full scored Avalanche inputs for a small selected set of cache row ids.
///
/// # Parameters
/// - `cache`: Shared SQLite cache wrapper.
/// - `ids`: Cache row ids to load.
///
/// # Returns
/// - `Result<Vec<ScoredAvalancheInput>, Box<dyn Error>>`: Decoded scored inputs in the same order as `ids`.
///
/// # Expected Output
/// - Loads a small selected row set from SQLite and preserves caller order.
pub(crate) fn load_cached_scored_inputs_by_ids(
    cache: &AvalancheCacheGuard,
    ids: &[i64],
) -> Result<Vec<ScoredAvalancheInput>, Box<dyn Error>> {
    if ids.is_empty() {
        return Ok(Vec::new());
    }
    let rows = load_cached_scored_input_rows_by_ids(cache, ids)?;
    let mut by_id = HashMap::with_capacity(rows.len());
    for row in rows {
        by_id.insert(row.id, row);
    }
    let mut ordered = Vec::with_capacity(ids.len());
    for id in ids {
        let row = by_id
            .remove(id)
            .ok_or_else(|| format!("missing cached scored-input row id {}", id))?;
        ordered.push(deserialize_scored_avalanche_input_row(row)?);
    }
    Ok(ordered)
}

/// Counts cached scored-input rows for one analysis batch.
///
/// # Parameters
/// - `cache`: Shared SQLite cache wrapper.
/// - `batch_number`: One-based analysis batch number.
///
/// # Returns
/// - `Result<usize, Box<dyn Error>>`: Number of cached scored-input rows in the batch.
///
/// # Expected Output
/// - Reads one aggregate count from SQLite.
pub(crate) fn count_cached_scored_inputs(
    cache: &AvalancheCacheGuard,
    batch_number: usize,
) -> Result<usize, Box<dyn Error>> {
    let batch_number = i32::try_from(batch_number).map_err(|_| "batch number exceeds i32 range")?;
    let mut connection = cache.pool()?.get()?;
    let count = avalanche_cache_inputs::table
        .filter(avalanche_cache_inputs::batch_number.eq(batch_number))
        .select(count_star())
        .first::<i64>(&mut connection)?;
    usize::try_from(count).map_err(|_| "cached scored-input count exceeds usize range".into())
}

/// Converts a selected Avalanche sample into one cached SQLite row.
///
/// # Parameters
/// - `batch_number`: One-based analysis batch number.
/// - `sample`: Finalized selected sample to persist.
/// - `engine`: Engine configuration controlling recursive source bits.
///
/// # Returns
/// - `Result<NewCachedAvalancheSample, Box<dyn Error>>`: Cached sample row ready for Diesel.
///
/// # Expected Output
/// - Returns an owned insert payload for SQLite.
fn serialize_selected_sample_for_cache(
    batch_number: usize,
    sample: &SelectedAvalancheSample,
    engine: &EngineConfig,
) -> Result<NewCachedAvalancheSample, Box<dyn Error>> {
    let recursive_bits = PackedBits::from_bools(recursive_tier_bits(sample, engine));
    Ok(NewCachedAvalancheSample {
        batch_number: i32::try_from(batch_number).map_err(|_| "batch number exceeds i32 range")?,
        tier_index: i32::try_from(sample.tier_index).map_err(|_| "tier index exceeds i32 range")?,
        sample_index: i32::try_from(sample.sample_index)
            .map_err(|_| "sample index exceeds i32 range")?,
        input_count: i32::try_from(sample.input_count)
            .map_err(|_| "input count exceeds i32 range")?,
        average_score_pct: sample.average_score_pct,
        top_beam_score: sample.top_beam_score,
        top_beam_match_pct: sample.top_beam_match_pct,
        best_match_pct: sample.best_match_pct,
        majority_vote_match_pct: sample.majority_vote_match_pct,
        majority_vote_ones_match_pct: sample.majority_vote_ones_match_pct,
        best_bits: PackedBits::from_bools(&sample.best_bits)
            .bytes_le()
            .to_vec(),
        best_bits_bit_len: i32::try_from(sample.best_bits.len())
            .map_err(|_| "best bit length exceeds i32 range")?,
        majority_vote_bits: PackedBits::from_bools(&sample.majority_vote_bits)
            .bytes_le()
            .to_vec(),
        majority_vote_bits_bit_len: i32::try_from(sample.majority_vote_bits.len())
            .map_err(|_| "majority-vote bit length exceeds i32 range")?,
        recursive_bits: recursive_bits.bytes_le().to_vec(),
        recursive_bits_bit_len: i32::try_from(recursive_bits.len())
            .map_err(|_| "recursive bit length exceeds i32 range")?,
        beam_results_json: serde_json::to_string(&sample.beam_results)?,
        center_biases_json: serde_json::to_string(&sample.center_biases)?,
    })
}

/// Inserts finalized selected samples for one batch/tier into the SQLite cache.
///
/// # Parameters
/// - `cache`: Shared SQLite cache wrapper.
/// - `batch_number`: One-based analysis batch number.
/// - `samples`: Finalized samples to persist.
/// - `engine`: Engine configuration controlling recursive source bits.
///
/// # Returns
/// - `Result<(), Box<dyn Error>>`: `Ok(())` on success.
///
/// # Expected Output
/// - Writes finalized sample rows to SQLite using page-sized Diesel insert batches.
pub(crate) fn insert_cached_selected_samples(
    cache: &AvalancheCacheGuard,
    batch_number: usize,
    samples: &[SelectedAvalancheSample],
    engine: &EngineConfig,
) -> Result<(), Box<dyn Error>> {
    if samples.is_empty() {
        return Ok(());
    }
    let rows = samples
        .iter()
        .map(|sample| serialize_selected_sample_for_cache(batch_number, sample, engine))
        .collect::<Result<Vec<_>, _>>()?;
    let mut connection = cache.pool()?.get()?;
    for row_chunk in rows.chunks(cache.page_rows_usize().max(1)) {
        diesel::insert_into(avalanche_cache_samples::table)
            .values(row_chunk)
            .execute(&mut connection)?;
    }
    Ok(())
}

/// Loads cached selected-sample rows for one batch/tier page.
///
/// # Parameters
/// - `cache`: Shared SQLite cache wrapper.
/// - `batch_number`: One-based analysis batch number.
/// - `tier_index`: One-based Avalanche tier index.
/// - `offset_rows`: Number of rows to skip before reading.
/// - `limit_rows`: Maximum rows to return.
///
/// # Returns
/// - `Result<Vec<CachedAvalancheSampleRow>, Box<dyn Error>>`: One page of cached selected samples.
///
/// # Expected Output
/// - Reads a bounded page of finalized sample rows from SQLite.
pub(crate) fn load_cached_selected_sample_rows_page(
    cache: &AvalancheCacheGuard,
    batch_number: usize,
    tier_index: usize,
    offset_rows: i64,
    limit_rows: i64,
) -> Result<Vec<CachedAvalancheSampleRow>, Box<dyn Error>> {
    let batch_number = i32::try_from(batch_number).map_err(|_| "batch number exceeds i32 range")?;
    let tier_index = i32::try_from(tier_index).map_err(|_| "tier index exceeds i32 range")?;
    let mut connection = cache.pool()?.get()?;
    let rows = avalanche_cache_samples::table
        .filter(avalanche_cache_samples::batch_number.eq(batch_number))
        .filter(avalanche_cache_samples::tier_index.eq(tier_index))
        .order(avalanche_cache_samples::sample_index.asc())
        .limit(limit_rows)
        .offset(offset_rows)
        .select(CachedAvalancheSampleRow::as_select())
        .load::<CachedAvalancheSampleRow>(&mut connection)?;
    Ok(rows)
}

/// Loads raw cached selected-sample rows for a selected id set.
///
/// # Parameters
/// - `cache`: Shared SQLite cache wrapper.
/// - `ids`: Cache row ids to load.
///
/// # Returns
/// - `Result<Vec<CachedAvalancheSampleRow>, Box<dyn Error>>`: Loaded cached sample rows.
///
/// # Expected Output
/// - Reads the selected sample row set from SQLite without preserving caller order.
pub(crate) fn load_cached_selected_sample_rows_by_ids(
    cache: &AvalancheCacheGuard,
    ids: &[i64],
) -> Result<Vec<CachedAvalancheSampleRow>, Box<dyn Error>> {
    if ids.is_empty() {
        return Ok(Vec::new());
    }
    let mut connection = cache.pool()?.get()?;
    let rows = avalanche_cache_samples::table
        .filter(avalanche_cache_samples::id.eq_any(ids))
        .select(CachedAvalancheSampleRow::as_select())
        .load::<CachedAvalancheSampleRow>(&mut connection)?;
    Ok(rows)
}

/// Counts cached selected-sample rows for one batch/tier.
///
/// # Parameters
/// - `cache`: Shared SQLite cache wrapper.
/// - `batch_number`: One-based analysis batch number.
/// - `tier_index`: One-based Avalanche tier index.
///
/// # Returns
/// - `Result<usize, Box<dyn Error>>`: Number of cached finalized sample rows.
///
/// # Expected Output
/// - Reads one aggregate count from SQLite.
pub(crate) fn count_cached_selected_samples(
    cache: &AvalancheCacheGuard,
    batch_number: usize,
    tier_index: usize,
) -> Result<usize, Box<dyn Error>> {
    let batch_number = i32::try_from(batch_number).map_err(|_| "batch number exceeds i32 range")?;
    let tier_index = i32::try_from(tier_index).map_err(|_| "tier index exceeds i32 range")?;
    let mut connection = cache.pool()?.get()?;
    let count = avalanche_cache_samples::table
        .filter(avalanche_cache_samples::batch_number.eq(batch_number))
        .filter(avalanche_cache_samples::tier_index.eq(tier_index))
        .select(count_star())
        .first::<i64>(&mut connection)?;
    usize::try_from(count).map_err(|_| "cached selected-sample count exceeds usize range".into())
}

/// Rebuilds a finalized selected sample from one cached SQLite row.
///
/// # Parameters
/// - `row`: Cached SQLite selected-sample row.
///
/// # Returns
/// - `Result<SelectedAvalancheSample, Box<dyn Error>>`: Fully reconstructed selected sample.
///
/// # Expected Output
/// - Returns the decoded selected sample; no stdout/stderr output.
pub(crate) fn deserialize_selected_avalanche_sample_row(
    row: CachedAvalancheSampleRow,
) -> Result<SelectedAvalancheSample, Box<dyn Error>> {
    let beam_results =
        serde_json::from_str::<Vec<AvalancheCombinationBeamResult>>(&row.beam_results_json)?;
    let center_biases =
        serde_json::from_str::<Vec<AvalancheCenterBiasEntry>>(&row.center_biases_json)?;
    Ok(SelectedAvalancheSample {
        sample_index: usize::try_from(row.sample_index)
            .map_err(|_| "cached sample index exceeds usize range")?,
        tier_index: usize::try_from(row.tier_index)
            .map_err(|_| "cached tier index exceeds usize range")?,
        input_count: usize::try_from(row.input_count)
            .map_err(|_| "cached input count exceeds usize range")?,
        average_score_pct: row.average_score_pct,
        beam_results,
        majority_vote_bits: PackedBits::from_bytes_le(
            &row.majority_vote_bits,
            usize::try_from(row.majority_vote_bits_bit_len)
                .map_err(|_| "cached majority-vote bit length exceeds usize range")?,
        )
        .to_bools(),
        majority_vote_match_pct: row.majority_vote_match_pct,
        majority_vote_ones_match_pct: row.majority_vote_ones_match_pct,
        best_bits: PackedBits::from_bytes_le(
            &row.best_bits,
            usize::try_from(row.best_bits_bit_len)
                .map_err(|_| "cached best-bit length exceeds usize range")?,
        )
        .to_bools(),
        top_beam_score: row.top_beam_score,
        top_beam_match_pct: row.top_beam_match_pct,
        best_match_pct: row.best_match_pct,
        center_biases,
        node: AvalancheNode::from_packed_bits(
            PackedBits::from_bytes_le(
                &row.recursive_bits,
                usize::try_from(row.recursive_bits_bit_len)
                    .map_err(|_| "cached recursive bit length exceeds usize range")?,
            ),
            vec![
                0.0;
                usize::try_from(row.recursive_bits_bit_len)
                    .map_err(|_| "cached recursive bit length exceeds usize range")?
            ],
        ),
    })
}

/// Loads cached scored-input pages with keyset pagination and detailed timing logs.
///
/// # Parameters
/// - `total_rows`: Total cached row count expected from the scan.
/// - `page_rows`: Number of application rows to decode per page.
/// - `progress_label`: Human-readable label used for interval progress logging.
/// - `load_page`: Callback that loads and decodes one page after the provided row id cursor.
///
/// # Returns
/// - `Result<Vec<T>, String>`: Flattened decoded page items in page order.
///
/// # Expected Output
/// - Prints interval progress plus connection/sql/decode timing summaries while loading cached pages.
pub(crate) fn load_cached_scored_input_pages_with_progress<T, F>(
    total_rows: usize,
    page_rows: usize,
    progress_label: &str,
    mut load_page: F,
) -> Result<Vec<T>, String>
where
    F: FnMut(Option<i64>) -> Result<CachedKeysetPage<T>, String>,
{
    if total_rows == 0 {
        return Ok(Vec::new());
    }

    let page_rows = page_rows.max(1);
    let total_pages = total_rows.div_ceil(page_rows);
    println!("{progress_label}: loading {total_pages} cached page(s) via keyset pagination");

    let progress_started_at = Instant::now();
    let log_interval = Duration::from_secs(5);
    let mut next_log_at = log_interval;
    let mut last_row_id = None;
    let mut page_count = 0usize;
    let mut flattened = Vec::with_capacity(total_rows);
    let mut timing_totals = CachedPageLoadTiming::default();

    while flattened.len() < total_rows {
        let page = load_page(last_row_id)?;
        let page_rows_loaded = page.items.len();
        if page_rows_loaded == 0 {
            break;
        }
        last_row_id = page.last_row_id;
        page_count += 1;
        timing_totals += page.timing;
        flattened.extend(page.items);

        let elapsed = progress_started_at.elapsed();
        if elapsed >= next_log_at || flattened.len() >= total_rows || page_count == total_pages {
            let pct = ((page_count as f64) * 100.0 / (total_pages as f64)).min(100.0);
            println!(
                "{progress_label}: loaded {:.5}% ({}/{}) cached page(s), rows {} of {}, connection-wait {:.3}s sql {:.3}s decode {:.3}s total {:.3}s",
                pct,
                page_count,
                total_pages,
                flattened.len().min(total_rows),
                total_rows,
                timing_totals.connection_wait.as_secs_f64(),
                timing_totals.sql_query.as_secs_f64(),
                timing_totals.row_decode.as_secs_f64(),
                timing_totals.total().as_secs_f64(),
            );
            while elapsed >= next_log_at {
                next_log_at += log_interval;
            }
        }
    }
    if flattened.len() != total_rows {
        return Err(format!(
            "{progress_label}: loaded {} of {} cached rows before keyset pagination exhausted the batch",
            flattened.len(),
            total_rows
        ));
    }
    println!(
        "{progress_label}: completed {} cached page(s) with {} row(s) in {:.3}s (connection-wait {:.3}s sql {:.3}s decode {:.3}s)",
        page_count,
        flattened.len(),
        progress_started_at.elapsed().as_secs_f64(),
        timing_totals.connection_wait.as_secs_f64(),
        timing_totals.sql_query.as_secs_f64(),
        timing_totals.row_decode.as_secs_f64(),
    );
    Ok(flattened)
}

/// Loads lightweight cached recursive sample summaries for one batch/tier.
///
/// # Parameters
/// - `cache`: Shared SQLite cache wrapper.
/// - `batch_number`: One-based analysis batch number.
/// - `tier_index`: One-based Avalanche tier index.
///
/// # Returns
/// - `Result<Vec<CachedRecursiveSampleSummary>, Box<dyn Error>>`: Lightweight cached recursive sample summaries.
///
/// # Expected Output
/// - Streams cached sample rows from SQLite and returns only the metadata needed for recursive selection.
pub(crate) fn load_cached_recursive_sample_summaries(
    cache: &AvalancheCacheGuard,
    batch_number: usize,
    tier_index: usize,
) -> Result<Vec<CachedRecursiveSampleSummary>, Box<dyn Error>> {
    let total_rows = count_cached_selected_samples(cache, batch_number, tier_index)?;
    let mut offset_rows = 0i64;
    let mut summaries = Vec::with_capacity(total_rows);
    while summaries.len() < total_rows {
        let rows = load_cached_selected_sample_rows_page(
            cache,
            batch_number,
            tier_index,
            offset_rows,
            cache.page_rows_i64(),
        )?;
        if rows.is_empty() {
            break;
        }
        offset_rows +=
            i64::try_from(rows.len()).map_err(|_| "row page length exceeds i64 range")?;
        for row in rows {
            summaries.push(CachedRecursiveSampleSummary { id: row.id });
        }
    }
    Ok(summaries)
}

/// Builds tier statistics directly from cached selected-sample rows.
///
/// # Parameters
/// - `cache`: Shared SQLite cache wrapper.
/// - `batch_number`: One-based analysis batch number.
/// - `tier_index`: One-based Avalanche tier index.
/// - `group_size`: Number of source items grouped into each sample for the tier.
/// - `source_kind`: Human-readable description of the source data for the tier.
///
/// # Returns
/// - `Result<AvalancheTierStatistics, Box<dyn Error>>`: Per-tier sample accuracy summary.
///
/// # Expected Output
/// - Reads cached selected-sample rows from SQLite and returns per-tier analytics.
pub(crate) fn build_cached_avalanche_tier_statistics(
    cache: &AvalancheCacheGuard,
    batch_number: usize,
    tier_index: usize,
    group_size: usize,
    source_kind: &str,
) -> Result<AvalancheTierStatistics, Box<dyn Error>> {
    let total_rows = count_cached_selected_samples(cache, batch_number, tier_index)?;
    let mut offset_rows = 0i64;
    let mut sample_stats = Vec::with_capacity(total_rows);
    while sample_stats.len() < total_rows {
        let rows = load_cached_selected_sample_rows_page(
            cache,
            batch_number,
            tier_index,
            offset_rows,
            cache.page_rows_i64(),
        )?;
        if rows.is_empty() {
            break;
        }
        offset_rows +=
            i64::try_from(rows.len()).map_err(|_| "row page length exceeds i64 range")?;
        sample_stats.extend(rows.into_iter().map(|row| AvalancheTierSampleStat {
            sample_index: usize::try_from(row.sample_index).unwrap_or(0),
            input_count: usize::try_from(row.input_count).unwrap_or(0),
            average_score_pct: row.average_score_pct,
            beam_match_pct: row.top_beam_match_pct,
            majority_vote_match_pct: Some(row.majority_vote_match_pct),
            best_match_pct: row.best_match_pct,
        }));
    }
    Ok(AvalancheTierStatistics {
        tier_index,
        sample_count: sample_stats.len(),
        group_size,
        source_kind: source_kind.to_string(),
        sample_stats,
    })
}
