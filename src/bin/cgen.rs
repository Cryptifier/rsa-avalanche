/// Eclipse Public License 2.0
/// SPDX-License-Identifier: EPL-2.0
/// Copyright (c) 2026 Nicholas LaRoche <nlaroche@cryptifier.dev>
use std::{
    cmp::min,
    collections::HashSet,
    error::Error,
    fs::File,
    io::{BufWriter, Write},
    num::NonZeroUsize,
    path::Path,
    path::PathBuf,
};

use clap::Parser;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use rayon::prelude::*;

#[derive(Parser, Debug)]
#[command(
    name = "cgen",
    about = "Generate compressed combination indices and write them to CSV",
    author,
    version
)]
struct Args {
    /// Number of available values `Q`, producing combinations from `0..Q`
    #[arg(short = 'q', long, value_parser = clap::value_parser!(u64).range(1..))]
    universe_size: u64,

    /// Number of selected values `N` in each combination
    #[arg(short = 'n', long, value_parser = clap::value_parser!(u64).range(1..))]
    combination_size: u64,

    /// Maximum number of combinations `K` to write
    #[arg(short = 'k', long, default_value_t = 100u64)]
    limit: u64,

    /// Deterministic seed used to sample unique random combinations with ChaCha20
    #[arg(long)]
    seed: u64,

    /// Output CSV path
    #[arg(short = 'o', long, default_value = "data/cgen_output.csv")]
    output: PathBuf,
}

/// Parses CLI arguments and writes the requested combination index.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `Result<(), Box<dyn Error>>`: `Ok(())` on success or an error when the request is invalid or the CSV cannot be written.
///
/// # Expected Output
/// - Prints a short generation summary to stdout and writes a CSV file.
fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    run(Args::parse())
}

/// Validates CLI arguments and emits the compressed combination index CSV.
///
/// # Parameters
/// - `args`: Parsed CLI arguments describing `Q`, `N`, `K`, and the output path.
///
/// # Returns
/// - `Result<(), Box<dyn Error>>`: `Ok(())` after the CSV is written or an error if validation fails.
///
/// # Expected Output
/// - Prints a summary line to stdout and writes the CSV file to disk.
fn run(args: Args) -> Result<(), Box<dyn Error + Send + Sync>> {
    let universe_size = usize::try_from(args.universe_size)
        .map_err(|_| "universe_size exceeds usize range on this platform")?;
    let combination_size = usize::try_from(args.combination_size)
        .map_err(|_| "combination_size exceeds usize range on this platform")?;
    let limit =
        usize::try_from(args.limit).map_err(|_| "limit exceeds usize range on this platform")?;

    if combination_size > universe_size {
        return Err(format!(
            "combination_size ({}) cannot exceed universe_size ({})",
            combination_size, universe_size
        )
        .into());
    }

    let total_combinations = checked_binomial(universe_size, combination_size)
        .ok_or("total combination count overflowed u128 during random sampling")?;
    let combinations_written = write_combination_index_csv(
        universe_size,
        combination_size,
        limit,
        total_combinations,
        args.seed,
        &args.output,
    )?;
    let total_display = total_combinations.to_string();

    println!(
        "Wrote {} random compressed combinations of {} from {} to {} with seed {} (total possible: {})",
        combinations_written,
        combination_size,
        universe_size,
        args.output.display(),
        args.seed,
        total_display
    );
    Ok(())
}

/// Writes lexicographically ordered combinations as compressed CSV records.
///
/// # Parameters
/// - `universe_size`: Number of available values `Q`, selecting from `0..Q`.
/// - `combination_size`: Number of selected values `N` in each combination.
/// - `limit`: Maximum number of combinations `K` to emit.
/// - `total_combinations`: Exact count of available combinations.
/// - `seed`: ChaCha20 seed used to select unique random combination ranks.
/// - `output_path`: Destination CSV file path.
///
/// # Returns
/// - `Result<usize, Box<dyn Error>>`: Number of rows written to the CSV file.
///
/// # Expected Output
/// - Writes a CSV file with a header and one row per randomly selected combination; no stderr output on success.
fn write_combination_index_csv(
    universe_size: usize,
    combination_size: usize,
    limit: usize,
    total_combinations: u128,
    seed: u64,
    output_path: &Path,
) -> Result<usize, Box<dyn Error + Send + Sync>> {
    let file = File::create(output_path)?;
    let mut writer = BufWriter::new(file);
    writeln!(writer, "combination_index,compressed_indices_hex")?;

    let row_count = resolve_row_count(total_combinations, limit);
    if row_count == 0 {
        writer.flush()?;
        return Ok(0);
    }

    let selected_ranks = select_random_combination_ranks(total_combinations, row_count, seed);
    let rows_per_chunk = preferred_chunk_row_count(combination_size);
    let worker_count = available_worker_count();
    let chunks_per_batch = worker_count.saturating_mul(4).max(1);
    let batch_row_count = rows_per_chunk.saturating_mul(chunks_per_batch);
    let mut batch_start = 0usize;

    while batch_start < row_count {
        let batch_end = min(row_count, batch_start.saturating_add(batch_row_count));
        let chunk_starts: Vec<usize> = (batch_start..batch_end).step_by(rows_per_chunk).collect();
        let rendered_chunks: Result<Vec<String>, Box<dyn Error + Send + Sync>> = chunk_starts
            .into_par_iter()
            .map(|chunk_start| {
                let chunk_end = min(batch_end, chunk_start.saturating_add(rows_per_chunk));
                render_combination_csv_chunk(
                    universe_size,
                    combination_size,
                    &selected_ranks[chunk_start..chunk_end],
                )
            })
            .collect();

        for chunk in rendered_chunks? {
            writer.write_all(chunk.as_bytes())?;
        }
        batch_start = batch_end;
    }

    writer.flush()?;
    Ok(row_count)
}

/// Computes `n choose k`, returning `None` if intermediate arithmetic overflows `u128`.
///
/// # Parameters
/// - `n`: Total number of values.
/// - `k`: Number of selected values.
///
/// # Returns
/// - `Option<u128>`: Exact binomial coefficient when it fits, otherwise `None`.
///
/// # Expected Output
/// - Returns the computed coefficient; no stdout/stderr output.
fn checked_binomial(n: usize, k: usize) -> Option<u128> {
    if k > n {
        return Some(0);
    }

    let reduced_k = k.min(n - k);
    let mut result = 1u128;
    for step in 0..reduced_k {
        let numerator = (n - step) as u128;
        let denominator = (step + 1) as u128;
        result = checked_mul_div_exact(result, numerator, denominator)?;
    }
    Some(result)
}

/// Resolves how many CSV rows should be written for the requested combination set.
///
/// # Parameters
/// - `total_combinations`: Exact number of combinations available for sampling.
/// - `limit`: Maximum number of combinations requested by the caller.
///
/// # Returns
/// - `usize`: Number of rows that should be emitted to the CSV.
///
/// # Expected Output
/// - Returns the bounded row count; no stdout/stderr output.
fn resolve_row_count(total_combinations: u128, limit: usize) -> usize {
    let bounded = min(total_combinations, limit as u128);
    usize::try_from(bounded).expect("bounded combination count should fit in usize")
}

/// Chooses a chunk size that balances Rayon scheduling overhead against row cost.
///
/// # Parameters
/// - `combination_size`: Number of selected values `N` per generated row.
///
/// # Returns
/// - `usize`: Preferred number of rows to render per parallel chunk.
///
/// # Expected Output
/// - Returns a chunk size hint; no stdout/stderr output.
fn preferred_chunk_row_count(combination_size: usize) -> usize {
    let scaled = 32_768usize / combination_size.max(1);
    scaled.clamp(256, 8_192)
}

/// Detects how many worker threads should participate in CSV rendering.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `usize`: Number of available worker threads, with a minimum of one.
///
/// # Expected Output
/// - Returns the worker count; no stdout/stderr output.
fn available_worker_count() -> usize {
    std::thread::available_parallelism()
        .unwrap_or(NonZeroUsize::MIN)
        .get()
}

/// Renders one contiguous range of CSV rows for parallel batch writing.
///
/// # Parameters
/// - `universe_size`: Number of available values `Q`.
/// - `combination_size`: Number of selected values `N`.
/// - `ranks`: Ordered lexicographic ranks to render into CSV rows.
///
/// # Returns
/// - `Result<String, Box<dyn Error + Send + Sync>>`: CSV text for the requested chunk.
///
/// # Expected Output
/// - Returns chunk-local CSV rows with trailing newlines; no stdout/stderr output.
fn render_combination_csv_chunk(
    universe_size: usize,
    combination_size: usize,
    ranks: &[u128],
) -> Result<String, Box<dyn Error + Send + Sync>> {
    let mut chunk = String::with_capacity(ranks.len().saturating_mul(32));
    for &rank in ranks {
        let combination = combination_for_rank(universe_size, combination_size, rank)?;
        let encoded = encode_indices_gap_varint(&combination)?;
        chunk.push_str(&rank.to_string());
        chunk.push(',');
        chunk.push_str(&hex::encode(encoded));
        chunk.push('\n');
    }
    Ok(chunk)
}

/// Resolves the lexicographic `rank`-th combination without iterating earlier rows.
///
/// # Parameters
/// - `total_combinations`: Exact number of combinations available for sampling.
/// - `row_count`: Number of unique ranks to draw.
/// - `seed`: ChaCha20 seed used for deterministic sampling.
///
/// # Returns
/// - `Vec<u128>`: Sorted unique lexicographic combination ranks.
///
/// # Expected Output
/// - Returns `row_count` unique ranks in ascending order; no stdout/stderr output.
fn select_random_combination_ranks(
    total_combinations: u128,
    row_count: usize,
    seed: u64,
) -> Vec<u128> {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let mut selected = HashSet::with_capacity(row_count.saturating_mul(2));
    while selected.len() < row_count {
        selected.insert(rng.gen_range(0..total_combinations));
    }

    let mut ranks = selected.into_iter().collect::<Vec<_>>();
    ranks.sort_unstable();
    ranks
}

/// Resolves the lexicographic `rank`-th combination without iterating earlier rows.
///
/// # Parameters
/// - `universe_size`: Number of available values `Q`.
/// - `combination_size`: Number of selected values `N`.
/// - `rank`: Zero-based lexicographic combination index.
///
/// # Returns
/// - `Result<Vec<usize>, Box<dyn Error + Send + Sync>>`: Sorted indices for the requested row.
///
/// # Expected Output
/// - Returns one combination; no stdout/stderr output.
fn combination_for_rank(
    universe_size: usize,
    combination_size: usize,
    rank: u128,
) -> Result<Vec<usize>, Box<dyn Error + Send + Sync>> {
    if combination_size > universe_size {
        return Err(format!(
            "combination_size ({}) cannot exceed universe_size ({})",
            combination_size, universe_size
        )
        .into());
    }
    if combination_size == 0 {
        if rank == 0 {
            return Ok(Vec::new());
        }
        return Err("combination rank out of range".into());
    }

    let mut combination = Vec::with_capacity(combination_size);
    let mut next_minimum = 0usize;
    let mut remaining_rank = rank;

    for position in 0..combination_size {
        let remaining_slots = combination_size - position - 1;
        let max_candidate = universe_size - (combination_size - position);
        let mut chosen_value = None;

        for candidate in next_minimum..=max_candidate {
            let suffix_count = capped_binomial(
                universe_size - candidate - 1,
                remaining_slots,
                remaining_rank.saturating_add(1),
            );
            if remaining_rank < suffix_count {
                chosen_value = Some(candidate);
                next_minimum = candidate + 1;
                break;
            }
            remaining_rank -= suffix_count;
        }

        if let Some(candidate) = chosen_value {
            combination.push(candidate);
        } else {
            return Err("combination rank out of range".into());
        }
    }

    if remaining_rank != 0 {
        return Err("combination rank out of range".into());
    }
    Ok(combination)
}

/// Computes a binomial coefficient with an upper cap for rank comparisons.
///
/// # Parameters
/// - `n`: Total number of values.
/// - `k`: Number of selected values.
/// - `cap`: Maximum value that needs to be distinguished by the caller.
///
/// # Returns
/// - `u128`: Exact `n choose k` when it is below `cap`, otherwise `cap`.
///
/// # Expected Output
/// - Returns a capped coefficient; no stdout/stderr output.
fn capped_binomial(n: usize, k: usize, cap: u128) -> u128 {
    if cap == 0 {
        return 0;
    }
    if k > n {
        return 0;
    }

    let reduced_k = k.min(n - k);
    let mut result = 1u128;
    for step in 0..reduced_k {
        let numerator = (n - step) as u128;
        let denominator = (step + 1) as u128;
        let Some(next_value) = checked_mul_div_exact(result, numerator, denominator) else {
            return cap;
        };
        result = min(next_value, cap);
        if result >= cap {
            return cap;
        }
    }
    result
}

/// Multiplies by `numerator`, divides by `denominator`, and preserves exactness.
///
/// # Parameters
/// - `value`: Current multiplicative accumulator.
/// - `numerator`: Multiplicative term applied before division.
/// - `denominator`: Exact divisor for the intermediate product.
///
/// # Returns
/// - `Option<u128>`: Updated value or `None` when multiplication overflows.
///
/// # Expected Output
/// - Returns the exact transformed accumulator; no stdout/stderr output.
fn checked_mul_div_exact(value: u128, numerator: u128, denominator: u128) -> Option<u128> {
    let left_gcd = gcd_u128(numerator, denominator);
    let reduced_numerator = numerator / left_gcd;
    let reduced_denominator = denominator / left_gcd;
    let right_gcd = gcd_u128(value, reduced_denominator);
    let reduced_value = value / right_gcd;
    let final_denominator = reduced_denominator / right_gcd;
    debug_assert_eq!(final_denominator, 1);
    reduced_value.checked_mul(reduced_numerator)
}

/// Computes the greatest common divisor for `u128` values.
///
/// # Parameters
/// - `left`: First value.
/// - `right`: Second value.
///
/// # Returns
/// - `u128`: Greatest common divisor of the inputs.
///
/// # Expected Output
/// - Returns the divisor; no stdout/stderr output.
fn gcd_u128(mut left: u128, mut right: u128) -> u128 {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left
}

/// Encodes a sorted combination as gap values using unsigned LEB128 bytes.
///
/// # Parameters
/// - `indices`: Sorted combination indices to encode.
///
/// # Returns
/// - `Result<Vec<u8>, Box<dyn Error>>`: Gap-encoded varint byte sequence.
///
/// # Expected Output
/// - Returns the compressed bytes; no stdout/stderr output.
fn encode_indices_gap_varint(indices: &[usize]) -> Result<Vec<u8>, Box<dyn Error + Send + Sync>> {
    let mut encoded = Vec::new();
    let mut previous = None;

    for &index in indices {
        let gap = match previous {
            Some(prev) => index
                .checked_sub(prev + 1)
                .ok_or("indices must be strictly increasing")?,
            None => index,
        };
        let gap_u64 = u64::try_from(gap).map_err(|_| "gap exceeds u64 range")?;
        push_varint(gap_u64, &mut encoded);
        previous = Some(index);
    }

    Ok(encoded)
}

/// Appends a single unsigned integer to a buffer using LEB128 encoding.
///
/// # Parameters
/// - `value`: Unsigned integer to encode.
/// - `buffer`: Output byte buffer receiving the encoded bytes.
///
/// # Returns
/// - `()`: No direct return value.
///
/// # Expected Output
/// - Appends bytes to `buffer`; no stdout/stderr output.
fn push_varint(mut value: u64, buffer: &mut Vec<u8>) {
    loop {
        let mut byte = (value & 0x7f) as u8;
        value >>= 7;
        if value != 0 {
            byte |= 0x80;
        }
        buffer.push(byte);
        if value == 0 {
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        checked_binomial, combination_for_rank, encode_indices_gap_varint, resolve_row_count,
        select_random_combination_ranks, write_combination_index_csv,
    };
    use std::{fs, path::PathBuf, time::SystemTime};

    /// Decodes a gap-encoded varint combination for test verification.
    ///
    /// # Parameters
    /// - `encoded`: LEB128 bytes produced by `encode_indices_gap_varint`.
    /// - `combination_size`: Number of indices expected in the decoded output.
    ///
    /// # Returns
    /// - `Vec<usize>`: Decoded sorted combination indices.
    ///
    /// # Expected Output
    /// - Returns decoded indices; no stdout/stderr output.
    fn decode_indices_gap_varint(encoded: &[u8], combination_size: usize) -> Vec<usize> {
        let mut decoded = Vec::with_capacity(combination_size);
        let mut offset = 0usize;
        let mut current = 0usize;

        for idx in 0..combination_size {
            let (gap, used) = decode_varint(&encoded[offset..]);
            offset += used;
            current = if idx == 0 { gap } else { current + gap + 1 };
            decoded.push(current);
        }

        assert_eq!(offset, encoded.len());
        decoded
    }

    /// Decodes a single unsigned LEB128 value for tests.
    ///
    /// # Parameters
    /// - `encoded`: Byte slice beginning at a varint boundary.
    ///
    /// # Returns
    /// - `(usize, usize)`: Decoded value and the number of consumed bytes.
    ///
    /// # Expected Output
    /// - Returns the decoded value and width; no stdout/stderr output.
    fn decode_varint(encoded: &[u8]) -> (usize, usize) {
        let mut value = 0usize;
        let mut shift = 0usize;
        for (offset, byte) in encoded.iter().copied().enumerate() {
            value |= usize::from(byte & 0x7f) << shift;
            if byte & 0x80 == 0 {
                return (value, offset + 1);
            }
            shift += 7;
        }
        panic!("unterminated varint")
    }

    /// Builds a unique path in the process temp directory for CSV tests.
    ///
    /// # Parameters
    /// - None.
    ///
    /// # Returns
    /// - `PathBuf`: Unique file path under the OS temp directory.
    ///
    /// # Expected Output
    /// - Returns a path; no stdout/stderr output.
    fn temp_csv_path() -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .expect("system time should be after UNIX_EPOCH")
            .as_nanos();
        std::env::temp_dir().join(format!("cgen_test_{}_{}.csv", std::process::id(), nanos))
    }

    #[test]
    fn encodes_and_decodes_combination_indices() {
        let encoded = encode_indices_gap_varint(&[2, 3, 9, 14]).expect("combination should encode");
        let decoded = decode_indices_gap_varint(&encoded, 4);
        assert_eq!(decoded, vec![2, 3, 9, 14]);
    }

    #[test]
    fn resolves_ranked_combinations_in_lexicographic_order() {
        let combinations = (0..4)
            .map(|rank| combination_for_rank(5, 3, rank).expect("rank should decode"))
            .collect::<Vec<_>>();
        assert_eq!(
            combinations,
            vec![vec![0, 1, 2], vec![0, 1, 3], vec![0, 1, 4], vec![0, 2, 3]]
        );
    }

    #[test]
    fn writes_limited_combination_csv() {
        let output_path = temp_csv_path();
        let rows_written = write_combination_index_csv(5, 3, 3, 10, 7, &output_path)
            .expect("csv should be written");
        assert_eq!(rows_written, 3);

        let csv = fs::read_to_string(&output_path).expect("csv should be readable");
        let lines: Vec<&str> = csv.lines().collect();
        assert_eq!(lines.len(), 4);
        assert_eq!(lines[0], "combination_index,compressed_indices_hex");
        assert_eq!(lines[1], "4,000101");
        assert_eq!(lines[2], "5,000200");
        assert_eq!(lines[3], "8,010100");

        fs::remove_file(&output_path).expect("temporary csv should be removed");
    }

    #[test]
    fn computes_small_binomial_coefficients() {
        assert_eq!(checked_binomial(5, 3), Some(10));
        assert_eq!(checked_binomial(10, 0), Some(1));
    }

    #[test]
    fn caps_written_rows_at_total_combination_count() {
        assert_eq!(resolve_row_count(10, 50), 10);
    }

    #[test]
    fn selects_seeded_unique_ranks() {
        let ranks = select_random_combination_ranks(10, 5, 11);
        assert_eq!(ranks, vec![1, 3, 5, 7, 9]);
    }
}
