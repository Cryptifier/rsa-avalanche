/// Eclipse Public License 2.0
/// SPDX-License-Identifier: EPL-2.0
/// Copyright (c) 2026 Nicholas LaRoche <nlaroche@cryptifier.dev>
use std::{
    error::Error,
    fs::File,
    io::{BufWriter, Write},
    path::Path,
    path::PathBuf,
};

use clap::Parser;

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
fn main() -> Result<(), Box<dyn Error>> {
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
fn run(args: Args) -> Result<(), Box<dyn Error>> {
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

    let total_combinations = checked_binomial(universe_size, combination_size);
    let combinations_written =
        write_combination_index_csv(universe_size, combination_size, limit, &args.output)?;
    let total_display = total_combinations
        .map(|value| value.to_string())
        .unwrap_or_else(|| "overflow".to_string());

    println!(
        "Wrote {} compressed combinations of {} from {} to {} (total possible: {})",
        combinations_written,
        combination_size,
        universe_size,
        args.output.display(),
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
/// - `output_path`: Destination CSV file path.
///
/// # Returns
/// - `Result<usize, Box<dyn Error>>`: Number of rows written to the CSV file.
///
/// # Expected Output
/// - Writes a CSV file with a header and one row per generated combination; no stderr output on success.
fn write_combination_index_csv(
    universe_size: usize,
    combination_size: usize,
    limit: usize,
    output_path: &Path,
) -> Result<usize, Box<dyn Error>> {
    let file = File::create(output_path)?;
    let mut writer = BufWriter::new(file);
    writeln!(writer, "combination_index,compressed_indices_hex")?;

    if limit == 0 {
        writer.flush()?;
        return Ok(0);
    }

    let mut rows_written = 0usize;
    for (combination_index, combination) in
        CombinationIterator::new(universe_size, combination_size).enumerate()
    {
        if combination_index >= limit {
            break;
        }
        let encoded = encode_indices_gap_varint(&combination)?;
        writeln!(writer, "{},{}", combination_index, hex::encode(encoded))?;
        rows_written += 1;
    }

    writer.flush()?;
    Ok(rows_written)
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
        result = result.checked_mul(numerator)?;
        result /= denominator;
    }
    Some(result)
}

/// Iterates lexicographically over all `N`-element combinations drawn from `0..Q`.
///
/// # Parameters
/// - `universe_size`: Number of available values `Q`.
/// - `combination_size`: Number of selected values `N`.
///
/// # Returns
/// - `CombinationIterator`: Iterator yielding sorted index vectors.
///
/// # Expected Output
/// - Returns the iterator; no stdout/stderr output.
struct CombinationIterator {
    universe_size: usize,
    current: Option<Vec<usize>>,
}

impl CombinationIterator {
    /// Builds an iterator over sorted combinations.
    ///
    /// # Parameters
    /// - `universe_size`: Number of available values `Q`.
    /// - `combination_size`: Number of selected values `N`.
    ///
    /// # Returns
    /// - `Self`: Iterator positioned at the first combination when one exists.
    ///
    /// # Expected Output
    /// - Returns a new iterator; no stdout/stderr output.
    fn new(universe_size: usize, combination_size: usize) -> Self {
        let current = if combination_size == 0 {
            Some(Vec::new())
        } else if combination_size > universe_size {
            None
        } else {
            Some((0..combination_size).collect())
        };

        Self {
            universe_size,
            current,
        }
    }
}

impl Iterator for CombinationIterator {
    type Item = Vec<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        let combination = self.current.clone()?;
        if combination.is_empty() {
            self.current = None;
            return Some(combination);
        }

        let mut next_combination = combination.clone();
        let width = next_combination.len();
        let mut advanced = false;
        for idx in (0..width).rev() {
            let max_value = self.universe_size - (width - idx);
            if next_combination[idx] < max_value {
                next_combination[idx] += 1;
                for reset_idx in (idx + 1)..width {
                    next_combination[reset_idx] = next_combination[reset_idx - 1] + 1;
                }
                advanced = true;
                break;
            }
        }

        self.current = if advanced {
            Some(next_combination)
        } else {
            None
        };
        Some(combination)
    }
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
fn encode_indices_gap_varint(indices: &[usize]) -> Result<Vec<u8>, Box<dyn Error>> {
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
        CombinationIterator, checked_binomial, encode_indices_gap_varint,
        write_combination_index_csv,
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
    fn iterates_first_combinations_in_lexicographic_order() {
        let combinations: Vec<Vec<usize>> = CombinationIterator::new(5, 3).take(4).collect();
        assert_eq!(
            combinations,
            vec![vec![0, 1, 2], vec![0, 1, 3], vec![0, 1, 4], vec![0, 2, 3]]
        );
    }

    #[test]
    fn writes_limited_combination_csv() {
        let output_path = temp_csv_path();
        let rows_written =
            write_combination_index_csv(5, 3, 3, &output_path).expect("csv should be written");
        assert_eq!(rows_written, 3);

        let csv = fs::read_to_string(&output_path).expect("csv should be readable");
        let lines: Vec<&str> = csv.lines().collect();
        assert_eq!(lines.len(), 4);
        assert_eq!(lines[0], "combination_index,compressed_indices_hex");
        assert_eq!(lines[1], "0,000000");
        assert_eq!(lines[2], "1,000001");
        assert_eq!(lines[3], "2,000002");

        fs::remove_file(&output_path).expect("temporary csv should be removed");
    }

    #[test]
    fn computes_small_binomial_coefficients() {
        assert_eq!(checked_binomial(5, 3), Some(10));
        assert_eq!(checked_binomial(10, 0), Some(1));
    }
}
