///! Eclipse Public License 2.0
///! SPDX-License-Identifier: EPL-2.0
///! Copyright (c) 2025 Nicholas LaRoche <nlaroche@cryptifier.dev>
use bigdecimal::{BigDecimal, FromPrimitive, One};
use clap::Parser;
use std::{error::Error, str::FromStr};

#[derive(Parser, Debug)]
#[command(
    name = "test_cosine",
    about = "Generate cosine of a BigDecimal number.",
    author,
    version
)]
struct Args {
    /// Whether sizing is driven by prime bits or exact modulus bits
    #[arg(long, default_value = "32.000000")]
    input: String,

    #[arg(long, default_value = "80")]
    digits: i64,
}

fn pow10_neg(scale: i64) -> BigDecimal {
    BigDecimal::new(1.into(), scale)
}

fn cos_bigdecimal(x: BigDecimal, digits: i64) -> BigDecimal {
    let tolerance = pow10_neg(digits + 8);

    let mut sum = BigDecimal::one();
    let mut term = BigDecimal::one();

    let x2 = &x * &x;
    for n in 1..20_000_i64 {
        // term *= -x^2 / ((2n - 1)(2n))
        let denom = BigDecimal::from_i64((2 * n - 1) * (2 * n)).unwrap();

        term = -(&term * &x2) / denom;
        term = term.with_scale(digits + 16);

        sum += &term;
        sum = sum.with_scale(digits + 16);

        if term.abs() < tolerance {
            break;
        }
    }

    sum.with_scale(digits)
}

/// Entry point.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `Result<(), Box<dyn Error>>`: `Ok(())` on success or an error on failure.
///
/// # Expected Output
/// - Writes the cosine value to stdout.
fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    let x = BigDecimal::from_str(args.input.as_str()).expect("invalid decimal input");
    let y = cos_bigdecimal(x, args.digits);
    println!("{y}");

    Ok(())
}
