use num_bigint::{BigInt, BigUint};
use num_integer::Integer;
use num_traits::{One, Signed, Zero};
use rand::RngCore;
use std::time::Instant;

use crate::rng::RngChoice;

/// Selects the first odd public exponent `e >= start` that is coprime with `phi`.
///
/// # Parameters
/// - `start`: Starting exponent candidate (will be adjusted to odd if even).
/// - `phi`: Euler's totient of the RSA modulus.
///
/// # Returns
/// - `BigUint`: The first odd `e` such that `gcd(e, phi) == 1`.
///
/// # Expected Output
/// - Returns a valid RSA public exponent; no side effects.
pub fn choose_exponent(start: u64, phi: &BigUint) -> BigUint {
    let mut candidate = BigUint::from(if start % 2 == 0 { start + 1 } else { start });
    let step = BigUint::from(2u8);

    while candidate.gcd(phi) != BigUint::one() {
        candidate += &step;
    }

    candidate
}

/// Computes the modular inverse of `a` modulo `modulus`, if it exists.
///
/// # Parameters
/// - `a`: Value to invert.
/// - `modulus`: Modulus for the inverse.
///
/// # Returns
/// - `Option<BigUint>`: `Some(inv)` if `a * inv ≡ 1 (mod modulus)`, otherwise `None`.
///
/// # Expected Output
/// - Returns `None` when `a` and `modulus` are not coprime; no side effects.
pub fn mod_inverse(a: &BigUint, modulus: &BigUint) -> Option<BigUint> {
    let a_int = BigInt::from(a.clone());
    let m_int = BigInt::from(modulus.clone());

    let egcd = a_int.extended_gcd(&m_int);
    if egcd.gcd != BigInt::one() {
        return None;
    }

    let mut x = egcd.x % &m_int;
    if x.is_negative() {
        x += m_int;
    }

    x.to_biguint()
}

/// Encodes a `BigUint` as a lowercase hexadecimal string.
///
/// # Parameters
/// - `value`: Integer to encode.
///
/// # Returns
/// - `String`: Hex string without `0x` prefix; `"0"` if the value is zero.
///
/// # Expected Output
/// - Returns a lowercase hex representation; no side effects.
pub fn to_hex(value: &BigUint) -> String {
    let bytes = value.to_bytes_be();
    if bytes.is_empty() || bytes.iter().all(|b| *b == 0) {
        return "0".to_string();
    }
    let mut hex = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        let _ = std::fmt::Write::write_fmt(&mut hex, format_args!("{:02x}", byte));
    }
    hex
}

/// Returns the number of bits required to represent `value`.
///
/// # Parameters
/// - `value`: Integer whose bit width is measured.
///
/// # Returns
/// - `u64`: Bit length (0 for zero).
///
/// # Expected Output
/// - Returns the bit length; no side effects.
pub fn bit_length(value: &BigUint) -> u64 {
    value.bits()
}

/// Computes the Shannon entropy of a Bernoulli bit distribution.
///
/// # Parameters
/// - `p`: Probability of observing a `1` (clamped to `[0.0, 1.0]`).
///
/// # Returns
/// - `f64`: Entropy in bits within `[0.0, 1.0]`.
///
/// # Expected Output
/// - Returns `0.0` when `p` is `0.0` or `1.0`; no side effects.
pub fn shannon_entropy_bit(p: f64) -> f64 {
    let p = p.clamp(0.0, 1.0);
    if p == 0.0 || p == 1.0 {
        return 0.0;
    }
    let q = 1.0 - p;
    -(p * p.log2() + q * q.log2())
}

/// Samples a probable prime with the specified bit width.
///
/// # Parameters
/// - `bits`: Desired bit width (must fit in `u64` range).
/// - `rng`: Random number generator for candidate selection.
///
/// # Returns
/// - `BigUint`: A probable prime with the requested bit width (odd).
///
/// # Expected Output
/// - Returns a probable prime; no stdout/stderr output.
pub fn random_prime_with_bits(bits: u32, rng: &mut RngChoice) -> BigUint {
    let min_prime = BigUint::from(3u8);
    loop {
        let mut candidate = random_biguint_bits(bits, rng);
        if candidate < min_prime {
            candidate = min_prime.clone();
        }
        if candidate.is_even() {
            candidate += BigUint::one();
        }
        if is_probable_prime_big(&candidate) {
            return candidate;
        }
    }
}

/// Generates a random `BigUint` with up to the requested bit width.
///
/// # Parameters
/// - `bits`: Requested bit width (0 yields 0).
/// - `rng`: Random number generator for bytes.
///
/// # Returns
/// - `BigUint`: Random value with the top bit set when possible.
///
/// # Expected Output
/// - Returns a random integer with the requested width; no side effects.
pub fn random_biguint_bits(bits: u32, rng: &mut RngChoice) -> BigUint {
    if bits == 0 {
        return BigUint::zero();
    }
    let bytes_len = ((bits as usize) + 7) / 8;
    let mut bytes = vec![0u8; bytes_len];
    rng.fill_bytes(&mut bytes);
    let leading_bits = (bits % 8) as u8;
    if leading_bits != 0 {
        let mask = (1u8 << leading_bits) - 1;
        bytes[0] &= mask;
    }
    // Ensure the top bit is set so the value uses the requested width when possible.
    let top_bit = if leading_bits == 0 {
        0x80
    } else {
        1u8 << (leading_bits - 1)
    };
    bytes[0] |= top_bit;
    BigUint::from_bytes_be(&bytes)
}

/// Performs a Miller-Rabin probable-prime test for `u64` values.
///
/// # Parameters
/// - `n`: Integer to test.
///
/// # Returns
/// - `bool`: `true` if `n` is a probable prime, `false` if composite.
///
/// # Expected Output
/// - Returns a deterministic answer for the selected bases; no side effects.

/// Computes Euler's totient from a factorization `(p, e)` list.
///
/// # Parameters
/// - `factors`: Prime power factors for `n`, as `(prime, exponent)`.
///
/// # Returns
/// - `BigUint`: `phi(n)` computed as `Π (p-1) * p^(e-1)`.
///
/// # Expected Output
/// - Returns the totient value; no side effects.
pub fn compute_totient(factors: &[(BigUint, u64)]) -> BigUint {
    let mut phi = BigUint::one();
    for (p, e) in factors {
        if *e == 0 {
            continue;
        }
        let term = (p - BigUint::one()) * p.pow((*e as u32).saturating_sub(1));
        phi *= term;
    }
    phi
}

/// Samples a random `BigUint` in the range `[0, upper)`.
///
/// # Parameters
/// - `upper`: Exclusive upper bound.
/// - `rng`: Random number generator for sampling.
///
/// # Returns
/// - `BigUint`: A uniformly sampled value below `upper` (or 0 if `upper` is 0).
///
/// # Expected Output
/// - Returns a random value below `upper`; no side effects.
pub fn random_biguint_below(upper: &BigUint, rng: &mut RngChoice) -> BigUint {
    if upper.is_zero() {
        return BigUint::zero();
    }
    let bits = upper.bits();
    loop {
        let candidate = random_biguint_bits(bits as u32, rng);
        if &candidate < upper {
            return candidate;
        }
    }
}

/// Computes a modular square root using Tonelli-Shanks for odd prime `p`.
///
/// # Parameters
/// - `a`: Value whose square root is sought.
/// - `p`: Odd prime modulus.
///
/// # Returns
/// - `BigUint`: A square root `r` such that `r^2 ≡ a (mod p)` when one exists.
///
/// # Expected Output
/// - Returns `0` for `a = 0`; returns `1` when no root exists per Legendre symbol.
pub fn modular_sqrt(a: &BigUint, p: &BigUint) -> BigUint {
    // Tonelli-Shanks for odd prime p; demo uses small-ish primes so this is fine.
    if a.is_zero() {
        return BigUint::zero();
    }
    if p == &BigUint::from(2u8) {
        return BigUint::zero();
    }
    if legendre_symbol(a, p) != BigInt::one() {
        return BigUint::one();
    }

    let mut q = p - BigUint::one();
    let mut s = 0u32;
    while (&q & BigUint::one()).is_zero() {
        q >>= 1;
        s += 1;
    }

    if s == 1 {
        return a.modpow(&((p + BigUint::one()) >> 2), p);
    }

    let mut z = BigUint::from(2u8);
    while legendre_symbol(&z, p) != BigInt::from(-1) {
        z += BigUint::one();
    }

    let mut m = s;
    let mut c = z.modpow(&q, p);
    let mut t = a.modpow(&q, p);
    let mut r = a.modpow(&((&q + BigUint::one()) >> 1), p);

    while t != BigUint::one() {
        let mut i = 1u32;
        let mut t2i = t.modpow(&BigUint::from(2u32), p);
        while t2i != BigUint::one() {
            t2i = t2i.modpow(&BigUint::from(2u32), p);
            i += 1;
            if i == m {
                break;
            }
        }
        let b = c.modpow(&BigUint::from(1u64 << (m - i - 1)), p);
        r = (&r * &b) % p;
        c = (&b * &b) % p;
        t = (&t * &c) % p;
        m = i;
    }
    r
}

/// Computes the Legendre symbol `(a | p)`.
///
/// # Parameters
/// - `a`: Value to test.
/// - `p`: Odd prime modulus.
///
/// # Returns
/// - `BigInt`: `1` if `a` is a quadratic residue, `-1` if non-residue, `0` if divisible.
///
/// # Expected Output
/// - Returns the Legendre symbol value; no side effects.
pub fn legendre_symbol(a: &BigUint, p: &BigUint) -> BigInt {
    let ls = a.modpow(&((p - BigUint::one()) >> 1), p);
    if ls.is_zero() {
        BigInt::zero()
    } else if ls == BigUint::one() {
        BigInt::one()
    } else {
        BigInt::from(-1)
    }
}

/// Attempts to factor `n` into prime powers before a deadline.
///
/// # Parameters
/// - `n`: Composite (or prime) integer to factor.
/// - `rng`: Random number generator used by Pollard Rho.
/// - `deadline`: Time limit for the factorization attempt.
///
/// # Returns
/// - `Option<Vec<(BigUint, u64)>>`: `Some` list of factors on success, `None` on timeout.
///
/// # Expected Output
/// - Returns a sorted, coalesced factor list when successful; no stdout/stderr output.
pub fn factor_composite_with_timeout(
    n: &BigUint,
    rng: &mut RngChoice,
    deadline: Instant,
) -> Option<Vec<(BigUint, u64)>> {
    let mut factors = Vec::new();
    if !factor_recursive(n.clone(), &mut factors, rng, deadline) {
        return None;
    }
    factors.sort_by(|a, b| a.0.cmp(&b.0));
    Some(coalesce_factors(factors))
}

/// Recursively factors `n`, populating `out` with prime factors.
///
/// # Parameters
/// - `n`: Integer to factor.
/// - `out`: Output list to be populated with `(prime, exponent)` pairs.
/// - `rng`: Random number generator for Pollard Rho steps.
/// - `deadline`: Time limit; the function returns `false` if exceeded.
///
/// # Returns
/// - `bool`: `true` if factorization completed before the deadline.
///
/// # Expected Output
/// - On success, `out` is extended with factors; no stdout/stderr output.
pub fn factor_recursive(
    n: BigUint,
    out: &mut Vec<(BigUint, u64)>,
    rng: &mut RngChoice,
    deadline: Instant,
) -> bool {
    if Instant::now() >= deadline {
        return false;
    }
    if n <= BigUint::one() {
        return true;
    }
    if is_probable_prime_big(&n) {
        out.push((n, 1));
        return true;
    }
    let Some(divisor) = pollard_rho(&n, rng, deadline) else {
        return false;
    };
    let other = &n / &divisor;
    factor_recursive(divisor, out, rng, deadline) && factor_recursive(other, out, rng, deadline)
}

/// Merges duplicate prime factors by summing their exponents.
///
/// # Parameters
/// - `factors`: Unsorted list of `(prime, exponent)` entries.
///
/// # Returns
/// - `Vec<(BigUint, u64)>`: Sorted list with merged exponents.
///
/// # Expected Output
/// - Returns an empty vector when given no factors; no side effects.
pub fn coalesce_factors(mut factors: Vec<(BigUint, u64)>) -> Vec<(BigUint, u64)> {
    if factors.is_empty() {
        return factors;
    }
    factors.sort_by(|a, b| a.0.cmp(&b.0));
    let mut merged: Vec<(BigUint, u64)> = Vec::new();
    let mut current = factors[0].clone();
    for item in factors.into_iter().skip(1) {
        if item.0 == current.0 {
            current.1 += item.1;
        } else {
            merged.push(current);
            current = item;
        }
    }
    merged.push(current);
    merged
}

/// Attempts to find a non-trivial factor of `n` using Pollard's Rho.
///
/// # Parameters
/// - `n`: Integer to factor (must be composite for success).
/// - `rng`: Random number generator for selecting the polynomial constant and seeds.
/// - `deadline`: Time limit for the search.
///
/// # Returns
/// - `Option<BigUint>`: `Some(factor)` if found before the deadline; otherwise `None`.
///
/// # Expected Output
/// - Returns `2` immediately when `n` is even; no stdout/stderr output.
pub fn pollard_rho(n: &BigUint, rng: &mut RngChoice, deadline: Instant) -> Option<BigUint> {
    if n.is_even() {
        return Some(BigUint::from(2u8));
    }
    let one = BigUint::one();
    let two = &one + &one;

    let mut c = random_biguint_below(n, rng);
    let mut x = random_biguint_below(n, rng);
    let mut y = x.clone();
    let f = |val: &BigUint, c: &BigUint, n: &BigUint| (val.modpow(&two, n) + c) % n;
    let mut iter: u64 = 0;

    while Instant::now() < deadline {
        iter += 1;
        x = f(&x, &c, n);
        y = f(&f(&y, &c, n), &c, n);
        let diff = if &x >= &y { &x - &y } else { &y - &x };
        let d = diff.gcd(n);
        if d != one && d != *n {
            return Some(d);
        }
        if d == *n || iter > 10_000 {
            c = random_biguint_below(n, rng);
            x = random_biguint_below(n, rng);
            y = x.clone();
            iter = 0;
        }
    }
    None
}

/// Performs a Miller-Rabin probable-prime test for `BigUint` values.
///
/// # Parameters
/// - `n`: Integer to test.
///
/// # Returns
/// - `bool`: `true` if `n` is a probable prime, `false` if composite.
///
/// # Expected Output
/// - Returns a deterministic answer for the selected bases; no side effects.
pub fn is_probable_prime_big(n: &BigUint) -> bool {
    if n <= &BigUint::from(3u8) {
        return *n == BigUint::from(2u8) || *n == BigUint::from(3u8);
    }
    if n.is_even() {
        return false;
    }

    const SMALL_PRIMES: [u64; 16] = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59];
    for p in SMALL_PRIMES {
        let p_big = BigUint::from(p);
        if n == &p_big {
            return true;
        }
        if (n % &p_big).is_zero() {
            return false;
        }
    }

    let one = BigUint::one();
    let two = &one + &one;
    let n_minus_one = n - &one;
    let (d, s) = decompose_big(n_minus_one.clone());

    const BASES: [u64; 7] = [2, 3, 5, 7, 11, 13, 17];
    'outer: for a in BASES {
        let a = BigUint::from(a);
        let mut x = a.modpow(&d, n);
        if x == one || x == n_minus_one {
            continue;
        }
        for _ in 1..s {
            x = x.modpow(&two, n);
            if x == n_minus_one {
                continue 'outer;
            }
        }
        return false;
    }
    true
}

/// Decomposes `value` into `d * 2^s` with `d` odd (BigUint variant).
///
/// # Parameters
/// - `value`: Integer to decompose.
///
/// # Returns
/// - `(BigUint, u32)`: Tuple of `(d, s)` such that `value = d * 2^s`.
///
/// # Expected Output
/// - Returns the odd component and exponent; no side effects.
pub fn decompose_big(mut value: BigUint) -> (BigUint, u32) {
    let mut s = 0u32;
    let one = BigUint::one();
    while (&value & &one).is_zero() {
        value >>= 1;
        s += 1;
    }
    (value, s)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::{RngChoice, RngMode};
    use std::time::{Duration, Instant};

    #[test]
    fn test_choose_exponent_coprime() {
        let phi = BigUint::from(40u8);
        let e = choose_exponent(3, &phi);
        assert_eq!(e, BigUint::from(3u8));
    }

    #[test]
    fn test_choose_exponent_skips_even() {
        let phi = BigUint::from(40u8);
        let e = choose_exponent(10, &phi);
        assert_eq!(e, BigUint::from(11u8));
    }

    #[test]
    fn test_mod_inverse_exists() {
        let a = BigUint::from(3u8);
        let m = BigUint::from(11u8);
        let inv = mod_inverse(&a, &m).expect("inverse missing");
        assert_eq!((&a * &inv) % &m, BigUint::one());
    }

    #[test]
    fn test_mod_inverse_missing() {
        let a = BigUint::from(6u8);
        let m = BigUint::from(12u8);
        assert!(mod_inverse(&a, &m).is_none());
    }

    #[test]
    fn test_to_hex_zero() {
        let v = BigUint::zero();
        assert_eq!(to_hex(&v), "0");
    }

    #[test]
    fn test_to_hex_value() {
        let v = BigUint::from_bytes_be(&[0xde, 0xad, 0xbe, 0xef]);
        assert_eq!(to_hex(&v), "deadbeef");
    }

    #[test]
    fn test_bit_length_zero() {
        let v = BigUint::zero();
        assert_eq!(bit_length(&v), 0);
    }

    #[test]
    fn test_bit_length_value() {
        let v = BigUint::from(10u8); // 1010
        assert_eq!(bit_length(&v), 4);
    }

    #[test]
    fn test_random_prime_with_bits_basic() {
        let mut rng = RngChoice::from_seed(RngMode::Standard, 7);
        let p = random_prime_with_bits(16, &mut rng);
        assert!(is_probable_prime_big(&p));
        assert!(p.bits() >= 16u64);
    }

    #[test]
    fn test_random_prime_with_bits_odd() {
        let mut rng = RngChoice::from_seed(RngMode::Standard, 9);
        let p = random_prime_with_bits(20, &mut rng);
        assert!(p.is_odd());
    }

    #[test]
    fn test_random_biguint_bits_zero() {
        let mut rng = RngChoice::from_seed(RngMode::Standard, 1);
        let v = random_biguint_bits(0, &mut rng);
        assert!(v.is_zero());
    }

    #[test]
    fn test_random_biguint_bits_range() {
        let mut rng = RngChoice::from_seed(RngMode::Standard, 2);
        let v = random_biguint_bits(8, &mut rng);
        assert!(v.bits() <= 8);
    }

    #[test]
    fn test_compute_totient_two_primes() {
        let factors = vec![(BigUint::from(3u8), 1), (BigUint::from(5u8), 1)];
        let phi = compute_totient(&factors);
        assert_eq!(phi, BigUint::from(8u8));
    }

    #[test]
    fn test_compute_totient_power() {
        let factors = vec![(BigUint::from(2u8), 3)];
        let phi = compute_totient(&factors);
        assert_eq!(phi, BigUint::from(4u8));
    }

    #[test]
    fn test_random_biguint_below() {
        let mut rng = RngChoice::from_seed(RngMode::Standard, 3);
        let upper = BigUint::from(10u8);
        for _ in 0..5 {
            let v = random_biguint_below(&upper, &mut rng);
            assert!(v < upper);
        }
    }

    #[test]
    fn test_random_biguint_below_zero() {
        let mut rng = RngChoice::from_seed(RngMode::Standard, 4);
        let upper = BigUint::zero();
        let v = random_biguint_below(&upper, &mut rng);
        assert!(v.is_zero());
    }

    #[test]
    fn test_modular_sqrt_residue() {
        let p = BigUint::from(11u8);
        let a = BigUint::from(9u8);
        let r = modular_sqrt(&a, &p);
        assert_eq!((&r * &r) % &p, a);
    }

    #[test]
    fn test_modular_sqrt_zero() {
        let p = BigUint::from(11u8);
        let a = BigUint::zero();
        let r = modular_sqrt(&a, &p);
        assert!(r.is_zero());
    }

    #[test]
    fn test_legendre_symbol_residue() {
        let p = BigUint::from(11u8);
        let a = BigUint::from(9u8);
        let ls = legendre_symbol(&a, &p);
        assert_eq!(ls, BigInt::one());
    }

    #[test]
    fn test_legendre_symbol_non_residue() {
        let p = BigUint::from(11u8);
        let a = BigUint::from(2u8);
        let ls = legendre_symbol(&a, &p);
        assert_eq!(ls, BigInt::from(-1));
    }

    #[test]
    fn test_is_probable_prime_big_prime() {
        let p = BigUint::from(101u8);
        assert!(is_probable_prime_big(&p));
    }

    #[test]
    fn test_is_probable_prime_big_composite() {
        let n = BigUint::from(121u8);
        assert!(!is_probable_prime_big(&n));
    }

    #[test]
    fn test_decompose_big_even() {
        let (d, s) = decompose_big(BigUint::from(40u8));
        assert_eq!(d, BigUint::from(5u8));
        assert_eq!(s, 3);
    }

    #[test]
    fn test_decompose_big_odd() {
        let (d, s) = decompose_big(BigUint::from(45u8));
        assert_eq!(d, BigUint::from(45u8));
        assert_eq!(s, 0);
    }

    #[test]
    fn test_pollard_rho_even() {
        let mut rng = RngChoice::from_seed(RngMode::Standard, 11);
        let n = BigUint::from(100u8);
        let deadline = Instant::now() + Duration::from_millis(50);
        let factor = pollard_rho(&n, &mut rng, deadline).expect("missing factor");
        assert_eq!(factor, BigUint::from(2u8));
    }

    #[test]
    fn test_pollard_rho_composite() {
        let mut rng = RngChoice::from_seed(RngMode::Standard, 12);
        let n = BigUint::from(8051u64); // 83 * 97
        let deadline = Instant::now() + Duration::from_secs(1);
        let factor = pollard_rho(&n, &mut rng, deadline).expect("missing factor");
        assert!(&n % &factor == BigUint::zero());
        assert!(factor != BigUint::one() && factor != n);
    }

    #[test]
    fn test_factor_recursive_composite() {
        let mut rng = RngChoice::from_seed(RngMode::Standard, 13);
        let mut factors = Vec::new();
        let deadline = Instant::now() + Duration::from_secs(1);
        assert!(factor_recursive(
            BigUint::from(12u8),
            &mut factors,
            &mut rng,
            deadline
        ));
        let mut values: Vec<BigUint> = factors.into_iter().map(|(p, _)| p).collect();
        values.sort();
        assert!(values.contains(&BigUint::from(2u8)));
        assert!(values.contains(&BigUint::from(3u8)));
    }

    #[test]
    fn test_factor_recursive_one() {
        let mut rng = RngChoice::from_seed(RngMode::Standard, 14);
        let mut factors = Vec::new();
        let deadline = Instant::now() + Duration::from_secs(1);
        assert!(factor_recursive(
            BigUint::one(),
            &mut factors,
            &mut rng,
            deadline
        ));
        assert!(factors.is_empty());
    }

    #[test]
    fn test_factor_composite_with_timeout() {
        let mut rng = RngChoice::from_seed(RngMode::Standard, 15);
        let n = BigUint::from(84u8);
        let deadline = Instant::now() + Duration::from_secs(1);
        let factors =
            factor_composite_with_timeout(&n, &mut rng, deadline).expect("missing factors");
        let product = factors
            .iter()
            .fold(BigUint::one(), |acc, (p, e)| acc * p.pow(*e as u32));
        assert_eq!(product, n);
    }

    #[test]
    fn test_factor_composite_with_timeout_prime() {
        let mut rng = RngChoice::from_seed(RngMode::Standard, 16);
        let n = BigUint::from(13u8);
        let deadline = Instant::now() + Duration::from_secs(1);
        let factors =
            factor_composite_with_timeout(&n, &mut rng, deadline).expect("missing factors");
        assert_eq!(factors.len(), 1);
        assert_eq!(factors[0].0, n);
        assert_eq!(factors[0].1, 1);
    }

    #[test]
    fn test_coalesce_factors_merges() {
        let factors = vec![
            (BigUint::from(3u8), 1),
            (BigUint::from(2u8), 1),
            (BigUint::from(3u8), 2),
        ];
        let merged = coalesce_factors(factors);
        assert_eq!(merged.len(), 2);
        assert_eq!(merged[0].0, BigUint::from(2u8));
        assert_eq!(merged[0].1, 1);
        assert_eq!(merged[1].0, BigUint::from(3u8));
        assert_eq!(merged[1].1, 3);
    }

    #[test]
    fn test_coalesce_factors_empty() {
        let merged = coalesce_factors(Vec::new());
        assert!(merged.is_empty());
    }
}
