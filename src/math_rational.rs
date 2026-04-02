use bigdecimal::BigDecimal;
use num_bigint::{BigInt, BigUint};
use num_integer::Integer;
use num_traits::{One, Signed, Zero};
use std::cmp::Ordering;
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// Represents errors that can occur while constructing or approximating rationals.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RationalError {
    ZeroDenominator,
    DivisionByZero,
    EmptyDecimal,
    InvalidDecimal(String),
    ExponentOutOfRange,
    InvalidBounds(&'static str),
}

impl fmt::Display for RationalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroDenominator => write!(f, "denominator must not be zero"),
            Self::DivisionByZero => write!(f, "division by zero"),
            Self::EmptyDecimal => write!(f, "decimal input must not be empty"),
            Self::InvalidDecimal(value) => write!(f, "invalid decimal input: {value}"),
            Self::ExponentOutOfRange => write!(f, "decimal exponent is out of range"),
            Self::InvalidBounds(message) => write!(f, "invalid approximation bounds: {message}"),
        }
    }
}

impl std::error::Error for RationalError {}

/// Defines numerator and denominator limits for rational approximation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ApproximationBounds {
    max_numerator: BigUint,
    max_denominator: BigUint,
}

impl ApproximationBounds {
    /// Creates approximation bounds from absolute size limits.
    ///
    /// # Parameters
    /// - `max_numerator`: Maximum allowed absolute numerator.
    /// - `max_denominator`: Maximum allowed positive denominator.
    ///
    /// # Returns
    /// - `Result<Self, RationalError>`: Bounds on success, or an error when the denominator limit is zero.
    ///
    /// # Expected Output
    /// - Returns validated bounds; no side effects.
    pub fn new(max_numerator: BigUint, max_denominator: BigUint) -> Result<Self, RationalError> {
        if max_denominator.is_zero() {
            return Err(RationalError::InvalidBounds(
                "max_denominator must be at least 1",
            ));
        }

        Ok(Self {
            max_numerator,
            max_denominator,
        })
    }

    /// Creates approximation bounds from numerator and denominator bit widths.
    ///
    /// # Parameters
    /// - `max_numerator_bits`: Maximum bit width for the absolute numerator (`0` allows only `0`).
    /// - `max_denominator_bits`: Maximum bit width for the positive denominator.
    ///
    /// # Returns
    /// - `Result<Self, RationalError>`: Bounds on success, or an error when the denominator width is zero.
    ///
    /// # Expected Output
    /// - Returns validated bounds derived from bit widths; no side effects.
    pub fn from_bit_widths(
        max_numerator_bits: u64,
        max_denominator_bits: u64,
    ) -> Result<Self, RationalError> {
        if max_denominator_bits == 0 {
            return Err(RationalError::InvalidBounds(
                "max_denominator_bits must be at least 1",
            ));
        }

        Self::new(
            max_value_for_bits(max_numerator_bits),
            max_value_for_bits(max_denominator_bits),
        )
    }

    /// Returns the maximum absolute numerator permitted by these bounds.
    ///
    /// # Parameters
    /// - None.
    ///
    /// # Returns
    /// - `&BigUint`: Absolute numerator limit.
    ///
    /// # Expected Output
    /// - Returns a shared reference; no side effects.
    pub fn max_numerator(&self) -> &BigUint {
        &self.max_numerator
    }

    /// Returns the maximum positive denominator permitted by these bounds.
    ///
    /// # Parameters
    /// - None.
    ///
    /// # Returns
    /// - `&BigUint`: Denominator limit.
    ///
    /// # Expected Output
    /// - Returns a shared reference; no side effects.
    pub fn max_denominator(&self) -> &BigUint {
        &self.max_denominator
    }
}

/// Stores a normalized rational value backed by arbitrary-precision integers.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BigRational {
    numerator: BigInt,
    denominator: BigInt,
}

impl BigRational {
    /// Builds a normalized rational from a numerator and denominator.
    ///
    /// # Parameters
    /// - `numerator`: Signed numerator.
    /// - `denominator`: Signed denominator, which must be non-zero.
    ///
    /// # Returns
    /// - `Result<Self, RationalError>`: Normalized rational on success, or an error when the denominator is zero.
    ///
    /// # Expected Output
    /// - Returns a reduced rational with a positive denominator; no side effects.
    pub fn new(numerator: BigInt, denominator: BigInt) -> Result<Self, RationalError> {
        if denominator.is_zero() {
            return Err(RationalError::ZeroDenominator);
        }

        Ok(Self::reduce(numerator, denominator))
    }

    /// Returns the additive identity.
    ///
    /// # Parameters
    /// - None.
    ///
    /// # Returns
    /// - `Self`: The rational value `0`.
    ///
    /// # Expected Output
    /// - Returns `0/1`; no side effects.
    pub fn zero() -> Self {
        Self {
            numerator: BigInt::zero(),
            denominator: BigInt::one(),
        }
    }

    /// Parses a decimal string into its exact rational representation.
    ///
    /// # Parameters
    /// - `decimal`: Decimal text, optionally signed and optionally using scientific notation.
    ///
    /// # Returns
    /// - `Result<Self, RationalError>`: Exact rational on success, or a parse error when invalid.
    ///
    /// # Expected Output
    /// - Returns a normalized exact representation; no side effects.
    pub fn from_decimal(decimal: &str) -> Result<Self, RationalError> {
        let (numerator, denominator) = parse_decimal_to_ratio(decimal)?;
        Self::new(numerator, denominator)
    }

    /// Converts a `BigDecimal` into its exact rational representation.
    ///
    /// # Parameters
    /// - `decimal`: Arbitrary-precision decimal value.
    ///
    /// # Returns
    /// - `Result<Self, RationalError>`: Exact rational on success.
    ///
    /// # Expected Output
    /// - Returns a normalized exact representation; no side effects.
    pub fn from_bigdecimal(decimal: &BigDecimal) -> Result<Self, RationalError> {
        let rendered = decimal.to_string();
        Self::from_decimal(&rendered)
    }

    /// Approximates a decimal string using explicit numerator and denominator size limits.
    ///
    /// # Parameters
    /// - `decimal`: Decimal text to approximate.
    /// - `max_numerator`: Maximum allowed absolute numerator.
    /// - `max_denominator`: Maximum allowed positive denominator.
    ///
    /// # Returns
    /// - `Result<Self, RationalError>`: Nearest admissible approximation, or an error when parsing or bounds validation fails.
    ///
    /// # Expected Output
    /// - Returns the nearest admissible rational; no side effects.
    pub fn approximate_decimal_with_bounds(
        decimal: &str,
        max_numerator: BigUint,
        max_denominator: BigUint,
    ) -> Result<Self, RationalError> {
        let bounds = ApproximationBounds::new(max_numerator, max_denominator)?;
        Self::from_decimal(decimal)?.approximate(&bounds)
    }

    /// Approximates a decimal string using numerator and denominator bit widths.
    ///
    /// # Parameters
    /// - `decimal`: Decimal text to approximate.
    /// - `max_numerator_bits`: Maximum bit width for the absolute numerator.
    /// - `max_denominator_bits`: Maximum bit width for the denominator.
    ///
    /// # Returns
    /// - `Result<Self, RationalError>`: Nearest admissible approximation, or an error when parsing or bounds validation fails.
    ///
    /// # Expected Output
    /// - Returns the nearest admissible rational; no side effects.
    pub fn approximate_decimal_with_bits(
        decimal: &str,
        max_numerator_bits: u64,
        max_denominator_bits: u64,
    ) -> Result<Self, RationalError> {
        let bounds =
            ApproximationBounds::from_bit_widths(max_numerator_bits, max_denominator_bits)?;
        Self::from_decimal(decimal)?.approximate(&bounds)
    }

    /// Approximates a `BigDecimal` using explicit numerator and denominator size limits.
    ///
    /// # Parameters
    /// - `decimal`: Decimal value to approximate.
    /// - `max_numerator`: Maximum allowed absolute numerator.
    /// - `max_denominator`: Maximum allowed positive denominator.
    ///
    /// # Returns
    /// - `Result<Self, RationalError>`: Nearest admissible approximation.
    ///
    /// # Expected Output
    /// - Returns the nearest admissible rational; no side effects.
    pub fn approximate_bigdecimal_with_bounds(
        decimal: &BigDecimal,
        max_numerator: BigUint,
        max_denominator: BigUint,
    ) -> Result<Self, RationalError> {
        let bounds = ApproximationBounds::new(max_numerator, max_denominator)?;
        Self::from_bigdecimal(decimal)?.approximate(&bounds)
    }

    /// Approximates a `BigDecimal` using numerator and denominator bit widths.
    ///
    /// # Parameters
    /// - `decimal`: Decimal value to approximate.
    /// - `max_numerator_bits`: Maximum bit width for the absolute numerator.
    /// - `max_denominator_bits`: Maximum bit width for the denominator.
    ///
    /// # Returns
    /// - `Result<Self, RationalError>`: Nearest admissible approximation.
    ///
    /// # Expected Output
    /// - Returns the nearest admissible rational; no side effects.
    pub fn approximate_bigdecimal_with_bits(
        decimal: &BigDecimal,
        max_numerator_bits: u64,
        max_denominator_bits: u64,
    ) -> Result<Self, RationalError> {
        let bounds =
            ApproximationBounds::from_bit_widths(max_numerator_bits, max_denominator_bits)?;
        Self::from_bigdecimal(decimal)?.approximate(&bounds)
    }

    /// Approximates this rational value under the supplied bounds.
    ///
    /// # Parameters
    /// - `bounds`: Absolute numerator and denominator limits.
    ///
    /// # Returns
    /// - `Result<Self, RationalError>`: Nearest admissible rational.
    ///
    /// # Expected Output
    /// - Returns a normalized approximation; no side effects.
    pub fn approximate(&self, bounds: &ApproximationBounds) -> Result<Self, RationalError> {
        if bounds.max_denominator().is_zero() {
            return Err(RationalError::InvalidBounds(
                "max_denominator must be at least 1",
            ));
        }

        if self.numerator.is_zero() {
            return Ok(Self::zero());
        }

        let abs_numerator = self
            .numerator
            .abs()
            .to_biguint()
            .expect("absolute numerator is always non-negative");
        let denominator = self
            .denominator
            .to_biguint()
            .expect("normalized denominator is always positive");

        if abs_numerator <= *bounds.max_numerator() && denominator <= *bounds.max_denominator() {
            return Ok(self.clone());
        }

        let (best_num, best_den) =
            best_positive_approximation(&abs_numerator, &denominator, bounds);
        let signed_num = if self.numerator.is_negative() {
            -BigInt::from(best_num)
        } else {
            BigInt::from(best_num)
        };

        Ok(Self::reduce(signed_num, BigInt::from(best_den)))
    }

    /// Returns the numerator of the rational.
    ///
    /// # Parameters
    /// - None.
    ///
    /// # Returns
    /// - `&BigInt`: Shared reference to the numerator.
    ///
    /// # Expected Output
    /// - Returns a shared reference; no side effects.
    pub fn numerator(&self) -> &BigInt {
        &self.numerator
    }

    /// Returns the positive denominator of the rational.
    ///
    /// # Parameters
    /// - None.
    ///
    /// # Returns
    /// - `&BigInt`: Shared reference to the denominator.
    ///
    /// # Expected Output
    /// - Returns a shared reference; no side effects.
    pub fn denominator(&self) -> &BigInt {
        &self.denominator
    }

    /// Returns `true` when the rational is exactly zero.
    ///
    /// # Parameters
    /// - None.
    ///
    /// # Returns
    /// - `bool`: `true` for zero and `false` otherwise.
    ///
    /// # Expected Output
    /// - Returns a boolean flag; no side effects.
    pub fn is_zero(&self) -> bool {
        self.numerator.is_zero()
    }

    /// Returns the multiplicative inverse when the rational is non-zero.
    ///
    /// # Parameters
    /// - None.
    ///
    /// # Returns
    /// - `Result<Self, RationalError>`: Reciprocal on success, or an error for zero.
    ///
    /// # Expected Output
    /// - Returns a normalized reciprocal; no side effects.
    pub fn checked_recip(&self) -> Result<Self, RationalError> {
        if self.numerator.is_zero() {
            return Err(RationalError::DivisionByZero);
        }

        Ok(Self::reduce(
            self.denominator.clone(),
            self.numerator.clone(),
        ))
    }

    /// Divides by another rational and reports division-by-zero explicitly.
    ///
    /// # Parameters
    /// - `rhs`: Divisor rational.
    ///
    /// # Returns
    /// - `Result<Self, RationalError>`: Quotient on success, or an error when the divisor is zero.
    ///
    /// # Expected Output
    /// - Returns a normalized quotient; no side effects.
    pub fn checked_div(&self, rhs: &Self) -> Result<Self, RationalError> {
        if rhs.numerator.is_zero() {
            return Err(RationalError::DivisionByZero);
        }

        Ok(Self::reduce(
            &self.numerator * &rhs.denominator,
            &self.denominator * &rhs.numerator,
        ))
    }

    /// Divides by a big integer and reports division-by-zero explicitly.
    ///
    /// # Parameters
    /// - `rhs`: Divisor integer.
    ///
    /// # Returns
    /// - `Result<Self, RationalError>`: Quotient on success, or an error when the divisor is zero.
    ///
    /// # Expected Output
    /// - Returns a normalized quotient; no side effects.
    pub fn checked_div_bigint(&self, rhs: &BigInt) -> Result<Self, RationalError> {
        if rhs.is_zero() {
            return Err(RationalError::DivisionByZero);
        }

        Ok(Self::reduce(
            self.numerator.clone(),
            &self.denominator * rhs,
        ))
    }

    /// Converts the rational to a big integer by taking the floor.
    ///
    /// # Parameters
    /// - None.
    ///
    /// # Returns
    /// - `BigInt`: Floor of the rational value.
    ///
    /// # Expected Output
    /// - Returns the floor using mathematical rounding toward negative infinity; no side effects.
    pub fn to_bigint(&self) -> BigInt {
        self.numerator.div_floor(&self.denominator)
    }

    /// Normalizes a rational with a non-zero denominator.
    ///
    /// # Parameters
    /// - `numerator`: Raw numerator.
    /// - `denominator`: Raw denominator.
    ///
    /// # Returns
    /// - `Self`: Reduced rational with a positive denominator.
    ///
    /// # Expected Output
    /// - Returns a normalized value; no side effects.
    fn reduce(mut numerator: BigInt, mut denominator: BigInt) -> Self {
        debug_assert!(!denominator.is_zero());

        if denominator.is_negative() {
            numerator = -numerator;
            denominator = -denominator;
        }

        if numerator.is_zero() {
            return Self::zero();
        }

        let gcd = numerator.abs().gcd(&denominator);
        Self {
            numerator: numerator / &gcd,
            denominator: denominator / gcd,
        }
    }
}

impl Default for BigRational {
    fn default() -> Self {
        Self::zero()
    }
}

impl From<BigInt> for BigRational {
    fn from(value: BigInt) -> Self {
        Self {
            numerator: value,
            denominator: BigInt::one(),
        }
    }
}

impl From<&BigInt> for BigRational {
    fn from(value: &BigInt) -> Self {
        Self::from(value.clone())
    }
}

impl TryFrom<BigDecimal> for BigRational {
    type Error = RationalError;

    fn try_from(value: BigDecimal) -> Result<Self, Self::Error> {
        Self::from_bigdecimal(&value)
    }
}

impl TryFrom<&BigDecimal> for BigRational {
    type Error = RationalError;

    fn try_from(value: &BigDecimal) -> Result<Self, Self::Error> {
        Self::from_bigdecimal(value)
    }
}

impl fmt::Display for BigRational {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.denominator == BigInt::one() {
            write!(f, "{}", self.numerator)
        } else {
            write!(f, "{}/{}", self.numerator, self.denominator)
        }
    }
}

impl PartialOrd for BigRational {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for BigRational {
    fn cmp(&self, other: &Self) -> Ordering {
        (&self.numerator * &other.denominator).cmp(&(&other.numerator * &self.denominator))
    }
}

impl Neg for BigRational {
    type Output = BigRational;

    fn neg(self) -> Self::Output {
        BigRational {
            numerator: -self.numerator,
            denominator: self.denominator,
        }
    }
}

impl Neg for &BigRational {
    type Output = BigRational;

    fn neg(self) -> Self::Output {
        BigRational {
            numerator: -self.numerator.clone(),
            denominator: self.denominator.clone(),
        }
    }
}

impl Add<&BigRational> for &BigRational {
    type Output = BigRational;

    fn add(self, rhs: &BigRational) -> Self::Output {
        BigRational::reduce(
            &self.numerator * &rhs.denominator + &rhs.numerator * &self.denominator,
            &self.denominator * &rhs.denominator,
        )
    }
}

impl Add<BigRational> for &BigRational {
    type Output = BigRational;

    fn add(self, rhs: BigRational) -> Self::Output {
        self + &rhs
    }
}

impl Add<&BigRational> for BigRational {
    type Output = BigRational;

    fn add(self, rhs: &BigRational) -> Self::Output {
        &self + rhs
    }
}

impl Add<BigRational> for BigRational {
    type Output = BigRational;

    fn add(self, rhs: BigRational) -> Self::Output {
        &self + &rhs
    }
}

impl Add<&BigInt> for &BigRational {
    type Output = BigRational;

    fn add(self, rhs: &BigInt) -> Self::Output {
        BigRational::reduce(
            &self.numerator + rhs * &self.denominator,
            self.denominator.clone(),
        )
    }
}

impl Add<BigInt> for &BigRational {
    type Output = BigRational;

    fn add(self, rhs: BigInt) -> Self::Output {
        self + &rhs
    }
}

impl Add<&BigInt> for BigRational {
    type Output = BigRational;

    fn add(self, rhs: &BigInt) -> Self::Output {
        &self + rhs
    }
}

impl Add<BigInt> for BigRational {
    type Output = BigRational;

    fn add(self, rhs: BigInt) -> Self::Output {
        &self + &rhs
    }
}

impl Sub<&BigRational> for &BigRational {
    type Output = BigRational;

    fn sub(self, rhs: &BigRational) -> Self::Output {
        BigRational::reduce(
            &self.numerator * &rhs.denominator - &rhs.numerator * &self.denominator,
            &self.denominator * &rhs.denominator,
        )
    }
}

impl Sub<BigRational> for &BigRational {
    type Output = BigRational;

    fn sub(self, rhs: BigRational) -> Self::Output {
        self - &rhs
    }
}

impl Sub<&BigRational> for BigRational {
    type Output = BigRational;

    fn sub(self, rhs: &BigRational) -> Self::Output {
        &self - rhs
    }
}

impl Sub<BigRational> for BigRational {
    type Output = BigRational;

    fn sub(self, rhs: BigRational) -> Self::Output {
        &self - &rhs
    }
}

impl Sub<&BigInt> for &BigRational {
    type Output = BigRational;

    fn sub(self, rhs: &BigInt) -> Self::Output {
        BigRational::reduce(
            &self.numerator - rhs * &self.denominator,
            self.denominator.clone(),
        )
    }
}

impl Sub<BigInt> for &BigRational {
    type Output = BigRational;

    fn sub(self, rhs: BigInt) -> Self::Output {
        self - &rhs
    }
}

impl Sub<&BigInt> for BigRational {
    type Output = BigRational;

    fn sub(self, rhs: &BigInt) -> Self::Output {
        &self - rhs
    }
}

impl Sub<BigInt> for BigRational {
    type Output = BigRational;

    fn sub(self, rhs: BigInt) -> Self::Output {
        &self - &rhs
    }
}

impl Mul<&BigRational> for &BigRational {
    type Output = BigRational;

    fn mul(self, rhs: &BigRational) -> Self::Output {
        BigRational::reduce(
            &self.numerator * &rhs.numerator,
            &self.denominator * &rhs.denominator,
        )
    }
}

impl Mul<BigRational> for &BigRational {
    type Output = BigRational;

    fn mul(self, rhs: BigRational) -> Self::Output {
        self * &rhs
    }
}

impl Mul<&BigRational> for BigRational {
    type Output = BigRational;

    fn mul(self, rhs: &BigRational) -> Self::Output {
        &self * rhs
    }
}

impl Mul<BigRational> for BigRational {
    type Output = BigRational;

    fn mul(self, rhs: BigRational) -> Self::Output {
        &self * &rhs
    }
}

impl Mul<&BigInt> for &BigRational {
    type Output = BigRational;

    fn mul(self, rhs: &BigInt) -> Self::Output {
        BigRational::reduce(&self.numerator * rhs, self.denominator.clone())
    }
}

impl Mul<BigInt> for &BigRational {
    type Output = BigRational;

    fn mul(self, rhs: BigInt) -> Self::Output {
        self * &rhs
    }
}

impl Mul<&BigInt> for BigRational {
    type Output = BigRational;

    fn mul(self, rhs: &BigInt) -> Self::Output {
        &self * rhs
    }
}

impl Mul<BigInt> for BigRational {
    type Output = BigRational;

    fn mul(self, rhs: BigInt) -> Self::Output {
        &self * &rhs
    }
}

impl Div<&BigRational> for &BigRational {
    type Output = BigRational;

    fn div(self, rhs: &BigRational) -> Self::Output {
        self.checked_div(rhs)
            .expect("attempted to divide a BigRational by zero")
    }
}

impl Div<BigRational> for &BigRational {
    type Output = BigRational;

    fn div(self, rhs: BigRational) -> Self::Output {
        self / &rhs
    }
}

impl Div<&BigRational> for BigRational {
    type Output = BigRational;

    fn div(self, rhs: &BigRational) -> Self::Output {
        &self / rhs
    }
}

impl Div<BigRational> for BigRational {
    type Output = BigRational;

    fn div(self, rhs: BigRational) -> Self::Output {
        &self / &rhs
    }
}

impl Div<&BigInt> for &BigRational {
    type Output = BigRational;

    fn div(self, rhs: &BigInt) -> Self::Output {
        self.checked_div_bigint(rhs)
            .expect("attempted to divide a BigRational by zero")
    }
}

impl Div<BigInt> for &BigRational {
    type Output = BigRational;

    fn div(self, rhs: BigInt) -> Self::Output {
        self / &rhs
    }
}

impl Div<&BigInt> for BigRational {
    type Output = BigRational;

    fn div(self, rhs: &BigInt) -> Self::Output {
        &self / rhs
    }
}

impl Div<BigInt> for BigRational {
    type Output = BigRational;

    fn div(self, rhs: BigInt) -> Self::Output {
        &self / &rhs
    }
}

/// Returns the largest unsigned value representable by the requested bit width.
///
/// # Parameters
/// - `bits`: Bit width to expand.
///
/// # Returns
/// - `BigUint`: `(2^bits) - 1`, or `0` when `bits == 0`.
///
/// # Expected Output
/// - Returns an integer bound; no side effects.
fn max_value_for_bits(bits: u64) -> BigUint {
    if bits == 0 {
        return BigUint::zero();
    }

    (BigUint::one() << (bits as usize)) - BigUint::one()
}

/// Parses a decimal string into an exact numerator and denominator pair.
///
/// # Parameters
/// - `decimal`: Decimal text, optionally signed and optionally using scientific notation.
///
/// # Returns
/// - `Result<(BigInt, BigInt), RationalError>`: Exact numerator and denominator on success.
///
/// # Expected Output
/// - Returns normalized raw parts suitable for rational construction; no side effects.
fn parse_decimal_to_ratio(decimal: &str) -> Result<(BigInt, BigInt), RationalError> {
    let trimmed = decimal.trim();
    if trimmed.is_empty() {
        return Err(RationalError::EmptyDecimal);
    }

    let (mantissa, exponent) = split_exponent(trimmed)?;
    let (negative, unsigned_mantissa) = match mantissa.as_bytes().first() {
        Some(b'+') => (false, &mantissa[1..]),
        Some(b'-') => (true, &mantissa[1..]),
        _ => (false, mantissa),
    };

    if unsigned_mantissa.is_empty() {
        return Err(RationalError::InvalidDecimal(trimmed.to_string()));
    }

    let mut parts = unsigned_mantissa.split('.');
    let whole = parts.next().unwrap_or_default();
    let fraction = parts.next().unwrap_or_default();
    if parts.next().is_some() || (whole.is_empty() && fraction.is_empty()) {
        return Err(RationalError::InvalidDecimal(trimmed.to_string()));
    }
    if !whole.as_bytes().iter().all(u8::is_ascii_digit)
        || !fraction.as_bytes().iter().all(u8::is_ascii_digit)
    {
        return Err(RationalError::InvalidDecimal(trimmed.to_string()));
    }

    let mut digits = String::with_capacity(whole.len() + fraction.len());
    digits.push_str(whole);
    digits.push_str(fraction);
    if digits.is_empty() {
        return Err(RationalError::InvalidDecimal(trimmed.to_string()));
    }

    let mut numerator = BigInt::parse_bytes(digits.as_bytes(), 10)
        .ok_or_else(|| RationalError::InvalidDecimal(trimmed.to_string()))?;
    if negative {
        numerator = -numerator;
    }

    let scale = (fraction.len() as i64)
        .checked_sub(exponent)
        .ok_or(RationalError::ExponentOutOfRange)?;

    if scale <= 0 {
        let multiplier = BigInt::from(pow10_biguint((-scale) as u64));
        return Ok((numerator * multiplier, BigInt::one()));
    }

    Ok((numerator, BigInt::from(pow10_biguint(scale as u64))))
}

/// Splits a decimal string into mantissa and exponent components.
///
/// # Parameters
/// - `decimal`: Raw decimal text.
///
/// # Returns
/// - `Result<(&str, i64), RationalError>`: Mantissa plus parsed exponent.
///
/// # Expected Output
/// - Returns borrowed mantissa text and exponent value; no side effects.
fn split_exponent(decimal: &str) -> Result<(&str, i64), RationalError> {
    match decimal.find(['e', 'E']) {
        Some(index) => {
            let mantissa = &decimal[..index];
            let exponent_text = &decimal[index + 1..];
            if exponent_text.is_empty() {
                return Err(RationalError::InvalidDecimal(decimal.to_string()));
            }
            let exponent = exponent_text
                .parse::<i64>()
                .map_err(|_| RationalError::ExponentOutOfRange)?;
            Ok((mantissa, exponent))
        }
        None => Ok((decimal, 0)),
    }
}

/// Raises ten to an unsigned exponent using arbitrary-precision arithmetic.
///
/// # Parameters
/// - `exponent`: Power applied to the base `10`.
///
/// # Returns
/// - `BigUint`: `10^exponent`.
///
/// # Expected Output
/// - Returns an exact power of ten; no side effects.
fn pow10_biguint(exponent: u64) -> BigUint {
    let mut result = BigUint::one();
    let mut base = BigUint::from(10u8);
    let mut exp = exponent;

    while exp > 0 {
        if exp & 1 == 1 {
            result *= &base;
        }
        exp >>= 1;
        if exp > 0 {
            base = &base * &base;
        }
    }

    result
}

/// Computes the continued-fraction partial quotients of a positive rational.
///
/// # Parameters
/// - `numerator`: Positive numerator.
/// - `denominator`: Positive denominator.
///
/// # Returns
/// - `Vec<BigUint>`: Continued-fraction partial quotients.
///
/// # Expected Output
/// - Returns the exact finite continued fraction; no side effects.
fn continued_fraction(numerator: &BigUint, denominator: &BigUint) -> Vec<BigUint> {
    let mut n = numerator.clone();
    let mut d = denominator.clone();
    let mut partials = Vec::new();

    while !d.is_zero() {
        let (q, r) = n.div_rem(&d);
        partials.push(q);
        n = d;
        d = r;
    }

    partials
}

/// Selects the nearest admissible positive approximation under the given bounds.
///
/// # Parameters
/// - `target_num`: Absolute numerator of the target value.
/// - `target_den`: Positive denominator of the target value.
/// - `bounds`: Absolute numerator and denominator limits.
///
/// # Returns
/// - `(BigUint, BigUint)`: Best admissible numerator and denominator.
///
/// # Expected Output
/// - Returns the nearest admissible positive fraction; no side effects.
fn best_positive_approximation(
    target_num: &BigUint,
    target_den: &BigUint,
    bounds: &ApproximationBounds,
) -> (BigUint, BigUint) {
    let partials = continued_fraction(target_num, target_den);
    let mut best_num = BigUint::zero();
    let mut best_den = BigUint::one();

    let mut prev2_num = BigUint::zero();
    let mut prev2_den = BigUint::one();
    let mut prev1_num = BigUint::one();
    let mut prev1_den = BigUint::zero();

    for partial in partials {
        if let Some(step) = max_admissible_step(
            &partial, &prev2_num, &prev2_den, &prev1_num, &prev1_den, bounds,
        ) {
            let candidate_num = &prev2_num + &step * &prev1_num;
            let candidate_den = &prev2_den + &step * &prev1_den;
            if candidate_num <= *bounds.max_numerator()
                && candidate_den <= *bounds.max_denominator()
                && candidate_is_better(
                    target_num,
                    target_den,
                    &candidate_num,
                    &candidate_den,
                    &best_num,
                    &best_den,
                )
            {
                best_num = candidate_num;
                best_den = candidate_den;
            }
        }

        let current_num = &partial * &prev1_num + &prev2_num;
        let current_den = &partial * &prev1_den + &prev2_den;
        if current_num <= *bounds.max_numerator()
            && current_den <= *bounds.max_denominator()
            && candidate_is_better(
                target_num,
                target_den,
                &current_num,
                &current_den,
                &best_num,
                &best_den,
            )
        {
            best_num = current_num.clone();
            best_den = current_den.clone();
        }

        prev2_num = prev1_num;
        prev2_den = prev1_den;
        prev1_num = current_num;
        prev1_den = current_den;
    }

    (best_num, best_den)
}

/// Computes the largest admissible semiconvergent step for the current continued-fraction block.
///
/// # Parameters
/// - `partial`: Continued-fraction coefficient for the current block.
/// - `prev2_num`: Numerator of the convergent two steps back.
/// - `prev2_den`: Denominator of the convergent two steps back.
/// - `prev1_num`: Numerator of the previous convergent.
/// - `prev1_den`: Denominator of the previous convergent.
/// - `bounds`: Absolute numerator and denominator limits.
///
/// # Returns
/// - `Option<BigUint>`: Largest admissible step, or `None` when no positive step fits.
///
/// # Expected Output
/// - Returns the largest admissible semiconvergent multiplier; no side effects.
fn max_admissible_step(
    partial: &BigUint,
    prev2_num: &BigUint,
    prev2_den: &BigUint,
    prev1_num: &BigUint,
    prev1_den: &BigUint,
    bounds: &ApproximationBounds,
) -> Option<BigUint> {
    let mut limit = partial.clone();

    if *prev2_num > *bounds.max_numerator() || *prev2_den > *bounds.max_denominator() {
        return None;
    }

    if !prev1_num.is_zero() {
        limit = limit.min((bounds.max_numerator() - prev2_num) / prev1_num);
    }

    if !prev1_den.is_zero() {
        limit = limit.min((bounds.max_denominator() - prev2_den) / prev1_den);
    }

    if limit.is_zero() { None } else { Some(limit) }
}

/// Determines whether one admissible candidate is closer to the target than another.
///
/// # Parameters
/// - `target_num`: Absolute numerator of the target value.
/// - `target_den`: Positive denominator of the target value.
/// - `candidate_num`: Candidate numerator.
/// - `candidate_den`: Candidate denominator.
/// - `best_num`: Incumbent best numerator.
/// - `best_den`: Incumbent best denominator.
///
/// # Returns
/// - `bool`: `true` when the candidate is strictly better under the module's tie-break rules.
///
/// # Expected Output
/// - Returns a comparison result; no side effects.
fn candidate_is_better(
    target_num: &BigUint,
    target_den: &BigUint,
    candidate_num: &BigUint,
    candidate_den: &BigUint,
    best_num: &BigUint,
    best_den: &BigUint,
) -> bool {
    let candidate_error = absolute_difference(target_num, target_den, candidate_num, candidate_den);
    let best_error = absolute_difference(target_num, target_den, best_num, best_den);

    let left = &candidate_error * best_den;
    let right = &best_error * candidate_den;
    match left.cmp(&right) {
        Ordering::Less => true,
        Ordering::Greater => false,
        Ordering::Equal => match candidate_den.cmp(best_den) {
            Ordering::Less => true,
            Ordering::Greater => false,
            Ordering::Equal => candidate_num < best_num,
        },
    }
}

/// Computes the absolute numerator of the difference between two positive rationals.
///
/// # Parameters
/// - `target_num`: Target numerator.
/// - `target_den`: Target denominator.
/// - `candidate_num`: Candidate numerator.
/// - `candidate_den`: Candidate denominator.
///
/// # Returns
/// - `BigUint`: Absolute value of `target_num/target_den - candidate_num/candidate_den`, scaled by `target_den * candidate_den`.
///
/// # Expected Output
/// - Returns the unsigned difference numerator; no side effects.
fn absolute_difference(
    target_num: &BigUint,
    target_den: &BigUint,
    candidate_num: &BigUint,
    candidate_den: &BigUint,
) -> BigUint {
    let left = target_num * candidate_den;
    let right = candidate_num * target_den;
    if left >= right {
        left - right
    } else {
        right - left
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    fn br(num: i64, den: i64) -> BigRational {
        BigRational::new(BigInt::from(num), BigInt::from(den)).expect("valid test rational")
    }

    fn brute_force_best(target: &BigRational, bounds: &ApproximationBounds) -> BigRational {
        let abs_target =
            BigRational::new(target.numerator.abs(), target.denominator.clone()).unwrap();
        let max_num = bounds.max_numerator().to_u64_digits();
        let max_den = bounds.max_denominator().to_u64_digits();
        let numerator_limit = *max_num.first().unwrap_or(&0);
        let denominator_limit = *max_den.first().unwrap_or(&0);

        let mut best = BigRational::zero();
        for den in 1..=denominator_limit {
            for num in 0..=numerator_limit {
                let candidate = BigRational::new(BigInt::from(num), BigInt::from(den)).unwrap();
                if candidate_is_better_for_tests(&abs_target, &candidate, &best) {
                    best = candidate;
                }
            }
        }

        if target.numerator.is_negative() {
            -best
        } else {
            best
        }
    }

    fn candidate_is_better_for_tests(
        target: &BigRational,
        candidate: &BigRational,
        best: &BigRational,
    ) -> bool {
        let candidate_delta = (&target.numerator * &candidate.denominator
            - &candidate.numerator * &target.denominator)
            .abs();
        let best_delta =
            (&target.numerator * &best.denominator - &best.numerator * &target.denominator).abs();
        let left = candidate_delta * &best.denominator;
        let right = best_delta * &candidate.denominator;
        match left.cmp(&right) {
            Ordering::Less => true,
            Ordering::Greater => false,
            Ordering::Equal => match candidate.denominator.cmp(&best.denominator) {
                Ordering::Less => true,
                Ordering::Greater => false,
                Ordering::Equal => candidate.numerator < best.numerator,
            },
        }
    }

    #[test]
    fn new_reduces_and_normalizes_signs() {
        let value = BigRational::new(BigInt::from(-6), BigInt::from(-8)).unwrap();
        assert_eq!(value.numerator(), &BigInt::from(3));
        assert_eq!(value.denominator(), &BigInt::from(4));
    }

    #[test]
    fn from_decimal_parses_plain_decimal() {
        let value = BigRational::from_decimal("123.4500").unwrap();
        assert_eq!(value, br(2469, 20));
    }

    #[test]
    fn from_decimal_parses_scientific_notation() {
        let value = BigRational::from_decimal("-1.25e3").unwrap();
        assert_eq!(value, BigRational::from(BigInt::from(-1250)));
    }

    #[test]
    fn from_bigdecimal_parses_exact_value() {
        let value = BigDecimal::from_str("123.4500").unwrap();
        assert_eq!(BigRational::from_bigdecimal(&value).unwrap(), br(2469, 20));
        assert_eq!(BigRational::try_from(&value).unwrap(), br(2469, 20));
    }

    #[test]
    fn from_decimal_rejects_invalid_values() {
        assert!(matches!(
            BigRational::from_decimal(""),
            Err(RationalError::EmptyDecimal)
        ));
        assert!(matches!(
            BigRational::from_decimal("12.3.4"),
            Err(RationalError::InvalidDecimal(_))
        ));
        assert!(matches!(
            BigRational::from_decimal("abc"),
            Err(RationalError::InvalidDecimal(_))
        ));
    }

    #[test]
    fn arithmetic_with_rationals_is_reduced() {
        let a = br(1, 3);
        let b = br(5, 6);

        assert_eq!(&a + &b, br(7, 6));
        assert_eq!(&a - &b, br(-1, 2));
        assert_eq!(&a * &b, br(5, 18));
        assert_eq!(&a / &b, br(2, 5));
    }

    #[test]
    fn arithmetic_with_bigints_is_supported() {
        let value = br(7, 3);
        let three = BigInt::from(3);
        let two = BigInt::from(2);

        assert_eq!(&value + &three, br(16, 3));
        assert_eq!(&value - &three, br(-2, 3));
        assert_eq!(&value * &two, br(14, 3));
        assert_eq!(&value / &two, br(7, 6));
    }

    #[test]
    fn checked_division_reports_zero_divisors() {
        let value = br(7, 5);
        let zero = BigInt::zero();

        assert_eq!(
            value.checked_div(&BigRational::zero()),
            Err(RationalError::DivisionByZero)
        );
        assert_eq!(
            value.checked_div_bigint(&zero),
            Err(RationalError::DivisionByZero)
        );
        assert_eq!(
            BigRational::zero().checked_recip(),
            Err(RationalError::DivisionByZero)
        );
    }

    #[test]
    fn to_bigint_uses_floor_for_positive_and_negative_values() {
        assert_eq!(br(7, 3).to_bigint(), BigInt::from(2));
        assert_eq!(br(-7, 3).to_bigint(), BigInt::from(-3));
        assert_eq!(br(9, 3).to_bigint(), BigInt::from(3));
    }

    #[test]
    fn approximate_returns_exact_value_when_already_within_bounds() {
        let exact = BigRational::from_decimal("3.125").unwrap();
        let bounds =
            ApproximationBounds::new(BigUint::from(100u32), BigUint::from(100u32)).unwrap();
        assert_eq!(exact.approximate(&bounds).unwrap(), br(25, 8));
    }

    #[test]
    fn approximate_decimal_with_value_bounds_finds_nearest_fraction() {
        let value = BigRational::approximate_decimal_with_bounds(
            "0.6176470588235294117647",
            BigUint::from(5u32),
            BigUint::from(8u32),
        )
        .unwrap();
        assert_eq!(value, br(5, 8));
    }

    #[test]
    fn approximate_decimal_with_bit_bounds_supports_width_based_limits() {
        let value = BigRational::approximate_decimal_with_bits("3.1415926535", 5, 5).unwrap();
        assert_eq!(value, br(22, 7));
    }

    #[test]
    fn approximate_bigdecimal_with_value_bounds_finds_nearest_fraction() {
        let value = BigDecimal::from_str("0.6176470588235294117647").unwrap();
        let approximation = BigRational::approximate_bigdecimal_with_bounds(
            &value,
            BigUint::from(5u32),
            BigUint::from(8u32),
        )
        .unwrap();
        assert_eq!(approximation, br(5, 8));
    }

    #[test]
    fn approximate_bigdecimal_with_bit_bounds_supports_width_based_limits() {
        let value = BigDecimal::from_str("3.1415926535").unwrap();
        let approximation = BigRational::approximate_bigdecimal_with_bits(&value, 5, 5).unwrap();
        assert_eq!(approximation, br(22, 7));
    }

    #[test]
    fn approximate_decimal_handles_negative_inputs() {
        let value = BigRational::approximate_decimal_with_bounds(
            "-2.718281828",
            BigUint::from(20u32),
            BigUint::from(10u32),
        )
        .unwrap();
        assert_eq!(value, br(-19, 7));
    }

    #[test]
    fn approximate_decimal_with_zero_numerator_budget_returns_zero() {
        let value = BigRational::approximate_decimal_with_bounds(
            "0.875",
            BigUint::zero(),
            BigUint::from(128u32),
        )
        .unwrap();
        assert_eq!(value, BigRational::zero());
    }

    #[test]
    fn approximation_matches_bruteforce_for_small_bounds() {
        let samples = ["0.1", "0.3333", "1.4142", "2.71828", "-0.875", "5.125"];

        for sample in samples {
            let exact = BigRational::from_decimal(sample).unwrap();
            for max_num in 0u32..=8 {
                for max_den in 1u32..=8 {
                    let bounds =
                        ApproximationBounds::new(BigUint::from(max_num), BigUint::from(max_den))
                            .unwrap();
                    let approximated = exact.approximate(&bounds).unwrap();
                    let brute_force = brute_force_best(&exact, &bounds);
                    assert_eq!(
                        approximated, brute_force,
                        "sample={sample}, max_num={max_num}, max_den={max_den}"
                    );
                }
            }
        }
    }

    #[test]
    fn approximation_matches_bruteforce_for_small_bit_widths() {
        let exact = BigRational::from_decimal("0.7272727272").unwrap();
        let bounds = ApproximationBounds::from_bit_widths(3, 3).unwrap();
        let approximated = exact.approximate(&bounds).unwrap();
        let brute_force = brute_force_best(&exact, &bounds);
        assert_eq!(approximated, brute_force);
    }

    #[test]
    fn display_omits_denominator_for_integers() {
        assert_eq!(BigRational::from(BigInt::from(42)).to_string(), "42");
        assert_eq!(br(7, 3).to_string(), "7/3");
    }
}
