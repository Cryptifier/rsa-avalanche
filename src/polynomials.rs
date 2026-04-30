/// Eclipse Public License 2.0
/// SPDX-License-Identifier: EPL-2.0
/// Copyright (c) 2025 Nicholas LaRoche <nlaroche@cryptifier.dev>
use num_bigint::{BigInt, BigUint};
use num_traits::{One, Signed, Zero};

/// Dense univariate integer polynomial stored in ascending coefficient order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Poly {
    coefficients: Vec<BigInt>,
}

/// Backward-compatible alias for older lattice code paths.
pub type IntegerPolynomial = Poly;

impl Poly {
    /// Builds a polynomial and removes trailing zero coefficients.
    ///
    /// # Parameters
    /// - `coefficients`: Coefficients in ascending order by degree.
    ///
    /// # Returns
    /// - `Self`: Normalized polynomial representation.
    ///
    /// # Expected Output
    /// - Returns a polynomial value object; no side effects.
    pub fn new(coefficients: Vec<BigInt>) -> Self {
        Self {
            coefficients: normalize_coefficients(coefficients),
        }
    }

    /// Builds a constant polynomial.
    ///
    /// # Parameters
    /// - `constant`: Constant coefficient.
    ///
    /// # Returns
    /// - `Self`: Polynomial equal to `constant`.
    ///
    /// # Expected Output
    /// - Returns a polynomial value object; no side effects.
    pub fn constant(constant: BigInt) -> Self {
        Self::new(vec![constant])
    }

    /// Builds a constant polynomial.
    ///
    /// # Parameters
    /// - `constant`: Constant coefficient.
    ///
    /// # Returns
    /// - `Self`: Polynomial equal to `constant`.
    ///
    /// # Expected Output
    /// - Returns a polynomial value object; no side effects.
    pub fn from_constant(constant: BigInt) -> Self {
        Self::constant(constant)
    }

    /// Builds the polynomial `x`.
    ///
    /// # Parameters
    /// - None.
    ///
    /// # Returns
    /// - `Self`: The monomial `x`.
    ///
    /// # Expected Output
    /// - Returns a polynomial value object; no side effects.
    pub fn x() -> Self {
        Self::new(vec![BigInt::zero(), BigInt::one()])
    }

    /// Builds the monomial `coefficient * x^degree`.
    ///
    /// # Parameters
    /// - `coefficient`: Leading coefficient for the monomial.
    /// - `degree`: Degree of the monomial.
    ///
    /// # Returns
    /// - `Self`: Monomial polynomial.
    ///
    /// # Expected Output
    /// - Returns a polynomial value object; no side effects.
    pub fn monomial(coefficient: BigInt, degree: usize) -> Self {
        let mut coefficients = vec![BigInt::zero(); degree + 1];
        coefficients[degree] = coefficient;
        Self::new(coefficients)
    }

    /// Returns the normalized coefficient slice in ascending degree order.
    ///
    /// # Parameters
    /// - None.
    ///
    /// # Returns
    /// - `&[BigInt]`: Borrowed coefficient slice.
    ///
    /// # Expected Output
    /// - Returns the internal coefficient view; no side effects.
    pub fn coefficients(&self) -> &[BigInt] {
        &self.coefficients
    }

    /// Returns the coefficient for `x^i`, or zero when `i` is out of range.
    ///
    /// # Parameters
    /// - `i`: Degree whose coefficient is requested.
    ///
    /// # Returns
    /// - `BigInt`: Coefficient value for `x^i`.
    ///
    /// # Expected Output
    /// - Returns a coefficient value; no side effects.
    pub fn coeff(&self, i: usize) -> BigInt {
        self.coefficients
            .get(i)
            .cloned()
            .unwrap_or_else(BigInt::zero)
    }

    /// Returns whether the polynomial is identically zero.
    ///
    /// # Parameters
    /// - None.
    ///
    /// # Returns
    /// - `bool`: `true` when every coefficient is zero.
    ///
    /// # Expected Output
    /// - Returns a boolean classification; no side effects.
    pub fn is_zero(&self) -> bool {
        self.coefficients.len() == 1 && self.coefficients[0].is_zero()
    }

    /// Returns the polynomial degree, or `0` for the zero polynomial.
    ///
    /// # Parameters
    /// - None.
    ///
    /// # Returns
    /// - `usize`: Highest represented degree after normalization.
    ///
    /// # Expected Output
    /// - Returns the degree classification; no side effects.
    pub fn degree(&self) -> usize {
        self.coefficients.len().saturating_sub(1)
    }

    /// Returns the leading coefficient when the polynomial is non-zero.
    ///
    /// # Parameters
    /// - None.
    ///
    /// # Returns
    /// - `Option<&BigInt>`: Leading coefficient, or `None` for the zero polynomial.
    ///
    /// # Expected Output
    /// - Returns a borrowed coefficient reference; no side effects.
    pub fn leading_coefficient(&self) -> Option<&BigInt> {
        if self.is_zero() {
            None
        } else {
            self.coefficients.last()
        }
    }

    /// Returns the sum of two polynomials.
    ///
    /// # Parameters
    /// - `other`: Polynomial added to `self`.
    ///
    /// # Returns
    /// - `Self`: Sum polynomial.
    ///
    /// # Expected Output
    /// - Returns a new polynomial value; no side effects.
    pub fn add(&self, other: &Self) -> Self {
        let degree = self.coefficients.len().max(other.coefficients.len());
        let mut coefficients = vec![BigInt::zero(); degree];

        for (index, coefficient) in self.coefficients.iter().enumerate() {
            coefficients[index] += coefficient;
        }
        for (index, coefficient) in other.coefficients.iter().enumerate() {
            coefficients[index] += coefficient;
        }

        Self::new(coefficients)
    }

    /// Returns the product of two polynomials.
    ///
    /// # Parameters
    /// - `other`: Polynomial multiplied with `self`.
    ///
    /// # Returns
    /// - `Self`: Product polynomial.
    ///
    /// # Expected Output
    /// - Returns a new polynomial value; no side effects.
    pub fn mul(&self, other: &Self) -> Self {
        if self.is_zero() || other.is_zero() {
            return Self::constant(BigInt::zero());
        }

        let mut coefficients =
            vec![BigInt::zero(); self.coefficients.len() + other.coefficients.len() - 1];

        for (left_degree, left_coefficient) in self.coefficients.iter().enumerate() {
            for (right_degree, right_coefficient) in other.coefficients.iter().enumerate() {
                coefficients[left_degree + right_degree] += left_coefficient * right_coefficient;
            }
        }

        Self::new(coefficients)
    }

    /// Raises the polynomial to a non-negative integer power.
    ///
    /// # Parameters
    /// - `exponent`: Power applied to the polynomial.
    ///
    /// # Returns
    /// - `Self`: `self^exponent`.
    ///
    /// # Expected Output
    /// - Returns a new polynomial value; no side effects.
    pub fn pow(&self, exponent: usize) -> Self {
        if exponent == 0 {
            return Self::constant(BigInt::one());
        }

        let mut result = Self::constant(BigInt::one());
        let mut base = self.clone();
        let mut remaining = exponent;

        while remaining > 0 {
            if remaining % 2 == 1 {
                result = result.mul(&base);
            }
            remaining /= 2;
            if remaining > 0 {
                base = base.mul(&base);
            }
        }

        result
    }

    /// Multiplies the polynomial by `x^shift`.
    ///
    /// # Parameters
    /// - `shift`: Number of degrees to shift upward.
    ///
    /// # Returns
    /// - `Self`: Shifted polynomial.
    ///
    /// # Expected Output
    /// - Returns a new polynomial value; no side effects.
    pub fn mul_x_pow(&self, shift: usize) -> Self {
        if self.is_zero() {
            return Self::constant(BigInt::zero());
        }
        if shift == 0 {
            return self.clone();
        }

        let mut coefficients = vec![BigInt::zero(); shift];
        coefficients.extend(self.coefficients.iter().cloned());
        Self::new(coefficients)
    }

    /// Multiplies the polynomial by `x^shift`.
    ///
    /// # Parameters
    /// - `shift`: Number of degrees to shift upward.
    ///
    /// # Returns
    /// - `Self`: Shifted polynomial.
    ///
    /// # Expected Output
    /// - Returns a new polynomial value; no side effects.
    pub fn shift(&self, shift: usize) -> Self {
        self.mul_x_pow(shift)
    }

    /// Multiplies every coefficient by the same integer scalar.
    ///
    /// # Parameters
    /// - `scalar`: Coefficient multiplier.
    ///
    /// # Returns
    /// - `Self`: Scaled polynomial.
    ///
    /// # Expected Output
    /// - Returns a new polynomial value; no side effects.
    pub fn scale(&self, scalar: &BigInt) -> Self {
        if scalar.is_zero() || self.is_zero() {
            return Self::constant(BigInt::zero());
        }

        Self::new(self.coefficients.iter().map(|value| value * scalar).collect())
    }

    /// Substitutes `x <- factor * x`.
    ///
    /// # Parameters
    /// - `factor`: Integer scaling applied to the input variable.
    ///
    /// # Returns
    /// - `Self`: Polynomial after the input scaling.
    ///
    /// # Expected Output
    /// - Returns a new polynomial value; no side effects.
    pub fn scale_variable(&self, factor: &BigInt) -> Self {
        if self.is_zero() {
            return Self::constant(BigInt::zero());
        }

        let mut scaled_coefficients = Vec::with_capacity(self.coefficients.len());
        let mut factor_power = BigInt::one();

        for coefficient in &self.coefficients {
            scaled_coefficients.push(coefficient * &factor_power);
            factor_power *= factor;
        }

        Self::new(scaled_coefficients)
    }

    /// Substitutes `x <- factor * x`.
    ///
    /// # Parameters
    /// - `factor`: Integer scaling applied to the input variable.
    ///
    /// # Returns
    /// - `Self`: Polynomial after the input scaling.
    ///
    /// # Expected Output
    /// - Returns a new polynomial value; no side effects.
    pub fn scale_input(&self, factor: &BigInt) -> Self {
        self.scale_variable(factor)
    }

    /// Reverses `scale_variable` when each coefficient is exactly divisible by `factor^i`.
    ///
    /// # Parameters
    /// - `factor`: Integer scaling originally applied to the input variable.
    ///
    /// # Returns
    /// - `Self`: Unscaled polynomial recovered from `f(factor * x)`.
    ///
    /// # Expected Output
    /// - Returns a new polynomial value; no side effects.
    pub fn unscale_variable_exact(&self, factor: &BigInt) -> Self {
        assert!(
            !factor.is_zero(),
            "cannot exactly unscale a polynomial with a zero variable factor"
        );

        let mut coefficients = Vec::with_capacity(self.coefficients.len());
        let mut factor_power = BigInt::one();

        for coefficient in &self.coefficients {
            assert!(
                coefficient.is_zero() || (coefficient % &factor_power) == BigInt::zero(),
                "scaled polynomial coefficient is not divisible by the expected factor power"
            );
            coefficients.push(coefficient / &factor_power);
            factor_power *= factor;
        }

        Self::new(coefficients)
    }

    /// Evaluates the polynomial at an integer point using Horner's method.
    ///
    /// # Parameters
    /// - `x`: Evaluation point.
    ///
    /// # Returns
    /// - `BigInt`: Polynomial value at `x`.
    ///
    /// # Expected Output
    /// - Returns the computed integer value; no side effects.
    pub fn eval(&self, x: &BigInt) -> BigInt {
        self.coefficients
            .iter()
            .rev()
            .fold(BigInt::zero(), |acc, coefficient| acc * x + coefficient)
    }

    /// Evaluates the polynomial at an integer point using Horner's method.
    ///
    /// # Parameters
    /// - `x`: Evaluation point.
    ///
    /// # Returns
    /// - `BigInt`: Polynomial value at `x`.
    ///
    /// # Expected Output
    /// - Returns the computed integer value; no side effects.
    pub fn evaluate(&self, x: &BigInt) -> BigInt {
        self.eval(x)
    }

    /// Returns exact integer roots for low-degree polynomials when a closed-form test is available.
    ///
    /// # Parameters
    /// - None.
    ///
    /// # Returns
    /// - `Option<Vec<BigInt>>`: `Some(roots)` for degrees `0`, `1`, and `2`, or `None` for higher degrees.
    ///
    /// # Expected Output
    /// - Returns sorted unique integer roots for supported degrees; no side effects.
    pub fn exact_integer_roots_low_degree(&self) -> Option<Vec<BigInt>> {
        match self.degree() {
            0 => Some(Vec::new()),
            1 => Some(self.exact_integer_roots_linear()),
            2 => Some(self.exact_integer_roots_quadratic()),
            _ => None,
        }
    }

    /// Returns the coefficients padded with trailing zeroes to `width`.
    ///
    /// # Parameters
    /// - `width`: Target coefficient vector width.
    ///
    /// # Returns
    /// - `Vec<BigInt>`: Padded coefficient vector.
    ///
    /// # Expected Output
    /// - Returns a new coefficient vector; no side effects.
    pub fn padded_coefficients(&self, width: usize) -> Vec<BigInt> {
        assert!(
            width >= self.coefficients.len(),
            "width must cover the polynomial degree"
        );

        let mut coefficients = self.coefficients.clone();
        coefficients.resize(width, BigInt::zero());
        coefficients
    }

    /// Solves a linear polynomial exactly over the integers.
    fn exact_integer_roots_linear(&self) -> Vec<BigInt> {
        let a = self.coeff(1);
        let b = self.coeff(0);

        if a.is_zero() {
            return Vec::new();
        }

        let numerator = -b;
        if (&numerator % &a) != BigInt::zero() {
            return Vec::new();
        }

        vec![numerator / a]
    }

    /// Solves a quadratic polynomial exactly over the integers.
    fn exact_integer_roots_quadratic(&self) -> Vec<BigInt> {
        let a = self.coeff(2);
        let b = self.coeff(1);
        let c = self.coeff(0);

        if a.is_zero() {
            return Self::new(vec![c, b]).exact_integer_roots_linear();
        }

        let discriminant = &b * &b - BigInt::from(4u8) * &a * &c;
        let Some(sqrt_discriminant) = exact_bigint_square_root(&discriminant) else {
            return Vec::new();
        };

        let denominator = BigInt::from(2u8) * a;
        let mut roots = Vec::new();

        for numerator in [-&b + &sqrt_discriminant, -&b - &sqrt_discriminant] {
            if (&numerator % &denominator) == BigInt::zero() {
                let root = numerator / &denominator;
                if !roots.contains(&root) {
                    roots.push(root);
                }
            }
        }

        roots.sort();
        roots
    }
}

/// Removes trailing zero coefficients while preserving a single zero for the zero polynomial.
fn normalize_coefficients(mut coefficients: Vec<BigInt>) -> Vec<BigInt> {
    if coefficients.is_empty() {
        return vec![BigInt::zero()];
    }

    while coefficients.len() > 1 && coefficients.last().is_some_and(BigInt::is_zero) {
        coefficients.pop();
    }

    if coefficients.is_empty() {
        vec![BigInt::zero()]
    } else {
        coefficients
    }
}

/// Returns the exact square root of a non-negative `BigInt` when it is a perfect square.
fn exact_bigint_square_root(value: &BigInt) -> Option<BigInt> {
    if value.is_negative() {
        return None;
    }

    let value_uint: BigUint = value.to_biguint()?;
    let sqrt = value_uint.sqrt();

    if &sqrt * &sqrt == value_uint {
        Some(BigInt::from(sqrt))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn polynomial_normalizes_trailing_zeroes() {
        let polynomial = Poly::new(vec![BigInt::from(4), BigInt::from(0), BigInt::from(0)]);

        assert_eq!(polynomial.coefficients(), &[BigInt::from(4)]);
        assert_eq!(polynomial.degree(), 0);
    }

    #[test]
    fn polynomial_tracks_zero_degree_as_zero() {
        let polynomial = Poly::new(vec![]);

        assert!(polynomial.is_zero());
        assert_eq!(polynomial.degree(), 0);
        assert_eq!(polynomial.leading_coefficient(), None);
    }

    #[test]
    fn polynomial_scale_input_updates_each_degree() {
        let polynomial = Poly::new(vec![BigInt::from(3), BigInt::from(-2), BigInt::from(1)]);

        let scaled = polynomial.scale_variable(&BigInt::from(5));

        assert_eq!(
            scaled.coefficients(),
            &[BigInt::from(3), BigInt::from(-10), BigInt::from(25)]
        );
    }

    #[test]
    fn polynomial_unscale_recovers_original_coefficients() {
        let polynomial = Poly::new(vec![BigInt::from(3), BigInt::from(-2), BigInt::from(1)]);
        let scaled = polynomial.scale_variable(&BigInt::from(5));

        let recovered = scaled.unscale_variable_exact(&BigInt::from(5));

        assert_eq!(recovered, polynomial);
    }

    #[test]
    fn polynomial_pow_handles_zero_exponent() {
        let polynomial = Poly::new(vec![BigInt::from(7), BigInt::from(1)]);

        let result = polynomial.pow(0);

        assert_eq!(result, Poly::constant(BigInt::from(1)));
    }

    #[test]
    fn polynomial_mul_and_eval_work_with_negative_coefficients() {
        let left = Poly::new(vec![BigInt::from(1), BigInt::from(1)]);
        let right = Poly::new(vec![BigInt::from(-2), BigInt::from(1)]);
        let product = left.mul(&right);

        assert_eq!(
            product.coefficients(),
            &[BigInt::from(-2), BigInt::from(-1), BigInt::from(1)]
        );
        assert_eq!(product.eval(&BigInt::from(3)), BigInt::from(4));
    }

    #[test]
    fn polynomial_mul_x_pow_matches_shift() {
        let polynomial = Poly::new(vec![BigInt::from(2), BigInt::from(3)]);

        assert_eq!(
            polynomial.mul_x_pow(2),
            Poly::new(vec![
                BigInt::from(0),
                BigInt::from(0),
                BigInt::from(2),
                BigInt::from(3),
            ])
        );
    }

    #[test]
    fn polynomial_coeff_returns_zero_out_of_range() {
        let polynomial = Poly::new(vec![BigInt::from(2), BigInt::from(3)]);

        assert_eq!(polynomial.coeff(4), BigInt::zero());
    }

    #[test]
    fn polynomial_exact_integer_roots_linear_returns_integral_root() {
        let polynomial = Poly::new(vec![BigInt::from(-6), BigInt::from(2)]);

        let roots = polynomial.exact_integer_roots_low_degree();

        assert_eq!(roots, Some(vec![BigInt::from(3)]));
    }

    #[test]
    fn polynomial_exact_integer_roots_quadratic_returns_two_roots() {
        let polynomial = Poly::new(vec![BigInt::from(6), BigInt::from(-5), BigInt::from(1)]);

        let roots = polynomial.exact_integer_roots_low_degree();

        assert_eq!(roots, Some(vec![BigInt::from(2), BigInt::from(3)]));
    }

    #[test]
    fn polynomial_exact_integer_roots_quadratic_handles_repeated_root() {
        let polynomial = Poly::new(vec![BigInt::from(4), BigInt::from(-4), BigInt::from(1)]);

        let roots = polynomial.exact_integer_roots_low_degree();

        assert_eq!(roots, Some(vec![BigInt::from(2)]));
    }

    #[test]
    fn polynomial_exact_integer_roots_quadratic_rejects_non_square_discriminant() {
        let polynomial = Poly::new(vec![BigInt::from(1), BigInt::from(0), BigInt::from(1)]);

        let roots = polynomial.exact_integer_roots_low_degree();

        assert_eq!(roots, Some(Vec::new()));
    }
}
