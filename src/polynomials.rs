/// Eclipse Public License 2.0
/// SPDX-License-Identifier: EPL-2.0
/// Copyright (c) 2025 Nicholas LaRoche <nlaroche@cryptifier.dev>
use num_bigint::BigInt;
use num_traits::{One, Zero};

/// Dense univariate integer polynomial stored in ascending coefficient order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IntegerPolynomial {
    coefficients: Vec<BigInt>,
}

impl IntegerPolynomial {
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
    pub fn from_constant(constant: BigInt) -> Self {
        Self::new(vec![constant])
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

    /// Returns the polynomial degree when the polynomial is non-zero.
    ///
    /// # Parameters
    /// - None.
    ///
    /// # Returns
    /// - `Option<usize>`: Highest non-zero degree, or `None` for the zero polynomial.
    ///
    /// # Expected Output
    /// - Returns the degree classification; no side effects.
    pub fn degree(&self) -> Option<usize> {
        if self.is_zero() {
            None
        } else {
            Some(self.coefficients.len() - 1)
        }
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
        self.coefficients
            .iter()
            .rev()
            .fold(BigInt::zero(), |acc, coefficient| acc * x + coefficient)
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
            return Self::from_constant(BigInt::zero());
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
            return Self::from_constant(BigInt::one());
        }

        let mut result = Self::from_constant(BigInt::one());
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
            return Self::from_constant(BigInt::zero());
        }

        Self::new(self.coefficients.iter().map(|value| value * scalar).collect())
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
        if self.is_zero() {
            return Self::from_constant(BigInt::zero());
        }
        if shift == 0 {
            return self.clone();
        }

        let mut coefficients = vec![BigInt::zero(); shift];
        coefficients.extend(self.coefficients.iter().cloned());
        Self::new(coefficients)
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
        if self.is_zero() {
            return Self::from_constant(BigInt::zero());
        }

        let mut scaled_coefficients = Vec::with_capacity(self.coefficients.len());
        let mut factor_power = BigInt::one();

        for coefficient in &self.coefficients {
            scaled_coefficients.push(coefficient * &factor_power);
            factor_power *= factor;
        }

        Self::new(scaled_coefficients)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn polynomial_normalizes_trailing_zeroes() {
        let polynomial = IntegerPolynomial::new(vec![
            BigInt::from(4),
            BigInt::from(0),
            BigInt::from(0),
        ]);

        assert_eq!(polynomial.coefficients(), &[BigInt::from(4)]);
        assert_eq!(polynomial.degree(), Some(0));
    }

    #[test]
    fn polynomial_tracks_zero_degree_as_none() {
        let polynomial = IntegerPolynomial::new(vec![]);

        assert!(polynomial.is_zero());
        assert_eq!(polynomial.degree(), None);
        assert_eq!(polynomial.leading_coefficient(), None);
    }

    #[test]
    fn polynomial_scale_input_updates_each_degree() {
        let polynomial = IntegerPolynomial::new(vec![
            BigInt::from(3),
            BigInt::from(-2),
            BigInt::from(1),
        ]);

        let scaled = polynomial.scale_input(&BigInt::from(5));

        assert_eq!(
            scaled.coefficients(),
            &[BigInt::from(3), BigInt::from(-10), BigInt::from(25)]
        );
    }

    #[test]
    fn polynomial_pow_handles_zero_exponent() {
        let polynomial = IntegerPolynomial::new(vec![BigInt::from(7), BigInt::from(1)]);

        let result = polynomial.pow(0);

        assert_eq!(result, IntegerPolynomial::from_constant(BigInt::from(1)));
    }

    #[test]
    fn polynomial_mul_and_evaluate_work_with_negative_coefficients() {
        let left = IntegerPolynomial::new(vec![BigInt::from(1), BigInt::from(1)]);
        let right = IntegerPolynomial::new(vec![BigInt::from(-2), BigInt::from(1)]);
        let product = left.mul(&right);

        assert_eq!(
            product.coefficients(),
            &[BigInt::from(-2), BigInt::from(-1), BigInt::from(1)]
        );
        assert_eq!(product.evaluate(&BigInt::from(3)), BigInt::from(4));
    }
}
