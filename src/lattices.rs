/// Eclipse Public License 2.0
/// SPDX-License-Identifier: EPL-2.0
/// Copyright (c) 2025 Nicholas LaRoche <nlaroche@cryptifier.dev>
use bigdecimal::{BigDecimal, FromPrimitive};
use ndarray::{Array1, Array2};
use num_bigint::{BigInt, BigUint};
use num_integer::Integer;
use num_rational::BigRational;
use num_traits::{One, Signed, ToPrimitive, Zero};
use std::error::Error;
use std::fmt::{Display, Formatter};

use crate::poly::Poly;
use crate::polynomials::IntegerPolynomial;
use crate::math::cosine_bigdecimal;

/// Integer matrix used for lattice basis construction.
pub type BigIntMatrix = Array2<BigInt>;

/// Integer vector used for lattice rows and reduced basis vectors.
pub type BigIntVector = Array1<BigInt>;

/// Decimal matrix used for rotated lattice bases and decimal lattice metrics.
pub type BigDecimalMatrix = Array2<BigDecimal>;

/// Decimal vector used for rotated lattice rows.
pub type BigDecimalVector = Array1<BigDecimal>;

/// Error returned when lattice metric inputs are incompatible.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LatticeMetricError {
    ShapeMismatch {
        left_rows: usize,
        left_cols: usize,
        right_rows: usize,
        right_cols: usize,
    },
    InconsistentRowLength {
        row: usize,
        expected: usize,
        actual: usize,
    },
}

impl Display for LatticeMetricError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ShapeMismatch {
                left_rows,
                left_cols,
                right_rows,
                right_cols,
            } => write!(
                f,
                "frobenius inner product requires matching shapes, got ({left_rows}, {left_cols}) and ({right_rows}, {right_cols})"
            ),
            Self::InconsistentRowLength { row, expected, actual } => write!(
                f,
                "lattice row {row} has length {actual}, expected {expected}"
            ),
        }
    }
}

impl Error for LatticeMetricError {}

/// Error returned when a lattice rotation or decimal flattening request is malformed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LatticeTransformError {
    InvalidPlane {
        axis_a: usize,
        axis_b: usize,
        dimension: usize,
    },
    NegativeDigits {
        digits: i64,
    },
    InconsistentRowLength {
        row: usize,
        expected: usize,
        actual: usize,
    },
}

impl Display for LatticeTransformError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidPlane {
                axis_a,
                axis_b,
                dimension,
            } => write!(
                f,
                "rotation plane ({axis_a}, {axis_b}) is invalid for lattice dimension {dimension}"
            ),
            Self::NegativeDigits { digits } => {
                write!(f, "decimal digit count must be non-negative, got {digits}")
            }
            Self::InconsistentRowLength { row, expected, actual } => write!(
                f,
                "lattice row {row} has length {actual}, expected {expected}"
            ),
        }
    }
}

impl Error for LatticeTransformError {}

/// Error returned when a Coppersmith lattice request is malformed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CoppersmithError {
    MissingBound,
    MissingExponent,
    MissingDimension,
    InvalidModulus,
    ZeroBound,
    ZeroDegreePolynomial,
    NonMonicPolynomial,
    DimensionOverflow,
    DimensionTooSmall { minimum: usize, requested: usize },
    InvalidScaledVector { column: usize },
    UnknownPartExceedsPrime,
    PrimeDoesNotDivideModulus,
}

impl Display for CoppersmithError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingBound => write!(f, "coppersmith builder requires a positive search bound"),
            Self::MissingExponent => {
                write!(f, "coppersmith builder requires an m exponent parameter")
            }
            Self::MissingDimension => {
                write!(f, "coppersmith builder requires a target lattice dimension")
            }
            Self::InvalidModulus => write!(f, "rsa modulus must be greater than one"),
            Self::ZeroBound => write!(f, "coppersmith search bound must be greater than zero"),
            Self::ZeroDegreePolynomial => {
                write!(f, "coppersmith requires a non-constant monic polynomial")
            }
            Self::NonMonicPolynomial => {
                write!(f, "coppersmith requires the input polynomial to be monic")
            }
            Self::DimensionOverflow => write!(f, "requested lattice dimension overflowed usize"),
            Self::DimensionTooSmall { minimum, requested } => write!(
                f,
                "target lattice dimension {requested} is too small; minimum required is {minimum}"
            ),
            Self::InvalidScaledVector { column } => write!(
                f,
                "reduced basis column {column} could not be converted back to an unscaled polynomial"
            ),
            Self::UnknownPartExceedsPrime => {
                write!(f, "rsa unknown part x must be less than or equal to prime p")
            }
            Self::PrimeDoesNotDivideModulus => {
                write!(f, "provided rsa prime p does not divide the modulus")
            }
        }
    }
}

impl Error for CoppersmithError {}

/// Single lattice element `g_{i,j}(x)` generated for the Coppersmith basis.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoppersmithLatticeElement {
    /// Power index applied to `f(x)`.
    pub i: usize,
    /// Monomial shift index applied to `x^j`.
    pub j: usize,
    /// Unscaled polynomial `g_{i,j}(x)`.
    pub polynomial: IntegerPolynomial,
    /// Scaled polynomial `g_{i,j}(Xx)` used for lattice coefficients.
    pub scaled_polynomial: IntegerPolynomial,
    /// Coefficient row inserted into the lattice matrix.
    pub coefficient_vector: Vec<BigInt>,
}

/// Fully constructed Coppersmith lattice and its basis metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoppersmithLattice {
    /// Monic polynomial `f(x)` whose small root is sought modulo an RSA factor.
    pub polynomial: IntegerPolynomial,
    /// RSA modulus `N`.
    pub modulus: BigUint,
    /// Root search bound `X`.
    pub bound: BigUint,
    /// Exponent parameter `m`.
    pub m: usize,
    /// Total square lattice dimension `d`.
    pub dimension: usize,
    /// Number of extra `i = m` completion rows.
    pub completion_rows: usize,
    /// Ordered lattice elements used to assemble `basis`.
    pub elements: Vec<CoppersmithLatticeElement>,
    /// Square lattice basis matrix built from `g_{i,j}(Xx)` coefficients.
    pub basis: BigIntMatrix,
}

impl CoppersmithLattice {
    /// Runs LLL on the current lattice basis.
    ///
    /// # Parameters
    /// - None.
    ///
    /// # Returns
    /// - `Vec<BigIntVector>`: Reduced basis vectors in row order.
    ///
    /// # Expected Output
    /// - Returns a reduced basis; no side effects.
    pub fn reduce(&self) -> Vec<BigIntVector> {
        lll_reduce(&self.basis)
    }

    /// Converts reduced lattice vectors back into unscaled integer polynomials.
    ///
    /// # Parameters
    /// - `vectors`: Reduced basis vectors produced from this lattice.
    ///
    /// # Returns
    /// - `Result<Vec<IntegerPolynomial>, CoppersmithError>`: Unscaled polynomials or a scaling error.
    ///
    /// # Expected Output
    /// - Returns reconstructed polynomials; no side effects.
    pub fn reduced_polynomials(
        &self,
        vectors: &[BigIntVector],
    ) -> Result<Vec<IntegerPolynomial>, CoppersmithError> {
        vectors
            .iter()
            .map(|vector| self.vector_to_polynomial(vector))
            .collect()
    }

    /// Converts a single scaled lattice vector into an unscaled polynomial.
    ///
    /// # Parameters
    /// - `vector`: Reduced basis vector expressed in the lattice coordinates.
    ///
    /// # Returns
    /// - `Result<IntegerPolynomial, CoppersmithError>`: Reconstructed polynomial or a scaling error.
    ///
    /// # Expected Output
    /// - Returns the recovered polynomial; no side effects.
    pub fn vector_to_polynomial(
        &self,
        vector: &BigIntVector,
    ) -> Result<IntegerPolynomial, CoppersmithError> {
        let bound = BigInt::from(self.bound.clone());
        let mut factor_power = BigInt::one();
        let mut coefficients = Vec::with_capacity(vector.len());

        for (index, value) in vector.iter().enumerate() {
            if !value.is_zero() && (value % &factor_power) != BigInt::zero() {
                return Err(CoppersmithError::InvalidScaledVector { column: index });
            }

            coefficients.push(value / &factor_power);
            factor_power *= &bound;
        }

        Ok(IntegerPolynomial::new(coefficients))
    }

    /// Computes the Frobenius inner product of this lattice basis with another basis.
    ///
    /// # Parameters
    /// - `other`: Lattice whose basis matrix is compared against `self`.
    ///
    /// # Returns
    /// - `Result<BigDecimal, LatticeMetricError>`: Exact Frobenius inner product or a shape error.
    ///
    /// # Expected Output
    /// - Returns an exact `BigDecimal` metric value; no side effects.
    pub fn frobenius_inner_product_basis(
        &self,
        other: &Self,
    ) -> Result<BigDecimal, LatticeMetricError> {
        frobenius_inner_product_bigdecimal(&self.basis, &other.basis)
    }

    /// Rotates this lattice basis within a selected 2D coordinate plane using decimal arithmetic.
    ///
    /// # Parameters
    /// - `axis_a`: First coordinate axis participating in the rotation plane.
    /// - `axis_b`: Second coordinate axis participating in the rotation plane.
    /// - `theta`: Rotation angle in radians.
    /// - `digits`: Decimal digits retained in the trigonometric coefficients and output entries.
    ///
    /// # Returns
    /// - `Result<BigDecimalMatrix, LatticeTransformError>`: Rotated decimal basis or a validation error.
    ///
    /// # Expected Output
    /// - Returns a rotated decimal matrix; no side effects.
    pub fn rotate_basis_plane_bigdecimal(
        &self,
        axis_a: usize,
        axis_b: usize,
        theta: &BigDecimal,
        digits: i64,
    ) -> Result<BigDecimalMatrix, LatticeTransformError> {
        rotate_lattice_plane_bigdecimal(&self.basis, axis_a, axis_b, theta, digits)
    }
}

/// Builder for a univariate Coppersmith lattice geared toward RSA partial-factor attacks.
#[derive(Debug, Clone)]
pub struct CoppersmithLatticeBuilder {
    modulus: BigUint,
    polynomial: IntegerPolynomial,
    bound: Option<BigUint>,
    m: Option<usize>,
    dimension: Option<usize>,
}

impl CoppersmithLatticeBuilder {
    /// Starts a builder from an RSA modulus and a monic integer polynomial.
    ///
    /// # Parameters
    /// - `modulus`: RSA modulus `N`.
    /// - `polynomial`: Monic polynomial `f(x)` used to build the lattice.
    ///
    /// # Returns
    /// - `Self`: Builder with the mandatory base inputs captured.
    ///
    /// # Expected Output
    /// - Returns a builder value; no side effects.
    pub fn new(modulus: BigUint, polynomial: IntegerPolynomial) -> Self {
        Self {
            modulus,
            polynomial,
            bound: None,
            m: None,
            dimension: None,
        }
    }

    /// Sets the root bound `X`.
    ///
    /// # Parameters
    /// - `bound`: Positive bound on the absolute root size.
    ///
    /// # Returns
    /// - `Self`: Updated builder.
    ///
    /// # Expected Output
    /// - Returns the updated builder; no side effects.
    pub fn with_bound(mut self, bound: BigUint) -> Self {
        self.bound = Some(bound);
        self
    }

    /// Sets the exponent parameter `m`.
    ///
    /// # Parameters
    /// - `m`: Exponent applied in the Coppersmith basis construction.
    ///
    /// # Returns
    /// - `Self`: Updated builder.
    ///
    /// # Expected Output
    /// - Returns the updated builder; no side effects.
    pub fn with_exponent(mut self, m: usize) -> Self {
        self.m = Some(m);
        self
    }

    /// Sets the target square lattice dimension `d`.
    ///
    /// # Parameters
    /// - `dimension`: Number of lattice rows and columns in the resulting basis.
    ///
    /// # Returns
    /// - `Self`: Updated builder.
    ///
    /// # Expected Output
    /// - Returns the updated builder; no side effects.
    pub fn with_dimension(mut self, dimension: usize) -> Self {
        self.dimension = Some(dimension);
        self
    }

    /// Builds the Coppersmith lattice and all `g_{i,j}(x)` rows.
    ///
    /// # Parameters
    /// - None.
    ///
    /// # Returns
    /// - `Result<CoppersmithLattice, CoppersmithError>`: Constructed lattice or a validation error.
    ///
    /// # Expected Output
    /// - Returns a lattice description and basis matrix; no side effects.
    pub fn build(self) -> Result<CoppersmithLattice, CoppersmithError> {
        validate_coppersmith_inputs(
            &self.modulus,
            &self.polynomial,
            self.bound.as_ref(),
            self.m,
            self.dimension,
        )?;

        let bound = self.bound.expect("validated bound");
        let m = self.m.expect("validated exponent");
        let dimension = self.dimension.expect("validated dimension");
        let degree = self.polynomial.degree();
        let minimum_dimension = degree
            .checked_mul(m)
            .ok_or(CoppersmithError::DimensionOverflow)?;
        let completion_rows = dimension - minimum_dimension;
        let bound_int = BigInt::from(bound.clone());
        let polynomial_powers = build_polynomial_powers(&self.polynomial, m);
        let mut elements = Vec::with_capacity(dimension);

        for i in 0..m {
            let modulus_power = pow_biguint_usize(&self.modulus, m - i);
            let scalar = BigInt::from(modulus_power);

            for j in 0..degree {
                let polynomial = polynomial_powers[i].shift(j).scale(&scalar);
                let scaled_polynomial = polynomial.scale_input(&bound_int);
                let coefficient_vector = scaled_polynomial.padded_coefficients(dimension);

                elements.push(CoppersmithLatticeElement {
                    i,
                    j,
                    polynomial,
                    scaled_polynomial,
                    coefficient_vector,
                });
            }
        }

        for j in 0..completion_rows {
            let polynomial = polynomial_powers[m].shift(j);
            let scaled_polynomial = polynomial.scale_input(&bound_int);
            let coefficient_vector = scaled_polynomial.padded_coefficients(dimension);

            elements.push(CoppersmithLatticeElement {
                i: m,
                j,
                polynomial,
                scaled_polynomial,
                coefficient_vector,
            });
        }

        let basis = as_matrix(
            &elements
                .iter()
                .map(|element| Array1::from(element.coefficient_vector.clone()))
                .collect::<Vec<_>>(),
        );

        debug_assert_eq!(basis.nrows(), dimension);
        debug_assert_eq!(basis.ncols(), dimension);

        Ok(CoppersmithLattice {
            polynomial: self.polynomial,
            modulus: self.modulus,
            bound,
            m,
            dimension,
            completion_rows,
            elements,
            basis,
        })
    }
}

/// Input bundle for the RSA partial-prime Coppersmith runner.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RsaCoppersmithInput {
    /// RSA modulus `N = pq`.
    pub modulus: BigUint,
    /// Prime factor `p` whose low-order portion is treated as unknown.
    ///
    /// Set this to `0` to skip factor-based validation and use `known_prefix` directly.
    pub prime: BigUint,
    /// Known high-order prefix used to build `f(x) = x + known_prefix` when `prime == 0`.
    pub known_prefix: BigUint,
    /// Exact unknown low-order part `x` used to derive the known prefix `p - x`.
    pub unknown_part: BigUint,
    /// Exponent parameter `m` for the lattice basis.
    pub m: usize,
    /// Total lattice dimension `d`.
    pub dimension: usize,
}

/// Output produced by the RSA partial-prime Coppersmith runner.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RsaCoppersmithRun {
    /// Known high-order prefix `p - x` used in `f(x) = x + (p - x)`.
    pub known_prefix: BigUint,
    /// Root search bound derived from the bit width of `x`.
    pub bound: BigUint,
    /// Coppersmith lattice assembled for this RSA instance.
    pub lattice: CoppersmithLattice,
    /// LLL-reduced basis vectors.
    pub reduced_basis: Vec<BigIntVector>,
    /// Reduced basis vectors converted back to integer polynomials.
    pub reduced_polynomials: Vec<IntegerPolynomial>,
    /// Recovered low-order part, when a candidate root is found in the reduced basis.
    ///
    /// When `prime != 0`, the candidate must also reconstruct a factor of the RSA modulus.
    pub recovered_unknown: Option<BigUint>,
}

impl RsaCoppersmithRun {
    /// Returns whether the run recovered the supplied unknown part exactly.
    ///
    /// # Parameters
    /// - `expected_unknown`: The expected low-order value `x`.
    ///
    /// # Returns
    /// - `bool`: `true` when `recovered_unknown == Some(expected_unknown.clone())`.
    ///
    /// # Expected Output
    /// - Returns a boolean match result; no side effects.
    pub fn recovered_expected_unknown(&self, expected_unknown: &BigUint) -> bool {
        self.recovered_unknown.as_ref() == Some(expected_unknown)
    }
}

/// Runs the RSA partial-prime Coppersmith workflow for a known `p` and low-order tail `x`.
///
/// # Parameters
/// - `input`: RSA modulus, optional prime factor, known prefix, unknown tail, and lattice parameters.
///
/// # Returns
/// - `Result<RsaCoppersmithRun, CoppersmithError>`: Run artifacts or a validation error.
///
/// # Expected Output
/// - Returns the lattice, reduced basis, and recovered candidate root; no side effects.
pub fn run_rsa_coppersmith(
    input: &RsaCoppersmithInput,
) -> Result<RsaCoppersmithRun, CoppersmithError> {
    if input.modulus <= BigUint::one() {
        return Err(CoppersmithError::InvalidModulus);
    }
    let prime_supplied = !input.prime.is_zero();

    if prime_supplied {
        if input.unknown_part > input.prime {
            return Err(CoppersmithError::UnknownPartExceedsPrime);
        }
        if (&input.modulus % &input.prime) != BigUint::zero() {
            return Err(CoppersmithError::PrimeDoesNotDivideModulus);
        }
    }

    let known_prefix = if prime_supplied {
        &input.prime - &input.unknown_part
    } else {
        input.known_prefix.clone()
    };
    let bound = rsa_unknown_bound(&input.unknown_part);
    let polynomial = IntegerPolynomial::new(vec![BigInt::from(known_prefix.clone()), BigInt::one()]);
    let lattice = CoppersmithLatticeBuilder::new(input.modulus.clone(), polynomial)
        .with_bound(bound.clone())
        .with_exponent(input.m)
        .with_dimension(input.dimension)
        .build()?;
    let reduced_basis = lattice.reduce();
    let reduced_polynomials = lattice.reduced_polynomials(&reduced_basis)?;
    let recovered_unknown = if prime_supplied {
        brute_force_rsa_unknown(&known_prefix, &input.modulus, &reduced_polynomials, &bound)
    } else {
        brute_force_small_root(&reduced_polynomials, &bound)
    };

    Ok(RsaCoppersmithRun {
        known_prefix,
        bound,
        lattice,
        reduced_basis,
        reduced_polynomials,
        recovered_unknown,
    })
}

/// Runs the educational univariate Coppersmith flow against a monic polynomial.
///
/// # Parameters
/// - `f`: Monic univariate polynomial whose small roots are sought.
/// - `n`: Positive modulus used in the congruence test.
/// - `x_bound`: Positive bound `X` used both for lattice scaling and brute-force root checks.
/// - `m`: Main Coppersmith exponent parameter.
/// - `t`: Number of extra `x^j f(x)^m` shift rows.
///
/// # Returns
/// - `Vec<BigInt>`: Sorted unique roots recovered from the reduced basis.
///
/// # Expected Output
/// - Returns candidate small roots without side effects.
pub fn univariate_coppersmith(
    f: &Poly,
    n: &BigInt,
    x_bound: &BigInt,
    m: usize,
    t: usize,
) -> Vec<BigInt> {
    assert!(n > &BigInt::one(), "modulus n must be greater than one");
    assert!(
        x_bound > &BigInt::zero(),
        "x_bound must be greater than zero"
    );

    let d = f.degree();
    assert!(d > 0, "f(x) must have degree at least one");
    assert_eq!(f.coeff(d), BigInt::one(), "f(x) should be monic");

    let modulus = BigUint::try_from(n.clone()).expect("univariate_coppersmith requires n >= 0");
    let bound = BigUint::try_from(x_bound.clone())
        .expect("univariate_coppersmith requires x_bound >= 0");
    let dimension = d
        .checked_mul(m)
        .and_then(|value| value.checked_add(t))
        .expect("coppersmith lattice dimension overflow");

    let lattice = CoppersmithLatticeBuilder::new(modulus, f.clone())
        .with_bound(bound)
        .with_exponent(m)
        .with_dimension(dimension)
        .build()
        .expect("univariate_coppersmith inputs must form a valid lattice");
    let reduced = lattice.reduce();

    let mut roots = Vec::<BigInt>::new();

    for row in reduced.iter().take(8) {
        let h_scaled = Poly::new(row.to_vec());
        let h = h_scaled.unscale_variable_exact(x_bound);

        for r in candidate_integer_roots(&h, x_bound) {
            if f.eval(&r).mod_floor(n).is_zero() && !roots.contains(&r) {
                roots.push(r);
            }
        }
    }

    roots.sort();
    roots
}

/// Computes the integer dot product of two lattice vectors.
#[cfg(test)]
fn dot_i(a: &BigIntVector, b: &BigIntVector) -> BigInt {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| x * y)
        .fold(BigInt::zero(), |acc, v| acc + v)
}

/// Computes the rational dot product of two Gram-Schmidt vectors.
fn dot_q(a: &[BigRational], b: &[BigRational]) -> BigRational {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| x * y)
        .fold(BigRational::zero(), |acc, v| acc + v)
}

/// Rounds a rational value to the nearest integer using symmetric half-up behavior.
fn nearest_integer(x: &BigRational) -> BigInt {
    let n = x.numer().clone();
    let d = x.denom().clone();

    if n.sign() == num_bigint::Sign::Minus {
        -nearest_integer(&(-x.clone()))
    } else {
        (&n + (&d / 2)) / d
    }
}

/// Computes the Gram-Schmidt orthogonalization for the current basis.
fn gram_schmidt(
    basis: &[BigIntVector],
) -> (
    Vec<Vec<BigRational>>,
    Vec<Vec<BigRational>>,
    Vec<BigRational>,
) {
    let n = basis.len();
    let dim = basis[0].len();

    let mut b_star = vec![vec![BigRational::zero(); dim]; n];
    let mut mu = vec![vec![BigRational::zero(); n]; n];
    let mut norm = vec![BigRational::zero(); n];

    for i in 0..n {
        let mut v: Vec<BigRational> = basis[i]
            .iter()
            .map(|x| BigRational::from_integer(x.clone()))
            .collect();

        for j in 0..i {
            mu[i][j] = dot_q(
                &basis[i]
                    .iter()
                    .map(|x| BigRational::from_integer(x.clone()))
                    .collect::<Vec<_>>(),
                &b_star[j],
            ) / &norm[j];

            for k in 0..dim {
                v[k] -= &mu[i][j] * &b_star[j][k];
            }
        }

        norm[i] = dot_q(&v, &v);
        b_star[i] = v;
    }

    (b_star, mu, norm)
}

/// Applies a size-reduction step during LLL.
fn size_reduce(basis: &mut [BigIntVector], k: usize, l: usize, q: &BigInt) {
    if q.is_zero() {
        return;
    }

    let row_l = basis[l].clone();

    for i in 0..basis[k].len() {
        basis[k][i] -= q * &row_l[i];
    }
}

/// Runs LLL reduction with the default Lovasz parameter `3/4`.
///
/// # Parameters
/// - `input`: Lattice basis matrix whose rows are the basis vectors.
///
/// # Returns
/// - `Vec<BigIntVector>`: Reduced basis vectors.
///
/// # Expected Output
/// - Returns a reduced basis; no side effects.
pub fn lll_reduce(input: &BigIntMatrix) -> Vec<BigIntVector> {
    lll_reduce_delta(input, BigRational::new(BigInt::from(3), BigInt::from(4)))
}

/// Runs LLL reduction with an explicit Lovasz parameter.
///
/// # Parameters
/// - `input`: Lattice basis matrix whose rows are the basis vectors.
/// - `delta`: Lovasz parameter in `(1/4, 1)`.
///
/// # Returns
/// - `Vec<BigIntVector>`: Reduced basis vectors.
///
/// # Expected Output
/// - Returns a reduced basis; no side effects.
pub fn lll_reduce_delta(input: &BigIntMatrix, delta: BigRational) -> Vec<BigIntVector> {
    assert!(delta > BigRational::new(BigInt::from(1), BigInt::from(4)));
    assert!(delta < BigRational::one());

    let mut basis: Vec<BigIntVector> = input.rows().into_iter().map(|r| r.to_owned()).collect();

    if basis.is_empty() {
        return basis;
    }

    let n = basis.len();
    let mut k = 1usize;

    while k < n {
        let (_, mut mu, _) = gram_schmidt(&basis);

        for j in (0..k).rev() {
            let q = nearest_integer(&mu[k][j]);

            if !q.is_zero() {
                size_reduce(&mut basis, k, j, &q);
            }
        }

        let (_, mu2, norm) = gram_schmidt(&basis);
        mu = mu2;

        let lhs = norm[k].clone();
        let rhs = (&delta - &mu[k][k - 1] * &mu[k][k - 1]) * &norm[k - 1];

        if lhs >= rhs {
            k += 1;
        } else {
            basis.swap(k, k - 1);

            if k > 1 {
                k -= 1;
            }
        }
    }

    basis
}

/// Computes the Frobenius inner product of two lattice matrices exactly as a `BigDecimal`.
///
/// # Parameters
/// - `left`: Left lattice matrix.
/// - `right`: Right lattice matrix with the same shape as `left`.
///
/// # Returns
/// - `Result<BigDecimal, LatticeMetricError>`: Exact Frobenius inner product or a shape error.
///
/// # Expected Output
/// - Returns an exact `BigDecimal` metric value; no side effects.
pub fn frobenius_inner_product_bigdecimal(
    left: &BigIntMatrix,
    right: &BigIntMatrix,
) -> Result<BigDecimal, LatticeMetricError> {
    validate_same_shape(left.nrows(), left.ncols(), right.nrows(), right.ncols())?;

    let sum = left
        .iter()
        .zip(right.iter())
        .map(|(left_value, right_value)| left_value * right_value)
        .fold(BigInt::zero(), |acc, value| acc + value);

    Ok(BigDecimal::from(sum))
}

/// Computes the Frobenius inner product of two reduced or non-reduced lattice row collections.
///
/// # Parameters
/// - `left`: Left row collection.
/// - `right`: Right row collection with the same matrix shape as `left`.
///
/// # Returns
/// - `Result<BigDecimal, LatticeMetricError>`: Exact Frobenius inner product or a shape error.
///
/// # Expected Output
/// - Returns an exact `BigDecimal` metric value; no side effects.
pub fn frobenius_inner_product_vectors_bigdecimal(
    left: &[BigIntVector],
    right: &[BigIntVector],
) -> Result<BigDecimal, LatticeMetricError> {
    let (left_rows, left_cols) = validate_bigint_vector_collection_shape_metric(left)?;
    let (right_rows, right_cols) = validate_bigint_vector_collection_shape_metric(right)?;
    validate_same_shape(left_rows, left_cols, right_rows, right_cols)?;

    let left_matrix = as_matrix(left);
    let right_matrix = as_matrix(right);
    frobenius_inner_product_bigdecimal(&left_matrix, &right_matrix)
}

/// Computes the Frobenius inner product of two decimal lattice matrices.
///
/// # Parameters
/// - `left`: Left decimal lattice matrix.
/// - `right`: Right decimal lattice matrix with the same shape as `left`.
///
/// # Returns
/// - `Result<BigDecimal, LatticeMetricError>`: Frobenius inner product or a shape error.
///
/// # Expected Output
/// - Returns a decimal metric value; no side effects.
pub fn frobenius_inner_product_decimal(
    left: &BigDecimalMatrix,
    right: &BigDecimalMatrix,
) -> Result<BigDecimal, LatticeMetricError> {
    validate_same_shape(left.nrows(), left.ncols(), right.nrows(), right.ncols())?;

    Ok(left
        .iter()
        .zip(right.iter())
        .fold(BigDecimal::zero(), |acc, (left_value, right_value)| {
            acc + left_value * right_value
        }))
}

/// Computes the Frobenius inner product of two decimal row-vector collections.
///
/// # Parameters
/// - `left`: Left decimal row collection.
/// - `right`: Right decimal row collection with the same matrix shape as `left`.
///
/// # Returns
/// - `Result<BigDecimal, LatticeMetricError>`: Frobenius inner product or a shape error.
///
/// # Expected Output
/// - Returns a decimal metric value; no side effects.
pub fn frobenius_inner_product_decimal_vectors(
    left: &[BigDecimalVector],
    right: &[BigDecimalVector],
) -> Result<BigDecimal, LatticeMetricError> {
    let (left_rows, left_cols) = validate_bigdecimal_vector_collection_shape_metric(left)?;
    let (right_rows, right_cols) = validate_bigdecimal_vector_collection_shape_metric(right)?;
    validate_same_shape(left_rows, left_cols, right_rows, right_cols)?;

    let left_matrix = as_decimal_matrix(left)?;
    let right_matrix = as_decimal_matrix(right)?;
    frobenius_inner_product_decimal(&left_matrix, &right_matrix)
}

/// Rotates an integer lattice matrix in a selected 2D coordinate plane using decimal arithmetic.
///
/// # Parameters
/// - `matrix`: Integer lattice matrix whose rows are basis vectors.
/// - `axis_a`: First coordinate axis participating in the rotation plane.
/// - `axis_b`: Second coordinate axis participating in the rotation plane.
/// - `theta`: Rotation angle in radians.
/// - `digits`: Decimal digits retained in the trigonometric coefficients and output entries.
///
/// # Returns
/// - `Result<BigDecimalMatrix, LatticeTransformError>`: Rotated decimal matrix or a validation error.
///
/// # Expected Output
/// - Returns a rotated decimal matrix; no side effects.
pub fn rotate_lattice_plane_bigdecimal(
    matrix: &BigIntMatrix,
    axis_a: usize,
    axis_b: usize,
    theta: &BigDecimal,
    digits: i64,
) -> Result<BigDecimalMatrix, LatticeTransformError> {
    let decimal = matrix.mapv(BigDecimal::from);
    rotate_decimal_lattice_plane_bigdecimal(&decimal, axis_a, axis_b, theta, digits)
}

/// Rotates an integer row-vector lattice in a selected 2D coordinate plane using decimal arithmetic.
///
/// # Parameters
/// - `vectors`: Integer row collection whose rows are basis vectors.
/// - `axis_a`: First coordinate axis participating in the rotation plane.
/// - `axis_b`: Second coordinate axis participating in the rotation plane.
/// - `theta`: Rotation angle in radians.
/// - `digits`: Decimal digits retained in the trigonometric coefficients and output entries.
///
/// # Returns
/// - `Result<Vec<BigDecimalVector>, LatticeTransformError>`: Rotated decimal vectors or a validation error.
///
/// # Expected Output
/// - Returns rotated decimal row vectors; no side effects.
pub fn rotate_lattice_vectors_plane_bigdecimal(
    vectors: &[BigIntVector],
    axis_a: usize,
    axis_b: usize,
    theta: &BigDecimal,
    digits: i64,
) -> Result<Vec<BigDecimalVector>, LatticeTransformError> {
    let matrix = as_matrix(vectors);
    let rotated = rotate_lattice_plane_bigdecimal(&matrix, axis_a, axis_b, theta, digits)?;
    decimal_matrix_to_vectors(&rotated)
}

/// Rotates a decimal lattice matrix in a selected 2D coordinate plane.
///
/// # Parameters
/// - `matrix`: Decimal lattice matrix whose rows are basis vectors.
/// - `axis_a`: First coordinate axis participating in the rotation plane.
/// - `axis_b`: Second coordinate axis participating in the rotation plane.
/// - `theta`: Rotation angle in radians.
/// - `digits`: Decimal digits retained in the trigonometric coefficients and output entries.
///
/// # Returns
/// - `Result<BigDecimalMatrix, LatticeTransformError>`: Rotated decimal matrix or a validation error.
///
/// # Expected Output
/// - Returns a rotated decimal matrix; no side effects.
pub fn rotate_decimal_lattice_plane_bigdecimal(
    matrix: &BigDecimalMatrix,
    axis_a: usize,
    axis_b: usize,
    theta: &BigDecimal,
    digits: i64,
) -> Result<BigDecimalMatrix, LatticeTransformError> {
    if digits < 0 {
        return Err(LatticeTransformError::NegativeDigits { digits });
    }

    let dimension = matrix.ncols();
    validate_rotation_plane(axis_a, axis_b, dimension)?;
    let precision = digits + 12;
    let (cos_theta, sin_theta) = rotation_coefficients(theta, precision)?;
    let mut rotated = matrix.clone();

    for row in 0..matrix.nrows() {
        let x = matrix[[row, axis_a]].clone();
        let y = matrix[[row, axis_b]].clone();
        let rotated_x = (&cos_theta * &x - &sin_theta * &y).with_scale(digits);
        let rotated_y = (&sin_theta * &x + &cos_theta * &y).with_scale(digits);
        rotated[[row, axis_a]] = rotated_x;
        rotated[[row, axis_b]] = rotated_y;
    }

    Ok(rotated)
}

/// Rotates a decimal row-vector lattice in a selected 2D coordinate plane.
///
/// # Parameters
/// - `vectors`: Decimal row collection whose rows are basis vectors.
/// - `axis_a`: First coordinate axis participating in the rotation plane.
/// - `axis_b`: Second coordinate axis participating in the rotation plane.
/// - `theta`: Rotation angle in radians.
/// - `digits`: Decimal digits retained in the trigonometric coefficients and output entries.
///
/// # Returns
/// - `Result<Vec<BigDecimalVector>, LatticeTransformError>`: Rotated decimal vectors or a validation error.
///
/// # Expected Output
/// - Returns rotated decimal row vectors; no side effects.
pub fn rotate_decimal_lattice_vectors_plane_bigdecimal(
    vectors: &[BigDecimalVector],
    axis_a: usize,
    axis_b: usize,
    theta: &BigDecimal,
    digits: i64,
) -> Result<Vec<BigDecimalVector>, LatticeTransformError> {
    let matrix = as_decimal_matrix_transform(vectors)?;
    let rotated = rotate_decimal_lattice_plane_bigdecimal(&matrix, axis_a, axis_b, theta, digits)?;
    decimal_matrix_to_vectors(&rotated)
}

/// Flattens a decimal lattice matrix back into integers by rounding `value * 10^digits`.
///
/// # Parameters
/// - `matrix`: Decimal lattice matrix to quantize.
/// - `digits`: Number of decimal digits to preserve before rounding into integers.
///
/// # Returns
/// - `Result<BigIntMatrix, LatticeTransformError>`: Quantized integer matrix or a validation error.
///
/// # Expected Output
/// - Returns an integer matrix; no side effects.
pub fn flatten_decimal_lattice_to_bigints(
    matrix: &BigDecimalMatrix,
    digits: i64,
) -> Result<BigIntMatrix, LatticeTransformError> {
    if digits < 0 {
        return Err(LatticeTransformError::NegativeDigits { digits });
    }

    let quantized: Vec<BigInt> = matrix
        .iter()
        .map(|value| round_scaled_bigdecimal_to_bigint(value, digits))
        .collect();

    Array2::from_shape_vec((matrix.nrows(), matrix.ncols()), quantized)
        .map_err(|_| LatticeTransformError::InconsistentRowLength {
            row: 0,
            expected: matrix.ncols(),
            actual: matrix.ncols(),
        })
}

/// Flattens decimal row vectors back into integer vectors by rounding `value * 10^digits`.
///
/// # Parameters
/// - `vectors`: Decimal row collection to quantize.
/// - `digits`: Number of decimal digits to preserve before rounding into integers.
///
/// # Returns
/// - `Result<Vec<BigIntVector>, LatticeTransformError>`: Quantized integer row vectors or a validation error.
///
/// # Expected Output
/// - Returns integer row vectors; no side effects.
pub fn flatten_decimal_lattice_vectors_to_bigints(
    vectors: &[BigDecimalVector],
    digits: i64,
) -> Result<Vec<BigIntVector>, LatticeTransformError> {
    let matrix = as_decimal_matrix_transform(vectors)?;
    let quantized = flatten_decimal_lattice_to_bigints(&matrix, digits)?;
    Ok(quantized
        .rows()
        .into_iter()
        .map(|row| row.to_owned())
        .collect())
}

/// Converts a row vector collection into an `ndarray` matrix.
///
/// # Parameters
/// - `vectors`: Row vectors to assemble into a matrix.
///
/// # Returns
/// - `BigIntMatrix`: Matrix with one row per vector.
///
/// # Expected Output
/// - Returns a matrix value; no side effects.
pub fn as_matrix(vectors: &[BigIntVector]) -> BigIntMatrix {
    if vectors.is_empty() {
        return Array2::from_shape_vec((0, 0), vec![]).expect("empty matrix");
    }

    let rows = vectors.len();
    let cols = vectors[0].len();

    let flat: Vec<BigInt> = vectors.iter().flat_map(|row| row.iter().cloned()).collect();

    Array2::from_shape_vec((rows, cols), flat).expect("valid lattice matrix shape")
}

/// Converts decimal row vectors into an `ndarray` matrix.
fn as_decimal_matrix(vectors: &[BigDecimalVector]) -> Result<BigDecimalMatrix, LatticeMetricError> {
    let (rows, cols) = validate_bigdecimal_vector_collection_shape_metric(vectors)?;
    let flat: Vec<BigDecimal> = vectors
        .iter()
        .flat_map(|row| row.iter().cloned())
        .collect();

    Array2::from_shape_vec((rows, cols), flat).map_err(|_| LatticeMetricError::InconsistentRowLength {
        row: 0,
        expected: cols,
        actual: cols,
    })
}

/// Converts decimal row vectors into an `ndarray` matrix for lattice transforms.
fn as_decimal_matrix_transform(
    vectors: &[BigDecimalVector],
) -> Result<BigDecimalMatrix, LatticeTransformError> {
    let (rows, cols) = validate_bigdecimal_vector_collection_shape_transform(vectors)?;
    let flat: Vec<BigDecimal> = vectors
        .iter()
        .flat_map(|row| row.iter().cloned())
        .collect();

    Array2::from_shape_vec((rows, cols), flat).map_err(|_| LatticeTransformError::InconsistentRowLength {
        row: 0,
        expected: cols,
        actual: cols,
    })
}

/// Converts a decimal matrix into owned decimal row vectors.
fn decimal_matrix_to_vectors(
    matrix: &BigDecimalMatrix,
) -> Result<Vec<BigDecimalVector>, LatticeTransformError> {
    Ok(matrix
        .rows()
        .into_iter()
        .map(|row| row.to_owned())
        .collect())
}

/// Validates that two matrix-like shapes match exactly.
fn validate_same_shape(
    left_rows: usize,
    left_cols: usize,
    right_rows: usize,
    right_cols: usize,
) -> Result<(), LatticeMetricError> {
    if left_rows == right_rows && left_cols == right_cols {
        Ok(())
    } else {
        Err(LatticeMetricError::ShapeMismatch {
            left_rows,
            left_cols,
            right_rows,
            right_cols,
        })
    }
}

/// Returns the effective matrix shape for an integer row-vector collection.
fn validate_bigint_vector_collection_shape_metric(
    vectors: &[BigIntVector],
) -> Result<(usize, usize), LatticeMetricError> {
    if vectors.is_empty() {
        return Ok((0, 0));
    }

    let cols = vectors[0].len();
    for (row_index, row) in vectors.iter().enumerate().skip(1) {
        if row.len() != cols {
            return Err(LatticeMetricError::InconsistentRowLength {
                row: row_index,
                expected: cols,
                actual: row.len(),
            });
        }
    }

    Ok((vectors.len(), cols))
}

/// Returns the effective matrix shape for a decimal row-vector collection.
fn validate_bigdecimal_vector_collection_shape_metric(
    vectors: &[BigDecimalVector],
) -> Result<(usize, usize), LatticeMetricError> {
    if vectors.is_empty() {
        return Ok((0, 0));
    }

    let cols = vectors[0].len();
    for (row_index, row) in vectors.iter().enumerate().skip(1) {
        if row.len() != cols {
            return Err(LatticeMetricError::InconsistentRowLength {
                row: row_index,
                expected: cols,
                actual: row.len(),
            });
        }
    }

    Ok((vectors.len(), cols))
}

/// Returns the effective matrix shape for a decimal row-vector collection for transforms.
fn validate_bigdecimal_vector_collection_shape_transform(
    vectors: &[BigDecimalVector],
) -> Result<(usize, usize), LatticeTransformError> {
    if vectors.is_empty() {
        return Ok((0, 0));
    }

    let cols = vectors[0].len();
    for (row_index, row) in vectors.iter().enumerate().skip(1) {
        if row.len() != cols {
            return Err(LatticeTransformError::InconsistentRowLength {
                row: row_index,
                expected: cols,
                actual: row.len(),
            });
        }
    }

    Ok((vectors.len(), cols))
}

/// Validates a 2D rotation plane against a lattice dimension.
fn validate_rotation_plane(
    axis_a: usize,
    axis_b: usize,
    dimension: usize,
) -> Result<(), LatticeTransformError> {
    if axis_a >= dimension || axis_b >= dimension || axis_a == axis_b {
        Err(LatticeTransformError::InvalidPlane {
            axis_a,
            axis_b,
            dimension,
        })
    } else {
        Ok(())
    }
}

/// Computes cosine and sine for a rotation angle using decimal approximations.
fn rotation_coefficients(
    theta: &BigDecimal,
    digits: i64,
) -> Result<(BigDecimal, BigDecimal), LatticeTransformError> {
    if digits < 0 {
        return Err(LatticeTransformError::NegativeDigits { digits });
    }

    Ok((
        cosine_bigdecimal(theta.clone(), digits),
        sine_bigdecimal(theta.clone(), digits),
    ))
}

/// Returns `10^-scale` as a `BigDecimal`.
fn pow10_neg_bigdecimal(scale: i64) -> BigDecimal {
    BigDecimal::new(1.into(), scale)
}

/// Returns the sine of `x` using a big decimal approximation with `digits` precision.
fn sine_bigdecimal(x: BigDecimal, digits: i64) -> BigDecimal {
    let tolerance = pow10_neg_bigdecimal(digits + 8);

    let mut sum = x.clone();
    let mut term = x.clone();
    let x2 = &x * &x;

    for n in 1..20_000_i64 {
        let denom = BigDecimal::from_i64((2 * n) * (2 * n + 1))
            .expect("small sine series denominator");
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

/// Rounds `value * 10^digits` to the nearest integer using symmetric half-up behavior.
fn round_scaled_bigdecimal_to_bigint(value: &BigDecimal, digits: i64) -> BigInt {
    let (coefficient, scale) = value.clone().into_bigint_and_exponent();
    let target = scale - digits;

    if target <= 0 {
        return coefficient * BigInt::from(10u8).pow((-target) as u32);
    }

    let divisor = BigInt::from(10u8).pow(target as u32);
    let quotient = &coefficient / &divisor;
    let remainder = &coefficient % &divisor;
    let doubled_remainder = remainder.abs() * 2;

    if doubled_remainder >= divisor {
        if coefficient.is_negative() {
            quotient - BigInt::one()
        } else {
            quotient + BigInt::one()
        }
    } else {
        quotient
    }
}

/// Validates the mandatory inputs for a Coppersmith lattice construction.
fn validate_coppersmith_inputs(
    modulus: &BigUint,
    polynomial: &IntegerPolynomial,
    bound: Option<&BigUint>,
    m: Option<usize>,
    dimension: Option<usize>,
) -> Result<(), CoppersmithError> {
    if modulus <= &BigUint::one() {
        return Err(CoppersmithError::InvalidModulus);
    }

    let bound = bound.ok_or(CoppersmithError::MissingBound)?;
    if bound.is_zero() {
        return Err(CoppersmithError::ZeroBound);
    }

    let m = m.ok_or(CoppersmithError::MissingExponent)?;
    let degree = polynomial.degree();

    if polynomial.is_zero() || degree == 0 {
        return Err(CoppersmithError::ZeroDegreePolynomial);
    }
    if polynomial.leading_coefficient() != Some(&BigInt::one()) {
        return Err(CoppersmithError::NonMonicPolynomial);
    }

    let dimension = dimension.ok_or(CoppersmithError::MissingDimension)?;
    let minimum_dimension = degree
        .checked_mul(m)
        .ok_or(CoppersmithError::DimensionOverflow)?;
    if dimension < minimum_dimension {
        return Err(CoppersmithError::DimensionTooSmall {
            minimum: minimum_dimension,
            requested: dimension,
        });
    }

    Ok(())
}

/// Precomputes `f(x)^0` through `f(x)^m`.
fn build_polynomial_powers(polynomial: &IntegerPolynomial, m: usize) -> Vec<IntegerPolynomial> {
    let mut powers = Vec::with_capacity(m + 1);
    powers.push(IntegerPolynomial::from_constant(BigInt::one()));

    for exponent in 1..=m {
        powers.push(polynomial.pow(exponent));
    }

    powers
}

/// Computes `base^exponent` for `BigUint` without truncating the exponent to `u32`.
fn pow_biguint_usize(base: &BigUint, exponent: usize) -> BigUint {
    if exponent == 0 {
        return BigUint::one();
    }

    let mut result = BigUint::one();
    let mut factor = base.clone();
    let mut remaining = exponent;

    while remaining > 0 {
        if remaining % 2 == 1 {
            result *= &factor;
        }
        remaining /= 2;
        if remaining > 0 {
            factor = &factor * &factor;
        }
    }

    result
}

/// Derives an RSA-friendly root bound from the bit width of the supplied unknown tail.
fn rsa_unknown_bound(unknown_part: &BigUint) -> BigUint {
    if unknown_part.is_zero() {
        BigUint::one()
    } else {
        BigUint::one() << unknown_part.bits()
    }
}

/// Searches the reduced polynomials for a candidate `x` that yields a factor of the modulus.
fn brute_force_rsa_unknown(
    known_prefix: &BigUint,
    modulus: &BigUint,
    reduced_polynomials: &[IntegerPolynomial],
    bound: &BigUint,
) -> Option<BigUint> {
    let mut candidate = BigUint::zero();

    while &candidate < bound {
        let candidate_int = BigInt::from(candidate.clone());
        let has_zero = reduced_polynomials
            .iter()
            .filter(|polynomial| !polynomial.is_zero())
            .any(|polynomial| polynomial.evaluate(&candidate_int).is_zero());

        if has_zero {
            let prime_candidate = known_prefix + &candidate;
            if prime_candidate > BigUint::one() && (modulus % &prime_candidate) == BigUint::zero() {
                return Some(candidate);
            }
        }

        candidate += BigUint::one();
    }

    None
}

/// Searches the reduced polynomials for the first candidate root in `[0, X)`.
fn brute_force_small_root(
    reduced_polynomials: &[IntegerPolynomial],
    bound: &BigUint,
) -> Option<BigUint> {
    let mut candidate = BigUint::zero();

    while &candidate < bound {
        let candidate_int = BigInt::from(candidate.clone());
        let has_zero = reduced_polynomials
            .iter()
            .filter(|polynomial| !polynomial.is_zero())
            .any(|polynomial| polynomial.evaluate(&candidate_int).is_zero());

        if has_zero {
            return Some(candidate);
        }

        candidate += BigUint::one();
    }

    None
}

/// Returns candidate integer roots, preferring exact low-degree extraction over bounded search.
fn candidate_integer_roots(h: &Poly, x_bound: &BigInt) -> Vec<BigInt> {
    if let Some(roots) = h.exact_integer_roots_low_degree() {
        return roots
            .into_iter()
            .filter(|root| root.abs() <= *x_bound)
            .collect();
    }

    brute_force_roots(h, x_bound)
}

/// Brute-forces exact integer roots of a polynomial inside `[-X, X]`.
fn brute_force_roots(h: &Poly, x_bound: &BigInt) -> Vec<BigInt> {
    let Some(bound) = x_bound.abs().to_i64() else {
        return Vec::new();
    };

    let mut roots = Vec::new();

    for x in -bound..=bound {
        let candidate = BigInt::from(x);
        if h.eval(&candidate).is_zero() {
            roots.push(candidate);
        }
    }

    roots
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_lll_small_basis() {
        let basis = array![
            [BigInt::from(105), BigInt::from(821), BigInt::from(404)],
            [BigInt::from(331), BigInt::from(569), BigInt::from(74)],
            [BigInt::from(511), BigInt::from(322), BigInt::from(912)],
        ];

        let reduced = lll_reduce(&basis);

        assert_eq!(reduced.len(), 3);
        assert_eq!(reduced[0].len(), 3);
        assert_eq!(dot_i(&reduced[0], &reduced[0]).sign(), num_bigint::Sign::Plus);
    }

    #[test]
    fn frobenius_inner_product_bigdecimal_returns_exact_integer_value() {
        let left = array![
            [BigInt::from(1), BigInt::from(2)],
            [BigInt::from(3), BigInt::from(4)],
        ];
        let right = array![
            [BigInt::from(5), BigInt::from(6)],
            [BigInt::from(7), BigInt::from(8)],
        ];

        let product = frobenius_inner_product_bigdecimal(&left, &right).expect("product");

        assert_eq!(product, BigDecimal::from(BigInt::from(70)));
    }

    #[test]
    fn frobenius_inner_product_vectors_bigdecimal_supports_reduced_bases() {
        let basis = array![
            [BigInt::from(1), BigInt::from(1)],
            [BigInt::from(1), BigInt::from(0)],
        ];
        let reduced = lll_reduce(&basis);

        let product =
            frobenius_inner_product_vectors_bigdecimal(&reduced, &reduced).expect("product");

        assert_eq!(product, BigDecimal::from(BigInt::from(2)));
    }

    #[test]
    fn frobenius_inner_product_basis_uses_lattice_bases() {
        let polynomial = IntegerPolynomial::new(vec![BigInt::from(12), BigInt::one()]);
        let lattice = CoppersmithLatticeBuilder::new(BigUint::from(77u8), polynomial)
            .with_bound(BigUint::from(8u8))
            .with_exponent(2)
            .with_dimension(4)
            .build()
            .expect("lattice");

        let product = lattice
            .frobenius_inner_product_basis(&lattice)
            .expect("product");

        assert_eq!(product, BigDecimal::from(BigInt::from(40396513u32)));
    }

    #[test]
    fn frobenius_inner_product_bigdecimal_rejects_shape_mismatch() {
        let left = array![[BigInt::from(1), BigInt::from(2)]];
        let right = array![
            [BigInt::from(1), BigInt::from(2)],
            [BigInt::from(3), BigInt::from(4)],
        ];

        let error = frobenius_inner_product_bigdecimal(&left, &right).expect_err("shape mismatch");

        assert_eq!(
            error,
            LatticeMetricError::ShapeMismatch {
                left_rows: 1,
                left_cols: 2,
                right_rows: 2,
                right_cols: 2,
            }
        );
    }

    #[test]
    fn frobenius_inner_product_decimal_supports_rotated_decimal_matrices() {
        let left = Array2::from_shape_vec(
            (1, 2),
            vec!["1.5".parse::<BigDecimal>().unwrap(), BigDecimal::from(2)],
        )
        .expect("decimal matrix");
        let right = Array2::from_shape_vec(
            (1, 2),
            vec!["3.0".parse::<BigDecimal>().unwrap(), BigDecimal::from(-4)],
        )
        .expect("decimal matrix");

        let product = frobenius_inner_product_decimal(&left, &right).expect("product");

        assert_eq!(product, "-3.5".parse::<BigDecimal>().unwrap());
    }

    #[test]
    fn rotate_lattice_plane_bigdecimal_quarter_turn_flattens_to_integer_rotation() {
        let basis = array![
            [BigInt::from(1), BigInt::from(0)],
            [BigInt::from(0), BigInt::from(1)],
        ];
        let theta = "1.57079632679489661923".parse::<BigDecimal>().expect("theta");

        let rotated = rotate_lattice_plane_bigdecimal(&basis, 0, 1, &theta, 18).expect("rotate");
        let flattened = flatten_decimal_lattice_to_bigints(&rotated, 0).expect("flatten");

        assert_eq!(
            flattened,
            array![
                [BigInt::from(0), BigInt::from(1)],
                [BigInt::from(-1), BigInt::from(0)],
            ]
        );
    }

    #[test]
    fn rotate_lattice_vectors_plane_bigdecimal_preserves_self_frobenius_after_flattening() {
        let basis = array![
            [BigInt::from(1), BigInt::from(1)],
            [BigInt::from(1), BigInt::from(0)],
        ];
        let reduced = lll_reduce(&basis);
        let theta = "1.57079632679489661923".parse::<BigDecimal>().expect("theta");

        let rotated =
            rotate_lattice_vectors_plane_bigdecimal(&reduced, 0, 1, &theta, 18).expect("rotate");
        let flattened = flatten_decimal_lattice_vectors_to_bigints(&rotated, 0).expect("flatten");
        let original_product =
            frobenius_inner_product_vectors_bigdecimal(&reduced, &reduced).expect("product");
        let rotated_product =
            frobenius_inner_product_vectors_bigdecimal(&flattened, &flattened).expect("product");

        assert_eq!(original_product, rotated_product);
    }

    #[test]
    fn flatten_decimal_lattice_to_bigints_preserves_configured_digits() {
        let matrix = Array2::from_shape_vec(
            (1, 2),
            vec![
                "1.2345".parse::<BigDecimal>().unwrap(),
                "-0.006".parse::<BigDecimal>().unwrap(),
            ],
        )
        .expect("decimal matrix");

        let flattened = flatten_decimal_lattice_to_bigints(&matrix, 3).expect("flatten");

        assert_eq!(
            flattened,
            array![[BigInt::from(1235), BigInt::from(-6)]]
        );
    }

    #[test]
    fn rotate_lattice_plane_bigdecimal_rejects_invalid_plane() {
        let basis = array![[BigInt::from(1), BigInt::from(0)]];
        let theta = BigDecimal::from(0);

        let error = rotate_lattice_plane_bigdecimal(&basis, 1, 1, &theta, 8).expect_err("plane");

        assert_eq!(
            error,
            LatticeTransformError::InvalidPlane {
                axis_a: 1,
                axis_b: 1,
                dimension: 2,
            }
        );
    }

    #[test]
    fn flatten_decimal_lattice_to_bigints_rejects_negative_digits() {
        let matrix = Array2::from_shape_vec((0, 0), Vec::<BigDecimal>::new()).expect("matrix");

        let error =
            flatten_decimal_lattice_to_bigints(&matrix, -1).expect_err("negative digits");

        assert_eq!(error, LatticeTransformError::NegativeDigits { digits: -1 });
    }

    #[test]
    fn coppersmith_builder_creates_expected_linear_basis() {
        let polynomial = IntegerPolynomial::new(vec![BigInt::from(12), BigInt::one()]);
        let lattice = CoppersmithLatticeBuilder::new(BigUint::from(77u8), polynomial)
            .with_bound(BigUint::from(8u8))
            .with_exponent(2)
            .with_dimension(4)
            .build()
            .expect("lattice");

        assert_eq!(lattice.dimension, 4);
        assert_eq!(lattice.completion_rows, 2);
        assert_eq!(lattice.elements.len(), 4);
        assert_eq!(lattice.basis.nrows(), 4);
        assert_eq!(lattice.basis.ncols(), 4);

        assert_eq!(lattice.elements[0].i, 0);
        assert_eq!(lattice.elements[0].j, 0);
        assert_eq!(
            lattice.elements[0].polynomial.coefficients(),
            &[BigInt::from(5929)]
        );

        assert_eq!(lattice.elements[1].i, 1);
        assert_eq!(lattice.elements[1].j, 0);
        assert_eq!(
            lattice.elements[1].scaled_polynomial.coefficients(),
            &[BigInt::from(924), BigInt::from(616)]
        );

        assert_eq!(lattice.elements[2].i, 2);
        assert_eq!(lattice.elements[2].j, 0);
        assert_eq!(
            lattice.elements[2].coefficient_vector,
            vec![
                BigInt::from(144),
                BigInt::from(192),
                BigInt::from(64),
                BigInt::from(0)
            ]
        );

        assert_eq!(lattice.elements[3].i, 2);
        assert_eq!(lattice.elements[3].j, 1);
        assert_eq!(
            lattice.elements[3].coefficient_vector,
            vec![
                BigInt::from(0),
                BigInt::from(1152),
                BigInt::from(1536),
                BigInt::from(512)
            ]
        );
    }

    #[test]
    fn univariate_coppersmith_finds_small_linear_root() {
        let polynomial = Poly::new(vec![BigInt::from(-3), BigInt::one()]);

        let roots = univariate_coppersmith(
            &polynomial,
            &BigInt::from(97u8),
            &BigInt::from(4u8),
            2,
            2,
        );

        assert_eq!(roots, vec![BigInt::from(3)]);
    }

    #[test]
    fn univariate_coppersmith_finds_negative_root_inside_bound() {
        let polynomial = Poly::new(vec![BigInt::from(2), BigInt::one()]);

        let roots = univariate_coppersmith(
            &polynomial,
            &BigInt::from(97u8),
            &BigInt::from(3u8),
            2,
            2,
        );

        assert_eq!(roots, vec![BigInt::from(-2)]);
    }

    #[test]
    fn coppersmith_builder_rejects_non_monic_polynomial() {
        let polynomial = IntegerPolynomial::new(vec![BigInt::from(3), BigInt::from(2)]);
        let error = CoppersmithLatticeBuilder::new(BigUint::from(91u8), polynomial)
            .with_bound(BigUint::from(4u8))
            .with_exponent(2)
            .with_dimension(2)
            .build()
            .expect_err("non-monic should fail");

        assert_eq!(error, CoppersmithError::NonMonicPolynomial);
    }

    #[test]
    fn coppersmith_builder_rejects_constant_polynomial() {
        let polynomial = IntegerPolynomial::from_constant(BigInt::from(5));
        let error = CoppersmithLatticeBuilder::new(BigUint::from(91u8), polynomial)
            .with_bound(BigUint::from(4u8))
            .with_exponent(2)
            .with_dimension(2)
            .build()
            .expect_err("constant polynomial should fail");

        assert_eq!(error, CoppersmithError::ZeroDegreePolynomial);
    }

    #[test]
    fn coppersmith_builder_rejects_zero_bound() {
        let polynomial = IntegerPolynomial::new(vec![BigInt::from(7), BigInt::one()]);
        let error = CoppersmithLatticeBuilder::new(BigUint::from(91u8), polynomial)
            .with_bound(BigUint::zero())
            .with_exponent(2)
            .with_dimension(2)
            .build()
            .expect_err("zero bound should fail");

        assert_eq!(error, CoppersmithError::ZeroBound);
    }

    #[test]
    fn coppersmith_builder_rejects_small_dimension() {
        let polynomial = IntegerPolynomial::new(vec![BigInt::from(5), BigInt::zero(), BigInt::one()]);
        let error = CoppersmithLatticeBuilder::new(BigUint::from(91u8), polynomial)
            .with_bound(BigUint::from(4u8))
            .with_exponent(2)
            .with_dimension(3)
            .build()
            .expect_err("dimension should fail");

        assert_eq!(
            error,
            CoppersmithError::DimensionTooSmall {
                minimum: 4,
                requested: 3
            }
        );
    }

    #[test]
    fn coppersmith_vector_to_polynomial_requires_column_scaling() {
        let polynomial = IntegerPolynomial::new(vec![BigInt::from(12), BigInt::one()]);
        let lattice = CoppersmithLatticeBuilder::new(BigUint::from(77u8), polynomial)
            .with_bound(BigUint::from(8u8))
            .with_exponent(2)
            .with_dimension(4)
            .build()
            .expect("lattice");
        let invalid = Array1::from(vec![
            BigInt::from(1),
            BigInt::from(3),
            BigInt::from(64),
            BigInt::from(0),
        ]);

        let error = lattice
            .vector_to_polynomial(&invalid)
            .expect_err("vector should fail");

        assert_eq!(error, CoppersmithError::InvalidScaledVector { column: 1 });
    }

    #[test]
    fn rsa_coppersmith_recovers_small_unknown_tail() {
        let input = RsaCoppersmithInput {
            modulus: BigUint::from(11413u32),
            prime: BigUint::from(101u8),
            known_prefix: BigUint::zero(),
            unknown_part: BigUint::from(5u8),
            m: 3,
            dimension: 6,
        };

        let run = run_rsa_coppersmith(&input).expect("rsa run");

        assert_eq!(run.known_prefix, BigUint::from(96u8));
        assert_eq!(run.bound, BigUint::from(8u8));
        assert!(run.recovered_expected_unknown(&BigUint::from(5u8)));
        assert!(!run.reduced_polynomials.is_empty());
    }

    #[test]
    fn rsa_coppersmith_handles_zero_unknown_tail() {
        let input = RsaCoppersmithInput {
            modulus: BigUint::from(11413u32),
            prime: BigUint::from(101u8),
            known_prefix: BigUint::zero(),
            unknown_part: BigUint::zero(),
            m: 2,
            dimension: 4,
        };

        let run = run_rsa_coppersmith(&input).expect("rsa run");

        assert_eq!(run.bound, BigUint::one());
        assert!(run.recovered_expected_unknown(&BigUint::zero()));
    }

    #[test]
    fn rsa_coppersmith_rejects_oversized_unknown_tail() {
        let input = RsaCoppersmithInput {
            modulus: BigUint::from(11413u32),
            prime: BigUint::from(101u8),
            known_prefix: BigUint::zero(),
            unknown_part: BigUint::from(102u8),
            m: 2,
            dimension: 4,
        };

        let error = run_rsa_coppersmith(&input).expect_err("unknown tail should fail");

        assert_eq!(error, CoppersmithError::UnknownPartExceedsPrime);
    }

    #[test]
    fn rsa_coppersmith_rejects_prime_not_in_modulus() {
        let input = RsaCoppersmithInput {
            modulus: BigUint::from(11413u32),
            prime: BigUint::from(103u8),
            known_prefix: BigUint::zero(),
            unknown_part: BigUint::from(5u8),
            m: 2,
            dimension: 4,
        };

        let error = run_rsa_coppersmith(&input).expect_err("prime divisibility should fail");

        assert_eq!(error, CoppersmithError::PrimeDoesNotDivideModulus);
    }

    #[test]
    fn rsa_coppersmith_allows_zero_prime_when_known_prefix_is_supplied() {
        let input = RsaCoppersmithInput {
            modulus: BigUint::from(11413u32),
            prime: BigUint::zero(),
            known_prefix: BigUint::from(96u8),
            unknown_part: BigUint::from(5u8),
            m: 3,
            dimension: 6,
        };

        let run = run_rsa_coppersmith(&input).expect("rsa run");

        assert_eq!(run.known_prefix, BigUint::from(96u8));
        assert!(run.recovered_expected_unknown(&BigUint::from(5u8)));
    }
}
