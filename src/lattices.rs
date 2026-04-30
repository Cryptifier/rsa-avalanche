/// Eclipse Public License 2.0
/// SPDX-License-Identifier: EPL-2.0
/// Copyright (c) 2025 Nicholas LaRoche <nlaroche@cryptifier.dev>
use ndarray::{Array1, Array2};
use num_bigint::{BigInt, BigUint};
use num_rational::BigRational;
use num_traits::{One, Zero};
use std::error::Error;
use std::fmt::{Display, Formatter};

use crate::polynomials::IntegerPolynomial;

/// Integer matrix used for lattice basis construction.
pub type BigIntMatrix = Array2<BigInt>;

/// Integer vector used for lattice rows and reduced basis vectors.
pub type BigIntVector = Array1<BigInt>;

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
        let degree = self
            .polynomial
            .degree()
            .ok_or(CoppersmithError::ZeroDegreePolynomial)?;
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
    let degree = polynomial
        .degree()
        .ok_or(CoppersmithError::ZeroDegreePolynomial)?;

    if degree == 0 {
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
