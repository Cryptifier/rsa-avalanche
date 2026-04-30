/// Eclipse Public License 2.0
/// SPDX-License-Identifier: EPL-2.0
/// Copyright (c) 2025 Nicholas LaRoche <nlaroche@cryptifier.dev>
use ndarray::{Array1, Array2};
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Zero};

pub type BigIntMatrix = Array2<BigInt>;
pub type BigIntVector = Array1<BigInt>;

#[allow(dead_code)]
fn dot_i(a: &BigIntVector, b: &BigIntVector) -> BigInt {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| x * y)
        .fold(BigInt::zero(), |acc, v| acc + v)
}

fn dot_q(a: &[BigRational], b: &[BigRational]) -> BigRational {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| x * y)
        .fold(BigRational::zero(), |acc, v| acc + v)
}

fn nearest_integer(x: &BigRational) -> BigInt {
    let n = x.numer().clone();
    let d = x.denom().clone();

    if n.sign() == num_bigint::Sign::Minus {
        -nearest_integer(&(-x.clone()))
    } else {
        (&n + (&d / 2)) / d
    }
}

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

fn size_reduce(basis: &mut [BigIntVector], k: usize, l: usize, q: &BigInt) {
    if q.is_zero() {
        return;
    }

    let row_l = basis[l].clone();

    for i in 0..basis[k].len() {
        basis[k][i] -= q * &row_l[i];
    }
}

pub fn lll_reduce(input: &BigIntMatrix) -> Vec<BigIntVector> {
    lll_reduce_delta(input, BigRational::new(BigInt::from(3), BigInt::from(4)))
}

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
        let (_, mut mu, mut _norm) = gram_schmidt(&basis);

        for j in (0..k).rev() {
            let q = nearest_integer(&mu[k][j]);

            if !q.is_zero() {
                size_reduce(&mut basis, k, j, &q);
            }
        }

        let (_, mu2, _norm2) = gram_schmidt(&basis);
        mu = mu2;
        _norm = _norm2;

        let lhs = _norm[k].clone();
        let rhs = (&delta - &mu[k][k - 1] * &mu[k][k - 1]) * &_norm[k - 1];

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

pub fn as_matrix(vectors: &[BigIntVector]) -> BigIntMatrix {
    if vectors.is_empty() {
        return Array2::from_shape_vec((0, 0), vec![]).unwrap();
    }

    let rows = vectors.len();
    let cols = vectors[0].len();

    let flat: Vec<BigInt> = vectors.iter().flat_map(|row| row.iter().cloned()).collect();

    Array2::from_shape_vec((rows, cols), flat).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_lll_small_basis() {
        let basis = array![
            [BigInt::from(105), BigInt::from(821), BigInt::from(404)],
            [BigInt::from(331), BigInt::from(569), BigInt::from(074)],
            [BigInt::from(511), BigInt::from(322), BigInt::from(912)],
        ];

        let reduced = lll_reduce(&basis);

        assert_eq!(reduced.len(), 3);
        assert_eq!(reduced[0].len(), 3);
    }
}
