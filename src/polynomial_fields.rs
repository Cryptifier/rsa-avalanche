/// Eclipse Public License 2.0
/// SPDX-License-Identifier: EPL-2.0
/// Copyright (c) 2025 Nicholas LaRoche <nlaroche@cryptifier.dev>
use num_bigint::BigUint;
use num_traits::{One, ToPrimitive, Zero};

use crate::config::{PolynomialFieldConfig, PolynomialFieldsConfig};
use crate::math::is_probable_prime_big;

const MIN_POLY_BITS: u64 = 8;
const MAX_POLY_BITS: u64 = 64;
const POLY_DEGREE: usize = 3;
const MAX_FIELDS: usize = 16;

/// Polynomial field definition used to score ciphertext residues.
#[derive(Debug, Clone)]
pub struct PolynomialField {
    prime: BigUint,
    _seed: u64,
    coefficients: Vec<BigUint>,
}

/// Coordinate produced by evaluating a polynomial field.
#[derive(Debug, Clone)]
pub struct PolynomialCoordinate {
    /// Label for the coordinate (X, Y, Z, ...).
    pub label: String,
    /// Normalized value in the range [0.0, 1.0].
    pub value: f64,
}

#[cfg(feature = "cluster")]
use linfa::prelude::Predict;
#[cfg(feature = "pca")]
use linfa::{
    DatasetBase,
    prelude::{Fit, Transformer},
};
#[cfg(feature = "cluster")]
use linfa_clustering::KMeans;
#[cfg(feature = "pca")]
use linfa_reduction::Pca;
#[cfg(feature = "cluster")]
use ndarray::Array1;
#[cfg(feature = "pca")]
use ndarray::Array2;
#[cfg(feature = "pca")]
use plotters::prelude::*;
#[cfg(feature = "pca")]
use std::{error::Error, path::Path};

/// Builds polynomial fields from configuration.
///
/// # Parameters
/// - `config`: Polynomial field configuration loaded from JSON.
///
/// # Returns
/// - `Result<Vec<PolynomialField>, String>`: Parsed polynomial fields or an error message.
///
/// # Expected Output
/// - Returns validated fields without side effects.
pub fn build_polynomial_fields(
    config: &PolynomialFieldsConfig,
) -> Result<Vec<PolynomialField>, String> {
    if config.fields.len() > MAX_FIELDS {
        return Err(format!(
            "polynomial field count {} exceeds max {}",
            config.fields.len(),
            MAX_FIELDS
        ));
    }

    config
        .fields
        .iter()
        .map(PolynomialField::from_config)
        .collect()
}

/// Generates normalized coordinates for a ciphertext using the given fields.
///
/// # Parameters
/// - `ciphertext`: Ciphertext value to score.
/// - `fields`: Polynomial fields used to compute coordinates.
///
/// # Returns
/// - `Vec<PolynomialCoordinate>`: Coordinate list in field order.
///
/// # Expected Output
/// - Returns normalized coordinates; no stdout/stderr output.
pub fn coordinates_for_ciphertext(
    ciphertext: &BigUint,
    fields: &[PolynomialField],
) -> Vec<PolynomialCoordinate> {
    fields
        .iter()
        .enumerate()
        .map(|(idx, field)| PolynomialCoordinate {
            label: coordinate_label(idx),
            value: field.score_normalized(ciphertext),
        })
        .collect()
}

/// Projects coordinate vectors into lower dimensions using PCA.
///
/// # Parameters
/// - `coordinates`: Input coordinate vectors (each vector length must match field count).
/// - `components`: Number of PCA components to compute.
///
/// # Returns
/// - `Result<Array2<f64>, String>`: PCA-transformed coordinates or an error message.
///
/// # Expected Output
/// - Returns projected coordinates; no stdout/stderr output.
#[cfg(feature = "pca")]
pub fn pca_project_coordinates(
    coordinates: &[Vec<f64>],
    components: usize,
) -> Result<Array2<f64>, String> {
    if coordinates.is_empty() {
        return Err("no coordinate vectors provided".to_string());
    }
    if components == 0 {
        return Err("components must be >= 1".to_string());
    }

    let rows = coordinates.len();
    let cols = coordinates[0].len();
    if cols == 0 {
        return Err("coordinate vectors must not be empty".to_string());
    }
    if coordinates.iter().any(|row| row.len() != cols) {
        return Err("coordinate vectors have inconsistent lengths".to_string());
    }

    let flat: Vec<f64> = coordinates
        .iter()
        .flat_map(|row| row.iter().copied())
        .collect();
    let matrix = Array2::from_shape_vec((rows, cols), flat)
        .map_err(|err| format!("invalid coordinate matrix: {err}"))?;

    let dataset = DatasetBase::from(matrix);
    let model = Pca::params(components)
        .fit(&dataset)
        .map_err(|err| format!("pca fit failed: {err}"))?;
    let projected = model.transform(dataset);
    Ok(projected.records)
}

/// Runs k-means clustering on PCA-projected coordinates.
///
/// # Parameters
/// - `points`: PCA-projected coordinates (rows are samples).
/// - `k`: Number of clusters to fit.
///
/// # Returns
/// - `Result<Array1<usize>, String>`: Cluster labels for each sample.
///
/// # Expected Output
/// - Returns cluster labels; no stdout/stderr output.
#[cfg(feature = "cluster")]
pub fn kmeans_cluster(points: &Array2<f64>, k: usize) -> Result<Array1<usize>, String> {
    if k == 0 {
        return Err("cluster count must be >= 1".to_string());
    }
    if points.nrows() == 0 {
        return Err("no points provided".to_string());
    }
    let dataset = DatasetBase::from(points.clone());
    let model = KMeans::params(k)
        .fit(&dataset)
        .map_err(|err| format!("kmeans fit failed: {err}"))?;
    let labels = model.predict(&dataset);
    Ok(labels)
}

/// Writes a PNG scatter plot for clustered PCA coordinates.
///
/// # Parameters
/// - `points`: PCA-projected coordinates with at least two columns.
/// - `labels`: Cluster labels for each point.
/// - `path`: Output PNG path.
///
/// # Returns
/// - `Result<(), Box<dyn Error>>`: `Ok(())` on success or an error on failure.
///
/// # Expected Output
/// - Writes a PNG to `path`; no stdout/stderr output.
#[cfg(feature = "cluster")]
pub fn plot_clustered_png(
    points: &Array2<f64>,
    labels: &Array1<usize>,
    path: &Path,
) -> Result<(), Box<dyn Error>> {
    if points.ncols() < 2 || points.nrows() == 0 {
        return Ok(());
    }
    if labels.len() != points.nrows() {
        return Err("label count does not match point count".into());
    }

    let xs: Vec<f64> = points.column(0).iter().copied().collect();
    let ys: Vec<f64> = points.column(1).iter().copied().collect();
    let (min_x, max_x) = min_max_range(&xs);
    let (min_y, max_y) = min_max_range(&ys);

    let root = BitMapBackend::new(path, (1000, 800)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption("PCA Projection (Clustered)", ("sans-serif", 30).into_font())
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(min_x..max_x, min_y..max_y)?;

    chart.configure_mesh().x_desc("PC1").y_desc("PC2").draw()?;

    chart.draw_series(
        xs.iter()
            .zip(ys.iter())
            .zip(labels.iter())
            .map(|((x, y), label)| {
                let color = cluster_color(*label);
                Circle::new((*x, *y), 3, color.filled())
            }),
    )?;

    root.present()?;
    Ok(())
}

/// Runs PCA and k-means clustering, then writes a clustered scatter PNG.
///
/// # Parameters
/// - `coordinates`: Input coordinate vectors to project and cluster.
/// - `components`: Number of PCA components to compute (must be >= 2).
/// - `k`: Number of clusters to fit.
/// - `path`: Output PNG path.
///
/// # Returns
/// - `Result<(), Box<dyn Error>>`: `Ok(())` on success or an error on failure.
///
/// # Expected Output
/// - Writes a PNG to `path`; no stdout/stderr output.
#[cfg(feature = "cluster")]
pub fn cluster_coordinates_to_png(
    coordinates: &[Vec<f64>],
    components: usize,
    k: usize,
    path: &Path,
) -> Result<(), Box<dyn Error>> {
    if components < 2 {
        return Err("components must be >= 2 for plotting".into());
    }
    let projected = pca_project_coordinates(coordinates, components)
        .map_err(|err| format!("pca projection failed: {err}"))?;
    let labels = kmeans_cluster(&projected, k).map_err(|err| format!("kmeans failed: {err}"))?;
    plot_clustered_png(&projected, &labels, path)
}

/// Writes a PNG scatter plot for PCA coordinates using the first two components.
///
/// # Parameters
/// - `points`: PCA-projected coordinates with at least two columns.
/// - `path`: Output PNG path.
///
/// # Returns
/// - `Result<(), Box<dyn Error>>`: `Ok(())` on success or an error on failure.
///
/// # Expected Output
/// - Writes a PNG to `path`; no stdout/stderr output.
#[cfg(feature = "pca")]
pub fn plot_pca_png(points: &Array2<f64>, path: &Path) -> Result<(), Box<dyn Error>> {
    if points.ncols() < 2 || points.nrows() == 0 {
        return Ok(());
    }

    let xs: Vec<f64> = points.column(0).iter().copied().collect();
    let ys: Vec<f64> = points.column(1).iter().copied().collect();
    let (min_x, max_x) = min_max_range(&xs);
    let (min_y, max_y) = min_max_range(&ys);

    let root = BitMapBackend::new(path, (1000, 800)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption("PCA Projection", ("sans-serif", 30).into_font())
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(min_x..max_x, min_y..max_y)?;

    chart.configure_mesh().x_desc("PC1").y_desc("PC2").draw()?;

    chart.draw_series(
        xs.iter()
            .zip(ys.iter())
            .map(|(x, y)| Circle::new((*x, *y), 3, BLUE.mix(0.7).filled())),
    )?;

    root.present()?;
    Ok(())
}

impl PolynomialField {
    /// Builds a polynomial field from config, validating prime size and coefficients.
    ///
    /// # Parameters
    /// - `config`: Field config containing prime modulus and seed.
    ///
    /// # Returns
    /// - `Result<PolynomialField, String>`: Parsed field or an error message.
    ///
    /// # Expected Output
    /// - Returns a validated field; no stdout/stderr output.
    pub fn from_config(config: &PolynomialFieldConfig) -> Result<Self, String> {
        let bits = config.prime.bits();
        if bits < MIN_POLY_BITS || bits > MAX_POLY_BITS {
            return Err(format!(
                "prime modulus bit length {} outside {}..={}",
                bits, MIN_POLY_BITS, MAX_POLY_BITS
            ));
        }
        if config.prime.is_zero() || config.prime.is_one() {
            return Err("prime modulus must be > 1".to_string());
        }
        if !is_probable_prime_big(&config.prime) {
            return Err(format!("prime modulus {} is not prime", config.prime));
        }

        let coefficients = generate_coefficients(config.seed, &config.prime, POLY_DEGREE);
        Ok(Self {
            prime: config.prime.clone(),
            _seed: config.seed,
            coefficients,
        })
    }

    fn score_normalized(&self, ciphertext: &BigUint) -> f64 {
        let value = self.score(ciphertext);
        let max_value = if self.prime > BigUint::one() {
            &self.prime - BigUint::one()
        } else {
            BigUint::zero()
        };
        if max_value.is_zero() {
            return 0.0;
        }

        let value_u128 = value.to_u128().unwrap_or(u128::MAX);
        let max_u128 = max_value.to_u128().unwrap_or(u128::MAX);
        if max_u128 == 0 {
            return 0.0;
        }

        let ratio = (value_u128.min(max_u128)) as f64 / max_u128 as f64;
        ratio.clamp(0.0, 1.0)
    }

    fn score(&self, ciphertext: &BigUint) -> BigUint {
        let x = ciphertext % &self.prime;
        evaluate_polynomial(&self.coefficients, &x, &self.prime)
    }
}

/// Generates polynomial coefficients from a seed within a modulus.
///
/// # Parameters
/// - `seed`: Seed value used to derive coefficients.
/// - `modulus`: Prime modulus defining coefficient bounds.
/// - `degree`: Polynomial degree to generate.
///
/// # Returns
/// - `Vec<BigUint>`: Coefficients in ascending degree order.
///
/// # Expected Output
/// - Returns a deterministic coefficient list; no stdout/stderr output.
fn generate_coefficients(seed: u64, modulus: &BigUint, degree: usize) -> Vec<BigUint> {
    let mut state = seed;
    let mut coeffs = Vec::with_capacity(degree + 1);
    for _ in 0..=degree {
        state = lcg_next(state);
        let coeff = BigUint::from(state) % modulus;
        coeffs.push(coeff);
    }
    coeffs
}

/// Advances the local LCG used for coefficient generation.
///
/// # Parameters
/// - `state`: Current LCG state.
///
/// # Returns
/// - `u64`: Next LCG state.
///
/// # Expected Output
/// - Returns the next state; no stdout/stderr output.
fn lcg_next(state: u64) -> u64 {
    state.wrapping_mul(6364136223846793005).wrapping_add(1)
}

/// Evaluates a polynomial at `x` within the provided modulus.
///
/// # Parameters
/// - `coeffs`: Coefficients in ascending degree order.
/// - `x`: Input value to evaluate.
/// - `modulus`: Prime modulus for field arithmetic.
///
/// # Returns
/// - `BigUint`: Polynomial value reduced modulo `modulus`.
///
/// # Expected Output
/// - Returns the reduced polynomial value; no stdout/stderr output.
fn evaluate_polynomial(coeffs: &[BigUint], x: &BigUint, modulus: &BigUint) -> BigUint {
    let mut result = BigUint::zero();
    let mut power = BigUint::one();
    for coeff in coeffs {
        let term = (coeff * &power) % modulus;
        result = (result + term) % modulus;
        power = (&power * x) % modulus;
    }
    result
}

/// Returns a coordinate label for the given index.
///
/// # Parameters
/// - `index`: Coordinate index.
///
/// # Returns
/// - `String`: Coordinate label string.
///
/// # Expected Output
/// - Returns a label with no side effects.
fn coordinate_label(index: usize) -> String {
    const LABELS: [&str; 16] = [
        "X", "Y", "Z", "W", "V", "U", "T", "S", "R", "Q", "P", "O", "N", "M", "L", "K",
    ];
    if index < LABELS.len() {
        LABELS[index].to_string()
    } else {
        format!("C{}", index + 1)
    }
}

#[cfg(feature = "pca")]
/// Computes an inclusive min/max range for plotting.
///
/// # Parameters
/// - `values`: Input slice of values.
///
/// # Returns
/// - `(f64, f64)`: Min and max values (with padding if constant).
///
/// # Expected Output
/// - Returns a range tuple; no stdout/stderr output.
#[cfg(feature = "pca")]
fn min_max_range(values: &[f64]) -> (f64, f64) {
    if values.is_empty() {
        return (0.0, 1.0);
    }
    let mut min = values[0];
    let mut max = values[0];
    for val in values.iter().copied() {
        if val < min {
            min = val;
        }
        if val > max {
            max = val;
        }
    }
    if (max - min).abs() < f64::EPSILON {
        (min - 1.0, max + 1.0)
    } else {
        (min, max)
    }
}

#[cfg(feature = "cluster")]
fn cluster_color(label: usize) -> RGBColor {
    const COLORS: [RGBColor; 10] = [
        RGBColor(31, 119, 180),
        RGBColor(255, 127, 14),
        RGBColor(44, 160, 44),
        RGBColor(214, 39, 40),
        RGBColor(148, 103, 189),
        RGBColor(140, 86, 75),
        RGBColor(227, 119, 194),
        RGBColor(127, 127, 127),
        RGBColor(188, 189, 34),
        RGBColor(23, 190, 207),
    ];
    COLORS[label % COLORS.len()]
}

#[cfg(test)]
mod tests {
    use super::*;
    use insta::assert_yaml_snapshot;

    fn make_config(primes: &[u64], seeds: &[u64]) -> PolynomialFieldsConfig {
        let fields = primes
            .iter()
            .zip(seeds.iter())
            .map(|(prime, seed)| PolynomialFieldConfig {
                prime: BigUint::from(*prime),
                seed: *seed,
            })
            .collect();
        PolynomialFieldsConfig { fields }
    }

    fn snapshot_coords(ciphertext: u64, primes: &[u64], seeds: &[u64]) -> Vec<String> {
        let config = make_config(primes, seeds);
        let fields = build_polynomial_fields(&config).expect("fields");
        let coords = coordinates_for_ciphertext(&BigUint::from(ciphertext), &fields);
        coords
            .iter()
            .map(|coord| format!("{}:{:.8}", coord.label, coord.value))
            .collect()
    }

    #[test]
    fn test_coordinates_vector_1() {
        let out = snapshot_coords(123, &[251], &[1]);
        assert_yaml_snapshot!(out);
    }

    #[test]
    fn test_coordinates_vector_2() {
        let out = snapshot_coords(456, &[251, 257], &[1, 2]);
        assert_yaml_snapshot!(out);
    }

    #[test]
    fn test_coordinates_vector_3() {
        let out = snapshot_coords(789, &[251, 257, 263], &[1, 2, 3]);
        assert_yaml_snapshot!(out);
    }

    #[test]
    fn test_coordinates_vector_4() {
        let out = snapshot_coords(1024, &[251, 257, 263, 269], &[1, 2, 3, 4]);
        assert_yaml_snapshot!(out);
    }

    #[test]
    fn test_coordinates_vector_5() {
        let out = snapshot_coords(2048, &[251, 257, 263, 269, 271], &[1, 2, 3, 4, 5]);
        assert_yaml_snapshot!(out);
    }

    #[test]
    fn test_coordinates_vector_6() {
        let out = snapshot_coords(4096, &[251, 257, 263, 269, 271, 277], &[1, 2, 3, 4, 5, 6]);
        assert_yaml_snapshot!(out);
    }

    #[test]
    fn test_coordinates_vector_7() {
        let out = snapshot_coords(
            8192,
            &[251, 257, 263, 269, 271, 277, 281],
            &[1, 2, 3, 4, 5, 6, 7],
        );
        assert_yaml_snapshot!(out);
    }

    #[test]
    fn test_coordinates_vector_8() {
        let out = snapshot_coords(
            16384,
            &[251, 257, 263, 269, 271, 277, 281, 283],
            &[1, 2, 3, 4, 5, 6, 7, 8],
        );
        assert_yaml_snapshot!(out);
    }

    #[test]
    fn test_coordinates_vector_9() {
        let out = snapshot_coords(
            32768,
            &[251, 257, 263, 269, 271, 277, 281, 283, 293],
            &[1, 2, 3, 4, 5, 6, 7, 8, 9],
        );
        assert_yaml_snapshot!(out);
    }

    #[test]
    fn test_coordinates_vector_10() {
        let out = snapshot_coords(
            65535,
            &[251, 257, 263, 269, 271, 277, 281, 283, 293, 307],
            &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        );
        assert_yaml_snapshot!(out);
    }

    #[cfg(feature = "pca")]
    #[test]
    fn test_pca_projection_with_large_vectors() {
        let primes: Vec<u64> = vec![
            251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 349,
        ];
        let seeds: Vec<u64> = (1..=16).collect();
        let config = make_config(&primes, &seeds);
        let fields = build_polynomial_fields(&config).expect("fields");

        let base: BigUint = BigUint::one() << 120;
        let step: BigUint = BigUint::from(1_234_567u64);
        let mut vectors = Vec::with_capacity(100);
        for idx in 0..100u64 {
            let addend = &step * BigUint::from(idx);
            let mut ciphertext = base.clone();
            ciphertext += addend;
            let coords = coordinates_for_ciphertext(&ciphertext, &fields);
            vectors.push(coords.into_iter().map(|c| c.value).collect::<Vec<_>>());
        }

        let projected = pca_project_coordinates(&vectors, 2).expect("pca");
        assert_eq!(projected.nrows(), 100);
        assert_eq!(projected.ncols(), 2);
        assert!(projected.iter().all(|v| v.is_finite()));

        let path = Path::new("images/pca_projection.png");
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).expect("create images dir");
        }
        plot_pca_png(&projected, path).expect("plot");
        let meta = std::fs::metadata(path).expect("metadata");
        assert!(meta.len() > 0);
    }

    #[cfg(feature = "cluster")]
    #[test]
    fn test_cluster_projection_with_large_vectors() {
        let primes: Vec<u64> = vec![
            251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 349,
        ];
        let seeds: Vec<u64> = (1..=16).collect();
        let config = make_config(&primes, &seeds);
        let fields = build_polynomial_fields(&config).expect("fields");

        let base: BigUint = BigUint::one() << 128;
        let step: BigUint = BigUint::from(9_876_543u64);
        let mut vectors = Vec::with_capacity(100);
        for idx in 0..100u64 {
            let addend = &step * BigUint::from(idx);
            let mut ciphertext = base.clone();
            ciphertext += addend;
            let coords = coordinates_for_ciphertext(&ciphertext, &fields);
            vectors.push(coords.into_iter().map(|c| c.value).collect::<Vec<_>>());
        }

        let path = Path::new("images/pca_cluster_projection.png");
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).expect("create images dir");
        }
        cluster_coordinates_to_png(&vectors, 2, 4, path).expect("cluster plot");
        let meta = std::fs::metadata(path).expect("metadata");
        assert!(meta.len() > 0);
    }
}
