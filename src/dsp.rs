/**
 * DSP module for detecting ramp signals in binned data.
 * A ramp signal is defined as a sequence of values that increase by 1 from the mean.
 * The module provides functions to find ramps and calculate their strength.
 */

fn mean_i32(data: &[i32]) -> f64 {
    let sum: i32 = data.iter().sum();
    sum as f64 / data.len() as f64
}

fn mean_f64(data: &[f64]) -> f64 {
    let sum: f64 = data.iter().sum();
    sum / data.len() as f64
}

pub fn find_ramp_signals(
    bins: &[i32],
    ramp_length: usize,
    tolerance: i32,
) -> Vec<(usize, usize, Vec<i32>)> {
    let avg = mean_i32(bins).round() as i32;
    let mut ramps = Vec::new();
    let n = bins.len();
    let mut i = 0;
    while i + ramp_length <= n {
        let mut match_ramp = true;
        let mut ramp_vals = Vec::new();
        for j in 0..ramp_length {
            let expected = avg + (j as i32) + 1;
            if (bins[i + j] - expected).abs() > tolerance {
                match_ramp = false;
                break;
            }
            ramp_vals.push(bins[i + j]);
        }
        if match_ramp {
            ramps.push((i, ramp_length, ramp_vals));
            i += ramp_length; // Skip overlapping
        } else {
            i += 1;
        }
    }
    ramps
}

pub fn find_ramp_signals_f64(
    bins: &[f64],
    ramp_length: usize,
    step: f64,
    tolerance: f64,
) -> Vec<(usize, usize, Vec<f64>)> {
    if bins.is_empty() || ramp_length == 0 {
        return Vec::new();
    }
    let avg = mean_f64(bins);
    let center_offset = (ramp_length as f64 - 1.0) / 2.0;
    let mut ramps = Vec::new();
    let n = bins.len();
    let mut i = 0;
    while i + ramp_length <= n {
        let mut match_ramp = true;
        let mut ramp_vals = Vec::new();
        for j in 0..ramp_length {
            let expected = avg + ((j as f64) - center_offset) * step;
            if (bins[i + j] - expected).abs() > tolerance {
                match_ramp = false;
                break;
            }
            ramp_vals.push(bins[i + j]);
        }
        if match_ramp {
            ramps.push((i, ramp_length, ramp_vals));
            i += ramp_length;
        } else {
            i += 1;
        }
    }
    ramps
}

pub fn ramp_signal_strength(ramps: &[(usize, usize, Vec<i32>)]) -> usize {
    ramps.iter().map(|(_, len, _)| *len).sum()
}

pub fn ramp_signal_strength_f64(ramps: &[(usize, usize, Vec<f64>)]) -> usize {
    ramps.iter().map(|(_, len, _)| *len).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ramp_detection() {
        let bins = vec![11, 12, 13, 14, 8, 9, 10, 11, 12];
        // mean is 11, so ramp is 12, 13, 14
        let ramps = find_ramp_signals(&bins, 3, 0);
        assert!(!ramps.is_empty(), "No ramps found: {:?}", ramps);
    }

    #[test]
    fn test_ramp_strength() {
        let ramps = vec![(0, 3, vec![12, 13, 14]), (5, 3, vec![9, 10, 11])];
        let strength = ramp_signal_strength(&ramps);
        assert_eq!(strength, 6, "Expected strength 6, got {}", strength);
    }

    #[test]
    fn test_detect_and_print_ramp() {
        // Example bins: mean is 10, ramp should be 11, 12, 13
        let bins = vec![8, 9, 10, 11, 12, 13, 7, 8];
        let ramps = find_ramp_signals(&bins, 3, 0);
        println!("Detected ramps: {:?}", ramps);
        let strength = ramp_signal_strength(&ramps);
        println!("Signal strength: {}", strength);

        // Check that at least one ramp is detected and signal strength is correct
        assert!(!ramps.is_empty());
        assert!(strength > 0);
    }

    #[test]
    fn test_ramp_detection_float_centered() {
        let bins = vec![49.00, 49.05, 49.10];
        let ramps = find_ramp_signals_f64(&bins, 3, 0.05, 0.001);
        assert!(!ramps.is_empty(), "No float ramps found: {:?}", ramps);
        let strength = ramp_signal_strength_f64(&ramps);
        assert!(strength > 0);
    }
}
