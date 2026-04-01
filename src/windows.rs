use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use std::error::Error;
use std::fmt;

/// Default seed used by helper constructors when a caller does not provide one.
pub const DEFAULT_WINDOW_SEED: u64 = 0;

/// Indicates whether randomly requested partitions may overlap before gap filling.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum WindowOverlapPolicy {
    AllowOverlap,
    DisallowOverlap,
}

/// Indicates whether a partition came from the requested random set or from gap filling.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum WindowPartitionKind {
    Requested,
    GapFill,
}

/// Stores a partition start index, width, and origin within a generated window series.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct WindowPartition {
    index: usize,
    width: usize,
    kind: WindowPartitionKind,
}

impl WindowPartition {
    /// Creates a partition with an inclusive start index and width in bits.
    ///
    /// # Parameters
    /// - `index`: Inclusive starting bit index for the partition.
    /// - `width`: Width of the partition in bits.
    /// - `kind`: Origin of the partition within the generated series.
    ///
    /// # Returns
    /// - `WindowPartition`: Partition descriptor.
    ///
    /// # Expected Output
    /// - Returns a partition value; no side effects.
    pub fn new(index: usize, width: usize, kind: WindowPartitionKind) -> Self {
        Self { index, width, kind }
    }

    /// Returns the inclusive starting bit index for the partition.
    ///
    /// # Parameters
    /// - None.
    ///
    /// # Returns
    /// - `usize`: Partition start index.
    ///
    /// # Expected Output
    /// - Returns the stored start index; no side effects.
    pub fn index(&self) -> usize {
        self.index
    }

    /// Returns the width of the partition in bits.
    ///
    /// # Parameters
    /// - None.
    ///
    /// # Returns
    /// - `usize`: Partition width.
    ///
    /// # Expected Output
    /// - Returns the stored width; no side effects.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Returns the partition origin.
    ///
    /// # Parameters
    /// - None.
    ///
    /// # Returns
    /// - `WindowPartitionKind`: Origin of the partition.
    ///
    /// # Expected Output
    /// - Returns the stored kind; no side effects.
    pub fn kind(&self) -> WindowPartitionKind {
        self.kind
    }

    /// Returns the exclusive end bit index for the partition.
    ///
    /// # Parameters
    /// - None.
    ///
    /// # Returns
    /// - `usize`: Exclusive end index.
    ///
    /// # Expected Output
    /// - Returns the computed end index; no side effects.
    pub fn end_exclusive(&self) -> usize {
        self.index + self.width
    }
}

/// Stores a fully covered series of generated partitions over a bit range.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WindowPartitionSet {
    total_bits: usize,
    requested_count: usize,
    max_width: usize,
    seed: u64,
    overlap_policy: WindowOverlapPolicy,
    partitions: Vec<WindowPartition>,
}

impl WindowPartitionSet {
    /// Starts a builder for a new partition series.
    ///
    /// # Parameters
    /// - `total_bits`: Total number of bits that must be covered by the final series.
    ///
    /// # Returns
    /// - `WindowPartitionSetBuilder`: Builder with default seed and overlap-allowed mode.
    ///
    /// # Expected Output
    /// - Returns a builder value; no side effects.
    pub fn build(total_bits: usize) -> WindowPartitionSetBuilder {
        WindowPartitionSetBuilder::new(total_bits)
    }

    /// Returns the total number of bits covered by the series.
    ///
    /// # Parameters
    /// - None.
    ///
    /// # Returns
    /// - `usize`: Target bit width of the series.
    ///
    /// # Expected Output
    /// - Returns the configured bit width; no side effects.
    pub fn total_bits(&self) -> usize {
        self.total_bits
    }

    /// Returns the number of randomly requested partitions before gap filling.
    ///
    /// # Parameters
    /// - None.
    ///
    /// # Returns
    /// - `usize`: Requested random partition count.
    ///
    /// # Expected Output
    /// - Returns the stored count; no side effects.
    pub fn requested_count(&self) -> usize {
        self.requested_count
    }

    /// Returns the configured maximum partition width.
    ///
    /// # Parameters
    /// - None.
    ///
    /// # Returns
    /// - `usize`: Maximum width in bits.
    ///
    /// # Expected Output
    /// - Returns the stored maximum width; no side effects.
    pub fn max_width(&self) -> usize {
        self.max_width
    }

    /// Returns the `u64` seed used to create the series.
    ///
    /// # Parameters
    /// - None.
    ///
    /// # Returns
    /// - `u64`: Seed used with `ChaCha20Rng::seed_from_u64`.
    ///
    /// # Expected Output
    /// - Returns the stored seed; no side effects.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Returns the configured overlap policy.
    ///
    /// # Parameters
    /// - None.
    ///
    /// # Returns
    /// - `WindowOverlapPolicy`: Overlap behavior for requested partitions.
    ///
    /// # Expected Output
    /// - Returns the stored policy; no side effects.
    pub fn overlap_policy(&self) -> WindowOverlapPolicy {
        self.overlap_policy
    }

    /// Returns all partitions in ascending start-index order.
    ///
    /// # Parameters
    /// - None.
    ///
    /// # Returns
    /// - `&[WindowPartition]`: Complete partition slice including gap fillers.
    ///
    /// # Expected Output
    /// - Returns a shared slice; no side effects.
    pub fn partitions(&self) -> &[WindowPartition] {
        &self.partitions
    }

    /// Returns an iterator over the initially requested random partitions.
    ///
    /// # Parameters
    /// - None.
    ///
    /// # Returns
    /// - `impl Iterator<Item = &WindowPartition>`: Iterator over requested partitions.
    ///
    /// # Expected Output
    /// - Returns a filtered iterator; no side effects.
    pub fn requested_partitions(&self) -> impl Iterator<Item = &WindowPartition> {
        self.partitions
            .iter()
            .filter(|partition| partition.kind() == WindowPartitionKind::Requested)
    }

    /// Returns an iterator over the partitions added to fill uncovered gaps.
    ///
    /// # Parameters
    /// - None.
    ///
    /// # Returns
    /// - `impl Iterator<Item = &WindowPartition>`: Iterator over gap-fill partitions.
    ///
    /// # Expected Output
    /// - Returns a filtered iterator; no side effects.
    pub fn gap_fill_partitions(&self) -> impl Iterator<Item = &WindowPartition> {
        self.partitions
            .iter()
            .filter(|partition| partition.kind() == WindowPartitionKind::GapFill)
    }

    /// Returns the total number of stored partitions including gap fillers.
    ///
    /// # Parameters
    /// - None.
    ///
    /// # Returns
    /// - `usize`: Partition count.
    ///
    /// # Expected Output
    /// - Returns the stored length; no side effects.
    pub fn len(&self) -> usize {
        self.partitions.len()
    }

    /// Returns whether the series stores no partitions.
    ///
    /// # Parameters
    /// - None.
    ///
    /// # Returns
    /// - `bool`: `true` when the partition list is empty.
    ///
    /// # Expected Output
    /// - Returns a derived flag; no side effects.
    pub fn is_empty(&self) -> bool {
        self.partitions.is_empty()
    }

    /// Returns whether every bit in the configured range is covered.
    ///
    /// # Parameters
    /// - None.
    ///
    /// # Returns
    /// - `bool`: `true` when no bit is uncovered.
    ///
    /// # Expected Output
    /// - Returns a derived flag; no side effects.
    pub fn covers_all_bits(&self) -> bool {
        uncovered_ranges(self.total_bits, &self.partitions).is_empty()
    }

    /// Returns whether any stored partitions overlap.
    ///
    /// # Parameters
    /// - None.
    ///
    /// # Returns
    /// - `bool`: `true` when two partitions cover at least one common bit.
    ///
    /// # Expected Output
    /// - Returns a derived flag; no side effects.
    pub fn has_overlaps(&self) -> bool {
        partitions_overlap(&self.partitions)
    }
}

/// Builds a partition series with seeded ChaCha20-based generation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WindowPartitionSetBuilder {
    total_bits: usize,
    requested_count: usize,
    max_width: usize,
    seed: u64,
    overlap_policy: WindowOverlapPolicy,
}

impl WindowPartitionSetBuilder {
    /// Creates a builder with default seed and overlap-allowed generation.
    ///
    /// # Parameters
    /// - `total_bits`: Total number of bits that must be covered by the final series.
    ///
    /// # Returns
    /// - `WindowPartitionSetBuilder`: Builder with default configuration.
    ///
    /// # Expected Output
    /// - Returns a builder value; no side effects.
    pub fn new(total_bits: usize) -> Self {
        Self {
            total_bits,
            requested_count: 0,
            max_width: total_bits.max(1),
            seed: DEFAULT_WINDOW_SEED,
            overlap_policy: WindowOverlapPolicy::AllowOverlap,
        }
    }

    /// Sets the deterministic seed used to initialize `ChaCha20Rng`.
    ///
    /// # Parameters
    /// - `seed`: Seed value passed to `ChaCha20Rng::seed_from_u64`.
    ///
    /// # Returns
    /// - `WindowPartitionSetBuilder`: Updated builder.
    ///
    /// # Expected Output
    /// - Returns an updated builder; no side effects.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Sets the number of random partitions to request before filling gaps.
    ///
    /// # Parameters
    /// - `requested_count`: Number of random partitions to generate before enforcing full coverage.
    ///
    /// # Returns
    /// - `WindowPartitionSetBuilder`: Updated builder.
    ///
    /// # Expected Output
    /// - Returns an updated builder; no side effects.
    pub fn count(mut self, requested_count: usize) -> Self {
        self.requested_count = requested_count;
        self
    }

    /// Sets the maximum width for generated partitions.
    ///
    /// # Parameters
    /// - `max_width`: Upper bound for partition width in bits.
    ///
    /// # Returns
    /// - `WindowPartitionSetBuilder`: Updated builder.
    ///
    /// # Expected Output
    /// - Returns an updated builder; no side effects.
    pub fn max_width(mut self, max_width: usize) -> Self {
        self.max_width = max_width;
        self
    }

    /// Sets the overlap policy used for the requested random partitions.
    ///
    /// # Parameters
    /// - `policy`: Overlap behavior for the requested random partitions.
    ///
    /// # Returns
    /// - `WindowPartitionSetBuilder`: Updated builder.
    ///
    /// # Expected Output
    /// - Returns an updated builder; no side effects.
    pub fn overlap_policy(mut self, policy: WindowOverlapPolicy) -> Self {
        self.overlap_policy = policy;
        self
    }

    /// Configures the builder to allow overlaps among requested partitions.
    ///
    /// # Parameters
    /// - None.
    ///
    /// # Returns
    /// - `WindowPartitionSetBuilder`: Updated builder.
    ///
    /// # Expected Output
    /// - Returns an updated builder; no side effects.
    pub fn allow_overlap(self) -> Self {
        self.overlap_policy(WindowOverlapPolicy::AllowOverlap)
    }

    /// Configures the builder to disallow overlaps among requested partitions.
    ///
    /// # Parameters
    /// - None.
    ///
    /// # Returns
    /// - `WindowPartitionSetBuilder`: Updated builder.
    ///
    /// # Expected Output
    /// - Returns an updated builder; no side effects.
    pub fn disallow_overlap(self) -> Self {
        self.overlap_policy(WindowOverlapPolicy::DisallowOverlap)
    }

    /// Generates a fully covered partition set using the configured options.
    ///
    /// # Parameters
    /// - None.
    ///
    /// # Returns
    /// - `Result<WindowPartitionSet, WindowBuildError>`: Fully covered partition series or a configuration error.
    ///
    /// # Expected Output
    /// - Returns the generated partition series; no side effects.
    pub fn generate(self) -> Result<WindowPartitionSet, WindowBuildError> {
        validate_builder(&self)?;

        let effective_max_width = self.max_width.min(self.total_bits);
        let mut rng = ChaCha20Rng::seed_from_u64(self.seed);
        let mut partitions = match self.overlap_policy {
            WindowOverlapPolicy::AllowOverlap => build_requested_partitions_with_overlap(
                self.total_bits,
                self.requested_count,
                effective_max_width,
                &mut rng,
            ),
            WindowOverlapPolicy::DisallowOverlap => build_requested_partitions_without_overlap(
                self.total_bits,
                self.requested_count,
                effective_max_width,
                &mut rng,
            ),
        };

        let gaps = uncovered_ranges(self.total_bits, &partitions);
        for gap in gaps {
            partitions.extend(split_gap_into_partitions(
                gap.index,
                gap.width,
                effective_max_width,
                &mut rng,
            ));
        }

        partitions.sort_by_key(|partition| (partition.index, partition.kind, partition.width));

        Ok(WindowPartitionSet {
            total_bits: self.total_bits,
            requested_count: self.requested_count,
            max_width: effective_max_width,
            seed: self.seed,
            overlap_policy: self.overlap_policy,
            partitions,
        })
    }
}

/// Describes configuration errors for partition generation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WindowBuildError {
    TotalBitsMustBePositive,
    MaxWidthMustBePositive,
    RequestedCountExceedsTotalBits {
        requested_count: usize,
        total_bits: usize,
    },
}

impl fmt::Display for WindowBuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TotalBitsMustBePositive => {
                write!(f, "total_bits must be positive")
            }
            Self::MaxWidthMustBePositive => {
                write!(f, "max_width must be positive")
            }
            Self::RequestedCountExceedsTotalBits {
                requested_count,
                total_bits,
            } => write!(
                f,
                "requested_count ({requested_count}) exceeds total_bits ({total_bits}) for non-overlapping partitions"
            ),
        }
    }
}

impl Error for WindowBuildError {}

/// Generates a fully covered partition set with the default seed and overlap-allowed mode.
///
/// # Parameters
/// - `total_bits`: Total number of bits that must be covered by the final series.
/// - `requested_count`: Number of random partitions to request before gap filling.
/// - `max_width`: Maximum width for any generated partition.
///
/// # Returns
/// - `Result<WindowPartitionSet, WindowBuildError>`: Fully covered partition series or a configuration error.
///
/// # Expected Output
/// - Returns the generated partition series; no side effects.
pub fn generate_windows(
    total_bits: usize,
    requested_count: usize,
    max_width: usize,
) -> Result<WindowPartitionSet, WindowBuildError> {
    WindowPartitionSet::build(total_bits)
        .count(requested_count)
        .max_width(max_width)
        .allow_overlap()
        .generate()
}

/// Generates a fully covered partition set with a caller-provided seed and overlap-allowed mode.
///
/// # Parameters
/// - `total_bits`: Total number of bits that must be covered by the final series.
/// - `requested_count`: Number of random partitions to request before gap filling.
/// - `max_width`: Maximum width for any generated partition.
/// - `seed`: Seed passed to `ChaCha20Rng::seed_from_u64`.
///
/// # Returns
/// - `Result<WindowPartitionSet, WindowBuildError>`: Fully covered partition series or a configuration error.
///
/// # Expected Output
/// - Returns the generated partition series; no side effects.
pub fn generate_windows_with_seed(
    total_bits: usize,
    requested_count: usize,
    max_width: usize,
    seed: u64,
) -> Result<WindowPartitionSet, WindowBuildError> {
    WindowPartitionSet::build(total_bits)
        .seed(seed)
        .count(requested_count)
        .max_width(max_width)
        .allow_overlap()
        .generate()
}

/// Generates a fully covered non-overlapping partition set with the default seed.
///
/// # Parameters
/// - `total_bits`: Total number of bits that must be covered by the final series.
/// - `requested_count`: Number of random partitions to request before gap filling.
/// - `max_width`: Maximum width for any generated partition.
///
/// # Returns
/// - `Result<WindowPartitionSet, WindowBuildError>`: Fully covered partition series or a configuration error.
///
/// # Expected Output
/// - Returns the generated partition series; no side effects.
pub fn generate_non_overlapping_windows(
    total_bits: usize,
    requested_count: usize,
    max_width: usize,
) -> Result<WindowPartitionSet, WindowBuildError> {
    WindowPartitionSet::build(total_bits)
        .count(requested_count)
        .max_width(max_width)
        .disallow_overlap()
        .generate()
}

/// Generates a fully covered non-overlapping partition set with a caller-provided seed.
///
/// # Parameters
/// - `total_bits`: Total number of bits that must be covered by the final series.
/// - `requested_count`: Number of random partitions to request before gap filling.
/// - `max_width`: Maximum width for any generated partition.
/// - `seed`: Seed passed to `ChaCha20Rng::seed_from_u64`.
///
/// # Returns
/// - `Result<WindowPartitionSet, WindowBuildError>`: Fully covered partition series or a configuration error.
///
/// # Expected Output
/// - Returns the generated partition series; no side effects.
pub fn generate_non_overlapping_windows_with_seed(
    total_bits: usize,
    requested_count: usize,
    max_width: usize,
    seed: u64,
) -> Result<WindowPartitionSet, WindowBuildError> {
    WindowPartitionSet::build(total_bits)
        .seed(seed)
        .count(requested_count)
        .max_width(max_width)
        .disallow_overlap()
        .generate()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct BitRange {
    index: usize,
    width: usize,
}

/// Validates builder configuration before partition generation.
///
/// # Parameters
/// - `builder`: Builder configuration to validate.
///
/// # Returns
/// - `Result<(), WindowBuildError>`: `Ok(())` when the configuration is valid.
///
/// # Expected Output
/// - Returns a validation result; no side effects.
fn validate_builder(builder: &WindowPartitionSetBuilder) -> Result<(), WindowBuildError> {
    if builder.total_bits == 0 {
        return Err(WindowBuildError::TotalBitsMustBePositive);
    }

    if builder.max_width == 0 {
        return Err(WindowBuildError::MaxWidthMustBePositive);
    }

    if builder.overlap_policy == WindowOverlapPolicy::DisallowOverlap
        && builder.requested_count > builder.total_bits
    {
        return Err(WindowBuildError::RequestedCountExceedsTotalBits {
            requested_count: builder.requested_count,
            total_bits: builder.total_bits,
        });
    }

    Ok(())
}

/// Builds the initially requested random partitions when overlaps are allowed.
///
/// # Parameters
/// - `total_bits`: Total number of bits in the target domain.
/// - `requested_count`: Number of requested random partitions.
/// - `max_width`: Maximum partition width in bits.
/// - `rng`: Seeded ChaCha20 RNG used for random choices.
///
/// # Returns
/// - `Vec<WindowPartition>`: Requested random partitions.
///
/// # Expected Output
/// - Returns the generated requested partitions; no side effects.
fn build_requested_partitions_with_overlap(
    total_bits: usize,
    requested_count: usize,
    max_width: usize,
    rng: &mut ChaCha20Rng,
) -> Vec<WindowPartition> {
    let mut partitions = Vec::with_capacity(requested_count);

    for _ in 0..requested_count {
        let index = rng.gen_range(0..total_bits);
        let width = rng.gen_range(1..=max_width.min(total_bits - index));
        partitions.push(WindowPartition::new(
            index,
            width,
            WindowPartitionKind::Requested,
        ));
    }

    partitions
}

/// Builds the initially requested random partitions when overlaps are disallowed.
///
/// # Parameters
/// - `total_bits`: Total number of bits in the target domain.
/// - `requested_count`: Number of requested random partitions.
/// - `max_width`: Maximum partition width in bits.
/// - `rng`: Seeded ChaCha20 RNG used for random choices.
///
/// # Returns
/// - `Vec<WindowPartition>`: Non-overlapping requested partitions.
///
/// # Expected Output
/// - Returns the generated requested partitions; no side effects.
fn build_requested_partitions_without_overlap(
    total_bits: usize,
    requested_count: usize,
    max_width: usize,
    rng: &mut ChaCha20Rng,
) -> Vec<WindowPartition> {
    if requested_count == 0 {
        return Vec::new();
    }

    let widths = choose_non_overlapping_widths(total_bits, requested_count, max_width, rng);
    let occupied_bits = widths.iter().sum::<usize>();
    let gaps = distribute_gap_bits(total_bits - occupied_bits, requested_count + 1, rng);

    let mut partitions = Vec::with_capacity(requested_count);
    let mut cursor = gaps[0];

    for (width, trailing_gap) in widths.into_iter().zip(gaps.into_iter().skip(1)) {
        partitions.push(WindowPartition::new(
            cursor,
            width,
            WindowPartitionKind::Requested,
        ));
        cursor += width + trailing_gap;
    }

    partitions
}

/// Chooses non-overlapping requested widths while preserving capacity for the remaining partitions.
///
/// # Parameters
/// - `total_bits`: Total number of bits in the target domain.
/// - `requested_count`: Number of requested random partitions.
/// - `max_width`: Maximum partition width in bits.
/// - `rng`: Seeded ChaCha20 RNG used for random choices.
///
/// # Returns
/// - `Vec<usize>`: Widths for the requested partitions.
///
/// # Expected Output
/// - Returns the generated widths; no side effects.
fn choose_non_overlapping_widths(
    total_bits: usize,
    requested_count: usize,
    max_width: usize,
    rng: &mut ChaCha20Rng,
) -> Vec<usize> {
    let mut widths = Vec::with_capacity(requested_count);
    let mut used_bits = 0usize;

    for offset in 0..requested_count {
        let remaining_partitions = requested_count - offset - 1;
        let remaining_capacity = total_bits - used_bits;
        let max_width_for_current = max_width.min(remaining_capacity - remaining_partitions);
        let width = rng.gen_range(1..=max_width_for_current);
        widths.push(width);
        used_bits += width;
    }

    widths
}

/// Distributes uncovered capacity into random gap sizes around requested partitions.
///
/// # Parameters
/// - `gap_bits`: Number of uncovered bits to distribute.
/// - `slot_count`: Number of gap slots before, between, and after partitions.
/// - `rng`: Seeded ChaCha20 RNG used for random choices.
///
/// # Returns
/// - `Vec<usize>`: Gap widths whose sum equals `gap_bits`.
///
/// # Expected Output
/// - Returns the generated gap widths; no side effects.
fn distribute_gap_bits(gap_bits: usize, slot_count: usize, rng: &mut ChaCha20Rng) -> Vec<usize> {
    let mut gaps = vec![0usize; slot_count];

    for _ in 0..gap_bits {
        let slot = rng.gen_range(0..slot_count);
        gaps[slot] += 1;
    }

    gaps.shuffle(rng);
    gaps
}

/// Splits a fully uncovered gap into random-width partitions that preserve full coverage.
///
/// # Parameters
/// - `index`: Inclusive start index of the uncovered gap.
/// - `width`: Width of the uncovered gap in bits.
/// - `max_width`: Maximum partition width in bits.
/// - `rng`: Seeded ChaCha20 RNG used for random choices.
///
/// # Returns
/// - `Vec<WindowPartition>`: Gap-filling partitions that exactly cover the gap.
///
/// # Expected Output
/// - Returns the generated gap-fill partitions; no side effects.
fn split_gap_into_partitions(
    index: usize,
    width: usize,
    max_width: usize,
    rng: &mut ChaCha20Rng,
) -> Vec<WindowPartition> {
    let mut partitions = Vec::new();
    let mut cursor = index;
    let mut remaining = width;

    while remaining > 0 {
        let chunk_width = rng.gen_range(1..=max_width.min(remaining));
        partitions.push(WindowPartition::new(
            cursor,
            chunk_width,
            WindowPartitionKind::GapFill,
        ));
        cursor += chunk_width;
        remaining -= chunk_width;
    }

    partitions
}

/// Returns the uncovered bit ranges within the target domain.
///
/// # Parameters
/// - `total_bits`: Total number of bits in the target domain.
/// - `partitions`: Partitions to inspect for coverage.
///
/// # Returns
/// - `Vec<BitRange>`: Uncovered ranges in ascending index order.
///
/// # Expected Output
/// - Returns derived gap ranges; no side effects.
fn uncovered_ranges(total_bits: usize, partitions: &[WindowPartition]) -> Vec<BitRange> {
    let mut covered = vec![false; total_bits];

    for partition in partitions {
        for bit_index in partition.index..partition.end_exclusive() {
            covered[bit_index] = true;
        }
    }

    let mut gaps = Vec::new();
    let mut next_index = 0usize;

    while next_index < total_bits {
        if covered[next_index] {
            next_index += 1;
            continue;
        }

        let gap_start = next_index;
        while next_index < total_bits && !covered[next_index] {
            next_index += 1;
        }

        gaps.push(BitRange {
            index: gap_start,
            width: next_index - gap_start,
        });
    }

    gaps
}

/// Reports whether any partitions overlap after sorting by start index.
///
/// # Parameters
/// - `partitions`: Partitions to inspect for overlap.
///
/// # Returns
/// - `bool`: `true` when two partitions share at least one covered bit.
///
/// # Expected Output
/// - Returns a derived flag; no side effects.
fn partitions_overlap(partitions: &[WindowPartition]) -> bool {
    let mut ordered = partitions.iter().collect::<Vec<_>>();
    ordered.sort_by_key(|partition| (partition.index, partition.end_exclusive()));

    let mut current_end = 0usize;
    let mut has_seen_partition = false;

    for partition in ordered {
        if has_seen_partition && partition.index < current_end {
            return true;
        }

        current_end = current_end.max(partition.end_exclusive());
        has_seen_partition = true;
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn non_overlapping_builder_covers_every_bit_without_overlap() {
        let windows = WindowPartitionSet::build(64)
            .seed(7)
            .count(9)
            .max_width(8)
            .disallow_overlap()
            .generate()
            .expect("windows should build");

        assert_eq!(windows.total_bits(), 64);
        assert_eq!(windows.seed(), 7);
        assert_eq!(windows.requested_count(), 9);
        assert!(windows.covers_all_bits());
        assert!(!windows.has_overlaps());
        assert!(uncovered_ranges(windows.total_bits(), windows.partitions()).is_empty());
    }

    #[test]
    fn helper_non_overlapping_generation_covers_every_bit_without_overlap() {
        let windows = generate_non_overlapping_windows_with_seed(512, 32, 19, 11)
            .expect("windows should build");

        assert_eq!(windows.total_bits(), 512);
        assert!(windows.covers_all_bits());
        assert!(!windows.has_overlaps());
        assert!(windows
            .partitions()
            .iter()
            .all(|partition| partition.width() <= 19));
    }

    #[test]
    fn overlapping_generation_still_preserves_full_coverage() {
        let windows = generate_windows_with_seed(128, 24, 13, 29).expect("windows should build");

        assert_eq!(windows.total_bits(), 128);
        assert!(windows.covers_all_bits());
        assert!(windows
            .partitions()
            .iter()
            .all(|partition| partition.end_exclusive() <= 128));
    }

    #[test]
    fn builder_is_deterministic_for_the_same_seed() {
        let left = WindowPartitionSet::build(96)
            .seed(41)
            .count(12)
            .max_width(10)
            .disallow_overlap()
            .generate()
            .expect("left windows should build");
        let right = WindowPartitionSet::build(96)
            .seed(41)
            .count(12)
            .max_width(10)
            .disallow_overlap()
            .generate()
            .expect("right windows should build");

        assert_eq!(left, right);
    }

    #[test]
    fn builder_rejects_impossible_non_overlapping_request_counts() {
        let error = WindowPartitionSet::build(8)
            .count(9)
            .max_width(3)
            .disallow_overlap()
            .generate()
            .expect_err("configuration should be rejected");

        assert_eq!(
            error,
            WindowBuildError::RequestedCountExceedsTotalBits {
                requested_count: 9,
                total_bits: 8,
            }
        );
    }
}
