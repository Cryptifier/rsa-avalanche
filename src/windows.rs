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

    /// Starts a query builder for selecting a random subset of partitions from this set.
    ///
    /// # Parameters
    /// - None.
    ///
    /// # Returns
    /// - `WindowPartitionQueryBuilder<'_>`: Builder configured to query this partition set.
    ///
    /// # Expected Output
    /// - Returns a query builder; no side effects.
    pub fn query(&self) -> WindowPartitionQueryBuilder<'_> {
        WindowPartitionQueryBuilder::new(self)
    }
}

/// Stores a mutable query result containing a random subset of partitions.
#[derive(Clone, Debug, PartialEq)]
pub struct WindowPartitionQuery {
    total_bits: usize,
    percentage: f64,
    target_bits: usize,
    max_query_bits: usize,
    seed: u64,
    overlap_policy: WindowOverlapPolicy,
    partitions: Vec<WindowPartition>,
}

impl WindowPartitionQuery {
    /// Returns the total bit width of the source domain used by the query.
    ///
    /// # Parameters
    /// - None.
    ///
    /// # Returns
    /// - `usize`: Total bit width for the queried domain.
    ///
    /// # Expected Output
    /// - Returns the stored domain width; no side effects.
    pub fn total_bits(&self) -> usize {
        self.total_bits
    }

    /// Returns the percentage used to determine the target number of queried bits.
    ///
    /// # Parameters
    /// - None.
    ///
    /// # Returns
    /// - `f64`: Percentage in the inclusive range `0.0..=1.0`.
    ///
    /// # Expected Output
    /// - Returns the stored percentage; no side effects.
    pub fn percentage(&self) -> f64 {
        self.percentage
    }

    /// Returns the maximum bit budget applied to the query result.
    ///
    /// # Parameters
    /// - None.
    ///
    /// # Returns
    /// - `usize`: Maximum number of covered bits allowed for the query.
    ///
    /// # Expected Output
    /// - Returns the stored bit budget; no side effects.
    pub fn max_query_bits(&self) -> usize {
        self.max_query_bits
    }

    /// Returns the target number of covered bits derived from the percentage and bit budget.
    ///
    /// # Parameters
    /// - None.
    ///
    /// # Returns
    /// - `usize`: Target number of covered bits for the query result.
    ///
    /// # Expected Output
    /// - Returns the stored target bit count; no side effects.
    pub fn target_bits(&self) -> usize {
        self.target_bits
    }

    /// Returns the `u64` seed used to initialize the query RNG.
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

    /// Returns the overlap policy used for the query result.
    ///
    /// # Parameters
    /// - None.
    ///
    /// # Returns
    /// - `WindowOverlapPolicy`: Overlap behavior applied to the query result.
    ///
    /// # Expected Output
    /// - Returns the stored overlap policy; no side effects.
    pub fn overlap_policy(&self) -> WindowOverlapPolicy {
        self.overlap_policy
    }

    /// Returns the queried partitions in ascending start-index order.
    ///
    /// # Parameters
    /// - None.
    ///
    /// # Returns
    /// - `&[WindowPartition]`: Queried partitions.
    ///
    /// # Expected Output
    /// - Returns a shared slice; no side effects.
    pub fn partitions(&self) -> &[WindowPartition] {
        &self.partitions
    }

    /// Returns the number of queried partitions.
    ///
    /// # Parameters
    /// - None.
    ///
    /// # Returns
    /// - `usize`: Number of partitions stored in the query result.
    ///
    /// # Expected Output
    /// - Returns the stored length; no side effects.
    pub fn len(&self) -> usize {
        self.partitions.len()
    }

    /// Returns whether the query result stores no partitions.
    ///
    /// # Parameters
    /// - None.
    ///
    /// # Returns
    /// - `bool`: `true` when no partitions are stored.
    ///
    /// # Expected Output
    /// - Returns a derived flag; no side effects.
    pub fn is_empty(&self) -> bool {
        self.partitions.is_empty()
    }

    /// Returns the number of unique bits covered by the query result.
    ///
    /// # Parameters
    /// - None.
    ///
    /// # Returns
    /// - `usize`: Number of uniquely covered bits.
    ///
    /// # Expected Output
    /// - Returns a derived count; no side effects.
    pub fn covered_bits(&self) -> usize {
        covered_bit_count(self.total_bits, &self.partitions)
    }

    /// Returns whether any stored query partitions overlap.
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

    /// Adds a partition to the query result after validating bounds and overlap policy.
    ///
    /// # Parameters
    /// - `partition`: Partition to add to the query result.
    ///
    /// # Returns
    /// - `Result<(), WindowQueryError>`: `Ok(())` when the partition is accepted.
    ///
    /// # Expected Output
    /// - Mutates the stored query partitions when successful.
    pub fn add_partition(&mut self, partition: WindowPartition) -> Result<(), WindowQueryError> {
        validate_query_partition(
            self.total_bits,
            self.overlap_policy,
            &self.partitions,
            &partition,
        )?;
        self.partitions.push(partition);
        self.partitions
            .sort_by_key(|entry| (entry.index(), entry.kind(), entry.width()));
        Ok(())
    }

    /// Removes the first matching partition from the query result.
    ///
    /// # Parameters
    /// - `partition`: Partition value to remove from the query result.
    ///
    /// # Returns
    /// - `bool`: `true` when a matching partition was removed.
    ///
    /// # Expected Output
    /// - Mutates the stored query partitions when a match exists.
    pub fn remove_partition(&mut self, partition: &WindowPartition) -> bool {
        if let Some(position) = self.partitions.iter().position(|entry| entry == partition) {
            self.partitions.remove(position);
            return true;
        }

        false
    }

    /// Removes the partition at the provided position.
    ///
    /// # Parameters
    /// - `position`: Index within the stored partition vector.
    ///
    /// # Returns
    /// - `Option<WindowPartition>`: Removed partition when the position exists.
    ///
    /// # Expected Output
    /// - Mutates the stored query partitions when the position exists.
    pub fn remove_partition_at(&mut self, position: usize) -> Option<WindowPartition> {
        if position < self.partitions.len() {
            return Some(self.partitions.remove(position));
        }

        None
    }
}

/// Builds a percentage-based random query over an existing partition set.
#[derive(Clone, Debug)]
pub struct WindowPartitionQueryBuilder<'a> {
    source: &'a WindowPartitionSet,
    percentage: Option<f64>,
    max_query_bits: Option<usize>,
    seed: u64,
    overlap_policy: WindowOverlapPolicy,
}

impl<'a> WindowPartitionQueryBuilder<'a> {
    /// Creates a query builder over an existing partition set.
    ///
    /// # Parameters
    /// - `source`: Partition set used as the query source.
    ///
    /// # Returns
    /// - `WindowPartitionQueryBuilder<'a>`: Query builder with the source seed and overlap policy.
    ///
    /// # Expected Output
    /// - Returns a query builder; no side effects.
    pub fn new(source: &'a WindowPartitionSet) -> Self {
        Self {
            source,
            percentage: None,
            max_query_bits: None,
            seed: source.seed(),
            overlap_policy: source.overlap_policy(),
        }
    }

    /// Sets the percentage of the source bit domain to target during the query.
    ///
    /// # Parameters
    /// - `percentage`: Fraction in the inclusive range `0.0..=1.0`.
    ///
    /// # Returns
    /// - `WindowPartitionQueryBuilder<'a>`: Updated query builder.
    ///
    /// # Expected Output
    /// - Returns an updated query builder; no side effects.
    pub fn percentage(mut self, percentage: f64) -> Self {
        self.percentage = Some(percentage);
        self
    }

    /// Sets the maximum number of bits that may be covered by the query result.
    ///
    /// # Parameters
    /// - `max_query_bits`: Maximum number of covered bits allowed for the query.
    ///
    /// # Returns
    /// - `WindowPartitionQueryBuilder<'a>`: Updated query builder.
    ///
    /// # Expected Output
    /// - Returns an updated query builder; no side effects.
    pub fn max_bits(mut self, max_query_bits: usize) -> Self {
        self.max_query_bits = Some(max_query_bits);
        self
    }

    /// Sets the deterministic seed used to initialize `ChaCha20Rng` for the query.
    ///
    /// # Parameters
    /// - `seed`: Seed value passed to `ChaCha20Rng::seed_from_u64`.
    ///
    /// # Returns
    /// - `WindowPartitionQueryBuilder<'a>`: Updated query builder.
    ///
    /// # Expected Output
    /// - Returns an updated query builder; no side effects.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Sets the overlap policy used while constructing the query result.
    ///
    /// # Parameters
    /// - `policy`: Overlap behavior for queried partitions.
    ///
    /// # Returns
    /// - `WindowPartitionQueryBuilder<'a>`: Updated query builder.
    ///
    /// # Expected Output
    /// - Returns an updated query builder; no side effects.
    pub fn overlap_policy(mut self, policy: WindowOverlapPolicy) -> Self {
        self.overlap_policy = policy;
        self
    }

    /// Configures the query result to allow overlaps.
    ///
    /// # Parameters
    /// - None.
    ///
    /// # Returns
    /// - `WindowPartitionQueryBuilder<'a>`: Updated query builder.
    ///
    /// # Expected Output
    /// - Returns an updated query builder; no side effects.
    pub fn allow_overlap(self) -> Self {
        self.overlap_policy(WindowOverlapPolicy::AllowOverlap)
    }

    /// Configures the query result to disallow overlaps.
    ///
    /// # Parameters
    /// - None.
    ///
    /// # Returns
    /// - `WindowPartitionQueryBuilder<'a>`: Updated query builder.
    ///
    /// # Expected Output
    /// - Returns an updated query builder; no side effects.
    pub fn disallow_overlap(self) -> Self {
        self.overlap_policy(WindowOverlapPolicy::DisallowOverlap)
    }

    /// Generates a random query result biased toward non-contiguous partitions when available.
    ///
    /// # Parameters
    /// - None.
    ///
    /// # Returns
    /// - `Result<WindowPartitionQuery, WindowQueryError>`: Query result or a query configuration error.
    ///
    /// # Expected Output
    /// - Returns the generated query result; no side effects.
    pub fn generate(self) -> Result<WindowPartitionQuery, WindowQueryError> {
        validate_query_builder(&self)?;

        let percentage = self
            .percentage
            .expect("validated query builder must contain a percentage");
        let max_query_bits = self
            .max_query_bits
            .expect("validated query builder must contain max_query_bits")
            .min(self.source.total_bits());
        let target_bits = target_query_bits(self.source.total_bits(), percentage, max_query_bits);
        let mut rng = ChaCha20Rng::seed_from_u64(self.seed);
        let mut remaining_indices = (0..self.source.partitions().len()).collect::<Vec<_>>();
        let mut selected = Vec::new();

        while covered_bit_count(self.source.total_bits(), &selected) < target_bits
            && !remaining_indices.is_empty()
        {
            let covered_bits = covered_bit_count(self.source.total_bits(), &selected);
            let remaining_budget = max_query_bits.saturating_sub(covered_bits);
            let mut preferred = Vec::new();
            let mut fallback = Vec::new();

            for candidate_index in &remaining_indices {
                let candidate = &self.source.partitions()[*candidate_index];
                let added_bits =
                    additional_covered_bit_count(self.source.total_bits(), &selected, candidate);

                if added_bits == 0 || added_bits > remaining_budget {
                    continue;
                }

                if self.overlap_policy == WindowOverlapPolicy::DisallowOverlap
                    && partition_overlaps_any(candidate, &selected)
                {
                    continue;
                }

                if partition_is_non_contiguous_with_selection(candidate, &selected) {
                    preferred.push(*candidate_index);
                } else {
                    fallback.push(*candidate_index);
                }
            }

            let pool = if !preferred.is_empty() {
                &preferred
            } else {
                &fallback
            };

            if pool.is_empty() {
                break;
            }

            let chosen_candidate = pool[rng.gen_range(0..pool.len())];
            selected.push(self.source.partitions()[chosen_candidate].clone());
            remaining_indices.retain(|index| *index != chosen_candidate);
        }

        selected.sort_by_key(|entry| (entry.index(), entry.kind(), entry.width()));

        Ok(WindowPartitionQuery {
            total_bits: self.source.total_bits(),
            percentage,
            target_bits,
            max_query_bits,
            seed: self.seed,
            overlap_policy: self.overlap_policy,
            partitions: selected,
        })
    }
}

/// Builds a partition series with seeded ChaCha20-based generation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WindowPartitionSetBuilder {
    total_bits: usize,
    requested_count: usize,
    max_width: Option<usize>,
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
            max_width: None,
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
        self.max_width = Some(max_width);
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

        let effective_max_width = self
            .max_width
            .expect("validated builder must contain max_width")
            .min(self.total_bits);
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
    MaxWidthMustBeSpecified,
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
            Self::MaxWidthMustBeSpecified => {
                write!(f, "max_width must be specified by the builder")
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

/// Describes configuration and mutation errors for partition queries.
#[derive(Clone, Debug, PartialEq)]
pub enum WindowQueryError {
    PercentageMustBeSpecified,
    PercentageMustBeFinite,
    PercentageOutOfRange,
    MaxQueryBitsMustBeSpecified,
    MaxQueryBitsMustBePositive,
    PartitionWidthMustBePositive,
    PartitionOutOfBounds,
    PartitionOverlapNotAllowed,
}

impl fmt::Display for WindowQueryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PercentageMustBeSpecified => {
                write!(f, "percentage must be specified by the query builder")
            }
            Self::PercentageMustBeFinite => {
                write!(f, "percentage must be a finite value")
            }
            Self::PercentageOutOfRange => {
                write!(f, "percentage must be in the inclusive range 0.0..=1.0")
            }
            Self::MaxQueryBitsMustBeSpecified => {
                write!(f, "max_query_bits must be specified by the query builder")
            }
            Self::MaxQueryBitsMustBePositive => {
                write!(f, "max_query_bits must be positive")
            }
            Self::PartitionWidthMustBePositive => {
                write!(f, "partition width must be positive")
            }
            Self::PartitionOutOfBounds => {
                write!(f, "partition must fit within the query bit domain")
            }
            Self::PartitionOverlapNotAllowed => {
                write!(f, "partition overlaps an existing query partition")
            }
        }
    }
}

impl Error for WindowQueryError {}

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

/// Queries a random subset of partitions by bit percentage using the source seed.
///
/// # Parameters
/// - `source`: Partition set used as the query source.
/// - `percentage`: Fraction of the source bit domain to target.
/// - `max_query_bits`: Maximum number of covered bits allowed for the query result.
///
/// # Returns
/// - `Result<WindowPartitionQuery, WindowQueryError>`: Query result or a query configuration error.
///
/// # Expected Output
/// - Returns the generated query result; no side effects.
pub fn query_windows_by_percentage(
    source: &WindowPartitionSet,
    percentage: f64,
    max_query_bits: usize,
) -> Result<WindowPartitionQuery, WindowQueryError> {
    source
        .query()
        .percentage(percentage)
        .max_bits(max_query_bits)
        .generate()
}

/// Queries a random subset of partitions by bit percentage using a caller-provided seed.
///
/// # Parameters
/// - `source`: Partition set used as the query source.
/// - `percentage`: Fraction of the source bit domain to target.
/// - `max_query_bits`: Maximum number of covered bits allowed for the query result.
/// - `seed`: Seed passed to `ChaCha20Rng::seed_from_u64`.
///
/// # Returns
/// - `Result<WindowPartitionQuery, WindowQueryError>`: Query result or a query configuration error.
///
/// # Expected Output
/// - Returns the generated query result; no side effects.
pub fn query_windows_by_percentage_with_seed(
    source: &WindowPartitionSet,
    percentage: f64,
    max_query_bits: usize,
    seed: u64,
) -> Result<WindowPartitionQuery, WindowQueryError> {
    source
        .query()
        .percentage(percentage)
        .max_bits(max_query_bits)
        .seed(seed)
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

    let max_width = builder
        .max_width
        .ok_or(WindowBuildError::MaxWidthMustBeSpecified)?;

    if max_width == 0 {
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

/// Validates query builder configuration before random query generation.
///
/// # Parameters
/// - `builder`: Query builder configuration to validate.
///
/// # Returns
/// - `Result<(), WindowQueryError>`: `Ok(())` when the configuration is valid.
///
/// # Expected Output
/// - Returns a validation result; no side effects.
fn validate_query_builder(
    builder: &WindowPartitionQueryBuilder<'_>,
) -> Result<(), WindowQueryError> {
    let percentage = builder
        .percentage
        .ok_or(WindowQueryError::PercentageMustBeSpecified)?;

    if !percentage.is_finite() {
        return Err(WindowQueryError::PercentageMustBeFinite);
    }

    if !(0.0..=1.0).contains(&percentage) {
        return Err(WindowQueryError::PercentageOutOfRange);
    }

    let max_query_bits = builder
        .max_query_bits
        .ok_or(WindowQueryError::MaxQueryBitsMustBeSpecified)?;

    if max_query_bits == 0 {
        return Err(WindowQueryError::MaxQueryBitsMustBePositive);
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

/// Validates a caller-supplied query partition against bounds and overlap policy.
///
/// # Parameters
/// - `total_bits`: Total number of bits in the query domain.
/// - `overlap_policy`: Overlap behavior for the query result.
/// - `existing`: Existing query partitions.
/// - `candidate`: Partition to validate.
///
/// # Returns
/// - `Result<(), WindowQueryError>`: `Ok(())` when the partition is valid.
///
/// # Expected Output
/// - Returns a validation result; no side effects.
fn validate_query_partition(
    total_bits: usize,
    overlap_policy: WindowOverlapPolicy,
    existing: &[WindowPartition],
    candidate: &WindowPartition,
) -> Result<(), WindowQueryError> {
    if candidate.width() == 0 {
        return Err(WindowQueryError::PartitionWidthMustBePositive);
    }

    if candidate.index() >= total_bits || candidate.end_exclusive() > total_bits {
        return Err(WindowQueryError::PartitionOutOfBounds);
    }

    if overlap_policy == WindowOverlapPolicy::DisallowOverlap
        && partition_overlaps_any(candidate, existing)
    {
        return Err(WindowQueryError::PartitionOverlapNotAllowed);
    }

    Ok(())
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

/// Computes the target number of covered bits for a percentage-based query.
///
/// # Parameters
/// - `total_bits`: Total number of bits in the query domain.
/// - `percentage`: Fraction of the bit domain to target.
/// - `max_query_bits`: Maximum number of covered bits allowed for the query result.
///
/// # Returns
/// - `usize`: Target number of covered bits.
///
/// # Expected Output
/// - Returns a derived bit count; no side effects.
fn target_query_bits(total_bits: usize, percentage: f64, max_query_bits: usize) -> usize {
    if percentage == 0.0 || total_bits == 0 {
        return 0;
    }

    let percentage_bits = ((total_bits as f64) * percentage).ceil() as usize;
    percentage_bits.min(max_query_bits).min(total_bits)
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

/// Returns the number of uniquely covered bits across the provided partitions.
///
/// # Parameters
/// - `total_bits`: Total number of bits in the inspected domain.
/// - `partitions`: Partitions to inspect.
///
/// # Returns
/// - `usize`: Number of unique covered bits.
///
/// # Expected Output
/// - Returns a derived count; no side effects.
fn covered_bit_count(total_bits: usize, partitions: &[WindowPartition]) -> usize {
    let mut covered = vec![false; total_bits];

    for partition in partitions {
        for bit_index in partition.index()..partition.end_exclusive() {
            covered[bit_index] = true;
        }
    }

    covered.into_iter().filter(|is_covered| *is_covered).count()
}

/// Returns the additional unique coverage contributed by a candidate partition.
///
/// # Parameters
/// - `total_bits`: Total number of bits in the inspected domain.
/// - `selected`: Already selected partitions.
/// - `candidate`: Candidate partition to measure.
///
/// # Returns
/// - `usize`: Number of newly covered bits contributed by `candidate`.
///
/// # Expected Output
/// - Returns a derived count; no side effects.
fn additional_covered_bit_count(
    total_bits: usize,
    selected: &[WindowPartition],
    candidate: &WindowPartition,
) -> usize {
    let mut covered = vec![false; total_bits];

    for partition in selected {
        for bit_index in partition.index()..partition.end_exclusive() {
            covered[bit_index] = true;
        }
    }

    let mut added_bits = 0usize;
    for bit_index in candidate.index()..candidate.end_exclusive() {
        if !covered[bit_index] {
            covered[bit_index] = true;
            added_bits += 1;
        }
    }

    added_bits
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

/// Reports whether a candidate overlaps any partition in an existing selection.
///
/// # Parameters
/// - `candidate`: Candidate partition to inspect.
/// - `partitions`: Existing partitions to compare against.
///
/// # Returns
/// - `bool`: `true` when the candidate overlaps any partition in `partitions`.
///
/// # Expected Output
/// - Returns a derived flag; no side effects.
fn partition_overlaps_any(candidate: &WindowPartition, partitions: &[WindowPartition]) -> bool {
    partitions
        .iter()
        .any(|existing| partitions_overlap_pair(candidate, existing))
}

/// Reports whether a candidate is separated by at least one uncovered bit from all selected partitions.
///
/// # Parameters
/// - `candidate`: Candidate partition to inspect.
/// - `selected`: Existing selected partitions.
///
/// # Returns
/// - `bool`: `true` when the candidate is neither overlapping nor adjacent to any selected partition.
///
/// # Expected Output
/// - Returns a derived flag; no side effects.
fn partition_is_non_contiguous_with_selection(
    candidate: &WindowPartition,
    selected: &[WindowPartition],
) -> bool {
    selected
        .iter()
        .all(|existing| !partitions_touch_or_overlap(candidate, existing))
}

/// Reports whether two partitions overlap.
///
/// # Parameters
/// - `left`: First partition.
/// - `right`: Second partition.
///
/// # Returns
/// - `bool`: `true` when the partitions share at least one covered bit.
///
/// # Expected Output
/// - Returns a derived flag; no side effects.
fn partitions_overlap_pair(left: &WindowPartition, right: &WindowPartition) -> bool {
    left.index() < right.end_exclusive() && right.index() < left.end_exclusive()
}

/// Reports whether two partitions overlap or touch contiguously.
///
/// # Parameters
/// - `left`: First partition.
/// - `right`: Second partition.
///
/// # Returns
/// - `bool`: `true` when the partitions overlap or have no gap between them.
///
/// # Expected Output
/// - Returns a derived flag; no side effects.
fn partitions_touch_or_overlap(left: &WindowPartition, right: &WindowPartition) -> bool {
    left.index() <= right.end_exclusive() && right.index() <= left.end_exclusive()
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

    #[test]
    fn builder_requires_explicit_max_width() {
        let error = WindowPartitionSet::build(32)
            .seed(5)
            .count(4)
            .disallow_overlap()
            .generate()
            .expect_err("configuration should be rejected");

        assert_eq!(error, WindowBuildError::MaxWidthMustBeSpecified);
    }

    #[test]
    fn query_builder_respects_bit_budget_and_non_overlap_policy() {
        let windows = generate_windows_with_seed(128, 24, 12, 73).expect("windows should build");
        let query = windows
            .query()
            .percentage(0.25)
            .max_bits(20)
            .seed(19)
            .disallow_overlap()
            .generate()
            .expect("query should build");

        assert!(!query.is_empty());
        assert!(query.covered_bits() <= 20);
        assert!(query.covered_bits() <= query.target_bits());
        assert!(!query.has_overlaps());
        assert!(query
            .partitions()
            .iter()
            .all(|partition| partition.end_exclusive() <= windows.total_bits()));
    }

    #[test]
    fn query_builder_prefers_non_contiguous_partitions_when_available() {
        let source = WindowPartitionSet {
            total_bits: 32,
            requested_count: 3,
            max_width: 4,
            seed: 17,
            overlap_policy: WindowOverlapPolicy::DisallowOverlap,
            partitions: vec![
                WindowPartition::new(0, 4, WindowPartitionKind::Requested),
                WindowPartition::new(4, 4, WindowPartitionKind::Requested),
                WindowPartition::new(12, 4, WindowPartitionKind::Requested),
            ],
        };

        let query = source
            .query()
            .percentage(0.25)
            .max_bits(8)
            .seed(3)
            .disallow_overlap()
            .generate()
            .expect("query should build");

        assert_eq!(query.len(), 2);
        assert!(!partitions_touch_or_overlap(
            &query.partitions()[0],
            &query.partitions()[1]
        ));
    }

    #[test]
    fn query_result_allows_manual_add_and_remove_operations() {
        let source = WindowPartitionSet {
            total_bits: 32,
            requested_count: 2,
            max_width: 4,
            seed: 23,
            overlap_policy: WindowOverlapPolicy::DisallowOverlap,
            partitions: vec![
                WindowPartition::new(0, 4, WindowPartitionKind::Requested),
                WindowPartition::new(12, 4, WindowPartitionKind::Requested),
            ],
        };
        let mut query = source
            .query()
            .percentage(0.125)
            .max_bits(4)
            .seed(9)
            .disallow_overlap()
            .generate()
            .expect("query should build");
        let added = WindowPartition::new(20, 4, WindowPartitionKind::Requested);

        query
            .add_partition(added.clone())
            .expect("disjoint partition should be accepted");
        assert!(query.partitions().contains(&added));
        assert!(query.remove_partition(&added));
        assert!(!query.partitions().contains(&added));

        let overlapping = WindowPartition::new(1, 3, WindowPartitionKind::Requested);
        let add_error = query
            .add_partition(overlapping)
            .expect_err("overlapping partition should be rejected");
        assert_eq!(add_error, WindowQueryError::PartitionOverlapNotAllowed);

        let removed = query.remove_partition_at(0);
        assert!(removed.is_some());
    }

    #[test]
    fn query_builder_requires_explicit_percentage_and_max_bits() {
        let windows =
            generate_non_overlapping_windows_with_seed(64, 8, 8, 31).expect("windows should build");

        let error = windows
            .query()
            .seed(11)
            .disallow_overlap()
            .generate()
            .expect_err("query configuration should be rejected");

        assert_eq!(error, WindowQueryError::PercentageMustBeSpecified);
    }
}
