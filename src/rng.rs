use rand::rngs::{OsRng, StdRng};
use rand::{CryptoRng, RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;

/// RNG mode selection for standard vs cryptographic generators.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RngMode {
    Standard,
    Crypto,
}

/// Wrapper over supported RNGs for runtime selection.
#[derive(Debug)]
pub enum RngChoice {
    Standard(StdRng),
    Crypto(ChaCha20Rng),
}

impl RngChoice {
    /// Creates an RNG instance seeded from a `u64`.
    ///
    /// # Parameters
    /// - `mode`: RNG mode to instantiate.
    /// - `seed`: Seed used to initialize the generator.
    ///
    /// # Returns
    /// - `RngChoice`: Seeded RNG instance.
    ///
    /// # Expected Output
    /// - Returns a seeded RNG; no side effects.
    pub fn from_seed(mode: RngMode, seed: u64) -> Self {
        match mode {
            RngMode::Standard => Self::Standard(StdRng::seed_from_u64(seed)),
            RngMode::Crypto => Self::Crypto(ChaCha20Rng::seed_from_u64(seed)),
        }
    }

    /// Creates an RNG instance seeded from OS entropy.
    ///
    /// # Parameters
    /// - `mode`: RNG mode to instantiate.
    ///
    /// # Returns
    /// - `Result<RngChoice, rand::Error>`: RNG instance or an entropy error.
    ///
    /// # Expected Output
    /// - Pulls entropy from the OS when successful.
    pub fn from_entropy(mode: RngMode) -> Result<Self, rand::Error> {
        match mode {
            RngMode::Standard => Ok(Self::Standard(StdRng::from_rng(OsRng)?)),
            RngMode::Crypto => Ok(Self::Crypto(ChaCha20Rng::from_rng(OsRng)?)),
        }
    }

    /// Returns the current RNG mode.
    ///
    /// # Parameters
    /// - None.
    ///
    /// # Returns
    /// - `RngMode`: Mode of the wrapped RNG.
    ///
    /// # Expected Output
    /// - Returns the RNG mode; no side effects.
    pub fn mode(&self) -> RngMode {
        match self {
            Self::Standard(_) => RngMode::Standard,
            Self::Crypto(_) => RngMode::Crypto,
        }
    }
}

impl RngCore for RngChoice {
    fn next_u32(&mut self) -> u32 {
        match self {
            Self::Standard(rng) => rng.next_u32(),
            Self::Crypto(rng) => rng.next_u32(),
        }
    }

    fn next_u64(&mut self) -> u64 {
        match self {
            Self::Standard(rng) => rng.next_u64(),
            Self::Crypto(rng) => rng.next_u64(),
        }
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        match self {
            Self::Standard(rng) => rng.fill_bytes(dest),
            Self::Crypto(rng) => rng.fill_bytes(dest),
        }
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand::Error> {
        match self {
            Self::Standard(rng) => rng.try_fill_bytes(dest),
            Self::Crypto(rng) => rng.try_fill_bytes(dest),
        }
    }
}

impl CryptoRng for RngChoice {}
