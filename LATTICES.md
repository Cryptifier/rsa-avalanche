# Lattices

`src/lattices.rs` provides two related pieces of functionality:

- A generic univariate Coppersmith lattice builder for integer polynomials over an RSA modulus.
- An RSA-oriented helper that builds `f(x) = x + known_prefix` from a prime factor `p` whose low bits are treated as the unknown small root `x`.

The polynomial arithmetic used by the lattice code lives in `src/polynomials.rs`.

## Module Overview

The main public types are:

- `IntegerPolynomial` in `src/polynomials.rs`
- `CoppersmithLatticeBuilder`
- `CoppersmithLattice`
- `CoppersmithLatticeElement`
- `RsaCoppersmithInput`
- `RsaCoppersmithRun`
- `lll_reduce`

The lattice builder is geared toward RSA use cases:

- `modulus` is the RSA modulus `N`
- `polynomial` is a monic integer polynomial `f(x)`
- `bound` is the small-root bound `X`
- `m` is the Coppersmith exponent parameter
- `dimension` is the final square lattice dimension `d`

## Coppersmith Basis Shape

For a monic polynomial `f(x)` of degree `delta`, the builder constructs basis rows from:

```text
g_{i,j}(x) = N^(m - i) * x^j * f(x)^i
```

for:

- `0 <= i < m`
- `0 <= j < delta`

If `d > delta * m`, the builder appends completion rows using:

```text
g_{m,j}(x) = x^j * f(x)^m
```

for `0 <= j < d - delta * m`.

Each row is scaled as `g_{i,j}(Xx)` before its coefficients are inserted into the lattice matrix. This is what lets `vector_to_polynomial` and `reduced_polynomials` map reduced lattice vectors back into ordinary integer polynomials.

## Builder Usage

Use `CoppersmithLatticeBuilder` when you already know the polynomial you want to attack.

```rust
use num_bigint::{BigInt, BigUint};
use num_traits::One;
use rsademo::lattices::CoppersmithLatticeBuilder;
use rsademo::polynomials::IntegerPolynomial;

let modulus = BigUint::from(11413u32);
let polynomial = IntegerPolynomial::new(vec![
    BigInt::from(96u8),
    BigInt::one(),
]);

let lattice = CoppersmithLatticeBuilder::new(modulus, polynomial)
    .with_bound(BigUint::from(8u8))
    .with_exponent(3)
    .with_dimension(6)
    .build()?;

assert_eq!(lattice.dimension, 6);
assert_eq!(lattice.elements.len(), 6);
```

After the lattice is built:

- `lattice.elements` contains every generated `g_{i,j}(x)` row and its scaled form.
- `lattice.basis` is the square coefficient matrix used for LLL.
- `lattice.reduce()` runs LLL on that basis.
- `lattice.reduced_polynomials(&reduced_basis)` converts reduced vectors back into unscaled polynomials.

Example:

```rust
let reduced_basis = lattice.reduce();
let reduced_polynomials = lattice.reduced_polynomials(&reduced_basis)?;
```

## Inspecting Individual Lattice Rows

Each `CoppersmithLatticeElement` contains:

- `i`: the exponent used on `f(x)^i`
- `j`: the monomial shift `x^j`
- `polynomial`: the unscaled `g_{i,j}(x)`
- `scaled_polynomial`: the scaled `g_{i,j}(Xx)`
- `coefficient_vector`: the row actually placed in the lattice basis

This is useful when you want to verify the basis layout or test a specific `m` / `d` construction.

## RSA Helper Usage

Use `run_rsa_coppersmith` when you want the RSA-specific flow for a prime factor `p` with a small unknown low-order tail `x`.

The helper derives:

```text
known_prefix = p - x
f(x) = x + known_prefix
```

and then builds the corresponding Coppersmith lattice for the supplied `m` and `d`.

```rust
use num_bigint::BigUint;
use rsademo::lattices::{run_rsa_coppersmith, RsaCoppersmithInput};

let input = RsaCoppersmithInput {
    modulus: BigUint::from(11413u32),
    prime: BigUint::from(101u8),
    unknown_part: BigUint::from(5u8),
    m: 3,
    dimension: 6,
};

let run = run_rsa_coppersmith(&input)?;

assert_eq!(run.known_prefix, BigUint::from(96u8));
assert_eq!(run.recovered_unknown, Some(BigUint::from(5u8)));
```

The returned `RsaCoppersmithRun` includes:

- `known_prefix`
- `bound`
- `lattice`
- `reduced_basis`
- `reduced_polynomials`
- `recovered_unknown`

`bound` is derived from the bit width of `unknown_part`:

- if `x = 0`, the bound is `1`
- otherwise the bound is `2^(bits(x))`

## Validation Rules

The builder rejects malformed requests before constructing the basis:

- `modulus` must be greater than `1`
- `bound` must be present and non-zero
- `m` must be present
- `dimension` must be present
- `polynomial` must be monic
- `polynomial` must have degree at least `1`
- `dimension >= degree(f) * m`

The RSA helper adds RSA-specific checks:

- `prime <= modulus`
- `unknown_part <= prime`
- `prime` must divide `modulus`

## Choosing `m` and `d`

The code does not auto-tune `m` or `d`. You must choose them explicitly.

Practical guidance for the current implementation:

- Start with `d = degree(f) * m` if you want the minimal square lattice.
- Increase `d` when you want extra completion rows from `f(x)^m`.
- For simple linear RSA examples, larger values such as `m = 3`, `d = 6` are more reliable than the smallest admissible lattice.

## Current Recovery Behavior

`run_rsa_coppersmith` currently does two things after LLL:

1. It converts reduced lattice vectors back into ordinary integer polynomials.
2. It brute-forces candidates in `[0, X)` and accepts a candidate only if:
   - at least one reduced polynomial evaluates to zero at that candidate, and
   - `known_prefix + candidate` divides the RSA modulus.

This makes the current helper suitable for:

- validating the lattice construction
- experimenting with small-root setups
- unit testing RSA partial-prime examples

It is not yet a full symbolic small-root extractor over the reduced basis.

## Polynomial Utilities

`src/polynomials.rs` provides the integer polynomial operations the lattice builder depends on:

- `IntegerPolynomial::new`
- `IntegerPolynomial::from_constant`
- `IntegerPolynomial::monomial`
- `add`
- `mul`
- `pow`
- `scale`
- `shift`
- `scale_input`
- `evaluate`
- `padded_coefficients`

The coefficient order is ascending by degree:

```text
[c0, c1, c2]
```

means:

```text
c0 + c1*x + c2*x^2
```

## Unit Tests

The lattice and polynomial code is covered by unit tests in:

- `src/lattices.rs`
- `src/polynomials.rs`

The lattice tests cover:

- LLL sanity on a small basis
- exact `g_{i,j}` basis construction for a linear polynomial
- non-monic polynomial rejection
- constant polynomial rejection
- zero bound rejection
- insufficient dimension rejection
- invalid scaled-vector rejection
- RSA recovery for a small tail
- zero-tail RSA handling
- oversized tail rejection
- invalid prime/modulus pairing rejection

Run the library test suite with:

```bash
cargo test --lib
```
