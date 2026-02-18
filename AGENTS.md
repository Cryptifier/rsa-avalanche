# AGENTS Instructions

## Commenting Guidelines
- Use Rust doc comments (`///`) for public items and major internal routines.
- Each documented routine should include:
  - A one-line summary of what it does.
  - `# Parameters` with bullet-style `-` entries.
  - `# Returns` describing the return value (or `()` when applicable).
  - `# Expected Output` describing side effects (stdout/stderr/file writes) or stating there are none.
- Keep comments factual and aligned with the current implementation.
- Avoid duplicating obvious information that is already clear from the signature.
- Do not add verbose inline comments unless the logic is non-obvious.

## Modularity Guidelines
- Preserve the current module boundaries and responsibilities.
- Keep functionality in its existing module (`dsp`, `combiner`, `math`, `r_candidates`, and `bin/analysis`).
- Avoid moving functions across modules unless explicitly requested.
- Add new helpers in the most relevant existing module; do not create new modules without a clear need.
- Maintain clear separation between library code (`src/*.rs`) and CLI/analysis code (`src/bin/analysis.rs`).
