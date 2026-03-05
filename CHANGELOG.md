# Changelog

Weekly changelog derived from git history.

## 2026-W09 (2026-02-23 to 2026-03-01)
- 2026-03-01: Add majority-vote prediction row to bit similarity view
- 2026-03-01: Add bit-true probability timeline tab
- 2026-03-01: Align adjusted match percent with shifted comparisons
- 2026-03-01: Compute per-shift candidate decryptions from original ciphertext
- 2026-03-01: Render grouped shifted candidates in bit similarity view
- 2026-03-01: Keep log selection on sidebar refresh
- 2026-02-28: Persist bit similarity settings across logs
- 2026-02-28: Add sorting to bit similarity view
- 2026-02-28: Add timestamped session logs and viewer sidebar
- 2026-02-28: Adjust viewer bit square sizing
- 2026-02-28: Render bit values in viewer
- 2026-02-27: Add session viewer and bit similarity export
- 2026-02-27: Update demo and analysis docs
- 2026-02-27: Fixed a bug in analysis.rs where it can have issues with loading r candidates.
- 2026-02-27: Parallelize demo best-case and majority recovery
- 2026-02-27: Track running demo match averages
- 2026-02-27: Enhance demo encrypt loop with decryption and diff
- 2026-02-27: Add hex bit diff utility
- 2026-02-27: Add demo encrypt script and config
- 2026-02-27: Add demo CLI for speculative decryption
- 2026-02-27: Add verbose shifted batch script
- 2026-02-27: Report best-case message hex output
- 2026-02-27: Parallelize match entropy timeline
- 2026-02-27: Add optional ciphertext shift for analysis
- 2026-02-26: Added videos showing statistical glitches.
- 2026-02-26: Added mp4 videos showing statistical glitches.
- 2026-02-25: Gate match timeline charts behind --export
- 2026-02-25: Move make scripts under scripts
- 2026-02-25: Ignore JSON outputs
- 2026-02-25: Add analytics module and Ctrl-C session logging

## 2026-W08 (2026-02-16 to 2026-02-22)
- 2026-02-21: Updated ARCHITECTURE.md.
- 2026-02-21: Changed email address in README.md.
- 2026-02-21: Updated documentation and analysis code.
- 2026-02-21: Updated script and code for inverted message outputs.
- 2026-02-20: Updated to use cryptographic number generator with the --crypto-rng flag.
- 2026-02-20: Updated medium batch script.
- 2026-02-20: Fixed logs file.
- 2026-02-20: Added tests for speculative oracle testing that checks for the probability of success for the oracles on a per-bit basis.
- 2026-02-20: added base r candidates.
- 2026-02-20: Added script to run rgen.
- 2026-02-20: Add configurable r candidate limits and rgen tests
- 2026-02-17: Parallelize run_message_trial candidates
- 2026-02-17: Clarify r-based decryption in docs
- 2026-02-17: Parallelize enciphered export histogram writes
- 2026-02-17: Use BigUint prime generation
- 2026-02-17: Add rgen tool and centralize config schema
- 2026-02-17: Added AGENTS.md and comments for core routines.
- 2026-02-17: Changed and updated documentation.
- 2026-02-16: Parallelized the enciphered export functions over all iterations.
- 2026-02-16: Added code to analysis.rs and combiner.rs to compute based on oracles from r candidate decryptions.
- 2026-02-16: Updated config/rsa_config.json with the new options.
- 2026-02-16: Refactored code and implemented combiner experiment.

## 2026-W07 (2026-02-09 to 2026-02-15)
- 2026-02-15: Updated config for make_mp4.sh
- 2026-02-15: Changed options in make_mp4.sh to reflect changes to Python script.
- 2026-02-15: Added better config options and updated video scripts.
- 2026-02-15: Added post-processing Python script for viewing animation of highest matching bits.
- 2026-02-14: Initial commit of dsp.rs and test code for ramp detection.

## 2026-W06 (2026-02-02 to 2026-02-08)
- 2026-02-06: Next experiment.
- 2026-02-06: Added a bit position histogram.
- 2026-02-06: Cleaned up code and config.
- 2026-02-06: Updated to use invert_bits setting, and also updated config.
- 2026-02-06: Updated the configuration file and histogram bin number.
- 2026-02-06: Added historical factors for later. It looks like the factors all end in 3 for the best_r candidate.
- 2026-02-06: Added historical images showing notch and regular bell curve for either the best and worst r candidates.
- 2026-02-06: Added a histogram for plotting the overlapping bit percentages across all runs.
- 2026-02-06: Temporary changes removing inverted bits.
- 2026-02-05: Changed the logic to use HBC in a way to make e > phi(N).

## 2026-W04 (2026-01-19 to 2026-01-25)
- 2026-01-23: Added license file and author.
- 2026-01-22: Fixed typo.
- 2026-01-22: Fixed typo.
- 2026-01-22: Clarified.
- 2026-01-21: Formatting.
- 2026-01-21: Fixed some formatting.
- 2026-01-21: Updated README.md.
- 2026-01-21: Added test data.
- 2026-01-21: LaTeX.
- 2026-01-21: LaTeX.
- 2026-01-21: Added RSA experiment.
