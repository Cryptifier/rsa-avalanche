# Dataset Layout

This repository stores generated datasets under `./data` so they are easy to locate and separate from source code.

## Directories

- `data/`: Root folder for generated datasets and grid outputs.
- `data/rgen_grid/`: Grid CSVs generated from the small config (`config/rsa_config_small.json`). Filenames include the percent offset and size label (for example, `rgen_grid_small_pct_30.csv`).
- `data/rgen_grid_medium/`: Grid CSVs generated from medium configs (for example, `config/rsa_config_medium.json`). Use this directory when creating the medium-sized grid outputs so small and medium datasets do not mix.

## Purpose

The rgen grid data captures r-candidate CSVs across a range of modulus-size reductions (5, 10, 20, 30, 40, 50 percent smaller). This makes it easy to compare how candidate generation behaves as the target bit-length changes.
