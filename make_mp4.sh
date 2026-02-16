#!/usr/bin/env bash
python3 scripts/enciphered_bins_video.py \
  --input enciphered_decryption_bins.csv \
  --output enciphered_bins.mp4 \
  --metric float \
  --parallel \
  --z-scale linear \
  --z-percentile-high 0.50 \
  --z-percentile-low 0.45 \
  --smooth-window 3
