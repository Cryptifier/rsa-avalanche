#!/usr/bin/env bash
python3 scripts/enciphered_bins_video.py \
  --input enciphered_decryption_bins.csv \
  --output enciphered_bins.mp4 \
  --metric float \
  --parallel \
  --z-scale linear \
  --z-max 75 \
  --z-min 40 \
  --smooth-window 1
