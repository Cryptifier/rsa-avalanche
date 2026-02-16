#!/usr/bin/env bash
python3 scripts/enciphered_bins_video.py \
  --input enciphered_decryption_bins2.csv \
  --output enciphered_bins-2.mp4 \
  --metric float \
  --azim-step 1 \
  --parallel \
  --z-scale linear \
  --z-max 80 \
  --z-min 20 \
  --max-frames 600 \
  --smooth-window 1
