#!/usr/bin/env bash

echo "Running R-Candidate CNN..."

python3 scripts/r_candidate_cnn.py \
  --session logs/combined_session_20260305_072416.json \
  --config config/rsa_config_small_batch.json \
  --output pca_clusters3.png \
  --all-batches \
  --fc-layers 2 \
  --embedding-dim 32 \
  --epochs 100 \
  --batch-size 50 \
  --lr 0.01 \
  --seed 1000001 \
  --device cuda \
  --poly-degree 10 \
  --target-dim 4
