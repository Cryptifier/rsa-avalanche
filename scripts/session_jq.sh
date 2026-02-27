#!/usr/bin/env bash
set -euo pipefail

if ! command -v jq >/dev/null 2>&1; then
  if command -v pacman >/dev/null 2>&1; then
    echo "jq not found; installing with pacman..." >&2
    sudo pacman -S --noconfirm jq
  else
    echo "jq is required but not installed, and pacman was not found." >&2
    exit 1
  fi
fi

filter="${1:-.}"
file="${2:-session.json}"

if [ ! -f "$file" ]; then
  echo "Missing $file. Run the analysis CLI to generate session.json." >&2
  exit 1
fi

jq "$filter" "$file"
