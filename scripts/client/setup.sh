#!/usr/bin/env bash
# One-shot install + build for the FluxRT React client. Uses yarn only.
# Run after every `git pull` on the server:
#
#   bash scripts/client/setup.sh              # install + vendor + build
#   bash scripts/client/setup.sh --no-vendor  # skip MediaPipe vendoring (faster rebuilds)
#
set -euo pipefail

# Work from the client directory regardless of where this is invoked from.
cd "$(dirname "$0")"

if ! command -v yarn >/dev/null 2>&1; then
  echo "[ERROR] yarn not found. Enable it with 'corepack enable' (ships with Node)." >&2
  exit 1
fi

echo "[+] Installing dependencies with yarn..."
yarn install

if [ "${1:-}" = "--no-vendor" ]; then
  echo "[+] Skipping MediaPipe vendoring (--no-vendor)."
else
  echo "[+] Vendoring MediaPipe runtime + models (already-downloaded models are skipped)..."
  yarn vendor
fi

echo "[+] Building client (tsc + vite)..."
yarn build

echo "[+] Client ready -> scripts/client/dist (served by run_webrtc.py at /)."
