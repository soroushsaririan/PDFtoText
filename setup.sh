#!/usr/bin/env bash
set -e

cd "$(dirname "${BASH_SOURCE[0]}")"

if ! command -v brew >/dev/null 2>&1; then
  echo "Homebrew is required but not installed. Install it from https://brew.sh and re-run this script."
  exit 1
fi

echo "==> Installing tesseract (OCR engine for printed text/checkboxes)..."
brew list tesseract >/dev/null 2>&1 || brew install tesseract

PYTHON_BIN="$(command -v python3.11 || true)"
if [ -z "$PYTHON_BIN" ]; then
  echo "==> Python 3.11 not found, installing via Homebrew..."
  brew install python@3.11
  PYTHON_BIN="$(brew --prefix python@3.11)/bin/python3.11"
fi

echo "==> Creating virtual environment (.venv)..."
"$PYTHON_BIN" -m venv .venv

echo "==> Installing Python dependencies..."
./.venv/bin/pip install --upgrade pip -q
./.venv/bin/pip install -r requirements.txt

echo "==> Generating synthetic sample data..."
./.venv/bin/python src/generate_sample_data.py

echo "==> Running analysis on sample data..."
echo "    (first run downloads the Qwen2.5-VL handwriting model, ~7 GB -- this can take a while)"
./.venv/bin/python src/run_analysis.py

echo
echo "==> Done. Reports are in outputs/reports/, summary CSV is outputs/summary.csv"
echo "    To run on the real USB drive: ./.venv/bin/python src/run_analysis.py --base-dir /Volumes/USB --out-dir outputs_real"
