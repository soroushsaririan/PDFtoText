#!/usr/bin/env bash
set -e

cd "$(dirname "${BASH_SOURCE[0]}")"

if command -v apt-get >/dev/null 2>&1; then
  echo "==> Installing system dependencies (tesseract, Python 3.11) via apt..."
  sudo apt-get update -qq
  sudo apt-get install -y tesseract-ocr python3.11 python3.11-venv python3.11-dev
elif command -v dnf >/dev/null 2>&1; then
  echo "==> Installing system dependencies via dnf..."
  sudo dnf install -y tesseract python3.11
else
  echo "Could not detect apt or dnf on this system."
  echo "Install tesseract and Python 3.11 manually with your distro's package manager, then re-run this script."
  exit 1
fi

PYTHON_BIN="$(command -v python3.11)"
if [ -z "$PYTHON_BIN" ]; then
  echo "python3.11 still not found after install -- check your package manager's output above."
  exit 1
fi

echo "==> Creating virtual environment (.venv)..."
"$PYTHON_BIN" -m venv .venv

echo "==> Installing Python dependencies..."
./.venv/bin/pip install --upgrade pip -q
./.venv/bin/pip install -r requirements-linux.txt

if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  echo "==> NVIDIA GPU detected -- PyTorch will use it automatically."
else
  echo "==> No NVIDIA GPU detected -- running on CPU (slower, still works)."
fi

echo "==> Generating synthetic sample data..."
./.venv/bin/python src/generate_sample_data.py

echo "==> Running analysis on sample data..."
echo "    (first run downloads the Qwen2.5-VL handwriting model, ~7 GB -- this can take a while)"
./.venv/bin/python src/run_analysis.py

echo
echo "==> Done. Reports are in outputs/reports/, summary CSV is outputs/summary.csv"
echo "    To run on the real USB drive: ./.venv/bin/python src/run_analysis.py --base-dir /media/\$USER/USB --out-dir outputs_real"
