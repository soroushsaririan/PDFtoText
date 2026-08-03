#!/usr/bin/env bash
set -e

cd "$(dirname "${BASH_SOURCE[0]}")"

MIN_PY_MINOR=10

version_ok() {
  # $1: a python interpreter path/command. Returns 0 if it's Python 3.$MIN_PY_MINOR+.
  "$1" -c "import sys; exit(0 if sys.version_info >= (3, $MIN_PY_MINOR) else 1)" >/dev/null 2>&1
}

find_python() {
  for candidate in python3.11 python3.12 python3.13 python3; do
    if command -v "$candidate" >/dev/null 2>&1 && version_ok "$candidate"; then
      command -v "$candidate"
      return 0
    fi
  done
  return 1
}

echo "==> Installing tesseract..."
if command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update -qq || echo "    (some apt sources failed to refresh -- continuing with what's cached)"
  sudo apt-get install -y tesseract-ocr
elif command -v dnf >/dev/null 2>&1; then
  sudo dnf install -y tesseract
else
  echo "Could not detect apt or dnf on this system."
  echo "Install tesseract manually with your distro's package manager, then re-run this script."
  exit 1
fi

PYTHON_BIN="$(find_python || true)"

if [ -z "$PYTHON_BIN" ]; then
  echo "==> No suitable Python 3.$MIN_PY_MINOR+ found -- attempting to install python3.11..."
  if command -v apt-get >/dev/null 2>&1; then
    if sudo apt-get install -y python3.11 python3.11-venv python3.11-dev 2>/dev/null; then
      PYTHON_BIN="$(command -v python3.11)"
    else
      echo "    python3.11 isn't in this system's default apt repos (common on newer Ubuntu releases,"
      echo "    which ship python3.12 instead). Trying the deadsnakes PPA..."
      if sudo apt-get install -y software-properties-common 2>/dev/null \
        && sudo add-apt-repository -y ppa:deadsnakes/ppa 2>/dev/null \
        && sudo apt-get update -qq 2>/dev/null \
        && sudo apt-get install -y python3.11 python3.11-venv python3.11-dev 2>/dev/null; then
        PYTHON_BIN="$(command -v python3.11)"
      else
        echo "    Could not reach the deadsnakes PPA either (may be blocked on this network)."
      fi
    fi
  elif command -v dnf >/dev/null 2>&1; then
    sudo dnf install -y python3.11 || true
    PYTHON_BIN="$(command -v python3.11 || true)"
  fi
fi

if [ -z "$PYTHON_BIN" ]; then
  PYTHON_BIN="$(find_python || true)"
fi

if [ -z "$PYTHON_BIN" ]; then
  echo "Could not find or install any Python 3.$MIN_PY_MINOR+ interpreter."
  echo "Install one manually (e.g. python3.11, or via pyenv/conda) and re-run this script."
  exit 1
fi

PY_VERSION="$("$PYTHON_BIN" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
echo "==> Using $PYTHON_BIN (Python $PY_VERSION)"
if [ "$PY_VERSION" != "3.11" ]; then
  echo "    Note: this project was developed and tested on Python 3.11. $PY_VERSION should work fine"
  echo "    for this pipeline, but if you hit dependency issues, installing 3.11 specifically"
  echo "    (e.g. via the deadsnakes PPA or pyenv) is the safest fix."
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
