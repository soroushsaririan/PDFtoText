# handwriting_analysis

Analyzes handwritten weekly PT session forms, CRF visit forms, and exit
surveys collected for each study participant (VPT001, VPT002, ...) on the
study USB drive, and produces a per-participant report plus a combined
CSV for stats.

## Requirements

- macOS or Linux (Ubuntu/Debian or Fedora tested)
- Python 3.11
- ~15 GB free disk space (one-time handwriting-model download)
- A free Kaggle account -- only needed for the optional benchmark tools in `benchmark/`

## Setup

Two setup scripts, one per OS -- pick the one matching your machine. Both
do the same thing: install system dependencies, create a Python 3.11
virtual environment, install everything else, generate the synthetic
sample data, and run a first analysis pass against it. The first run also
downloads the handwriting model (~7 GB, one-time, cached under
`~/.cache/huggingface`).

### macOS setup

Requires [Homebrew](https://brew.sh).

```bash
git clone https://github.com/soroushsaririan/PDFtoText.git
cd PDFtoText
./setup.sh
```

### Linux setup

Ubuntu/Debian (via `apt`) or Fedora (via `dnf`) -- installs system packages
with `sudo`. Other distros aren't auto-detected: install `tesseract` and
Python 3.11 with your package manager, then run the steps in
`setup-linux.sh` by hand.

```bash
git clone https://github.com/soroushsaririan/PDFtoText.git
cd PDFtoText
./setup-linux.sh
```

The Linux script installs `torch`/`torchvision` from the default PyPI
index rather than the CPU-only build, so it picks up an NVIDIA GPU
automatically if drivers are present (falls back to CPU cleanly if not).
It tries `python3.11` first, falls back to the deadsnakes PPA if your
distro's default apt repos don't have it (common on newer Ubuntu releases,
which ship `python3.12` instead), and if that's unreachable too (e.g. a
locked-down network), falls back to whatever Python 3.10+ is already on
the system with a warning -- it doesn't just fail.

> If `git clone` says the destination already exists, either `cd` into
> the existing folder and run `git pull` instead of cloning again, or
> remove the old folder first (`rm -rf PDFtoText`) if it's not needed.

## Running it

Against the bundled sample data:

```bash
./.venv/bin/python src/run_analysis.py
```

Against the real USB drive (macOS mounts under `/Volumes/`, Linux
typically under `/media/$USER/` or `/mnt/`):

```bash
./.venv/bin/python src/run_analysis.py --base-dir /Volumes/USB --out-dir outputs_real       # macOS
./.venv/bin/python src/run_analysis.py --base-dir /media/$USER/USB --out-dir outputs_real   # Linux
```

Output goes to `outputs/reports/` (one markdown report per participant,
with pain/adherence/mileage pulled into a table) and `outputs/summary.csv`
(one row per participant-week, ready for Excel/Python/R). Anything found
that couldn't be matched to a participant/visit/week is listed in each
report under "Files Not Processed" and in `outputs/unmatched_files.csv`,
rather than being silently skipped.

## Command reference

| Flag | What it does |
|---|---|
| `--base-dir PATH` | Folder with `VPT001`, `VPT002`, ... subfolders (default: `sample_usb/`) |
| `--out-dir PATH` | Where reports/CSVs/cache are written (default: `outputs/`) |
| `--keywords-file PATH` | Override the keyword list (default: `config/keywords.txt`) |
| `--engine {vlm,trocr}` | Handwriting engine to use (default: `vlm`) |
| `--model NAME` | Use a different handwriting model, e.g. `Qwen/Qwen2.5-VL-7B-Instruct` or `microsoft/trocr-large-handwritten` |
| `--no-vlm` | Skip the handwriting model; printed/checkbox/circled fields only (fast) |
| `--participants ID ...` | Only process specific participants, e.g. `VPT001 VPT003` |
| `--force` | Ignore the OCR cache and redo everything |

## Adding keywords

Edit `config/keywords.txt` -- one phrase per line, `#` comments allowed.
Every run automatically searches for them across each participant's OCR'd
text and lists matches with context in that participant's report, plus a
combined `outputs/keyword_hits.csv`.

## Regenerating the sample data

```bash
./.venv/bin/python src/generate_sample_data.py
```

## Re-installing from scratch

Re-run `./setup.sh` (macOS) or `./setup-linux.sh` (Linux) -- both are safe
to run again.
