# handwriting_analysis

Analyzes handwritten weekly PT session forms, CRF visit forms, and exit
surveys collected for each study participant (VPT001, VPT002, ...) on the
study USB drive, and produces a per-participant report plus a combined
CSV for stats.

## Requirements

- macOS with [Homebrew](https://brew.sh)
- Python 3.11
- ~15 GB free disk space (one-time handwriting-model download)
- A free Kaggle account -- only needed for the optional benchmark tools in `benchmark/`

## Setup

```bash
git clone https://github.com/soroushsaririan/PDFtoText.git
cd PDFtoText
./setup.sh
```

This installs `tesseract`, creates a Python 3.11 virtual environment,
installs all dependencies, generates the synthetic sample data, and runs
a first analysis pass against it. The first run also downloads the
handwriting model (~7 GB, one-time, cached under `~/.cache/huggingface`).

## Running it

Against the bundled sample data:

```bash
./.venv/bin/python src/run_analysis.py
```

Against the real USB drive:

```bash
./.venv/bin/python src/run_analysis.py --base-dir /Volumes/USB --out-dir outputs_real
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
| `--model NAME` | Use a different handwriting model, e.g. `Qwen/Qwen2.5-VL-7B-Instruct` |
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

Re-run `./setup.sh` -- it's safe to run again.
