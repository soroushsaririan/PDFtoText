# handwriting_analysis

Analyzes the handwritten weekly PT session forms, CRF visit forms, and exit
surveys collected for each study participant (VPT001, VPT002, ...) on the
study USB drive, using an open-source handwriting-recognition model, and
produces a per-participant report plus a combined CSV for stats.

## What's here

```
handwriting_analysis/
  setup.sh                   <- run this first, see Quick start below
  requirements.txt
  config/
    keywords.txt            <- add your keywords here later (see below)
  src/
    pdf_form_kit.py          helper used only to draw the synthetic sample PDFs
    sample_data_content.py   fake data for the 5 sample participants
    generate_sample_data.py  builds sample_usb/ (synthetic test data)
    ocr_pipeline.py          the actual OCR engine (Tesseract + Qwen2.5-VL hybrid)
    field_extraction.py      turns OCR'd lines into structured CRF fields
    analyze_participant.py   walks one VPT folder, OCRs every PDF in it
    report_generation.py     renders a participant's results as markdown
    run_analysis.py          <- the script you run day-to-day, once set up
  sample_usb/                synthetic 5-participant dataset, generated (see below)
  outputs/                   generated: reports, summary.csv, OCR cache
```

## Quick start

Clone the repo and run the setup script -- it installs everything
(Homebrew's `tesseract`, a Python 3.11 virtual environment, all Python
dependencies), generates the synthetic sample data, and runs a first
analysis pass against it, all in one go:

```bash
git clone https://github.com/ssaririan/PDFtoText.git
cd PDFtoText
./setup.sh
```

Requires [Homebrew](https://brew.sh) already installed. The first run also
downloads the Qwen2.5-VL handwriting model (~7 GB, one-time, cached under
`~/.cache/huggingface`) -- make sure you have that much disk space free.

After that, day to day you just run (from inside the project folder):

```bash
./.venv/bin/python src/run_analysis.py
```

This writes:
- `outputs/reports/VPT001_report.md` (one per participant) - a readable
  summary of intake, every weekly session, the final visit, and the exit
  survey, with pain/adherence/mileage pulled out into a table.
- `outputs/summary.csv` - one row per participant-week, all fields as
  columns, ready to open in Excel or read into Python/R for stats.
- `outputs/rendered_pages/` and `outputs/ocr_cache/` - intermediate PNG
  renders and cached OCR JSON, so re-runs are fast (add `--force` to
  bypass the cache after re-running OCR-pipeline code changes).

## Running it on the real USB drive

```bash
./.venv/bin/python src/run_analysis.py --base-dir /Volumes/USB --out-dir outputs_real
```

`--base-dir` just needs to be a folder containing `VPT001/`, `VPT002/`, ...
subfolders. Within each participant's folder, `analyze_participant.py`
searches **recursively** and matches files by pattern rather than requiring
an exact layout, because the real drive isn't perfectly consistent from
participant to participant:

- Any PDF with "CRF visit N" in its name, anywhere in the folder, is a CRF
  visit (case-insensitive, doesn't need to start with the participant ID).
- Any PDF with "exit" in its name is the exit survey.
- Anything with "week N" in its name *or in the name of a containing
  folder* counts as that week's file -- this covers `Week N/` subfolders,
  files sitting directly in `weekly exercise log/` with the week number in
  the filename instead, and files nested inside an extra folder like
  `week 2 Email/`. `.jpg`/`.jpeg`/`.png` are read the same as `.pdf`. If a
  week has more than one matching file, all of them are OCR'd and combined
  rather than picking just one.
- `DARI FUNCTION`/`DARI RUNNING`/`Video` folders (gait-report exports,
  videos) are skipped entirely -- never descended into, regardless of what
  else is in them.
- A `.docx` matching a week number and containing no PDF/image for that
  week is treated as a missed-visit note (the "No Week N ... .docx"
  convention some folders use).
- **Anything else** -- a file that doesn't match any of the above -- is
  not silently dropped. It's collected and shown in that participant's
  report under "Files Not Processed", plus a combined
  `outputs/unmatched_files.csv` across everyone, so you can check whether
  something real was missed due to unexpected naming.

Useful flags:
- `--no-vlm` - skip the (slow) handwriting model and only pull
  printed/checkbox/circled-answer fields. Good for a fast first pass.
- `--model Qwen/Qwen2.5-VL-7B-Instruct` - use a bigger Qwen2.5-VL checkpoint
  instead of the default 3B one (see "Choosing the model" below).
- `--participants VPT001 VPT003` - only process specific participants.
- `--force` - ignore the OCR cache and redo everything.

## Adding keywords (for later)

`config/keywords.txt` is a plain list, one phrase per line (`#` comments
allowed), currently empty on purpose per your request. Whenever you add
terms there (e.g. `sharp pain`, `numbness`, `IT band`, `missed`), every
future run of `run_analysis.py` will automatically search for them across
all of a participant's OCR'd text (printed and handwritten) and list every
match with surrounding context in that participant's report, plus a
combined `outputs/keyword_hits.csv` across everyone.

## How the OCR works

Real filled-in CRF pages mix machine-printed labels/checkboxes with
handwritten answers on the same page, so a single OCR pass isn't enough.
`ocr_pipeline.py` does a hybrid pass per page:

1. **Tesseract** reads the whole page and reports a confidence per word.
   It's reliable on the printed form text, unreliable on handwriting. This
   is also how the pipeline knows *where* the printed labels and the
   handwritten regions are, and it's kept for that even though it's no
   longer doing the actual handwriting reading.
2. Any word Tesseract wasn't confident about is treated as a handwriting
   candidate: that region is cropped from the full-resolution page and
   read with **Qwen2.5-VL-3B-Instruct**, a vision-language model, instead
   of a plain character-recognition OCR model. Critically, the prompt
   includes the printed question text immediately before the field (e.g.
   `"Pain RIGHT NOW (0-10 scale):"`) as context, and asks the model to
   ignore any dotted fill-in-the-blank leader line and to reply with only
   the answer, or `"[illegible]"` if nothing is legible. A pure OCR model
   like TrOCR structurally can't use that kind of context; a
   vision-language model can, and the benchmark below is why this project
   switched to one.
3. Crossed-out corrections (someone wrote the wrong thing, scribbled over
   it, and wrote the right answer next to it) no longer need special-case
   pixel-density detection code the way they did with TrOCR: the prompt
   just tells the model to ignore crossed-out text and report only the
   final corrected answer, and it's generally able to do that directly
   from the image, the same way a person reading the form would --
   see `VPT005` in the sample data for a worked example. A lightweight
   ink-density check is still used to widen the crop when something looks
   like it might have a scratched-out correction nearby, so the model
   actually gets to see the correction in the same image.
4. Circled "Yes / No" answers aren't something a text model reads as
   text either way. The pipeline instead looks at which of the two words
   Tesseract *failed* to read at all (circling ink reliably breaks word
   recognition) and, as a fallback, checks for non-text-colored ink around
   each option. This part is unchanged and doesn't involve either model.
5. Unlike TrOCR, this model reports its own uncertainty directly as text
   (it says `"[illegible]"` when asked to and it can't read something)
   rather than needing a separate numeric confidence-score threshold --
   simpler, and the benchmark below shows it uses that option sensibly
   rather than only ever guessing.

### Choosing the model

`Qwen/Qwen2.5-VL-3B-Instruct` is the default. It needs more RAM and disk
than TrOCR did (~7 GB download, cached under `~/.cache/huggingface`) and is
slower per field, but is meaningfully more accurate -- see the benchmark
below. If you have the hardware for it, a bigger checkpoint like
`Qwen/Qwen2.5-VL-7B-Instruct` should do even better; switch with `--model
Qwen/Qwen2.5-VL-7B-Instruct` (not verified against this project's
benchmark, so accuracy/speed for that size is your own tradeoff to check).

## About `sample_usb/` (the synthetic test data)

Since I don't have access to real participant handwriting, `sample_usb/`
is generated (`src/generate_sample_data.py`) from made-up data for five
fictional participants, laid out exactly like the real drive, with pages
built to visually resemble the real CRF templates (`*.docx` files you
shared): printed labels/checkboxes plus "handwritten" answers rendered in
a macOS script font (a different one per participant) with small
per-character jitter so it isn't perfectly straight computer text.

Three participants (`VPT001`-`VPT003`) use normal, fairly legible
handwriting. Two are deliberately hard cases, to stress-test the pipeline
rather than flatter it:

- **`VPT004`** writes very messily throughout (heavy jitter, cramped/
  overlapping letters, a drifting baseline) -- expect a lot of fields to
  come back `"Uncertain"`/`"[illegible]"` for this participant. That's the
  correct, honest behavior, not a bug.
- **`VPT005`** has two fields where the participant wrote the wrong thing,
  scratched it out, and wrote the correct answer next to it. Week 2's
  version of this (a longer scratched-out phrase, "1.5 hours", followed by
  a clearly-separated "90") is resolved correctly. Week 3's version (a
  single scratched-out digit with the correction written right next to it,
  both quite small) is not -- the model reads the scribble itself as if it
  were a number ("8", the crossed-out value) rather than recognizing it as
  scratch marks and reporting the "3" written after it. This is a real,
  reproducible limitation of the 3B model on this specific kind of dense,
  small-scale scribble, not a pipeline bug -- rephrasing the prompt several
  different ways didn't fix it in testing. A bigger checkpoint
  (`Qwen/Qwen2.5-VL-7B-Instruct`) may do better here; that's untested
  locally since it didn't fit on this machine's free disk space at the
  time. Across all five participants' numeric fields checked against the
  known sample data, this is the *only* field that comes back confidently
  wrong (1 out of 171) -- everything else is either correct or honestly
  flagged `"Uncertain"`.

**Important caveat:** a script font rendered from vector text still does
not look like real pen handwriting (no pen pressure, natural stroke
variation, or personal letterforms), so this remains an imperfect proxy
for the real thing. Qwen2.5-VL, being a general vision-language model
rather than something narrowly trained on one handwriting dataset, handles
this synthetic font noticeably better than TrOCR did (don't be surprised
to see mostly-correct fields on `sample_usb/` now), but that's not the
same guarantee as "this will work equally well on genuinely messy pen
handwriting" -- the `benchmark/` tools further down test against real
handwriting specifically for that reason. What always validates cleanly
against this sample data (and should work the same on the real drive
regardless of handwriting quality) is everything that doesn't depend on
handwriting recognition at all: which checkboxes are checked, which
"Yes/No" answer is circled, all printed text, the file/folder discovery
logic, and the report/CSV generation. Regenerate the sample data any time
with:

```bash
./.venv/bin/python src/generate_sample_data.py
```

## Benchmarking the model against real handwriting

`benchmark/run_benchmark.py` answers a different question than `sample_usb/`
does: not "does the pipeline's plumbing work," but "how accurate is the
handwriting model actually, on genuine pen handwriting." It downloads real,
human-written word images with ground-truth transcriptions from the **IAM
Handwriting Database** (via a Kaggle mirror -- the same dataset TrOCR was
trained on), runs them through a model, and reports Character Error Rate
(CER), Word Error Rate (WER), and exact-match accuracy, plus the
best/worst individual examples.

```bash
./.venv/bin/python benchmark/run_benchmark.py --n 300
```

Requires a Kaggle account with an API token at `~/.kaggle/kaggle.json`
(free, from kaggle.com/settings). Data lives in `benchmark/data/`
(~1.3 GB, downloaded once) and results are saved to
`benchmark/results_<model>.csv`. Flags: `--n` (sample size, default 300),
`--seed` (for a reproducible sample), `--model` (e.g.
`microsoft/trocr-large-handwritten`, to compare against the default).

There's a second script, `benchmark/run_benchmark_vlm.py`, that runs the
*same* IAM sample through **Qwen2.5-VL-3B-Instruct** -- the model this
project's pipeline now actually uses -- for a direct comparison against
TrOCR. It runs fine from the main `.venv` (which already has everything
`ocr_pipeline.py` needs):

```bash
./.venv/bin/python benchmark/run_benchmark_vlm.py --n 300
```

Qwen2.5-VL-3B-Instruct is a ~7 GB download (cached under `~/.cache/huggingface`
once fetched) -- make sure there's disk space free before running this.

**Results from 300 real IAM word images (seed 42, restricted to words of
3+ characters -- see caveats below for why):**

| Model | CER | WER | Exact match |
|---|---|---|---|
| trocr-base-handwritten (pipeline default) | 0.326 | 0.745 | 47.2% |
| trocr-large-handwritten | 0.228 | 0.637 | 56.1% |
| **Qwen2.5-VL-3B-Instruct** | 0.252 | **0.344** | **66.5%** |

Qwen2.5-VL gets meaningfully more real words fully correct than either
TrOCR size (67% vs. 56% at best) and makes far fewer full-word errors
(WER almost half of TrOCR-large's) -- consistent with the theory that a
vision-language model's language understanding helps it disambiguate
messy strokes in a way a pure character-shape recognizer can't. Its raw
character-level accuracy (CER) is about the same as TrOCR-large, not
better, so it's the "gets the whole word right" behavior that's the real
win here, not marginally-cleaner letters.

Caveats worth knowing before trusting this table:
- All three numbers exclude words under 3 characters, because on the
  unfiltered 300-word sample Qwen2.5-VL's *raw* CER looks much worse
  (0.62) purely from an artifact of the metric: for a 1-character ground
  truth like `,` or `.`, replying `[illegible]` (which Qwen2.5-VL does a
  lot, and TrOCR never does at all) scores as a huge character-level
  error even though refusing to guess is the *correct*, safer behavior for
  this project (it maps directly onto this pipeline's own
  `"Uncertain"`/`"[illegible]"` convention). Full unfiltered numbers and
  every individual prediction are in `benchmark/results_*.csv`.
- Qwen2.5-VL also says `[illegible]` on about 7.5% of genuine 3+ character
  words it could plausibly have guessed at -- more conservative than
  TrOCR, which never refuses. Whether that's a good trade depends on
  whether you'd rather have a wrong guess or an honest "couldn't read
  this" for a given field.
- This still evaluates **isolated single words with no sentence/field
  context**. A real CRF field crop additionally has its printed label
  right next to it ("Pain RIGHT NOW (0-10 scale):"), which a
  vision-language model can read and use directly -- something TrOCR
  structurally cannot do. That's a further advantage for the VLM approach
  in this specific project that this word-level benchmark doesn't even
  capture yet.

**This benchmark is what justified switching the main pipeline over to
Qwen2.5-VL** -- `ocr_pipeline.py` now uses it as described above, in place
of TrOCR, with the field's printed label passed in as prompt context (which
this word-level benchmark doesn't even capture, so real-world field
accuracy should be better than the table above suggests). The
`run_benchmark.py` (TrOCR) and `run_benchmark_vlm.py` (Qwen2.5-VL) scripts
both remain in `benchmark/` as a way to re-check this decision later --
e.g. if a newer model comes out, or if you want to test whether a bigger
Qwen2.5-VL checkpoint is worth the slower inference for your data. Both
now run fine from the single main `.venv` (the separate `.venv-vlm`
mentioned in older notes is no longer needed -- it only existed because
the main environment used to be pinned to an older `transformers` for
TrOCR's tokenizer).

The practical takeaway: on genuine handwriting, expect real, multi-
character words to be read correctly noticeably more often than anything
you saw from the old TrOCR pipeline (no more "Download as PDFPrintable
version"-style hallucinations), but this is still not perfect on isolated
short answers -- treat every field this pipeline extracts as a draft to
verify, not a final transcription, especially for anything clinically
important.

## Re-installing from scratch

Just re-run `./setup.sh` -- it's safe to run again (skips `tesseract` if
already installed, recreates `.venv` if missing). What it does, if you'd
rather run the steps manually:

```bash
brew install tesseract
python3.11 -m venv .venv
./.venv/bin/pip install --upgrade pip
./.venv/bin/pip install -r requirements.txt
```
