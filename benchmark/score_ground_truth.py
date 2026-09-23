"""Run a handwriting engine/style/filter combo on the hand-checked
ground-truth sample and score it, or sweep several combos at once.

    ./.venv/bin/python benchmark/score_ground_truth.py                          # answer style, no filter (baseline)
    ./.venv/bin/python benchmark/score_ground_truth.py --style discriminative   # structured multi-field style
    ./.venv/bin/python benchmark/score_ground_truth.py --filter autocontrast
    ./.venv/bin/python benchmark/score_ground_truth.py --engine trocr
    ./.venv/bin/python benchmark/score_ground_truth.py --sweep                  # every style x filter combo below

Reads benchmark/real_ground_truth_sample.xlsx (every ground_truth cell must
already be filled in -- see the README section on ground-truth review),
re-runs the chosen engine/style/filter against each row's existing crop
image (no re-cropping, no touching the real participant data), and writes
benchmark/real_ground_truth_sample_scored_<engine>_<style>_<filter>.xlsx
with the engine's answer and a match column added. The ground-truth file
itself is never modified.

Non-"answer" styles (descriptive/discriminative/paired) return several
numbered fields, not a bare value -- DISCRIMINATIVE_VALUE_FIELD below says
which numbered field in prompts.py's _FIELDS holds the actual answer, and
extract_value() pulls just that line out before comparing to ground truth
(both the raw response and the extracted value are kept in the output sheet).
"""
import argparse
import re
import sys
from pathlib import Path

import pandas as pd
from PIL import Image, ImageEnhance, ImageOps

BENCH_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BENCH_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import prompts  # noqa: E402
from ocr_pipeline import (  # noqa: E402
    VLM_MAX_NEW_TOKENS,
    VLM_MAX_NEW_TOKENS_DEFAULT,
    VLM_MIN_CROP_HEIGHT,
    _trocr_read,
    _upscale_for_vlm,
)
from run_prompt_styles import _load_generator  # noqa: E402

SRC_PATH = BENCH_DIR / "real_ground_truth_sample.xlsx"

BLANK_WORDS = {"blank", "(blank)"}
ILLEGIBLE_WORDS = {"illegible", "(illegible)", "[illegible]"}

# Structured (non-"answer") styles ask for these numbered fields per
# prompts.py's _FIELDS -- this says which one carries the actual answer.
DISCRIMINATIVE_VALUE_FIELD = {
    "number": "VALUE AS WRITTEN",
    "pain": "VALUE 0-10",
    "yesno": "ANSWER",
    "text": "TEXT AS WRITTEN",
    "notes": "TRANSCRIPTION",
    "checkbox": "MARKED OPTIONS",
}

# A non-"answer" style has to describe/classify before answering, so it
# needs much more room than a bare-value "answer" reply.
STRUCTURED_MAX_NEW_TOKENS = 400

FILTERS = ("none", "autocontrast", "contrast", "sharpen")


def apply_filter(crop, name):
    if name == "none":
        return crop
    if name == "autocontrast":
        return ImageOps.autocontrast(crop, cutoff=1)
    if name == "contrast":
        return ImageEnhance.Contrast(crop).enhance(1.8)
    if name == "sharpen":
        return ImageEnhance.Sharpness(crop).enhance(2.5)
    raise ValueError(f"Unknown filter: {name!r}")


def extract_value(raw_text, field_kind, style):
    if style == "answer":
        return raw_text
    field_name = DISCRIMINATIVE_VALUE_FIELD.get(field_kind)
    if not field_name:
        return raw_text
    for line in raw_text.splitlines():
        line = line.strip()
        m = re.match(rf"^(?:\d+\.\s*)?{re.escape(field_name)}\s*:\s*(.*)$", line, re.I)
        if m:
            return m.group(1).strip()
    return None  # style didn't follow the numbered format -- couldn't find the field


def normalize(value):
    text = str(value).strip().strip("\"'").lower()
    text = re.sub(r"\s+", " ", text)
    if text in BLANK_WORDS:
        return "<blank>"
    if text in ILLEGIBLE_WORDS:
        return "<illegible>"
    text = re.sub(r"^(\d+)\.0$", r"\1", text)  # "4.0" == "4"
    return text


def run_one(df, engine, style, filt, min_height, generate, verbose=True):
    results, extracted = [], []
    for i, row in df.iterrows():
        crop_path = BENCH_DIR / row["crop"]
        crop = Image.open(crop_path).convert("RGB")
        crop = _upscale_for_vlm(crop, min_height=min_height)
        crop = apply_filter(crop, filt)
        budget = STRUCTURED_MAX_NEW_TOKENS if style != "answer" else \
            VLM_MAX_NEW_TOKENS.get(row["field_kind"], VLM_MAX_NEW_TOKENS_DEFAULT)
        prompt = prompts.build_prompt(label_context=row["label"], field_kind=row["field_kind"], style=style)
        try:
            raw = generate(crop, prompt, budget)
        except Exception as exc:
            raw = f"[ERROR: {exc}]"
        value = extract_value(raw, row["field_kind"], style)
        results.append(raw)
        extracted.append(value)
        if verbose:
            shown = value if style != "answer" else raw
            print(f"  {i + 1}/{len(df)}  {row['participant']}/{row['field_key']}: {shown!r}", flush=True)

    out = df.copy()
    out["raw_result"] = results
    out["result"] = extracted
    out["match"] = [normalize(g) == normalize(r) for g, r in zip(out["ground_truth"], out["result"])]
    return out


def summarize(label, scored):
    acc = scored["match"].mean()
    print(f"{label}: {scored['match'].sum()}/{len(scored)} correct ({acc:.0%})")
    return acc


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--engine", choices=("vlm", "trocr"), default="vlm",
                         help="Handwriting engine to score: 'vlm' (Qwen2.5-VL, default) or 'trocr'.")
    parser.add_argument("--style", choices=prompts.PROMPT_STYLES, default="answer",
                         help="Prompt style (ignored for --engine trocr, which takes no prompt).")
    parser.add_argument("--filter", choices=FILTERS, default="none",
                         help="Image preprocessing filter applied after upscaling, before the model sees it.")
    parser.add_argument("--min-height", type=int, default=VLM_MIN_CROP_HEIGHT,
                         help="Upscale any crop shorter than this many px before sending it to the model "
                              "(matches ocr_pipeline.VLM_MIN_CROP_HEIGHT by default). 0 disables upscaling.")
    parser.add_argument("--sweep", action="store_true",
                         help="Run every (style x filter) combo for --engine vlm and report all of them, "
                              "ranked, instead of just one run.")
    parser.add_argument("--limit", type=int, default=0,
                         help="Only score the first N rows (0 = all 44) -- for a quick timing/directional "
                              "read before committing to a slow style like discriminative/paired.")
    args = parser.parse_args()

    if not SRC_PATH.exists():
        print(f"Not found: {SRC_PATH}")
        return

    df = pd.read_excel(SRC_PATH)
    missing = df["ground_truth"].isna() | (df["ground_truth"].astype(str).str.strip() == "")
    if missing.any():
        print(f"{missing.sum()} row(s) still have no ground_truth -- fill those in first:")
        print(df[missing][["participant", "pdf", "field_key"]].to_string())
        return
    if args.limit:
        df = df.head(args.limit).reset_index(drop=True)

    print("Loading the model (first run downloads it) ...")
    if args.engine == "vlm":
        generate, model_name = _load_generator(None)
    else:
        model_name = "microsoft/trocr-base-handwritten"

        def generate(crop, prompt, max_new_tokens):
            text, _conf = _trocr_read(crop)
            return text
    print(f"Model: {model_name}\n")

    if not args.sweep:
        print(f"Engine: {args.engine}  Style: {args.style}  Filter: {args.filter}")
        scored = run_one(df, args.engine, args.style, args.filter, args.min_height, generate)
        summarize("\nOverall", scored)
        print("\nBy field kind:")
        print(scored.groupby("field_kind")["match"].agg(["sum", "count"]).to_string())
        out_path = BENCH_DIR / f"real_ground_truth_sample_scored_{args.engine}_{args.style}_{args.filter}.xlsx"
        scored.to_excel(out_path, index=False)
        print(f"\nScored sheet: {out_path}")
        return

    # --sweep: every style x filter combo, vlm only (trocr takes no prompt/filter axis worth sweeping here)
    combos = [(style, filt) for style in prompts.PROMPT_STYLES for filt in FILTERS]
    leaderboard = []
    for style, filt in combos:
        print(f"=== style={style} filter={filt} ===")
        scored = run_one(df, "vlm", style, filt, args.min_height, generate, verbose=False)
        acc = summarize("  Overall", scored)
        out_path = BENCH_DIR / f"real_ground_truth_sample_scored_vlm_{style}_{filt}.xlsx"
        scored.to_excel(out_path, index=False)
        leaderboard.append((acc, style, filt, out_path))
        print()

    leaderboard.sort(reverse=True)
    print("=" * 60)
    print("LEADERBOARD (best first)")
    for acc, style, filt, out_path in leaderboard:
        print(f"  {acc:.0%}  style={style:15s} filter={filt}")


if __name__ == "__main__":
    main()
