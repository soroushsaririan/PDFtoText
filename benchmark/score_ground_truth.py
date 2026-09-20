"""Run the model on the hand-checked ground-truth sample and score it.

    ./.venv/bin/python benchmark/score_ground_truth.py

Reads benchmark/real_ground_truth_sample.xlsx (every ground_truth cell must
already be filled in -- see the README section on ground-truth review),
re-runs the model against each row's existing crop image (no re-cropping,
no touching the real participant data), and writes
benchmark/real_ground_truth_sample_scored.xlsx with the model's answer and a
match column added. The ground-truth file itself is never modified.
"""
import argparse
import re
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

BENCH_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BENCH_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ocr_pipeline import (  # noqa: E402
    VLM_MAX_NEW_TOKENS,
    VLM_MAX_NEW_TOKENS_DEFAULT,
    VLM_MIN_CROP_HEIGHT,
    _upscale_for_vlm,
)
from run_prompt_styles import _load_generator  # noqa: E402

SRC_PATH = BENCH_DIR / "real_ground_truth_sample.xlsx"
OUT_PATH = BENCH_DIR / "real_ground_truth_sample_scored.xlsx"

BLANK_WORDS = {"blank", "(blank)"}
ILLEGIBLE_WORDS = {"illegible", "(illegible)", "[illegible]"}


def normalize(value):
    text = str(value).strip().strip("\"'").lower()
    text = re.sub(r"\s+", " ", text)
    if text in BLANK_WORDS:
        return "<blank>"
    if text in ILLEGIBLE_WORDS:
        return "<illegible>"
    text = re.sub(r"^(\d+)\.0$", r"\1", text)  # "4.0" == "4"
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-height", type=int, default=VLM_MIN_CROP_HEIGHT,
                         help="Upscale any crop shorter than this many px before sending it to the model "
                              "(matches ocr_pipeline.VLM_MIN_CROP_HEIGHT by default). 0 disables upscaling.")
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

    print(f"Upscaling: crops below {args.min_height}px tall get scaled up first" if args.min_height
          else "Upscaling: disabled")
    print("Loading the model (first run downloads it) ...")
    generate, model_name = _load_generator(None)
    print(f"Model: {model_name}\n")

    results = []
    for i, row in df.iterrows():
        crop_path = BENCH_DIR / row["crop"]
        crop = Image.open(crop_path).convert("RGB")
        crop = _upscale_for_vlm(crop, min_height=args.min_height)
        budget = VLM_MAX_NEW_TOKENS.get(row["field_kind"], VLM_MAX_NEW_TOKENS_DEFAULT)
        try:
            result = generate(crop, row["prompt"], budget)
        except Exception as exc:
            result = f"[ERROR: {exc}]"
        results.append(result)
        print(f"  {i + 1}/{len(df)}  {row['participant']}/{row['field_key']}: {result!r}")

    df["result"] = results
    df["match"] = [normalize(g) == normalize(r) for g, r in zip(df["ground_truth"], df["result"])]

    accuracy = df["match"].mean()
    print(f"\nOverall: {df['match'].sum()}/{len(df)} correct ({accuracy:.0%})")
    print("\nBy field kind:")
    print(df.groupby("field_kind")["match"].agg(["sum", "count"]).to_string())

    df.to_excel(OUT_PATH, index=False)
    print(f"\nScored sheet: {OUT_PATH}")


if __name__ == "__main__":
    main()
