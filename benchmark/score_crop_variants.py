"""Read every ground-truth field from several crop framings (field_reader's
CROP_VARIANTS) and several prompt styles, then score each single reading
and each way of combining them.

    ./.venv/bin/python benchmark/score_crop_variants.py
    ./.venv/bin/python benchmark/score_crop_variants.py --rescore   # re-score the saved readings, no model

Crops are rebuilt from the page renders saved in each crop folder's
pages/ subfolder, using the same label location the
original crops came from (the "label" variant is byte-identical to the
saved crop). All readings go to
real_ground_truth_sample_scored_crop_variants.xlsx -- one column per
variant x style -- so new ways of combining them can be scored without
re-running the model.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

BENCH_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH_DIR.parent / "src"))
sys.path.insert(0, str(BENCH_DIR))

import consensus  # noqa: E402
import field_reader  # noqa: E402
import prompts  # noqa: E402
from ocr_pipeline import VLM_MAX_NEW_TOKENS, VLM_MAX_NEW_TOKENS_DEFAULT, _upscale_for_vlm  # noqa: E402
from score_ground_truth import (  # noqa: E402
    SRC_PATH,
    STRUCTURED_MAX_NEW_TOKENS,
    _load_mlx_generator,
    extract_value,
    normalize,
)

STYLES = ("answer", "discriminative")


def out_path(styles):
    suffix = "" if tuple(styles) == STYLES else "_" + "_".join(styles)
    return BENCH_DIR / f"real_ground_truth_sample_scored_crop_variants{suffix}.xlsx"


def build_variants(df):
    pages = {}
    variants = []
    for _, row in df.iterrows():
        pages_dir = BENCH_DIR / Path(row["crop"]).parent / "pages"
        page_path = pages_dir / row["pdf"][:-4] / f"page_{int(row['page']):02d}.png"
        if page_path not in pages:
            image = field_reader.open_page(page_path)
            pages[page_path] = (image, field_reader.page_words(image), np.array(image.convert("L")))
        image, data, gray = pages[page_path]
        lines = field_reader.line_index(data)
        box, line_key = field_reader.label_hits(data, lines, row["label"])[int(row["occurrence"]) - 1]
        max_x1 = field_reader.next_label_start_x(data, lines.get(line_key, []), box[2])
        answer_x1 = max_x1 if max_x1 is not None else image.width
        ceiling = field_reader.ceiling_above(data, box, answer_x1)
        variants.append(field_reader.crop_variants(image, box, row["field_kind"], max_x1=max_x1, gray=gray,
                                                   ceiling=ceiling))
    return variants


def read_all(df, generate, styles=STYLES):
    variants = build_variants(df)
    readings = {f"{v}_{s}": [] for v in field_reader.CROP_VARIANTS for s in styles}
    started = time.time()
    for i, (row, crops) in enumerate(zip(df.itertuples(), variants), start=1):
        for variant in field_reader.CROP_VARIANTS:
            for style in styles:
                crop = crops.get(variant)
                if crop is None:
                    readings[f"{variant}_{style}"].append(None)
                    continue
                budget = STRUCTURED_MAX_NEW_TOKENS if style != "answer" else \
                    VLM_MAX_NEW_TOKENS.get(row.field_kind, VLM_MAX_NEW_TOKENS_DEFAULT)
                prompt = prompts.build_prompt(label_context=row.label, field_kind=row.field_kind, style=style)
                raw = generate(_upscale_for_vlm(crop), prompt, budget)
                readings[f"{variant}_{style}"].append(extract_value(raw, row.field_kind, style))
        shown = {k: v[-1] for k, v in readings.items()}
        print(f"  {i}/{len(df)} {row.participant}/{row.field_key} gt={row.ground_truth!r}: {shown}", flush=True)
    print(f"\n{time.time() - started:.0f}s for {len(df)} fields")
    out = df.copy()
    for col, values in readings.items():
        out[col] = values
    return out


def score(out):
    gt = out["ground_truth"].map(normalize)
    cols = [f"{v}_{s}" for v in field_reader.CROP_VARIANTS for s in STYLES if f"{v}_{s}" in out]
    used = out["field_key"] != "participant_id"

    def acc(values, mask=used):
        hits = [normalize(v) == g for v, g in zip(values, gt)]
        return np.mean([h for h, m in zip(hits, mask) if m])

    print(f"Single readings (accuracy on all {len(out)} / on the {used.sum()} non-participant_id fields):")
    for col in cols:
        print(f"  {col:24s} {acc(out[col], [True] * len(out)):.0%}  /  {acc(out[col]):.0%}")

    combos = {
        "vote: raised+ink (answer)": ["raised_answer", "ink_answer"],
        "vote: label+raised+ink (answer)": ["label_answer", "raised_answer", "ink_answer"],
        "vote: raised (answer+disc)": ["raised_answer", "raised_discriminative"],
        "vote: raised+ink (answer+disc)": ["raised_answer", "raised_discriminative",
                                          "ink_answer", "ink_discriminative"],
        "vote: all six": cols,
    }
    print("\nCombined (value / share auto-accepted / accuracy of auto-accepted), non-participant_id fields:")
    for name, combo_cols in combos.items():
        if any(c not in out for c in combo_cols):
            continue
        decisions = [consensus.decide([row[c] for c in combo_cols], field_kind=row["field_kind"])
                     for _, row in out.iterrows()]
        values = [d.value for d in decisions]
        accepted = np.array([d.status == consensus.AGREED for d in decisions])
        accepted_used = accepted & used.to_numpy()
        hits = np.array([normalize(v) == g for v, g in zip(values, gt)])
        print(f"  {name:36s} {hits[used].mean():.0%}   accepted {accepted_used.sum()}/{used.sum()}"
              f"   accepted-correct {hits[accepted_used].mean() if accepted_used.any() else float('nan'):.0%}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rescore", action="store_true", help="Score the saved readings without running the model.")
    parser.add_argument("--styles", nargs="+", choices=STYLES, default=list(STYLES),
                        help="Prompt styles to read with (answer alone is several times faster).")
    args = parser.parse_args()
    OUT_PATH = out_path(args.styles)

    if args.rescore:
        out = pd.read_excel(OUT_PATH)
        truth = pd.read_excel(SRC_PATH).set_index("crop")["ground_truth"]
        out["ground_truth"] = out["crop"].map(truth)
    else:
        df = pd.read_excel(SRC_PATH)
        truth = df["ground_truth"].astype(str).str.strip()
        missing = df["ground_truth"].isna() | (truth == "") | (truth.str.lower() == "skip")
        if missing.any():
            print(f"Skipping {missing.sum()} row(s) with no ground_truth yet or marked skip.")
            df = df[~missing].reset_index(drop=True)
        generate, model_name = _load_mlx_generator()
        print(f"Model: {model_name}\n")
        out = read_all(df, generate, args.styles)
        out.to_excel(OUT_PATH, index=False)
        print(f"Readings: {OUT_PATH}\n")
    score(out)


if __name__ == "__main__":
    main()
