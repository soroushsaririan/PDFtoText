import argparse
import random
import sys
from pathlib import Path

import jiwer
import pandas as pd
from PIL import Image

BENCH_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BENCH_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

WORDS_TXT = BENCH_DIR / "data" / "words_meta" / "words.txt"
IMAGES_ROOT = BENCH_DIR / "data" / "iam_words_full" / "iam_words" / "words"


def parse_words_txt(path):
    entries = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(" ", 8)
        if len(parts) < 9:
            continue
        word_id, status = parts[0], parts[1]
        transcription = parts[8]
        entries.append({"id": word_id, "status": status, "text": transcription})
    return entries


def image_path_for(word_id):
    parts = word_id.split("-")
    form_id = "-".join(parts[:2])
    form_prefix = parts[0]
    return IMAGES_ROOT / form_prefix / form_id / f"{word_id}.png"


def load_sample(n, seed):
    entries = parse_words_txt(WORDS_TXT)
    usable = [e for e in entries if e["status"] == "ok" and e["text"] and image_path_for(e["id"]).exists()]
    rng = random.Random(seed)
    rng.shuffle(usable)
    return usable[:n]


def run_benchmark(n, seed, model_override=None):
    import os

    if model_override:
        os.environ["TROCR_MODEL_NAME"] = model_override

    from ocr_pipeline import TROCR_MODEL_NAME, _trocr_read

    sample = load_sample(n, seed)
    print(f"Model      : {TROCR_MODEL_NAME}")
    print(f"Sample size: {len(sample)} words (status=ok) from IAM Words, seed={seed}")
    print()

    rows = []
    for i, entry in enumerate(sample, start=1):
        img = Image.open(image_path_for(entry["id"])).convert("RGB")
        try:
            pred, logprob = _trocr_read(img)
        except Exception as exc:
            pred, logprob = f"[ERROR: {exc}]", 0.0
        gt = entry["text"]
        cer = jiwer.cer(gt, pred) if gt else None
        wer = jiwer.wer(gt, pred) if gt else None
        rows.append({"id": entry["id"], "ground_truth": gt, "prediction": pred, "logprob": logprob, "cer": cer, "wer": wer})
        if i % 25 == 0 or i == len(sample):
            print(f"  {i}/{len(sample)} processed")

    df = pd.DataFrame(rows)
    model_slug = TROCR_MODEL_NAME.split("/")[-1]
    out_path = BENCH_DIR / f"results_{model_slug}.csv"
    df.to_csv(out_path, index=False)

    overall_cer = jiwer.cer(list(df["ground_truth"]), list(df["prediction"]))
    overall_wer = jiwer.wer(list(df["ground_truth"]), list(df["prediction"]))
    exact_match = (df["ground_truth"] == df["prediction"]).mean()

    print()
    print("=" * 60)
    print(f"Overall CER (character error rate): {overall_cer:.4f}")
    print(f"Overall WER (word error rate)      : {overall_wer:.4f}")
    print(f"Exact-match accuracy               : {exact_match:.4f}")
    print(f"Results saved to                   : {out_path}")
    print("=" * 60)

    print("\nBest 10 (lowest CER):")
    for _, r in df.sort_values("cer").head(10).iterrows():
        print(f"  gt={r['ground_truth']!r:20s} pred={r['prediction']!r:20s} cer={r['cer']:.2f}")

    print("\nWorst 10 (highest CER):")
    for _, r in df.sort_values("cer", ascending=False).head(10).iterrows():
        print(f"  gt={r['ground_truth']!r:20s} pred={r['prediction']!r:20s} cer={r['cer']:.2f}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark the project's TrOCR handwriting model against real IAM handwriting samples.")
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()
    run_benchmark(args.n, args.seed, args.model)
