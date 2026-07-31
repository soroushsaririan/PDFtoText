import argparse
import re
import sys
from pathlib import Path

import jiwer
import pandas as pd
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

BENCH_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH_DIR))

from run_benchmark import image_path_for, load_sample

PROMPT = (
    "Read the single handwritten word in this image. "
    "Reply with ONLY the exact text you see, no punctuation added, no explanation. "
    "If you truly cannot read it, reply with exactly: [illegible]"
)


def load_model(model_name):
    processor = AutoProcessor.from_pretrained(model_name)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, torch_dtype="auto")
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    model.to(device)
    model.eval()
    return processor, model, device


def transcribe(processor, model, device, pil_image):
    messages = [{"role": "user", "content": [{"type": "image", "image": pil_image}, {"type": "text", "text": PROMPT}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to(device)
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=24, do_sample=False)
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return clean_output(output_text)


def clean_output(text):
    text = text.strip()
    text = text.strip("\"'")
    text = re.sub(r"\s+", " ", text)
    return text


def run_benchmark(n, seed, model_name):
    sample = load_sample(n, seed)
    print(f"Model      : {model_name}")
    print(f"Sample size: {len(sample)} words (status=ok) from IAM Words, seed={seed}")
    print()

    processor, model, device = load_model(model_name)
    print(f"Device     : {device}")
    print()

    rows = []
    for i, entry in enumerate(sample, start=1):
        img = Image.open(image_path_for(entry["id"])).convert("RGB")
        try:
            pred = transcribe(processor, model, device, img)
        except Exception as exc:
            pred = f"[ERROR: {exc}]"
        gt = entry["text"]
        cer = jiwer.cer(gt, pred) if gt else None
        wer = jiwer.wer(gt, pred) if gt else None
        rows.append({"id": entry["id"], "ground_truth": gt, "prediction": pred, "cer": cer, "wer": wer})
        if i % 10 == 0 or i == len(sample):
            print(f"  {i}/{len(sample)} processed")

    df = pd.DataFrame(rows)
    model_slug = model_name.split("/")[-1].lower()
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
    parser = argparse.ArgumentParser(description="Benchmark a Qwen2.5-VL model against real IAM handwriting samples.")
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    args = parser.parse_args()
    run_benchmark(args.n, args.seed, args.model)
