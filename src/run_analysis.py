#!/usr/bin/env python
import argparse
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_participant import analyze_participant, discover_participants
from field_extraction import load_keywords
from report_generation import render_participant_report

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_ENV_VARS = {"mlx": "MLX_MODEL_NAME", "vlm": "VLM_MODEL_NAME", "trocr": "TROCR_MODEL_NAME"}


def default_engine():
    """mlx needs Apple Silicon and the mlx-vlm package; anywhere else the
    transformers-based 3B is the default."""
    import importlib.util
    import platform

    if platform.system() == "Darwin" and platform.machine() == "arm64" and importlib.util.find_spec("mlx_vlm"):
        return "mlx"
    return "vlm"


def build_weekly_rows(result):
    rows = []
    for wk in result["weeks"]:
        row = {"participant_id": result["participant_id"], "week": wk["week"], "status": wk["status"]}
        if wk["status"] == "completed":
            f = dict(wk["fields"])
            f.pop("participant_id", None)
            f["exercise_types"] = "; ".join(f.get("exercise_types") or [])
            f["cues_list"] = "; ".join(f.get("cues_list") or [])
            row.update(f)
            flagged = [r["field"] for r in result.get("review", [])
                       if r["form"] == "weekly" and r["week"] == wk["week"]]
            row["needs_review"] = "; ".join(flagged)
        rows.append(row)
    return rows


def build_log_rows(result):
    """One row per Exercise Log day, so the weekly totals can be checked
    against the cells they were read from."""
    rows = []
    for wk in result["weeks"]:
        for day in wk.get("log_days") or []:
            rows.append({"participant_id": result["participant_id"], "week": wk["week"], **day})
    return rows


def main():
    parser = argparse.ArgumentParser(description="OCR and analyze every participant folder under --base-dir.")
    parser.add_argument("--base-dir", default=str(PROJECT_ROOT / "sample_usb"),
                         help="Folder containing VPT001, VPT002, ... subfolders (a USB mount point, or sample_usb).")
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "outputs"), help="Where reports/CSVs/OCR cache are written.")
    parser.add_argument("--keywords-file", default=str(PROJECT_ROOT / "config" / "keywords.txt"))
    parser.add_argument("--model", default=None, help="Override the handwriting VLM, e.g. Qwen/Qwen2.5-VL-7B-Instruct.")
    parser.add_argument("--engine", choices=("mlx", "vlm", "trocr"), default=default_engine(),
                         help="Handwriting reading engine: 'mlx' (Qwen2.5-VL-7B 4-bit, Apple Silicon; default there), "
                              "'vlm' (Qwen2.5-VL-3B via transformers; default elsewhere) or 'trocr'.")
    parser.add_argument("--passes", type=int, default=2,
                         help="mlx only: read each handwritten field this many ways (crop/prompt combos) and flag "
                              "fields whose readings disagree for review. 1 = single read, no review flags.")
    parser.add_argument("--no-vlm", action="store_true", help="Skip the handwriting model; printed/checkbox fields only.")
    parser.add_argument("--force", action="store_true", help="Ignore the OCR cache and re-run OCR on every PDF.")
    parser.add_argument("--participants", nargs="*", default=None, help="Only process these participant IDs, e.g. VPT001 VPT002.")
    args = parser.parse_args()

    if args.model:
        # ocr_pipeline is already imported (via analyze_participant) and read
        # its model names from the environment at import time, so setting the
        # env var alone would be too late -- set the module attribute too.
        import ocr_pipeline

        os.environ[MODEL_ENV_VARS[args.engine]] = args.model
        setattr(ocr_pipeline, MODEL_ENV_VARS[args.engine], args.model)

    base_dir = Path(args.base_dir)
    out_dir = Path(args.out_dir)
    reports_dir = out_dir / "reports"
    image_dir = out_dir / "rendered_pages"
    ocr_cache_dir = out_dir / "ocr_cache"
    for d in (out_dir, reports_dir, image_dir, ocr_cache_dir):
        d.mkdir(parents=True, exist_ok=True)

    keywords = load_keywords(args.keywords_file)
    participants = discover_participants(base_dir)
    if args.participants:
        wanted = set(args.participants)
        participants = [p for p in participants if p.name in wanted]

    if not participants:
        print(f"No VPT participant folders found under: {base_dir}")
        return

    print(f"Base folder : {base_dir}")
    print(f"Participants: {[p.name for p in participants]}")
    print(f"Keywords    : {keywords or '(none configured yet)'}")
    from ocr_pipeline import _engine_model_name

    print(f"Engine      : {'disabled' if args.no_vlm else args.engine}")
    print(f"Handwriting model : {'disabled' if args.no_vlm else _engine_model_name(args.engine)}")
    if args.engine == "mlx" and not args.no_vlm:
        print(f"Passes      : {args.passes}")
    print()

    all_rows = []
    all_log_rows = []
    all_keyword_hits = []
    all_unmatched = []
    all_review = []
    for pdir in participants:
        print(f"Analyzing {pdir.name} ...")
        result = analyze_participant(
            pdir, image_dir=image_dir, ocr_cache_dir=ocr_cache_dir,
            keywords=keywords, use_vlm=not args.no_vlm, force=args.force, engine=args.engine, passes=args.passes,
        )
        report_md = render_participant_report(result, keywords_configured=bool(keywords))
        (reports_dir / f"{pdir.name}_report.md").write_text(report_md, encoding="utf-8")

        all_rows.extend(build_weekly_rows(result))
        all_log_rows.extend(build_log_rows(result))
        for hit in result["keyword_hits"]:
            all_keyword_hits.append({"participant_id": pdir.name, **hit})
        for f in result["unmatched_files"]:
            all_unmatched.append({"participant_id": pdir.name, "file": f})
        all_review.extend({"participant_id": pdir.name, **r} for r in result["review"])
        if result["unmatched_files"]:
            print(f"  {len(result['unmatched_files'])} file(s) not recognized -- see report or unmatched_files.csv")

    summary_df = pd.DataFrame(all_rows)
    summary_path = out_dir / "summary.csv"
    summary_df.to_csv(summary_path, index=False)

    if all_log_rows:
        pd.DataFrame(all_log_rows).to_csv(out_dir / "exercise_log_days.csv", index=False)

    if all_keyword_hits:
        pd.DataFrame(all_keyword_hits).to_csv(out_dir / "keyword_hits.csv", index=False)

    if all_unmatched:
        pd.DataFrame(all_unmatched).to_csv(out_dir / "unmatched_files.csv", index=False)

    if all_review:
        pd.DataFrame(all_review).to_csv(out_dir / "review_queue.csv", index=False)

    print()
    print(f"Reports written to : {reports_dir}")
    print(f"Weekly summary CSV : {summary_path}  ({len(summary_df)} rows)")
    if all_log_rows:
        print(f"Exercise log CSV   : {out_dir / 'exercise_log_days.csv'}  ({len(all_log_rows)} day rows)")
    if all_keyword_hits:
        print(f"Keyword hits CSV   : {out_dir / 'keyword_hits.csv'}  ({len(all_keyword_hits)} hits)")
    if all_review:
        print(f"Review queue CSV   : {out_dir / 'review_queue.csv'}  ({len(all_review)} fields to verify)")
    if all_unmatched:
        print(f"Unmatched files CSV: {out_dir / 'unmatched_files.csv'}  ({len(all_unmatched)} files not recognized)")


if __name__ == "__main__":
    main()
