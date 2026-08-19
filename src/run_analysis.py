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
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description="OCR and analyze every participant folder under --base-dir.")
    parser.add_argument("--base-dir", default=str(PROJECT_ROOT / "sample_usb"),
                         help="Folder containing VPT001, VPT002, ... subfolders (a USB mount point, or sample_usb).")
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "outputs"), help="Where reports/CSVs/OCR cache are written.")
    parser.add_argument("--keywords-file", default=str(PROJECT_ROOT / "config" / "keywords.txt"))
    parser.add_argument("--model", default=None, help="Override the handwriting VLM, e.g. Qwen/Qwen2.5-VL-7B-Instruct.")
    parser.add_argument("--engine", choices=("vlm", "trocr"), default="vlm",
                         help="Handwriting reading engine: 'vlm' (Qwen2.5-VL, default) or 'trocr' (microsoft/trocr-*-handwritten).")
    parser.add_argument("--no-vlm", action="store_true", help="Skip the handwriting model; printed/checkbox fields only.")
    parser.add_argument("--force", action="store_true", help="Ignore the OCR cache and re-run OCR on every PDF.")
    parser.add_argument("--participants", nargs="*", default=None, help="Only process these participant IDs, e.g. VPT001 VPT002.")
    args = parser.parse_args()

    if args.model:
        if args.engine == "trocr":
            os.environ["TROCR_MODEL_NAME"] = args.model
        else:
            os.environ["VLM_MODEL_NAME"] = args.model

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
    default_model = {"vlm": "Qwen/Qwen2.5-VL-3B-Instruct", "trocr": "microsoft/trocr-base-handwritten"}[args.engine]
    model_env_var = "TROCR_MODEL_NAME" if args.engine == "trocr" else "VLM_MODEL_NAME"
    print(f"Engine      : {'disabled' if args.no_vlm else args.engine}")
    print(f"Handwriting model : {'disabled' if args.no_vlm else os.environ.get(model_env_var, default_model)}")
    print()

    all_rows = []
    all_keyword_hits = []
    all_unmatched = []
    for pdir in participants:
        print(f"Analyzing {pdir.name} ...")
        result = analyze_participant(
            pdir, image_dir=image_dir, ocr_cache_dir=ocr_cache_dir,
            keywords=keywords, use_vlm=not args.no_vlm, force=args.force, engine=args.engine,
        )
        report_md = render_participant_report(result, keywords_configured=bool(keywords))
        (reports_dir / f"{pdir.name}_report.md").write_text(report_md, encoding="utf-8")

        all_rows.extend(build_weekly_rows(result))
        for hit in result["keyword_hits"]:
            all_keyword_hits.append({"participant_id": pdir.name, **hit})
        for f in result["unmatched_files"]:
            all_unmatched.append({"participant_id": pdir.name, "file": f})
        if result["unmatched_files"]:
            print(f"  {len(result['unmatched_files'])} file(s) not recognized -- see report or unmatched_files.csv")

    summary_df = pd.DataFrame(all_rows)
    summary_path = out_dir / "summary.csv"
    summary_df.to_csv(summary_path, index=False)

    if all_keyword_hits:
        pd.DataFrame(all_keyword_hits).to_csv(out_dir / "keyword_hits.csv", index=False)

    if all_unmatched:
        pd.DataFrame(all_unmatched).to_csv(out_dir / "unmatched_files.csv", index=False)

    print()
    print(f"Reports written to : {reports_dir}")
    print(f"Weekly summary CSV : {summary_path}  ({len(summary_df)} rows)")
    if all_keyword_hits:
        print(f"Keyword hits CSV   : {out_dir / 'keyword_hits.csv'}  ({len(all_keyword_hits)} hits)")
    if all_unmatched:
        print(f"Unmatched files CSV: {out_dir / 'unmatched_files.csv'}  ({len(all_unmatched)} files not recognized)")


if __name__ == "__main__":
    main()
