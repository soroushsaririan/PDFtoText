NOT_DOCUMENTED = "Not documented"
UNCERTAIN = "Uncertain"


def _fmt(v):
    return v if v not in (None, "") else NOT_DOCUMENTED


def _fields_table(fields, order=None):
    keys = order or list(fields.keys())
    lines = ["| Field | Value |", "|---|---|"]
    for k in keys:
        if k in fields:
            lines.append(f"| {k.replace('_', ' ')} | {_fmt(fields[k])} |")
    return "\n".join(lines)


def render_participant_report(result, keywords_configured):
    pid = result["participant_id"]
    lines = [f"# Participant {pid} - Handwriting Analysis Report", ""]

    if result["intake"]:
        lines.append("## Intake (Visit 1)")
        lines.append(f"_Source: {result['intake']['file']}_")
        lines.append("")
        lines.append(_fields_table(result["intake"]["fields"]))
        lines.append("")

    lines.append("## Weekly Sessions")
    lines.append("")
    lines.append(
        "| Week | Status | Days exercised | Too difficult | Confident form | "
        "Days ran | Mileage | Pain now | Pain usual | Pain worst |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for wk in sorted(result["weeks"], key=lambda w: (w["week"] is None, w["week"])):
        if wk["status"] == "missed":
            lines.append(f"| {wk['week']} | missed | - | - | - | - | - | - | - | - |")
            continue
        if wk["status"] == "no_file_found":
            lines.append(f"| {wk['week']} | no file found | - | - | - | - | - | - | - | - |")
            continue
        f = wk["fields"]
        lines.append(
            f"| {wk['week']} | completed | {_fmt(f.get('days_exercised'))} | "
            f"{_fmt(f.get('too_difficult'))} | {_fmt(f.get('confident_form'))} | "
            f"{_fmt(f.get('days_ran'))} | {_fmt(f.get('mileage'))} | "
            f"{_fmt(f.get('pain_now'))} | {_fmt(f.get('pain_usual'))} | {_fmt(f.get('pain_worst'))} |"
        )
    lines.append("")

    lines.append("### Weekly details")
    for wk in sorted(result["weeks"], key=lambda w: (w["week"] is None, w["week"])):
        lines.append(f"\n**Week {wk['week']}** ({wk['status']})")
        if wk["status"] == "missed":
            lines.append(f"- Note: {wk['note']}")
            continue
        if wk["status"] == "no_file_found":
            lines.append("- No PDF or missed-session note found for this week.")
            continue
        f = wk["fields"]
        lines.append(f"- Exercise types checked: {', '.join(f.get('exercise_types') or []) or NOT_DOCUMENTED}")
        if f.get("cues_used") == "Yes":
            lines.append(f"- Cues used: {', '.join(f.get('cues_list') or []) or UNCERTAIN}")
        lines.append(f"- Other exercise this week: {_fmt(f.get('other_exercise'))}")
        notes = f.get("notes", NOT_DOCUMENTED)
        lines.append(f"- Notes (OCR'd handwriting, verify against source PDF): {notes}")

    if result["intermediate_visits"]:
        lines.append("\n## Intermediate Visits")
        for v in result["intermediate_visits"]:
            lines.append(f"\n### Visit {v['visit']} ({v['file']})")
            lines.append(_fields_table(v["fields"]))

    if result["final"]:
        lines.append("\n## Final Visit")
        lines.append(f"_Source: {result['final']['file']}_\n")
        lines.append(_fields_table(result["final"]["fields"]))

    if result["exit"]:
        lines.append("\n## Exit Survey")
        lines.append(f"_Source: {result['exit']['file']}_\n")
        lines.append(_fields_table(result["exit"]["fields"]))

    lines.append("\n## Keyword Matches")
    if not keywords_configured:
        lines.append("_No keywords configured yet -- add terms to config/keywords.txt to enable this section._")
    elif not result["keyword_hits"]:
        lines.append("_No configured keywords were found in this participant's documents._")
    else:
        lines.append("| Keyword | Source file | Context |")
        lines.append("|---|---|---|")
        for hit in result["keyword_hits"]:
            lines.append(f"| {hit['keyword']} | {hit['source']} | ...{hit['context']}... |")

    unmatched = result.get("unmatched_files") or []
    if unmatched:
        lines.append("\n## Files Not Processed")
        lines.append(
            "_These files were found in this participant's folder but didn't match a recognized "
            "pattern (CRF visit N, Exit survey, or a Week N log), so they were not analyzed. Check "
            "whether any of them should have been -- naming may just be inconsistent._"
        )
        lines.append("")
        for f in unmatched:
            lines.append(f"- {f}")

    lines.append(
        "\n---\n_Handwriting fields were transcribed with an open-source handwriting-recognition "
        'model and may contain errors; anything the model was not confident about is reported as '
        '"Uncertain" or "[illegible]". Always verify clinically important values against the original PDF.'
        "_"
    )
    return "\n".join(lines)
