import report_generation as rg
from report_generation import NOT_DOCUMENTED


class TestFmt:
    def test_none_becomes_not_documented(self):
        assert rg._fmt(None) == NOT_DOCUMENTED

    def test_empty_string_becomes_not_documented(self):
        assert rg._fmt("") == NOT_DOCUMENTED

    def test_real_value_passes_through(self):
        assert rg._fmt("3") == "3"


class TestFieldsTable:
    def test_renders_a_markdown_table(self):
        table = rg._fields_table({"participant_id": "VPT001", "age": "34"})
        lines = table.splitlines()
        assert lines[0] == "| Field | Value |"
        assert lines[1] == "|---|---|"
        assert "| participant id | VPT001 |" in lines
        assert "| age | 34 |" in lines

    def test_explicit_order_is_honored_and_unknown_keys_skipped(self):
        table = rg._fields_table({"a": "1", "b": "2"}, order=["b", "a", "c"])
        body = [ln for ln in table.splitlines()[2:]]
        assert body == ["| b | 2 |", "| a | 1 |"]


def _minimal_result(**overrides):
    base = {
        "participant_id": "VPT001",
        "intake": None,
        "intermediate_visits": [],
        "final": None,
        "exit": None,
        "weeks": [],
        "keyword_hits": [],
        "unmatched_files": [],
    }
    base.update(overrides)
    return base


class TestRenderParticipantReport:
    def test_minimal_report_smoke(self):
        md = rg.render_participant_report(_minimal_result(), keywords_configured=False)
        assert md.startswith("# Participant VPT001")
        assert "## Weekly Sessions" in md
        assert "No keywords configured yet" in md

    def test_completed_week_row_and_detail_lines(self):
        result = _minimal_result(
            weeks=[
                {
                    "week": 1,
                    "status": "completed",
                    "note": None,
                    "fields": {
                        "days_exercised": "5",
                        "too_difficult": "No",
                        "confident_form": "Yes",
                        "days_ran": "3",
                        "mileage": "10",
                        "pain_now": "2",
                        "pain_usual": "3",
                        "pain_worst": "5",
                        "exercise_types": ["Hip Strength"],
                        "cues_used": "No",
                        "other_exercise": "No",
                        "notes": "felt good",
                    },
                    "file": "wk1.pdf",
                }
            ]
        )
        md = rg.render_participant_report(result, keywords_configured=True)
        assert "| 1 | completed | 5 | No | Yes | 3 | 10 | 2 | 3 | 5 |" in md
        assert "Exercise types checked: Hip Strength" in md
        assert "felt good" in md

    def test_missed_and_no_file_weeks_render_placeholder_rows(self):
        result = _minimal_result(
            weeks=[
                {"week": 2, "status": "missed", "note": "was sick", "fields": None, "file": "n.docx"},
                {"week": 3, "status": "no_file_found", "note": None, "fields": None},
            ]
        )
        md = rg.render_participant_report(result, keywords_configured=True)
        assert "| 2 | missed |" in md
        assert "| 3 | no file found |" in md
        assert "Note: was sick" in md

    def test_an_unreadable_week_renders_a_placeholder_row(self):
        result = _minimal_result(
            weeks=[
                {"week": 3, "status": "unreadable", "note": None, "fields": None,
                 "file": "Exercise Log week 3 .pdf"},
            ]
        )
        md = rg.render_participant_report(result, keywords_configured=True)
        assert "| 3 | unreadable |" in md
        assert "Exercise Log week 3 .pdf" in md
        assert "Files Not Processed" in md

    def test_keyword_hits_render_as_a_table(self):
        result = _minimal_result(
            keyword_hits=[{"keyword": "pain", "source": "wk1.pdf", "context": "sharp pain today"}]
        )
        md = rg.render_participant_report(result, keywords_configured=True)
        assert "| pain | wk1.pdf | ...sharp pain today... |" in md

    def test_unmatched_files_get_their_own_section(self):
        md = rg.render_participant_report(
            _minimal_result(unmatched_files=["weird.pdf"]), keywords_configured=True
        )
        assert "## Files Not Processed" in md
        assert "- weird.pdf" in md

    def test_intake_section_only_appears_when_present(self):
        without = rg.render_participant_report(_minimal_result(), keywords_configured=True)
        assert "## Intake" not in without
        with_intake = rg.render_participant_report(
            _minimal_result(intake={"file": "v1.pdf", "fields": {"age": "34"}}),
            keywords_configured=True,
        )
        assert "## Intake (Visit 1)" in with_intake
        assert "v1.pdf" in with_intake
