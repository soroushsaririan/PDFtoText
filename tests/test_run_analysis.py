"""The CSVs a run writes out, built from an analyze_participant result."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import run_analysis as run


def result(**over):
    base = {
        "participant_id": "VPT013",
        "weeks": [],
        "review": [],
        "unread_logs": [],
    }
    base.update(over)
    return base


class TestLogRows:
    def test_one_row_per_day_read(self):
        weeks = [{"week": 2, "status": "completed", "log_days": [
            {"source_page": "p/page_01.png", "day": 1, "completed": "CHECK", "running": None},
            {"source_page": "p/page_01.png", "day": 2, "completed": None, "running": "3 miles"},
        ]}]
        rows = run.build_log_rows(result(weeks=weeks))
        assert [r["day"] for r in rows] == [1, 2]
        assert rows[0]["participant_id"] == "VPT013" and rows[0]["week"] == 2
        assert rows[1]["running"] == "3 miles"

    def test_a_week_with_no_log_contributes_nothing(self):
        weeks = [{"week": 1, "status": "completed", "log_days": []},
                 {"week": 3, "status": "missed"}]
        assert run.build_log_rows(result(weeks=weeks)) == []


class TestWeeklyRows:
    def test_a_missed_week_still_gets_a_row(self):
        weeks = [{"week": 4, "status": "missed", "fields": None}]
        rows = run.build_weekly_rows(result(weeks=weeks))
        assert len(rows) == 1 and rows[0]["status"] == "missed"
        assert "mileage" not in rows[0]

    def test_flagged_fields_are_named_on_their_week(self):
        weeks = [{"week": 2, "status": "completed",
                  "fields": {"participant_id": "VPT013", "mileage": "35.28"}}]
        review = [{"form": "weekly", "week": 2, "field": "mileage"},
                  {"form": "weekly", "week": 3, "field": "pain_now"},
                  {"form": "CRF visit 1", "week": None, "field": "age"}]
        rows = run.build_weekly_rows(result(weeks=weeks, review=review))
        assert rows[0]["needs_review"] == "mileage"
        # participant_id comes from the folder name, not the handwriting
        assert "participant_id" not in rows[0] or rows[0]["participant_id"] == "VPT013"
