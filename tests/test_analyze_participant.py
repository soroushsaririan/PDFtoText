import analyze_participant as ap


def touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"%PDF-1.4\n")  # discovery only inspects names/extensions


class TestDiscoverParticipants:
    def test_returns_vpt_dirs_sorted_by_name(self, tmp_path):
        for name in ("VPT003", "VPT001", "VPT002"):
            (tmp_path / name).mkdir()
        assert [p.name for p in ap.discover_participants(tmp_path)] == [
            "VPT001",
            "VPT002",
            "VPT003",
        ]

    def test_ignores_wrong_case_suffixed_names_and_plain_files(self, tmp_path):
        (tmp_path / "VPT001").mkdir()
        (tmp_path / "vpt002").mkdir()
        (tmp_path / "VPT003_old").mkdir()
        (tmp_path / "notes").mkdir()
        (tmp_path / "VPT999").write_text("i am a file, not a folder")
        assert [p.name for p in ap.discover_participants(tmp_path)] == ["VPT001"]


class TestDiscoverFiles:
    def test_classifies_crf_exit_and_week_files(self, tmp_path):
        d = tmp_path / "VPT001"
        touch(d / "VPT001 CRF visit 1.pdf")
        touch(d / "VPT001 CRF visit 3.pdf")
        touch(d / "VPT001 CRF visit 2.pdf")
        touch(d / "VPT001 Exit survey.pdf")
        touch(d / "Week 1" / "session.pdf")
        touch(d / "Week 2" / "missed note.docx")

        got = ap.discover_files(d)

        assert [num for num, _ in got["crf_visits"]] == [1, 2, 3]
        assert got["exit_survey"].name == "VPT001 Exit survey.pdf"
        weeks = {w["week"]: w for w in got["weeks"]}
        assert weeks[1]["files"] and not weeks[1]["notes"]
        assert weeks[2]["notes"] and not weeks[2]["files"]
        assert got["unmatched"] == []

    def test_only_first_exit_candidate_wins_rest_are_unmatched(self, tmp_path):
        d = tmp_path / "VPT002"
        touch(d / "A Exit survey.pdf")
        touch(d / "B Exit form.pdf")
        got = ap.discover_files(d)
        assert got["exit_survey"].name == "A Exit survey.pdf"
        assert [p.name for p in got["unmatched"]] == ["B Exit form.pdf"]

    def test_unrecognized_file_lands_in_unmatched(self, tmp_path):
        d = tmp_path / "VPT003"
        touch(d / "random handout.pdf")
        got = ap.discover_files(d)
        assert [p.name for p in got["unmatched"]] == ["random handout.pdf"]

    def test_dari_and_video_subtrees_are_skipped_entirely(self, tmp_path):
        d = tmp_path / "VPT004"
        touch(d / "video" / "Week 1" / "clip.pdf")
        touch(d / "Dari translation" / "CRF visit 1.pdf")
        got = ap.discover_files(d)
        assert got["crf_visits"] == []
        assert got["weeks"] == []
        assert got["unmatched"] == []

    def test_week_number_read_from_filename_or_parent_folder(self, tmp_path):
        d = tmp_path / "VPT005"
        touch(d / "VPT005 Exercise Log Week 4.pdf")  # week in the filename
        touch(d / "week 2" / "log.pdf")              # week in the folder name
        got = ap.discover_files(d)
        assert sorted(w["week"] for w in got["weeks"]) == [2, 4]

    def test_image_files_count_as_week_documents(self, tmp_path):
        d = tmp_path / "VPT006"
        touch(d / "Week 1" / "scan.jpg")
        got = ap.discover_files(d)
        weeks = {w["week"]: w for w in got["weeks"]}
        assert [p.name for p in weeks[1]["files"]] == ["scan.jpg"]
