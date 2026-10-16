import fitz

import analyze_participant as ap


def touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"%PDF-1.4\n")  # discovery only inspects names/extensions


def write_pdf(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    doc.save(str(path))
    doc.close()


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

    def test_printed_email_in_a_week_folder_is_unmatched_not_a_form(self, tmp_path):
        d = tmp_path / "VPT007"
        touch(d / "VPT007 weekly exercise log" / "week 2" / "VPT week 2 email communication.pdf")
        touch(d / "VPT007 weekly exercise log" / "week 2" / "VPT007 Week 2 VPT survey.pdf")
        got = ap.discover_files(d)
        assert [p.name for p in got["unmatched"]] == ["VPT week 2 email communication.pdf"]
        assert [p.name for p in got["weeks"][0]["files"]] == ["VPT007 Week 2 VPT survey.pdf"]

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

    def test_no_session_marker_pdf_is_kept_out_of_the_real_week_files(self, tmp_path):
        d = tmp_path / "VPT020"
        touch(d / "week 1" / "VPT020 No week 1 VPT session.pdf")
        touch(d / "week 4" / "VPT022 No week4 VPT session.pdf")
        got = ap.discover_files(d)
        weeks = {w["week"]: w for w in got["weeks"]}
        assert weeks[1]["files"] == []
        assert [p.name for p in weeks[1]["missed_markers"]] == ["VPT020 No week 1 VPT session.pdf"]
        assert weeks[4]["files"] == []
        assert weeks[4]["missed_markers"]

    def test_no_session_marker_does_not_shadow_a_real_week_file(self, tmp_path):
        d = tmp_path / "VPT007"
        touch(d / "Week 2" / "VPT007 Exercise Log Week 2.pdf")
        touch(d / "Week 2" / "VPT007 No Week 2 VPT Session.pdf")
        got = ap.discover_files(d)
        weeks = {w["week"]: w for w in got["weeks"]}
        assert [p.name for p in weeks[2]["files"]] == ["VPT007 Exercise Log Week 2.pdf"]
        assert weeks[2]["missed_markers"]

    def test_empty_no_log_marker_dir_is_surfaced_as_unmatched(self, tmp_path):
        d = tmp_path / "VPT016"
        touch(d / "VPT016 CRF visit 1.pdf")
        (d / "No VPT016 Weekly Exercise Log").mkdir(parents=True)
        got = ap.discover_files(d)
        assert got["weeks"] == []
        assert [p.name for p in got["unmatched"]] == ["No VPT016 Weekly Exercise Log"]

    def test_marker_dir_with_real_files_inside_is_not_treated_as_empty(self, tmp_path):
        d = tmp_path / "VPT099"
        touch(d / "No VPT099 Weekly Exercise Log" / "week 1" / "session.pdf")
        got = ap.discover_files(d)
        assert got["unmatched"] == []
        weeks = {w["week"]: w for w in got["weeks"]}
        assert weeks[1]["files"]

    def test_duplicate_crf_visit_is_deduped_and_the_extra_copy_flagged(self, tmp_path):
        d = tmp_path / "VPT002"
        touch(d / "VPT002 CRF visit 3.pdf")
        touch(d / "VPT002 weekly exercise log" / "VPT002 CRF visit 3.pdf")
        got = ap.discover_files(d)
        assert [num for num, _ in got["crf_visits"]] == [3]
        # the root copy sorts first alphabetically and is the one kept
        kept_path = got["crf_visits"][0][1]
        assert kept_path.parent == d
        assert [p.name for p in got["unmatched"]] == ["VPT002 CRF visit 3.pdf"]

    def test_a_distinctly_named_repeat_visit_is_not_deduped_away(self, tmp_path):
        # Same visit number, but a different filename means a real second
        # document (e.g. the visit had to be re-administered), not a copy.
        d = tmp_path / "VPT018"
        touch(d / "VPT018 CRF visit 1.pdf")
        touch(d / "VPT018 CRF visit 1 repeat.pdf")
        got = ap.discover_files(d)
        assert sorted(num for num, _ in got["crf_visits"]) == [1, 1]
        assert got["unmatched"] == []


class TestAnalyzeParticipantMissedWeeks:
    def test_missed_week_note_comes_from_the_marker_pdfs_text(self, tmp_path):
        d = tmp_path / "VPT020"
        write_pdf(d / "week 1" / "VPT020 No week 1 VPT session.pdf", "Vpt020 No week 1 session")

        result = ap.analyze_participant(
            d, image_dir=tmp_path / "img", ocr_cache_dir=tmp_path / "cache", keywords=[],
        )

        assert len(result["weeks"]) == 1
        wk = result["weeks"][0]
        assert wk["status"] == "missed"
        assert "No week 1 session" in wk["note"]
        assert wk["fields"] is None


class TestWeeklyDuplicateFiles:
    """The same weekly scan sometimes arrives twice under different names --
    a phone photo plus a copy renamed ".pdf" (VPT014)."""

    def test_byte_identical_weekly_copies_are_ocrd_once(self, tmp_path):
        d = tmp_path / "VPT014"
        photo = d / "weekly log" / "Exercise Log week 1.JPG"
        renamed = d / "weekly log" / "Exercise Log week 1 .pdf"
        photo.parent.mkdir(parents=True, exist_ok=True)
        photo.write_bytes(b"\xff\xd8\xff\xe0 the same scan")
        renamed.write_bytes(b"\xff\xd8\xff\xe0 the same scan")

        got = ap.discover_files(d)

        assert [p.name for p in got["weeks"][0]["files"]] == ["Exercise Log week 1 .pdf"]
        assert [p.name for p in got["unmatched"]] == ["Exercise Log week 1.JPG"]

    def test_different_scans_for_one_week_are_both_kept(self, tmp_path):
        d = tmp_path / "VPT014"
        front = d / "weekly log" / "Exercise Log week 1 page 1.pdf"
        back = d / "weekly log" / "Exercise Log week 1 page 2.pdf"
        front.parent.mkdir(parents=True, exist_ok=True)
        front.write_bytes(b"%PDF-1.4 front side")
        back.write_bytes(b"%PDF-1.4 back side")

        got = ap.discover_files(d)

        assert len(got["weeks"][0]["files"]) == 2
        assert got["unmatched"] == []


class TestWhyAFileWasSetAside:
    """"9 files not recognized" read as nine lost records; most were
    duplicate scans. Every set-aside file now carries its reason."""

    def test_a_duplicate_scan_names_the_file_it_duplicates(self, tmp_path):
        d = tmp_path / "VPT014"
        (d / "VPT014 weekly Exercise log").mkdir(parents=True)
        for name in ("VPT014 Exercise Log week 1 .pdf", "VPT014 Exercise Log week 1.JPG"):
            (d / "VPT014 weekly Exercise log" / name).write_bytes(b"same scan bytes")
        got = ap.discover_files(d)
        [path] = got["unmatched"]
        assert path.name == "VPT014 Exercise Log week 1.JPG"
        assert got["unmatched_reasons"][path] == 'byte-identical copy of "VPT014 Exercise Log week 1 .pdf"'

    def test_a_printed_email_says_so(self, tmp_path):
        d = tmp_path / "VPT007"
        (d / "week 2").mkdir(parents=True)
        f = d / "week 2" / "VPT week 2 email communication.pdf"
        f.write_bytes(b"x")
        got = ap.discover_files(d)
        assert got["unmatched_reasons"][f] == "printed email, not a study form"

    def test_an_unrecognized_name_says_what_was_looked_for(self, tmp_path):
        d = tmp_path / "VPT001"
        d.mkdir()
        f = d / "random handout.pdf"
        f.write_bytes(b"x")
        got = ap.discover_files(d)
        assert "week number" in got["unmatched_reasons"][f]

    def test_a_second_copy_of_a_crf_visit_names_the_visit(self, tmp_path):
        d = tmp_path / "VPT002"
        (d / "VPT002 weekly exercise log").mkdir(parents=True)
        (d / "VPT002 CRF visit 3.pdf").write_bytes(b"a")
        (d / "VPT002 weekly exercise log" / "VPT002 CRF visit 3.pdf").write_bytes(b"b")
        got = ap.discover_files(d)
        [path] = got["unmatched"]
        assert got["unmatched_reasons"][path] == "second copy of CRF visit 3, filed in two places"

    def test_an_empty_no_log_folder_says_it_holds_nothing(self, tmp_path):
        d = tmp_path / "VPT016"
        marker = d / "No VPT016 Weekly Exercise Log"
        marker.mkdir(parents=True)
        got = ap.discover_files(d)
        assert got["unmatched_reasons"][marker] == "folder says there is no log and holds no file"
