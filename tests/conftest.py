"""Shared fixtures.

The OCR-facing code (field_extraction, report_generation) only ever touches
two shapes: a "page" with a .lines list, and a "line" with .text and .engine.
Rather than build real OcrLine/PageResult objects (which drags in torch et al.
transitively), the tests fake them with SimpleNamespace via these factories.
"""

from types import SimpleNamespace

import pytest


@pytest.fixture
def line():
    def _line(text, engine="printed"):
        return SimpleNamespace(text=text, engine=engine)

    return _line


@pytest.fixture
def page():
    def _page(*lines):
        return SimpleNamespace(lines=list(lines))

    return _page
