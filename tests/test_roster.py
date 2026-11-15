"""Name cleanup for OCR output (no Tesseract needed)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pickpov.roster import clean_name  # noqa: E402


def test_strips_junk_around_names():
    assert clean_name("tkzin,") == "tkzin"
    assert clean_name("| Darker") == "Darker"
    assert clean_name("DaviH {") == "DaviH"
    assert clean_name("—— luk xo") == "luk xo"
    assert clean_name("N DaviH |") == "DaviH"


def test_keeps_real_names():
    assert clean_name("luk xo") == "luk xo"
    assert clean_name("PROFEK") == "PROFEK"
    assert clean_name("") == ""
