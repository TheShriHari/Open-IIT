"""Visual door cues read from agent remarks (Step 7c)."""
import pytest

from remarks import remark_cues


@pytest.mark.parametrize("remark,cues", [
    ("ghar band tha; blue gate wala ghar, Masjid se right side", ["blue gate"]),
    ("neeli gate ke saamne", ["blue gate"]),
    ("kempu baagilu mane", ["red gate"]),
    ("hara rang ka gate", ["green gate"]),
    ("2nd floor, corner mane", ["corner house", "floor 2"]),
    ("gate band tha", []),
    ("mane khaali maadidaare", []),     # "maadidaare" = vacated, not a floor
])
def test_remark_cues(remark, cues):
    assert remark_cues(remark) == cues
