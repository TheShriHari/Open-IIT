"""The address parser finds the locality and the first landmark on known addresses."""
import pytest

from load import map_tables
from textparse import AddressParser

CASES = [   # (address text, town, locality_id, landmark type, relation)
    ("6th Cross, 5th Main, ಚರ್ಚ್ ಹತ್ತಿರ, Kuvempu Layt, Kaveripura - 960102",
     "T1", "T1-L03", "church", "NEAR"),
    ("H.NO. 356, NR. CHURCH, KUVEMPU LAYOUT, KAVERIPURA - 960104",
     "T1", "T1-L03", "church", "NEAR"),
    ("गली नं. 3, राशन की दुकान के बगल में, Tilak Ngr, Devgarh Nagar - 970204",
     "T2", "T2-L12", "ration_shop", "NEAR"),
    ("behind PDS Shop, Shivaji Nagar, Devgarh Nagar - 970202",
     "T2", "T2-L01", "ration_shop", "BEHIND"),
    ("blk a rd 2 milk dairy edurru palm meadows navanagara east - 980302",
     "T3", "T3-L10", "milk_dairy", "OPPOSITE"),
]


@pytest.fixture(scope="module")
def parser():
    return AddressParser(map_tables()[1])


@pytest.mark.parametrize("text,town,locality_id,lm_type,relation", CASES)
def test_parse_known_address(parser, text, town, locality_id, lm_type, relation):
    out = parser.parse(text, town)
    assert out["locality_id"] == locality_id
    assert out["landmarks"], "no landmark found"
    assert out["landmarks"][0]["type"] == lm_type
    assert out["landmarks"][0]["relation"] == relation
