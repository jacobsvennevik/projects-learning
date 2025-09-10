import pytest
import pandas as pd
from garmin_laps_parser import parse_laps

@pytest.fixture
def sample_laps_text():
    return '''Runder  Tid Samlet tid Distanse Gjennomsnittlig tempo …
1   6:03.3  6:03.3  1.00  6:03  --  135 146 1 3 -- -- …
2   6:01.5  12:05   0.94  6:26  --  146 161 2 3 -- -- …
…
52  0:03.0  1:03:02 0.01  3:24  --  175 175 0 0 -- -- …
Sammendrag 1:03:02 1:03:02 12.95 4:52'''


def test_parse_laps_shape(sample_laps_text):
    df = parse_laps(sample_laps_text)
    assert df.shape == (52, 28)


def test_parse_laps_time_conversion(sample_laps_text):
    df = parse_laps(sample_laps_text)
    assert df.iloc[0].time_s == pytest.approx(363.3, 0.01)  # 6:03.3


def test_parse_laps_distance_sum(sample_laps_text):
    df = parse_laps(sample_laps_text)
    assert df.distance_km.sum() == pytest.approx(12.95, 0.01) 