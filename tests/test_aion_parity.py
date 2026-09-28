"""The matching core reproduces AION-2's crossmatch outputs exactly.

``fixtures/aion_parity.npz`` holds synthetic catalogs and the indexes AION-2
built from them; ``fixtures/generate_aion_parity.py`` records it. Rows here are
AION-2's ``row_index`` values, so every column must agree element for element.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

from astro_crossmatch import (
    DegenerateCrossmatchConfig,
    EntitySelectionConfig,
    EntitywiseCrossmatchConfig,
    crossmatch,
    survey_coords_from_arrays,
)

FIXTURES = Path(__file__).with_name("fixtures")
sys.path.insert(0, str(FIXTURES))
from aion_parity_cases import (
    CASES,
    DEDUPE_RADIUS_ARCSEC,
    PAIR_RADIUS_OVERRIDE,
    RADIUS_ARCSEC,
    SURVEYS,
)

EXPECTED = dict(np.load(FIXTURES / "aion_parity.npz"))


def mode(kind, arguments):
    if kind == "degenerate":
        return DegenerateCrossmatchConfig(**arguments)
    arguments = dict(arguments)
    if "selection" in arguments:
        arguments["selection"] = EntitySelectionConfig(**arguments["selection"])
    return EntitywiseCrossmatchConfig(**arguments)


@pytest.fixture(scope="module")
def surveys():
    return [
        survey_coords_from_arrays(
            name, EXPECTED[f"input/{name}/ra"], EXPECTED[f"input/{name}/dec"]
        )
        for name in SURVEYS
    ]


@pytest.mark.parametrize("label", CASES)
def test_crossmatch_reproduces_aion_output(surveys, label):
    kind, arguments = CASES[label]
    table = crossmatch(
        surveys,
        radius_arcsec=RADIUS_ARCSEC,
        dedupe_radius_arcsec=DEDUPE_RADIUS_ARCSEC,
        pair_radius_overrides={PAIR_RADIUS_OVERRIDE[0]: PAIR_RADIUS_OVERRIDE[1]},
        mode=mode(kind, arguments),
    ).table
    expected = {
        key.removeprefix(f"{label}/"): value
        for key, value in EXPECTED.items()
        if key.startswith(f"{label}/")
    }
    assert table.column_names == list(expected)
    for column, values in expected.items():
        actual = table[column]
        if column.endswith("/row_index"):
            actual = actual.fill_null(-1).to_numpy()
        elif column == "disputed_reason":
            actual = np.array(actual.fill_null("").to_pylist())
        else:
            actual = actual.to_numpy()
        np.testing.assert_array_equal(actual, values, err_msg=column)


def test_fixture_exercises_duplicates_and_ambiguity():
    reasons = EXPECTED["entitywise_refuse_singleton/disputed_reason"]
    assert (reasons == "ambiguous_component").sum() > 100
    assert (reasons == "dedupe_disputed").sum() > 0
    refuse = EXPECTED["entitywise_refuse_singleton/a/row_index"]
    split = EXPECTED["entitywise_split_singleton/a/row_index"]
    assert len(refuse) != len(split)
