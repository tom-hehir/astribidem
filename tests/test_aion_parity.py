"""The matching core reproduces AION-2's crossmatch outputs exactly.

``fixtures/aion_parity.npz`` holds synthetic catalogs and the indexes AION-2
built from them; ``fixtures/generate_aion_parity.py`` records it. Rows here are
AION-2's ``row_index`` values, so every column must agree element for element.
Entities are compared in this package's order (first present survey, then its
row). Resolving saved, reloaded or streamed edges must give the same indexes, and
the edge audit must agree with AION-2's pair statistics and component census.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from astribidem import (
    DegenerateCrossmatchConfig,
    EntitySelectionConfig,
    EntitywiseCrossmatchConfig,
    audit_edges,
    build_edges,
    build_edges_to_directory,
    crossmatch,
    read_edges,
    resolve,
    survey_coords_from_arrays,
    write_edges,
)
from astribidem.banded import as_chunks, build_edges_by_band, write_band_layout
from astribidem.modes import entity_order

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


def in_entity_order(columns):
    """AION-2's recorded rows, reordered by first present survey then row.

    AION-2 lists entitywise entities in three blocks; this package orders
    every index by its entities' first present survey and that survey's row.
    The entities themselves must agree exactly.
    """
    rows = [values for name, values in columns.items() if name.endswith("/row_index")]
    order = entity_order(np.column_stack(rows))
    return {name: values[order] for name, values in columns.items()}


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


SETTINGS = {
    "radius_arcsec": RADIUS_ARCSEC,
    "dedupe_radius_arcsec": DEDUPE_RADIUS_ARCSEC,
    "pair_radius_overrides": {PAIR_RADIUS_OVERRIDE[0]: PAIR_RADIUS_OVERRIDE[1]},
}


@pytest.fixture(scope="module")
def edge_sources(surveys, tmp_path_factory):
    """Each way of handing all-pair candidate edges to ``resolve``."""
    in_memory = build_edges(surveys, **SETTINGS)
    saved = tmp_path_factory.mktemp("saved")
    write_edges(in_memory, saved)
    streamed = tmp_path_factory.mktemp("streamed")
    build_edges_to_directory(surveys, streamed, workers=3, chunk_rows=500, **SETTINGS)
    layout = tmp_path_factory.mktemp("layout")
    write_band_layout(
        layout,
        {
            name: as_chunks(
                EXPECTED[f"input/{name}/ra"], EXPECTED[f"input/{name}/dec"], 900
            )
            for name in SURVEYS
        },
        band_height_deg=3.0 / 3600,
    )
    banded = tmp_path_factory.mktemp("banded")
    build_edges_by_band(layout, banded, **SETTINGS)
    return {
        "in_memory": in_memory,
        "saved": read_edges(saved),
        "streamed": read_edges(streamed),
        "banded": read_edges(banded),
    }


@pytest.mark.parametrize(
    "source", ["crossmatch", "in_memory", "saved", "streamed", "banded"]
)
@pytest.mark.parametrize("label", CASES)
def test_reproduces_aion_output(surveys, edge_sources, label, source):
    kind, arguments = CASES[label]
    if source == "crossmatch":
        table = crossmatch(surveys, mode=mode(kind, arguments), **SETTINGS)
    else:
        table = resolve(edge_sources[source], mode(kind, arguments))
    expected = in_entity_order(
        {
            key.removeprefix(f"{label}/"): value
            for key, value in EXPECTED.items()
            if key.startswith(f"{label}/")
        }
    )
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


def test_audit_reproduces_aion_pair_statistics_and_component_census(edge_sources):
    audit = audit_edges(edge_sources["in_memory"])
    assert audit["pairs"] == json.loads(str(EXPECTED["audit/pairs"]))
    assert audit["components"] == json.loads(str(EXPECTED["audit/components"]))
    assert audit["components"]["n_ambiguous"] > 100
