"""Resolving saved edges segment by segment into one index file."""

import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from astro_crossmatch import (
    DegenerateCrossmatchConfig,
    EntitySelectionConfig,
    EntitywiseCrossmatchConfig,
    read_edges,
    resolve,
    write_edges,
)
from astro_crossmatch.banded import as_chunks, build_edges_by_band, write_band_layout
from astro_crossmatch.index_files import resolve_to_file

FIXTURES = Path(__file__).with_name("fixtures")
sys.path.insert(0, str(FIXTURES))
from aion_parity_cases import (
    DEDUPE_RADIUS_ARCSEC,
    PAIR_RADIUS_OVERRIDE,
    RADIUS_ARCSEC,
    SURVEYS,
)

FIXTURE = dict(np.load(FIXTURES / "aion_parity.npz"))
SETTINGS = {
    "radius_arcsec": RADIUS_ARCSEC,
    "dedupe_radius_arcsec": DEDUPE_RADIUS_ARCSEC,
    "pair_radius_overrides": {PAIR_RADIUS_OVERRIDE[0]: PAIR_RADIUS_OVERRIDE[1]},
}
MODES = [
    EntitywiseCrossmatchConfig(),
    EntitywiseCrossmatchConfig(resolver="split", disputed="drop"),
    EntitywiseCrossmatchConfig(resolver="sequential", priority=["c", "a", "b"]),
    EntitywiseCrossmatchConfig(
        selection=EntitySelectionConfig(min_surveys=2, must_include_surveys=["b"])
    ),
    DegenerateCrossmatchConfig(surveys=["b", "a", "c"]),
    DegenerateCrossmatchConfig(surveys=["c", "a"], policy="anchored_unique"),
]


@pytest.fixture(scope="module")
def banded(tmp_path_factory):
    layout = tmp_path_factory.mktemp("layout")
    write_band_layout(
        layout,
        {
            n: as_chunks(FIXTURE[f"input/{n}/ra"], FIXTURE[f"input/{n}/dec"], 800)
            for n in SURVEYS
        },
        band_height_deg=4.0 / 3600,
    )
    directory = tmp_path_factory.mktemp("edges")
    build_edges_by_band(layout, directory, **SETTINGS)
    return directory


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("batch_rows", [7, 65_536])
def test_merged_index_equals_resolving_everything_at_once(
    banded, tmp_path, mode, batch_rows
):
    path = tmp_path / "index.parquet"
    resolve_to_file(banded, mode, path, batch_rows=batch_rows)
    expected = resolve(read_edges(banded), mode).table
    assert pq.read_table(path).equals(expected, check_metadata=True)
    assert not list(tmp_path.glob(".*"))  # the per-segment files are removed


def test_parallel_segments_give_the_same_index(banded, tmp_path):
    mode = EntitywiseCrossmatchConfig(resolver="split")
    resolve_to_file(banded, mode, tmp_path / "serial.parquet")
    resolve_to_file(banded, mode, tmp_path / "parallel.parquet", processes=3)
    assert pq.read_table(tmp_path / "parallel.parquet").equals(
        pq.read_table(tmp_path / "serial.parquet"), check_metadata=True
    )


def test_unsorted_index_holds_the_same_entities_in_segment_order(banded, tmp_path):
    mode = EntitywiseCrossmatchConfig()
    resolve_to_file(banded, mode, tmp_path / "unsorted.parquet", sort=False)
    unsorted = pq.read_table(tmp_path / "unsorted.parquet")
    expected = resolve(read_edges(banded), mode).table
    assert unsorted.num_rows == expected.num_rows
    assert sorted(map(str, unsorted.to_pylist())) == sorted(
        map(str, expected.to_pylist())
    )
    assert not unsorted.equals(expected)
    assert unsorted.schema.metadata == expected.schema.metadata


def test_single_segment_edges_resolve_to_the_same_index(tmp_path):
    from astro_crossmatch import build_edges, survey_coords_from_arrays

    edges = build_edges(
        [
            survey_coords_from_arrays(
                n, FIXTURE[f"input/{n}/ra"], FIXTURE[f"input/{n}/dec"]
            )
            for n in SURVEYS
        ],
        **SETTINGS,
    )
    write_edges(edges, tmp_path / "edges")
    mode = EntitywiseCrossmatchConfig()
    resolve_to_file(tmp_path / "edges", mode, tmp_path / "index.parquet")
    assert pq.read_table(tmp_path / "index.parquet").equals(
        resolve(edges, mode).table, check_metadata=True
    )
