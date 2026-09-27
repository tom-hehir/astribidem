"""Band-by-band edge builds from a streamed layout."""

import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from astro_crossmatch import (
    EntitywiseCrossmatchConfig,
    build_edges,
    read_edges,
    read_segment,
    resolve,
    segment_names,
    survey_coords_from_arrays,
)
from astro_crossmatch.banded import (
    as_chunks,
    build_band,
    build_edges_by_band,
    prepare_band_build,
    read_layout,
    sweep_boundaries,
    write_band_layout,
)

FIXTURES = Path(__file__).with_name("fixtures")
sys.path.insert(0, str(FIXTURES))
from aion_parity_cases import (
    DEDUPE_RADIUS_ARCSEC,
    PAIR_RADIUS_OVERRIDE,
    RADIUS_ARCSEC,
    SURVEYS,
)

ARCSEC = 1 / 3600
FIXTURE = dict(np.load(FIXTURES / "aion_parity.npz"))
CATALOGS = {
    name: (FIXTURE[f"input/{name}/ra"], FIXTURE[f"input/{name}/dec"])
    for name in SURVEYS
}
SETTINGS = {
    "radius_arcsec": RADIUS_ARCSEC,
    "dedupe_radius_arcsec": DEDUPE_RADIUS_ARCSEC,
    "pair_radius_overrides": {PAIR_RADIUS_OVERRIDE[0]: PAIR_RADIUS_OVERRIDE[1]},
}


def in_memory(catalogs, **settings):
    return build_edges(
        [survey_coords_from_arrays(n, ra, dec) for n, (ra, dec) in catalogs.items()],
        **settings,
    )


def layout(tmp_path, catalogs, band_height_arcsec, chunk_rows=700):
    directory = tmp_path / "layout"
    write_band_layout(
        directory,
        {name: as_chunks(ra, dec, chunk_rows) for name, (ra, dec) in catalogs.items()},
        band_height_deg=band_height_arcsec * ARCSEC,
    )
    return directory


def assert_same_edges(left, right):
    for x, y in zip(left.surveys, right.surveys, strict=True):
        assert (x.name, x.n_rows, x.rows) == (y.name, y.n_rows, None)
        for field in ("dropped_rows", "kept_rows", "disputed_rows"):
            np.testing.assert_array_equal(getattr(x, field), getattr(y, field))
    assert list(left.pairs) == list(right.pairs)
    for key, x in left.pairs.items():
        for field in ("row_a", "row_b", "separation_arcsec"):
            np.testing.assert_array_equal(
                getattr(x, field), getattr(right.pairs[key], field)
            )


def test_layout_groups_rows_by_band_across_chunk_files(tmp_path):
    directory = layout(tmp_path, CATALOGS, 20.0)
    meta = read_layout(directory)
    assert [s["name"] for s in meta["surveys"]] == list(SURVEYS)
    assert meta["surveys"][0]["n_rows"] == len(CATALOGS["a"][0])
    index = pq.read_table(directory / "a" / "index.parquet").to_pydict()
    assert len(set(index["chunk"])) > 1 and len(set(index["band"])) > 1
    assert sum(index["n_rows"]) == len(CATALOGS["a"][0])


@pytest.mark.parametrize("band_height_arcsec", [1.5, 5.0, 60.0])
def test_band_build_reproduces_the_in_memory_edges(tmp_path, band_height_arcsec):
    edges_directory = tmp_path / "edges"
    build_edges_by_band(
        layout(tmp_path, CATALOGS, band_height_arcsec), edges_directory, **SETTINGS
    )
    names = segment_names(edges_directory)
    assert any(n.startswith("band-") for n in names)
    assert any(n.startswith("boundary-") for n in names)
    assert sorted(p.name for p in edges_directory.iterdir()) == sorted(
        [*names, "metadata.json"]
    )
    assert_same_edges(read_edges(edges_directory), in_memory(CATALOGS, **SETTINGS))


def test_band_tasks_can_run_in_separate_processes(tmp_path):
    single, parallel = tmp_path / "single", tmp_path / "parallel"
    directory = layout(tmp_path, CATALOGS, 5.0)
    build_edges_by_band(directory, single, **SETTINGS)
    build_edges_by_band(directory, parallel, band_processes=3, **SETTINGS)
    assert segment_names(single) == segment_names(parallel)
    assert_same_edges(read_edges(parallel), read_edges(single))


def test_controller_and_worker_steps_run_separately(tmp_path):
    edges_directory = tmp_path / "edges"
    bands = prepare_band_build(
        layout(tmp_path, CATALOGS, 5.0), edges_directory, **SETTINGS
    )
    for band in reversed(bands):  # any order
        build_band(edges_directory, band)
    sweep_boundaries(edges_directory)
    assert_same_edges(read_edges(edges_directory), in_memory(CATALOGS, **SETTINGS))


def test_resolving_each_segment_gives_the_in_memory_entities(tmp_path):
    edges_directory = tmp_path / "edges"
    build_edges_by_band(layout(tmp_path, CATALOGS, 5.0), edges_directory, **SETTINGS)
    mode = EntitywiseCrossmatchConfig(resolver="split")
    rows = sorted(
        json.dumps(r)
        for name in segment_names(edges_directory)
        for r in resolve(read_segment(edges_directory, name), mode).table.to_pylist()
    )
    whole = resolve(in_memory(CATALOGS, **SETTINGS), mode).table.to_pylist()
    assert rows == sorted(json.dumps(r) for r in whole)


def test_band_height_must_exceed_the_largest_radius(tmp_path):
    with pytest.raises(ValueError, match="must exceed the largest radius"):
        prepare_band_build(
            layout(tmp_path, CATALOGS, 1.0), tmp_path / "edges", **SETTINGS
        )


def test_sweep_fails_loudly_when_groups_span_many_bands(tmp_path):
    # A dense chain along declination crosses every band boundary.
    dec = np.arange(0, 200) * 0.9 * ARCSEC
    catalogs = {"a": (np.full(200, 10.0), dec)}
    directory = layout(tmp_path, catalogs, 2.0)
    with pytest.raises(RuntimeError, match="radius is too large"):
        build_edges_by_band(
            directory,
            tmp_path / "edges",
            radius_arcsec=1.0,
            dedupe_radius_arcsec={"a": 1.0},
            max_carried_rows=20,
        )
    build_edges_by_band(
        directory,
        tmp_path / "unbounded",
        radius_arcsec=1.0,
        dedupe_radius_arcsec={"a": 1.0},
    )
    assert_same_edges(
        read_edges(tmp_path / "unbounded"),
        in_memory(catalogs, radius_arcsec=1.0, dedupe_radius_arcsec={"a": 1.0}),
    )
