"""Every spatial route gives the in-memory index on dense, boundary-heavy inputs.

Three surveys are piled up on band boundaries, both poles and the RA wrap, so
groups cross bands, dedupe groups are pending in band tasks, and components
are large enough to be ambiguous. ``crossmatch``, resolving in-memory edges,
resolving reloaded banded edges and ``resolve_to_file`` must give identical
tables, metadata included, under every resolver and policy.
"""

import numpy as np
import pyarrow.parquet as pq
import pytest

from astribidem import (
    DegenerateCrossmatchConfig,
    EntitySelectionConfig,
    EntitywiseCrossmatchConfig,
    as_chunks,
    build_edges,
    build_edges_by_band,
    crossmatch,
    read_edges,
    resolve,
    resolve_to_file,
    write_band_layout,
)

pytestmark = pytest.mark.slow

BAND_HEIGHT_DEG = 0.002  # 7.2 arcsec: wider than every radius, so groups cross bands
SETTINGS = {
    "radius_arcsec": 4.0,
    "dedupe_radius_arcsec": 1.5,
    "radius_arcsec_overrides": {("a", "c"): 6.0},
}
MODES = [
    EntitywiseCrossmatchConfig(),
    EntitywiseCrossmatchConfig(resolver="split"),
    EntitywiseCrossmatchConfig(resolver="sequential", priority=["b", "a", "c"]),
    EntitywiseCrossmatchConfig(
        disputed="drop", selection=EntitySelectionConfig(min_surveys=2)
    ),
    DegenerateCrossmatchConfig(surveys=["a", "b", "c"]),
    DegenerateCrossmatchConfig(surveys=["b", "c"], policy="mutual_unique"),
    DegenerateCrossmatchConfig(surveys=["c", "a"], policy="anchored_nearest"),
]


def dense_catalog(rng, n):
    """Rows within 48 arcsec of a few band boundaries, at two RAs across the wrap."""
    last_band = int(180 / BAND_HEIGHT_DEG)
    bands = rng.choice([1, 2, 3, last_band // 2, last_band // 2 + 1, last_band], n)
    offset = rng.choice([-1, 1], n) * rng.uniform(0, 48 / 3600, n)
    dec = np.clip(-90 + bands * BAND_HEIGHT_DEG + offset, -90, 90)
    ra = rng.uniform(0, 0.004, n) + rng.choice([0.0, 359.99], n)
    return ra, dec


def assert_same(actual, expected):
    actual = actual.select(sorted(actual.column_names))
    expected = expected.select(sorted(expected.column_names))
    assert actual.equals(expected)
    assert actual.schema.metadata == expected.schema.metadata


@pytest.mark.parametrize("seed", range(4))
def test_every_route_gives_the_in_memory_index(tmp_path, seed):
    rng = np.random.default_rng(seed)
    catalogs = {name: dense_catalog(rng, int(rng.integers(150, 400))) for name in "abc"}
    write_band_layout(
        tmp_path / "layout",
        {name: as_chunks(ra, dec, 97) for name, (ra, dec) in catalogs.items()},
        band_height_deg=BAND_HEIGHT_DEG,
    )
    build_edges_by_band(tmp_path / "layout", tmp_path / "edges", **SETTINGS)
    in_memory = build_edges(catalogs, **SETTINGS)
    banded = read_edges(tmp_path / "edges")
    for mode in MODES:
        expected = crossmatch(catalogs, mode=mode, **SETTINGS)
        assert_same(resolve(in_memory, mode), expected)
        assert_same(resolve(banded, mode), expected)
        resolve_to_file(tmp_path / "edges", mode, tmp_path / "index.parquet")
        assert_same(pq.read_table(tmp_path / "index.parquet"), expected)
