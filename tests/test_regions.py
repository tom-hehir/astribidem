"""Region builds reproduce the in-memory edge build exactly."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from astro_crossmatch import (
    DegenerateCrossmatchConfig,
    EntitywiseCrossmatchConfig,
    build_edges,
    combine_segments,
    resolve,
    survey_coords_from_arrays,
)
from astro_crossmatch.edges import edge_settings
from astro_crossmatch.regions import (
    MARGIN_ABOVE,
    MARGIN_BELOW,
    OWNED,
    RegionRows,
    build_region,
    combine_handovers,
    finish_handover,
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


def band_regions(catalogs, band_height_deg, margin_deg):
    """Split in-memory catalogs into declination bands with their margins."""
    bands = {
        name: np.floor((dec + 90) / band_height_deg).astype(np.int64)
        for name, (_, dec) in catalogs.items()
    }
    lowest = min(b.min() for b in bands.values() if len(b))
    highest = max(b.max() for b in bands.values() if len(b))
    for k in range(lowest, highest + 1):
        lower = k * band_height_deg - 90
        upper = lower + band_height_deg
        region = {}
        for name, (ra, dec) in catalogs.items():
            band = bands[name]
            side = np.full(len(dec), 99, dtype=np.int8)
            side[band == k] = OWNED
            side[(band == k - 1) & (dec >= lower - margin_deg)] = MARGIN_BELOW
            side[(band == k + 1) & (dec <= upper + margin_deg)] = MARGIN_ABOVE
            take = np.flatnonzero(side != 99)
            region[name] = RegionRows(
                rows=take, ra=ra[take], dec=dec[take], side=side[take]
            )
        yield region


def banded_edges(catalogs, band_height_deg, **settings):
    n_rows = {name: len(dec) for name, (_, dec) in catalogs.items()}
    plan = edge_settings(n_rows, **settings)
    margin = plan.max_radius_arcsec * ARCSEC * (1 + 1e-6)
    segments, handovers = [], []
    for region in band_regions(catalogs, band_height_deg, margin):
        segment, handover = build_region(plan, region)
        segments.append(segment)
        handovers.append(handover)
    boundary = finish_handover(plan, combine_handovers(handovers))
    return combine_segments([*segments, boundary]), segments, boundary


def assert_same_edges(left, right):
    for x, y in zip(left.surveys, right.surveys, strict=True):
        assert (x.name, x.n_rows) == (y.name, y.n_rows)
        assert x.rows is None  # every row landed in exactly one segment
        for field in ("dropped_rows", "kept_rows", "disputed_rows"):
            np.testing.assert_array_equal(getattr(x, field), getattr(y, field))
    assert list(left.pairs) == list(right.pairs)
    for key, x in left.pairs.items():
        y = right.pairs[key]
        for field in ("row_a", "row_b", "separation_arcsec"):
            np.testing.assert_array_equal(getattr(x, field), getattr(y, field))


def canonical(table):
    return sorted(json.dumps(row) for row in table.to_pylist())


FIXTURE = dict(np.load(FIXTURES / "aion_parity.npz"))
PARITY_CATALOGS = {
    name: (FIXTURE[f"input/{name}/ra"], FIXTURE[f"input/{name}/dec"])
    for name in SURVEYS
}
PARITY_SETTINGS = {
    "radius_arcsec": RADIUS_ARCSEC,
    "dedupe_radius_arcsec": DEDUPE_RADIUS_ARCSEC,
    "pair_radius_overrides": {PAIR_RADIUS_OVERRIDE[0]: PAIR_RADIUS_OVERRIDE[1]},
}


def in_memory(catalogs, **settings):
    surveys = [
        survey_coords_from_arrays(name, ra, dec) for name, (ra, dec) in catalogs.items()
    ]
    return build_edges(surveys, **settings)


@pytest.mark.parametrize("band_height_arcsec", [1.0, 3.0, 20.0, 330.0])
def test_band_build_reproduces_the_in_memory_edges(band_height_arcsec):
    combined, _, boundary = banded_edges(
        PARITY_CATALOGS, band_height_arcsec * ARCSEC, **PARITY_SETTINGS
    )
    assert_same_edges(combined, in_memory(PARITY_CATALOGS, **PARITY_SETTINGS))
    # Thin bands send many groups across boundaries; thick ones very few.
    handed = sum(len(s.rows) for s in boundary.surveys)
    assert handed > 0


@pytest.mark.parametrize(
    "mode",
    [
        EntitywiseCrossmatchConfig(),
        EntitywiseCrossmatchConfig(resolver="split"),
        EntitywiseCrossmatchConfig(resolver="sequential", priority=["b", "a", "c"]),
        EntitywiseCrossmatchConfig(disputed="drop"),
        DegenerateCrossmatchConfig(surveys=["a", "b", "c"]),
        DegenerateCrossmatchConfig(surveys=["a", "b"], policy="anchored_nearest"),
    ],
)
def test_resolving_band_segments_separately_gives_the_same_entities(mode):
    _, segments, boundary = banded_edges(
        PARITY_CATALOGS, 3.0 * ARCSEC, **PARITY_SETTINGS
    )
    whole = resolve(in_memory(PARITY_CATALOGS, **PARITY_SETTINGS), mode)
    parts = [resolve(segment, mode) for segment in [*segments, boundary]]
    assert sorted(r for part in parts for r in canonical(part)) == canonical(whole)


@pytest.mark.parametrize("seed", range(4))
def test_random_crowded_catalogs_with_chains_across_bands(seed):
    rng = np.random.default_rng(seed)
    catalogs = {}
    for name, n in (("x", 600), ("y", 500)):
        # About one neighbour per arcsec-radius disc: many long chains.
        ra = 10 + rng.uniform(0, 30, n) * ARCSEC
        dec = rng.uniform(-15, 15, n) * ARCSEC
        catalogs[name] = (ra, dec)
    settings = {
        "radius_arcsec": 1.0,
        "dedupe_radius_arcsec": {"x": 0.6, "y": 0.0},
    }
    for band_height in (1.0, 2.5, 7.0):
        combined, _, _ = banded_edges(catalogs, band_height * ARCSEC, **settings)
        assert_same_edges(combined, in_memory(catalogs, **settings))


def test_a_chain_spanning_three_bands_is_finished_by_the_handover():
    # a0 - b0 - a1 - b1 - a2, 0.9 arcsec steps in declination across 1 arcsec bands.
    step = 0.9 * ARCSEC
    catalogs = {
        "a": (np.full(3, 10.0), np.array([0.05, 2, 4]) * step),
        "b": (np.full(2, 10.0), np.array([1, 3]) * step),
    }
    settings = {"radius_arcsec": 1.0, "dedupe_radius_arcsec": {"a": 0.0, "b": 0.0}}
    combined, segments, boundary = banded_edges(catalogs, 1.0 * ARCSEC, **settings)
    assert_same_edges(combined, in_memory(catalogs, **settings))
    assert all(len(s.pairs[frozenset("ab")].row_a) == 0 for s in segments)
    assert boundary.surveys[0].rows.tolist() == [0, 1, 2]
    assert boundary.pairs[frozenset("ab")].row_a.tolist() == [0, 1, 1, 2]
