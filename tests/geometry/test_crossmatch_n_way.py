"""N-way crossmatch tests (3 and 4 catalogs).

Verifies that the AstroBench adapters around Astral's anchored and all-pairs
mutual primitives agree with the independent astropy-based reference.
"""

from __future__ import annotations

import pytest

from astro_crossmatch.geometry import crossmatch_radec as crossmatch

from . import _astropy_reference as astropy_ref
from ._crossmatch_fixtures import (
    REGIMES,
    assert_groups_equal_as_sets,
    make_overlapping_catalogs,
)


@pytest.mark.parametrize("regime", REGIMES)
@pytest.mark.parametrize("n_catalogs", [3, 4])
@pytest.mark.parametrize("radius_arcsec", [0.5, 1.0, 5.0])
def test_one_way_n_way_matches_astropy(regime, n_catalogs, radius_arcsec):
    coords = make_overlapping_catalogs(
        regime,
        n_per_catalog=150,
        n_catalogs=n_catalogs,
        overlap_fraction=0.6,
        seed=10 + n_catalogs,
    )

    actual = crossmatch(coords, radius_arcsec=radius_arcsec)
    expected = astropy_ref.one_way(coords, radius_arcsec=radius_arcsec)

    assert_groups_equal_as_sets(actual, expected)


@pytest.mark.parametrize("regime", REGIMES)
@pytest.mark.parametrize("n_catalogs", [3, 4])
@pytest.mark.parametrize("radius_arcsec", [0.5, 1.0, 5.0])
def test_all_pairs_mutual_nearest_n_way_matches_astropy(
    regime, n_catalogs, radius_arcsec
):
    coords = make_overlapping_catalogs(
        regime,
        n_per_catalog=150,
        n_catalogs=n_catalogs,
        overlap_fraction=0.6,
        seed=20 + n_catalogs,
    )

    actual = crossmatch(
        coords,
        radius_arcsec=radius_arcsec,
        match_policy="mutual_nearest",
        matching_topology="all-pairs",
    )
    expected = astropy_ref.reciprocal(coords, radius_arcsec=radius_arcsec)

    assert_groups_equal_as_sets(actual, expected)


@pytest.mark.parametrize("n_catalogs", [3, 4])
def test_partial_overlap_filters_correctly(n_catalogs):
    """Sources present in only some catalogs must not appear in the result.

    Builds catalogs where the overlap fraction is small enough that there
    are *some* matched groups but most rows don't appear in all catalogs.
    The reference and the Astral-backed adapter must agree on which rows
    survive the all-catalogs intersection.
    """
    coords = make_overlapping_catalogs(
        "mid_latitude",
        n_per_catalog=120,
        n_catalogs=n_catalogs,
        overlap_fraction=0.3,
        seed=30 + n_catalogs,
    )

    actual = crossmatch(coords, radius_arcsec=1.0)
    expected = astropy_ref.one_way(coords, radius_arcsec=1.0)

    assert actual.shape[1] == expected.shape[1] > 0, (
        "Sanity check: this fixture should produce some matches."
    )
    assert actual.shape[1] < int(0.5 * 120), (
        "Sanity check: with overlap_fraction=0.3 and n_catalogs>=3 the "
        "intersection should be a small minority of rows."
    )
    assert_groups_equal_as_sets(actual, expected)
