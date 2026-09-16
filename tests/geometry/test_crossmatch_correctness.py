"""Correctness tests for AstroBench's Astral adapter against astropy.

Parametrized over four sky regimes (equator, mid-latitude, near-pole,
full-sphere) and four matching radii. The astropy-based reference lives in
``tests/datasets/_astropy_reference.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from astro_crossmatch.geometry import (
    crossmatch_radec as crossmatch,
    detect_collisions,
    detect_primary_collisions,
)

from . import _astropy_reference as astropy_ref
from ._crossmatch_fixtures import (
    REGIMES,
    assert_groups_equal_as_sets,
    make_overlapping_catalogs,
)

RADII_ARCSEC = (0.1, 1.0, 5.0, 60.0)


@pytest.mark.parametrize("regime", REGIMES)
@pytest.mark.parametrize("radius_arcsec", RADII_ARCSEC)
def test_one_way_two_catalogs_matches_astropy(regime, radius_arcsec):
    coords = make_overlapping_catalogs(regime, n_per_catalog=200, seed=0)

    actual = crossmatch(coords, radius_arcsec=radius_arcsec)
    expected = astropy_ref.one_way(coords, radius_arcsec=radius_arcsec)

    assert_groups_equal_as_sets(actual, expected)


@pytest.mark.parametrize("regime", REGIMES)
@pytest.mark.parametrize("radius_arcsec", RADII_ARCSEC)
def test_mutual_nearest_two_catalogs_matches_astropy(regime, radius_arcsec):
    coords = make_overlapping_catalogs(regime, n_per_catalog=200, seed=1)

    actual = crossmatch(
        coords,
        radius_arcsec=radius_arcsec,
        match_policy="mutual_nearest",
        matching_topology="all-pairs",
    )
    expected = astropy_ref.reciprocal(coords, radius_arcsec=radius_arcsec)

    assert_groups_equal_as_sets(actual, expected)


@pytest.mark.parametrize("regime", REGIMES)
@pytest.mark.parametrize("radius_arcsec", RADII_ARCSEC)
def test_primary_collisions_matches_astropy(regime, radius_arcsec):
    coords = make_overlapping_catalogs(regime, n_per_catalog=150, n_catalogs=3, seed=2)

    actual = detect_primary_collisions(coords, radius_arcsec=radius_arcsec)
    expected = astropy_ref.primary_collisions(coords, radius_arcsec=radius_arcsec)

    assert len(actual) == len(expected)
    for i, (a, e) in enumerate(zip(actual, expected)):
        np.testing.assert_array_equal(
            a, e, err_msg=f"primary collision mask differs for catalog {i}"
        )


@pytest.mark.parametrize("regime", REGIMES)
@pytest.mark.parametrize("radius_arcsec", RADII_ARCSEC)
def test_all_collisions_matches_astropy(regime, radius_arcsec):
    coords = make_overlapping_catalogs(regime, n_per_catalog=150, n_catalogs=3, seed=3)

    actual = detect_collisions(coords, radius_arcsec=radius_arcsec)
    expected = astropy_ref.all_collisions(coords, radius_arcsec=radius_arcsec)

    assert len(actual) == len(expected)
    for i, (a, e) in enumerate(zip(actual, expected)):
        np.testing.assert_array_equal(
            a, e, err_msg=f"all-collision mask differs for catalog {i}"
        )
