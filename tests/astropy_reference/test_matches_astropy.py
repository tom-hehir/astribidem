"""astribidem's nearest-neighbour matchers agree with an astropy reference.

Four sky regimes (equator, mid-latitude, near the poles, the full sphere) and
several radii. Anchored-nearest hub-and-spoke matching is compared with
astropy's one-way nearest neighbour, and the mutual-nearest subset join with
the groups every catalogue agrees on as the anchor.
"""

import pytest

from . import _astropy as astropy_ref
from ._support import (
    REGIMES,
    anchored_nearest,
    assert_groups_equal_as_sets,
    make_overlapping_catalogs,
    mutual_nearest,
)

RADII_ARCSEC = (0.1, 1.0, 5.0, 60.0)


@pytest.mark.parametrize("regime", REGIMES)
@pytest.mark.parametrize("radius_arcsec", RADII_ARCSEC)
def test_two_catalogs_anchored_nearest_matches_astropy(regime, radius_arcsec):
    coords = make_overlapping_catalogs(regime, n_per_catalog=200, seed=0)
    assert_groups_equal_as_sets(
        anchored_nearest(coords, radius_arcsec),
        astropy_ref.one_way(coords, radius_arcsec=radius_arcsec),
    )


@pytest.mark.parametrize("regime", REGIMES)
@pytest.mark.parametrize("radius_arcsec", RADII_ARCSEC)
def test_two_catalogs_mutual_nearest_matches_astropy(regime, radius_arcsec):
    coords = make_overlapping_catalogs(regime, n_per_catalog=200, seed=1)
    assert_groups_equal_as_sets(
        mutual_nearest(coords, radius_arcsec),
        astropy_ref.reciprocal(coords, radius_arcsec=radius_arcsec),
    )


@pytest.mark.parametrize("regime", REGIMES)
@pytest.mark.parametrize("n_catalogs", [3, 4])
@pytest.mark.parametrize("radius_arcsec", [0.5, 1.0, 5.0])
def test_n_catalogs_anchored_nearest_matches_astropy(regime, n_catalogs, radius_arcsec):
    coords = make_overlapping_catalogs(
        regime,
        n_per_catalog=150,
        n_catalogs=n_catalogs,
        overlap_fraction=0.6,
        seed=10 + n_catalogs,
    )
    assert_groups_equal_as_sets(
        anchored_nearest(coords, radius_arcsec),
        astropy_ref.one_way(coords, radius_arcsec=radius_arcsec),
    )


@pytest.mark.parametrize("regime", REGIMES)
@pytest.mark.parametrize("n_catalogs", [3, 4])
@pytest.mark.parametrize("radius_arcsec", [0.5, 1.0, 5.0])
def test_n_catalogs_mutual_nearest_matches_astropy(regime, n_catalogs, radius_arcsec):
    coords = make_overlapping_catalogs(
        regime,
        n_per_catalog=150,
        n_catalogs=n_catalogs,
        overlap_fraction=0.6,
        seed=20 + n_catalogs,
    )
    assert_groups_equal_as_sets(
        mutual_nearest(coords, radius_arcsec),
        astropy_ref.reciprocal(coords, radius_arcsec=radius_arcsec),
    )


@pytest.mark.parametrize("n_catalogs", [3, 4])
def test_rows_missing_from_some_catalogs_are_left_out(n_catalogs):
    coords = make_overlapping_catalogs(
        "mid_latitude",
        n_per_catalog=120,
        n_catalogs=n_catalogs,
        overlap_fraction=0.3,
        seed=30 + n_catalogs,
    )
    actual = anchored_nearest(coords, 1.0)
    expected = astropy_ref.one_way(coords, radius_arcsec=1.0)
    # The fixture must give some groups, but far fewer than one per row.
    assert 0 < actual.shape[1] == expected.shape[1] < 60
    assert_groups_equal_as_sets(actual, expected)
