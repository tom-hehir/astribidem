"""Empty and disjoint inputs, the RA wrap, antipodes and the radius boundary."""

import numpy as np
import pytest

from . import _astropy as astropy_ref
from ._support import (
    anchored_nearest,
    assert_groups_equal_as_sets,
    make_disjoint_catalogs,
    mutual_nearest,
)

MATCHERS = [anchored_nearest, mutual_nearest]
EMPTY = (np.array([], dtype=float), np.array([], dtype=float))


@pytest.mark.parametrize("match", MATCHERS)
def test_an_empty_catalog_gives_no_groups(match):
    one = (np.array([10.0]), np.array([10.0]))
    assert match((EMPTY, one), 1.0).shape == (2, 0)
    assert match((one, EMPTY), 1.0).shape == (2, 0)


def test_disjoint_catalogs_give_no_groups_within_a_small_radius():
    coords = make_disjoint_catalogs("mid_latitude", n_per_catalog=50, seed=99)
    actual = anchored_nearest(coords, 0.01)
    assert_groups_equal_as_sets(actual, astropy_ref.one_way(coords, 0.01))
    assert actual.shape[1] == 0


@pytest.mark.parametrize("match", MATCHERS)
def test_identical_positions_match(match):
    coords = ((np.array([45.0]), np.array([30.0])),) * 2
    np.testing.assert_array_equal(match(coords, 0.001), [[0], [0]])


@pytest.mark.parametrize("match", MATCHERS)
def test_rows_either_side_of_ra_zero_match(match):
    # 0.0001 degrees of RA at declination 10 is about 0.354 arcsec.
    coords = (
        (np.array([359.99995]), np.array([10.0])),
        (np.array([0.00005]), np.array([10.0])),
    )
    assert match(coords, 1.0).shape == (2, 1)
    assert match(coords, 0.1).shape == (2, 0)


@pytest.mark.parametrize("match", MATCHERS)
def test_antipodes_do_not_match(match):
    coords = (
        (np.array([0.0, 90.0]), np.array([0.0, 0.0])),
        (np.array([180.0, 270.0]), np.array([0.0, 0.0])),
    )
    assert match(coords, 3600.0).shape == (2, 0)


@pytest.mark.parametrize("match", MATCHERS)
@pytest.mark.parametrize("radius_arcsec", [0.5, 1.0, 5.0])
def test_radius_boundary(match, radius_arcsec):
    # Offsets along RA at the equator, where chord and angle are most likely
    # to disagree at the boundary.
    def pair(offset_arcsec):
        return (
            (np.array([0.0]), np.array([0.0])),
            (np.array([offset_arcsec / 3600.0]), np.array([0.0])),
        )

    assert match(pair(radius_arcsec - 1e-4), radius_arcsec).shape == (2, 1)
    assert match(pair(radius_arcsec + 1e-4), radius_arcsec).shape == (2, 0)


@pytest.mark.parametrize("match", MATCHERS)
def test_equatorial_ra_separation_is_not_underestimated(match):
    coords = (
        (np.array([0.0]), np.array([0.0])),
        (np.array([10.0 / 3600.0]), np.array([0.0])),
    )
    assert match(coords, 1.0).shape == (2, 0)
