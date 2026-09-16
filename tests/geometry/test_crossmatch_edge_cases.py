"""Edge case tests for ``ContrAst.datasets.crossmatch``.

Covers degenerate inputs (empty / single-row / no-match), RA wraparound,
antipodes, and the radius boundary behaviour.
"""

from __future__ import annotations

import numpy as np
import pytest

from astro_crossmatch.geometry import (
    crossmatch_radec as crossmatch,
    detect_collisions,
)

from . import _astropy_reference as astropy_ref
from ._crossmatch_fixtures import (
    assert_groups_equal_as_sets,
    make_disjoint_catalogs,
)

# --------------------------------------------------------------------------------------
# Empty / no-match
# --------------------------------------------------------------------------------------


def test_empty_query_returns_no_groups():
    coords = (
        (np.array([], dtype=float), np.array([], dtype=float)),
        (np.array([10.0]), np.array([10.0])),
    )
    actual = crossmatch(coords, radius_arcsec=1.0)
    assert actual.shape[1] == 0
    assert actual.shape[0] in (0, 2)


def test_empty_key_returns_no_groups():
    coords = (
        (np.array([10.0]), np.array([10.0])),
        (np.array([], dtype=float), np.array([], dtype=float)),
    )
    actual = crossmatch(coords, radius_arcsec=1.0)
    assert actual.shape[1] == 0


def test_disjoint_catalogs_yield_no_matches_within_small_radius():
    coords = make_disjoint_catalogs("mid_latitude", n_per_catalog=50, seed=99)
    actual = crossmatch(coords, radius_arcsec=0.01)  # 10 mas; effectively none
    expected = astropy_ref.one_way(coords, radius_arcsec=0.01)
    assert_groups_equal_as_sets(actual, expected)
    assert actual.shape[1] == 0


# --------------------------------------------------------------------------------------
# Single match
# --------------------------------------------------------------------------------------


def test_single_exact_match_two_catalogs():
    coords = (
        (np.array([45.0]), np.array([30.0])),
        (np.array([45.0]), np.array([30.0])),
    )
    actual = crossmatch(coords, radius_arcsec=0.001)
    assert actual.shape == (2, 1)
    assert int(actual[0, 0]) == 0
    assert int(actual[1, 0]) == 0


# --------------------------------------------------------------------------------------
# RA wraparound
# --------------------------------------------------------------------------------------


def test_ra_wraparound_at_zero_meridian():
    """Two sources straddling RA=0 must be matched as ~0.36 arcsec apart."""
    coords = (
        (np.array([359.99995]), np.array([10.0])),
        (np.array([0.00005]), np.array([10.0])),
    )
    # Separation in RA at dec=10 is 0.0001 deg * cos(10) ~= 0.354 arcsec
    actual_within = crossmatch(coords, radius_arcsec=1.0)
    actual_outside = crossmatch(coords, radius_arcsec=0.1)
    assert actual_within.shape == (2, 1)
    assert actual_outside.shape[1] == 0


# --------------------------------------------------------------------------------------
# Antipodes
# --------------------------------------------------------------------------------------


def test_antipodes_do_not_match():
    """Sources 180° apart must not be matched at any plausible radius."""
    coords = (
        (np.array([0.0, 90.0]), np.array([0.0, 0.0])),
        (np.array([180.0, 270.0]), np.array([0.0, 0.0])),
    )
    # All cross-catalog separations are 90° or 180°.
    actual = crossmatch(coords, radius_arcsec=3600.0)  # 1 degree -- still nowhere near
    assert actual.shape[1] == 0


# --------------------------------------------------------------------------------------
# Radius boundary
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("radius_arcsec", [0.5, 1.0, 5.0])
def test_radius_boundary_inclusion(radius_arcsec):
    """A pair separated by ``radius - eps`` matches; ``radius + eps`` does not.

    Tests at dec=0 with an RA-direction offset, where the chord-vs-angle
    relationship is most likely to expose floating-point boundary issues.
    """
    eps = 1e-4  # arcsec; well inside float64 precision at these scales
    inside_offset_deg = (radius_arcsec - eps) / 3600.0
    outside_offset_deg = (radius_arcsec + eps) / 3600.0

    coords_inside = (
        (np.array([0.0]), np.array([0.0])),
        (np.array([inside_offset_deg]), np.array([0.0])),
    )
    coords_outside = (
        (np.array([0.0]), np.array([0.0])),
        (np.array([outside_offset_deg]), np.array([0.0])),
    )

    inside = crossmatch(coords_inside, radius_arcsec=radius_arcsec)
    outside = crossmatch(coords_outside, radius_arcsec=radius_arcsec)

    assert inside.shape == (2, 1), f"radius={radius_arcsec}: inside should match"
    assert outside.shape[1] == 0, f"radius={radius_arcsec}: outside should not match"


# --------------------------------------------------------------------------------------
# Equator regression -- the bug we just fixed
# --------------------------------------------------------------------------------------


def test_equatorial_ra_separation_is_not_underestimated():
    """Pre-fix, the colatitude bug collapsed the equator to a point, so two
    equatorial sources tens of arcsec apart in RA would have been matched as
    if they were within 1 arcsec. This test makes that regression explicit.
    """
    coords = (
        (np.array([0.0]), np.array([0.0])),
        (np.array([10.0 / 3600.0]), np.array([0.0])),  # 10 arcsec away
    )
    actual = crossmatch(coords, radius_arcsec=1.0)
    assert actual.shape[1] == 0, (
        "Two equatorial sources separated by 10 arcsec must not match at 1 arcsec."
    )


# --------------------------------------------------------------------------------------
# Detect-collisions edge cases
# --------------------------------------------------------------------------------------


def test_detect_collisions_with_empty_catalog():
    coords = (
        (np.array([0.0, 1.0]), np.array([0.0, 0.0])),
        (np.array([], dtype=float), np.array([], dtype=float)),
    )
    masks = detect_collisions(coords, radius_arcsec=1.0)
    assert len(masks) == 2
    assert not masks[0].any()
    assert masks[1].size == 0


def test_detect_collisions_self_pair_excluded():
    """A row in catalog 0 must not be marked as colliding with itself.

    ``detect_collisions`` reports cross-catalog collisions only.
    """
    coords = (
        (np.array([0.0, 1.0]), np.array([0.0, 0.0])),
        (np.array([10.0]), np.array([10.0])),
    )
    masks = detect_collisions(coords, radius_arcsec=1.0)
    assert not masks[0].any()
    assert not masks[1].any()
