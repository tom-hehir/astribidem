"""Kernel contract tests against a brute-force O(n^2) oracle."""

from __future__ import annotations

import numpy as np
import pytest
from astro_crossmatch.kernel import (
    CatalogKernel,
    arcsec_to_chord,
    chord_to_arcsec,
    radec_to_xyz,
    resolve_workers,
)


def _brute_force_pairs(xyz_a, xyz_b, radius_arcsec):
    """Every (i, j, sep) with sep <= radius, lexsorted by (i, j)."""
    chord = np.linalg.norm(xyz_a[:, None, :] - xyz_b[None, :, :], axis=-1)
    sep = chord_to_arcsec(chord)
    i, j = np.nonzero(sep <= radius_arcsec + 1e-12)
    return i.astype(np.int64), j.astype(np.int64), sep[i, j]


def _random_patch(rng, n, *, ra0=150.0, dec0=0.0, span=0.05):
    ra = rng.uniform(ra0, ra0 + span, n)
    dec = rng.uniform(dec0, dec0 + span, n)
    return ra, dec


@pytest.mark.parametrize("radius", [0.5, 1.0, 5.0])
def test_all_pairs_matches_brute_force(radius):
    rng = np.random.default_rng(7)
    ra_a, dec_a = _random_patch(rng, 400)
    ra_b, dec_b = _random_patch(rng, 300)
    a = CatalogKernel(ra_a, dec_a)
    b = CatalogKernel(ra_b, dec_b)

    i, j, sep = a.all_pairs(b, radius)
    oi, oj, osep = _brute_force_pairs(a.xyz, b.xyz, radius)

    np.testing.assert_array_equal(i, oi)
    np.testing.assert_array_equal(j, oj)
    np.testing.assert_allclose(sep, osep, rtol=0, atol=1e-9)
    # Deterministic lexsorted emission.
    assert np.all(np.lexsort((j, i)) == np.arange(len(i)))


def test_self_pairs_matches_brute_force():
    rng = np.random.default_rng(11)
    ra, dec = _random_patch(rng, 500, span=0.01)
    kernel = CatalogKernel(ra, dec)

    i, j, sep = kernel.self_pairs(2.0)
    oi, oj, osep = _brute_force_pairs(kernel.xyz, kernel.xyz, 2.0)
    keep = oi < oj  # oracle emits both directions and identity pairs
    np.testing.assert_array_equal(i, oi[keep])
    np.testing.assert_array_equal(j, oj[keep])
    np.testing.assert_allclose(sep, osep[keep], rtol=0, atol=1e-9)
    assert np.all(i < j)


def test_self_pairs_includes_exact_duplicates():
    ra = np.array([10.0, 10.0, 11.0])
    dec = np.array([0.0, 0.0, 0.0])
    i, j, sep = CatalogKernel(ra, dec).self_pairs(0.5)
    assert i.tolist() == [0]
    assert j.tolist() == [1]
    assert sep[0] == 0.0


def test_boundary_behaviour_is_consistent_across_paths():
    # A pair ~one arcsecond apart; probe radii straddling and pinning the
    # measured separation. Whatever the float rounding does exactly at the
    # bound, the all-pairs, k-truncated, and self-match paths must agree, and
    # the strict above/below cases must include/exclude respectively.
    ra = np.array([180.0])
    dec = np.array([0.0])
    partner_ra = np.array([180.0 + 1.0 / 3600.0])
    a = CatalogKernel(ra, dec)
    b = CatalogKernel(partner_ra, dec)
    both = CatalogKernel(np.concatenate([ra, partner_ra]), np.zeros(2))

    _, _, sep = a.all_pairs(b, 2.0)
    sep0 = float(sep[0])

    for radius in (sep0 * (1 - 1e-9), sep0, sep0 * (1 + 1e-9)):
        n_full = len(a.all_pairs(b, radius)[0])
        n_k = len(a.all_pairs(b, radius, k=1)[0])
        n_self = len(both.self_pairs(radius)[0])
        assert n_full == n_k == n_self
    assert len(a.all_pairs(b, sep0 * (1 + 1e-9))[0]) == 1
    assert len(a.all_pairs(b, sep0 * (1 - 1e-9))[0]) == 0


@pytest.mark.parametrize("k", [1, 2])
def test_k_truncation_matches_brute_force(k):
    rng = np.random.default_rng(3)
    ra_a, dec_a = _random_patch(rng, 200, span=0.005)
    ra_b, dec_b = _random_patch(rng, 200, span=0.005)
    a = CatalogKernel(ra_a, dec_a)
    b = CatalogKernel(ra_b, dec_b)
    radius = 5.0

    i, j, sep = a.all_pairs(b, radius, k=k)
    oi, oj, osep = _brute_force_pairs(a.xyz, b.xyz, radius)
    for row in range(len(ra_a)):
        mine = sorted(sep[i == row])
        oracle = sorted(osep[oi == row])[:k]
        np.testing.assert_allclose(mine, oracle, rtol=0, atol=1e-9)


def test_chunked_join_is_bit_identical_to_serial():
    rng = np.random.default_rng(19)
    ra_a, dec_a = _random_patch(rng, 1000)
    ra_b, dec_b = _random_patch(rng, 800)
    a = CatalogKernel(ra_a, dec_a)
    b = CatalogKernel(ra_b, dec_b)

    serial = a.all_pairs(b, 3.0)
    chunked = a.all_pairs(b, 3.0, workers=4, chunk_rows=137)
    for s, c in zip(serial, chunked):
        np.testing.assert_array_equal(s, c)


def test_empty_inputs():
    empty = CatalogKernel(np.empty(0), np.empty(0))
    other = CatalogKernel(np.array([1.0]), np.array([0.0]))
    i, j, sep = empty.all_pairs(other, 1.0)
    assert len(i) == len(j) == len(sep) == 0
    i, j, sep = empty.self_pairs(1.0)
    assert len(i) == 0


def test_validation_errors():
    kernel = CatalogKernel(np.array([1.0]), np.array([0.0]))
    with pytest.raises(ValueError, match="radius_arcsec"):
        kernel.self_pairs(float("nan"))
    with pytest.raises(ValueError, match="radius_arcsec"):
        kernel.all_pairs(kernel, -1.0)
    with pytest.raises(ValueError, match="k must be >= 1"):
        kernel.all_pairs(kernel, 1.0, k=0)
    with pytest.raises(ValueError, match="workers"):
        resolve_workers(0)
    with pytest.raises(ValueError, match="shape"):
        CatalogKernel.from_xyz(np.zeros((3, 2)))


def test_radec_chord_roundtrip():
    sep = chord_to_arcsec(np.array([arcsec_to_chord(0.7)]))
    assert sep[0] == pytest.approx(0.7, abs=1e-12)
    xyz = radec_to_xyz(np.array([120.0]), np.array([-45.0]))
    assert np.linalg.norm(xyz[0]) == pytest.approx(1.0, abs=1e-15)
