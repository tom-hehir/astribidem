"""Random catalogues, astribidem calls in the reference's shape, and comparison."""

from __future__ import annotations

import numpy as np
import pyarrow as pa

from astribidem import (
    DegenerateCrossmatchConfig,
    SpatialLink,
    crossmatch,
    match_hub_and_spoke,
)

CoordPair = tuple[np.ndarray, np.ndarray]


REGIMES = ("equator", "mid_latitude", "near_pole", "full_sphere")


def _sample_regime(rng: np.random.Generator, n: int, regime: str) -> CoordPair:
    """Draw ``n`` (ra, dec) pairs in degrees from the named regime."""
    ra = rng.uniform(0.0, 360.0, size=n)
    if regime == "equator":
        dec = rng.uniform(-3.0, 3.0, size=n)
    elif regime == "mid_latitude":
        sign = rng.choice([-1.0, 1.0], size=n)
        dec = sign * rng.uniform(20.0, 60.0, size=n)
    elif regime == "near_pole":
        sign = rng.choice([-1.0, 1.0], size=n)
        dec = sign * rng.uniform(85.0, 89.5, size=n)
    elif regime == "full_sphere":
        u = rng.uniform(-1.0, 1.0, size=n)
        dec = np.degrees(np.arcsin(u))
    else:
        raise ValueError(f"Unknown regime: {regime!r}")
    return ra, dec


def _offset_in_degrees(
    ra: np.ndarray,
    dec: np.ndarray,
    offset_arcsec: np.ndarray,
    angle_rad: np.ndarray,
) -> CoordPair:
    """Apply a small great-circle offset in a random direction.

    The offset is scaled by ``1 / cos(dec)`` in RA so the angular distance
    is preserved (small-angle approximation, which is fine for sub-arcsec
    perturbations away from |dec| ~ 90°).
    """
    offset_deg = offset_arcsec / 3600.0
    cos_dec = np.cos(np.radians(np.clip(dec, -89.99, 89.99)))
    new_ra = (ra + offset_deg * np.cos(angle_rad) / cos_dec) % 360.0
    new_dec = np.clip(dec + offset_deg * np.sin(angle_rad), -90.0, 90.0)
    return new_ra, new_dec


def make_overlapping_catalogs(
    regime: str,
    n_per_catalog: int = 200,
    n_catalogs: int = 2,
    overlap_fraction: float = 0.7,
    offset_arcsec: float = 0.3,
    seed: int = 0,
) -> tuple[CoordPair, ...]:
    """Build ``n_catalogs`` catalogs sharing a partial common population.

    A "common" subset of size ``int(overlap_fraction * n_per_catalog)`` is
    drawn once and reused across catalogs (with sub-arcsec random offsets).
    The remainder of each catalog is independent random noise drawn from
    the same regime. Result is shuffled per-catalog to avoid the matched
    indices accidentally lining up.
    """
    if not (0.0 <= overlap_fraction <= 1.0):
        raise ValueError("overlap_fraction must be in [0, 1].")

    rng = np.random.default_rng(seed)
    n_common = int(round(overlap_fraction * n_per_catalog))
    n_unique = n_per_catalog - n_common

    common_ra, common_dec = _sample_regime(rng, n_common, regime)

    catalogs = []
    for _ in range(n_catalogs):
        offsets = rng.uniform(0.0, offset_arcsec, size=n_common)
        angles = rng.uniform(0.0, 2 * np.pi, size=n_common)
        ra_match, dec_match = _offset_in_degrees(common_ra, common_dec, offsets, angles)

        ra_unique, dec_unique = _sample_regime(rng, n_unique, regime)
        ra = np.concatenate([ra_match, ra_unique])
        dec = np.concatenate([dec_match, dec_unique])

        perm = rng.permutation(len(ra))
        catalogs.append((ra[perm], dec[perm]))

    return tuple(catalogs)


def make_disjoint_catalogs(
    regime: str, n_per_catalog: int = 50, n_catalogs: int = 2, seed: int = 0
) -> tuple[CoordPair, ...]:
    """Catalogs drawn independently from the regime; matches are accidental only."""
    rng = np.random.default_rng(seed)
    return tuple(_sample_regime(rng, n_per_catalog, regime) for _ in range(n_catalogs))


def assert_groups_equal_as_sets(actual: np.ndarray, expected: np.ndarray) -> None:
    """Compare two ``(k, m)`` group arrays as unordered sets of column tuples.

    astribidem and the astropy reference may return groups in different
    orders, so the sets of groups are compared.
    """
    if actual.size == 0 and expected.size == 0:
        return
    if actual.shape[0] != expected.shape[0]:
        raise AssertionError(
            f"Different number of catalogs: actual k={actual.shape[0]}, "
            f"expected k={expected.shape[0]}"
        )
    actual_set = {tuple(int(i) for i in actual[:, c]) for c in range(actual.shape[1])}
    expected_set = {
        tuple(int(i) for i in expected[:, c]) for c in range(expected.shape[1])
    }
    only_actual = actual_set - expected_set
    only_expected = expected_set - actual_set
    if only_actual or only_expected:
        raise AssertionError(
            f"Group mismatch.\n"
            f"  Only in actual:   {sorted(only_actual)[:10]}"
            f"{' (truncated)' if len(only_actual) > 10 else ''}\n"
            f"  Only in expected: {sorted(only_expected)[:10]}"
            f"{' (truncated)' if len(only_expected) > 10 else ''}"
        )


def _groups(index: pa.Table, names: list[str]) -> np.ndarray:
    if index.num_rows == 0:
        return np.empty((len(names), 0), dtype=np.int64)
    return np.stack([index[f"{name}/row_index"].to_numpy() for name in names])


def anchored_nearest(coords, radius_arcsec: float) -> np.ndarray:
    """Hub-and-spoke ``anchored_nearest`` from catalogue 0, dedupe disabled."""
    names = [f"catalog{i}" for i in range(len(coords))]
    index = match_hub_and_spoke(
        anchor=names[0],
        coordinates=dict(zip(names, coords, strict=True)),
        links={
            name: SpatialLink(radius_arcsec, "anchored_nearest") for name in names[1:]
        },
        dedupe_radius_arcsec=0.0,
    )
    return _groups(index, names)


def mutual_nearest(coords, radius_arcsec: float) -> np.ndarray:
    """A ``mutual_nearest`` subset join over every catalogue, dedupe disabled."""
    names = [f"catalog{i}" for i in range(len(coords))]
    index = crossmatch(
        dict(zip(names, coords, strict=True)),
        radius_arcsec=radius_arcsec,
        dedupe_radius_arcsec=0.0,
        mode=DegenerateCrossmatchConfig(surveys=names, policy="mutual_nearest"),
    )
    return _groups(index, names)
