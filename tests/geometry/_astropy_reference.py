"""Astropy-based reference implementations of crossmatching primitives.

Used by the crossmatch tests as a gold-standard oracle for AstroBench's
Astral-backed adapter. Keep this module test-only -- runtime code
must not import it.

All functions accept ``coords`` as a sequence of ``(ra, dec)`` numpy arrays
in degrees (matching the public crossmatch API).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord

CoordPair = tuple[np.ndarray, np.ndarray]


def _to_skycoord(ra_deg: np.ndarray, dec_deg: np.ndarray) -> SkyCoord:
    return SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg)


def one_way(coords: Sequence[CoordPair], radius_arcsec: float) -> np.ndarray:
    """Reference one-way crossmatch matching ``crossmatch.one_way_crossmatch``.

    For each row of ``coords[0]``, find the nearest neighbour in each
    subsequent catalog. Keep only rows where every catalog returned a match
    inside ``radius_arcsec``. Returns ``(k, m)`` array of indices.
    """
    if len(coords) == 0:
        return np.empty((0, 0), dtype=int)

    query = _to_skycoord(*coords[0])
    n_query = len(coords[0][0])

    if n_query == 0:
        return np.empty((len(coords), 0), dtype=int)

    columns: list[np.ndarray] = [np.arange(n_query, dtype=int)]
    masks: list[np.ndarray] = []
    for ra, dec in coords[1:]:
        if len(ra) == 0:
            return np.empty((len(coords), 0), dtype=int)
        keys = _to_skycoord(ra, dec)
        idx, sep, _ = query.match_to_catalog_sky(keys)
        columns.append(np.asarray(idx, dtype=int))
        masks.append(np.asarray(sep <= radius_arcsec * u.arcsec, dtype=bool))

    if not masks:
        # Single-catalog case: every row trivially matches itself.
        return np.stack(columns, axis=0)

    keep = np.logical_and.reduce(masks)
    return np.stack(columns, axis=1)[keep].T


def _one_way_anchored_at(
    coords: Sequence[CoordPair], anchor: int, radius_arcsec: float
) -> np.ndarray:
    """Run one-way crossmatch with ``coords[anchor]`` as the query.

    Returns indices in canonical column order (column ``j`` = catalog ``j``).
    """
    rotated: list[CoordPair] = [coords[anchor]] + [
        coords[j] for j in range(len(coords)) if j != anchor
    ]
    rotated_groups = one_way(rotated, radius_arcsec)
    if rotated_groups.shape[1] == 0:
        return np.empty((len(coords), 0), dtype=int)

    # Map rotated column index -> canonical catalog index.
    rotated_to_canonical = [anchor] + [j for j in range(len(coords)) if j != anchor]
    canonical = np.empty_like(rotated_groups)
    for rotated_col, canonical_col in enumerate(rotated_to_canonical):
        canonical[canonical_col] = rotated_groups[rotated_col]
    return canonical


def reciprocal(coords: Sequence[CoordPair], radius_arcsec: float) -> np.ndarray:
    """Reference reciprocal crossmatch matching ``reciprocal_crossmatch``.

    A group ``(i_0, ..., i_{k-1})`` survives iff, when each catalog in turn
    is used as the anchor, the resulting one-way match returns the same
    tuple. This is the multi-direction-consensus reciprocal match
    implemented by ``ContrAst.datasets.crossmatch._reciprocal_crossmatch``.
    """
    if len(coords) == 0:
        return np.empty((0, 0), dtype=int)
    if len(coords) == 1:
        return np.arange(len(coords[0][0]), dtype=int).reshape(1, -1)

    # Compute groups anchored at every catalog and intersect them.
    per_anchor: list[set] = []
    for anchor in range(len(coords)):
        groups = _one_way_anchored_at(coords, anchor, radius_arcsec)
        per_anchor.append(
            {tuple(int(i) for i in groups[:, c]) for c in range(groups.shape[1])}
        )

    common = set.intersection(*per_anchor) if per_anchor else set()
    if not common:
        return np.empty((len(coords), 0), dtype=int)

    # Sort by the first catalog's index for determinism.
    rows = sorted(common, key=lambda t: t[0])
    return np.asarray(rows, dtype=int).T


def primary_collisions(
    coords: Sequence[CoordPair], radius_arcsec: float
) -> list[np.ndarray]:
    """Reference for ``detect_primary_collisions``.

    Returns one boolean mask per catalog. The mask for ``coords[0]`` is True
    where a row has any neighbour in any other catalog within radius. The
    mask for ``coords[i]`` (i >= 1) is True where a row has a neighbour in
    ``coords[0]`` within radius.
    """
    masks = [np.zeros(len(coords[0][0]), dtype=bool)]
    primary = _to_skycoord(*coords[0])
    for ra, dec in coords[1:]:
        if len(ra) == 0:
            masks.append(np.zeros(0, dtype=bool))
            continue
        other = _to_skycoord(ra, dec)
        # other -> primary: which "other" rows hit the primary?
        if len(coords[0][0]) == 0:
            masks.append(np.zeros(len(ra), dtype=bool))
            continue
        idx_to_primary, sep_to_primary, _ = other.match_to_catalog_sky(primary)
        within = np.asarray(sep_to_primary <= radius_arcsec * u.arcsec, dtype=bool)
        masks.append(within)
        # Mark hit primary rows.
        masks[0][np.asarray(idx_to_primary)[within]] = True
    return masks


def all_collisions(
    coords: Sequence[CoordPair], radius_arcsec: float
) -> list[np.ndarray]:
    """Reference for ``detect_collisions``.

    Returns one boolean mask per catalog. ``masks[i][r]`` is True iff
    ``coords[i][r]`` has any neighbour in any *other* catalog within radius.
    """
    n = len(coords)
    skycoords = [_to_skycoord(ra, dec) for ra, dec in coords]
    masks = [np.zeros(len(coords[i][0]), dtype=bool) for i in range(n)]
    for i in range(n):
        if len(coords[i][0]) == 0:
            continue
        for j in range(n):
            if i == j or len(coords[j][0]) == 0:
                continue
            _, sep, _ = skycoords[i].match_to_catalog_sky(skycoords[j])
            masks[i] |= np.asarray(sep <= radius_arcsec * u.arcsec, dtype=bool)
    return masks
