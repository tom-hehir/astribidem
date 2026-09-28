"""Array-oriented matching adapters extracted from AstroBench.

The shared primitives own the matching semantics: the scipy cKDTree kernel
(:class:`~astribidem.kernel.CatalogKernel`),
the pairwise policy reductions (``one_to_one_pairs``), and the per-survey
geometric dedupe (``dedupe_survey``). This adapter exposes positional results
for callers working directly with in-memory arrays:

* the N-way *composition* of Astral's pairwise reductions over row positions
  (``anchor-pairs`` / ``all-pairs`` topology), preserved from the previous
  matching layer for in-memory group matching;
* collision *detection* — a separate masking operation, not a crossmatch
  mode. Prefer the entity-level index
  (:func:`~astribidem.crossmatch`)
  when "each physical object appears at most once" semantics are needed: the
  entity index resolves components explicitly instead of masking both sides
  of every collision.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Literal

import numpy as np

from astribidem.candidate_edges import DedupeOutcome
from astribidem.edges import dedupe_survey
from astribidem.kernel import (
    CatalogKernel,
    arcsec_to_chord,
    chord_to_arcsec,
    float64_coordinates,
    radec_to_xyz,
)
from astribidem.modes import one_to_one_pairs

MatchPolicy = Literal[
    "anchored_nearest",
    "mutual_nearest",
    "anchored_unique",
    "mutual_unique",
]
MatchingTopology = Literal["anchor-pairs", "all-pairs"]

MATCH_POLICIES: frozenset[str] = frozenset(
    {"anchored_nearest", "mutual_nearest", "anchored_unique", "mutual_unique"}
)
MATCHING_TOPOLOGIES: frozenset[str] = frozenset({"anchor-pairs", "all-pairs"})
ALL_PAIRS_POLICIES: frozenset[str] = frozenset({"mutual_nearest", "mutual_unique"})


def radius_arcsec_to_radius_cartesian(radius_arcsec: float) -> float:
    """Convert an angular radius in arcseconds to a unit-sphere chord length."""
    return float(arcsec_to_chord(radius_arcsec))


def radec_to_cartesian(
    ra: np.ndarray, dec: np.ndarray, *, degrees: bool = True
) -> np.ndarray:
    """Convert astronomical RA/Dec to Cartesian unit vectors.

    Args:
        ra: Right ascension (longitude around the celestial equator).
        dec: Declination (latitude; zero at the equator, positive northward).
        degrees: Whether ``ra``/``dec`` are in degrees (else radians).

    Returns:
        An ``(..., 3)`` array of unit vectors.
    """
    ra, dec = float64_coordinates(ra, dec)
    if not degrees:
        ra = np.rad2deg(ra)
        dec = np.rad2deg(dec)
    return radec_to_xyz(ra, dec)


def angular_separation_arcsec(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return angular separations (arcsec) between corresponding unit vectors."""
    chord = np.linalg.norm(np.asarray(a) - np.asarray(b), axis=-1)
    return chord_to_arcsec(chord)


def spatial_candidate_pairs(
    anchor_ra: np.ndarray,
    anchor_dec: np.ndarray,
    other_ra: np.ndarray,
    other_dec: np.ndarray,
    radius_arcsec: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Emit every in-radius pair between two coordinate arrays.

    This is the narrow public seam used by out-of-core readers: the caller
    decides how shards are scanned, while the exact
    :class:`CatalogKernel` remains the sole owner of pair generation.
    Candidate reduction is deliberately separate so candidates from every
    source shard can be resolved together rather than independently.
    """
    anchor = CatalogKernel(np.asarray(anchor_ra), np.asarray(anchor_dec))
    other = CatalogKernel(np.asarray(other_ra), np.asarray(other_dec))
    return anchor.all_pairs(other, radius_arcsec)


def resolve_spatial_candidate_pairs(
    anchor_rows: np.ndarray,
    other_rows: np.ndarray,
    separation_arcsec: np.ndarray,
    *,
    policy: MatchPolicy,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply an Astral one-to-one policy to a complete candidate relation."""
    return one_to_one_pairs(
        np.asarray(anchor_rows, dtype=np.int64),
        np.asarray(other_rows, dtype=np.int64),
        np.asarray(separation_arcsec, dtype=np.float64),
        policy,
    )


def _validate_topology(match_policy: str, matching_topology: str) -> None:
    if match_policy not in MATCH_POLICIES:
        raise ValueError(f"unknown match policy {match_policy!r}")
    if matching_topology not in MATCHING_TOPOLOGIES:
        raise ValueError(
            f"unknown matching_topology {matching_topology!r}; "
            f"expected one of {sorted(MATCHING_TOPOLOGIES)}"
        )
    if matching_topology == "all-pairs" and match_policy not in ALL_PAIRS_POLICIES:
        raise ValueError(
            "matching_topology='all-pairs' requires policy='mutual_nearest' "
            "or policy='mutual_unique'"
        )


def _pair_relation(
    kernel_a: CatalogKernel,
    kernel_b: CatalogKernel,
    radius_arcsec: float,
    match_policy: str,
) -> np.ndarray:
    """One pair's policy-reduced partner array: ``partner[row_a] = row_b | -1``.

    The kernel emits the complete in-radius candidate set; Astral's
    ``one_to_one_pairs`` applies the policy reduction.
    """
    row_a, row_b, sep = kernel_a.all_pairs(kernel_b, radius_arcsec)
    left, right, _ = one_to_one_pairs(row_a, row_b, sep, match_policy)
    partner = np.full(len(kernel_a), -1, dtype=np.int64)
    partner[left] = right
    return partner


def crossmatch_sky_vectors(
    coords: Sequence[np.ndarray],
    radius_arcsec: float,
    *,
    match_policy: MatchPolicy = "anchored_nearest",
    matching_topology: MatchingTopology = "anchor-pairs",
) -> np.ndarray:
    """Crossmatch N Cartesian sky-vector arrays with Astral's exact primitives.

    Args:
        coords: One ``(n_k, 3)`` unit-vector array per dataset. The first set
            is the anchor; matches are reported relative to it.
        radius_arcsec: Match radius in arcseconds (inclusive boundary).
        match_policy: One of Astral's four canonical spatial policies.
        matching_topology: ``anchor-pairs`` or ``all-pairs``.

    Returns:
        An ``(N, n_matches)`` int array; ``groups[k, m]`` is the row index into
        dataset ``k`` for the ``m``-th match.
    """
    if len(coords) < 2:
        raise ValueError("at least two catalogs are required")
    _validate_topology(match_policy, matching_topology)

    kernels = [CatalogKernel.from_xyz(np.asarray(c, dtype=np.float64)) for c in coords]
    n_catalogs = len(kernels)
    if matching_topology == "all-pairs":
        required_pairs = [
            (i, j) for i in range(n_catalogs - 1) for j in range(i + 1, n_catalogs)
        ]
    else:
        required_pairs = [(0, j) for j in range(1, n_catalogs)]

    relations = {
        (i, j): _pair_relation(kernels[i], kernels[j], radius_arcsec, match_policy)
        for i, j in required_pairs
    }

    n_anchor = len(kernels[0])
    group_indices = np.full((n_anchor, n_catalogs), -1, dtype=np.int64)
    group_indices[:, 0] = np.arange(n_anchor, dtype=np.int64)
    keep = np.ones(n_anchor, dtype=bool)
    for catalog_index in range(1, n_catalogs):
        partner = relations[(0, catalog_index)]
        group_indices[:, catalog_index] = partner
        keep &= partner >= 0

    if matching_topology == "all-pairs":
        # Every remaining pairwise relation must agree with the anchor-derived
        # group — the same all-pairs consistency the mode builders enforce.
        for left in range(1, n_catalogs - 1):
            for right in range(left + 1, n_catalogs):
                partner = relations[(left, right)]
                selected_left = group_indices[:, left]
                selected_right = group_indices[:, right]
                valid_left = selected_left >= 0
                agrees = np.zeros(n_anchor, dtype=bool)
                agrees[valid_left] = (
                    partner[selected_left[valid_left]] == selected_right[valid_left]
                )
                keep &= agrees

    return group_indices[keep].T


def crossmatch_radec(
    coords: Sequence[tuple[np.ndarray, np.ndarray]],
    radius_arcsec: float,
    *,
    match_policy: MatchPolicy = "anchored_nearest",
    matching_topology: MatchingTopology = "anchor-pairs",
) -> np.ndarray:
    """Crossmatch N datasets given as ``(ra, dec)`` arrays in degrees.

    See :func:`crossmatch_sky_vectors` for the policy and return semantics.
    """
    sky_vectors = tuple(radec_to_cartesian(ra, dec, degrees=True) for ra, dec in coords)
    return crossmatch_sky_vectors(
        sky_vectors,
        radius_arcsec=radius_arcsec,
        match_policy=match_policy,
        matching_topology=matching_topology,
    )


def get_match_masks(lengths: Iterable[int], groups: np.ndarray) -> list[np.ndarray]:
    """Return a per-dataset boolean keep-mask from a crossmatch index array."""
    masks = []
    for length, idxs in zip(lengths, groups):
        mask = np.zeros(length, dtype=bool)
        mask[idxs] = True
        masks.append(mask)
    return masks


# --------------------------------------------------------------------------------------
# Within-dataset dedupe (Astral's per-survey dedupe stage)
# --------------------------------------------------------------------------------------


def dedupe_radec(
    ra: np.ndarray,
    dec: np.ndarray,
    radius_arcsec: float,
    *,
    name: str = "dataset",
) -> tuple[DedupeOutcome, np.ndarray]:
    """De-duplicate one dataset's rows below its resolution floor.

    Runs Astral's per-survey dedupe: within-radius connected components that
    are cliques keep exactly one row (the lowest row); non-clique
    components are *disputed* — every member is flagged and excluded.

    Args:
        ra: Right ascension in degrees.
        dec: Declination in degrees.
        radius_arcsec: The dataset's resolution floor; ``0.0`` opts out
            (everything stays active).
        name: Dataset name, carried into the outcome for reporting.

    Returns:
        The :class:`DedupeOutcome` (dropped / kept / disputed rows)
        and the boolean active-row mask (``True`` = row survives).
    """
    ra, dec = float64_coordinates(ra, dec, name=name)
    if ra.ndim != 1 or dec.ndim != 1 or ra.shape != dec.shape:
        raise ValueError(f"survey {name!r}: ra/dec must be aligned 1-D arrays")
    if radius_arcsec <= 0 or len(ra) == 0:
        empty = np.empty(0, dtype=np.int64)
        return DedupeOutcome(
            name=name,
            n_rows=len(ra),
            dedupe_radius_arcsec=radius_arcsec,
            dropped_rows=empty,
            kept_rows=empty,
            disputed_rows=empty,
        ), np.ones(len(ra), dtype=bool)
    return dedupe_survey(name, radius_arcsec, CatalogKernel(ra, dec, name=name))


def dedupe_and_crossmatch_radec(
    coords: Sequence[tuple[np.ndarray, np.ndarray]],
    radius_arcsec: float,
    *,
    names: Sequence[str],
    dedupe_radii_arcsec: Sequence[float],
    match_policy: MatchPolicy = "anchored_nearest",
    matching_topology: MatchingTopology = "anchor-pairs",
) -> tuple[np.ndarray, list[DedupeOutcome]]:
    """Dedupe each dataset, crossmatch the survivors, report original indices.

    The dedupe stage runs Astral's per-survey dedupe on each dataset (a
    ``0.0`` radius opts that dataset out), then the active rows are
    crossmatched. Group indices are mapped back to *original* row positions,
    so callers index their unfiltered tables directly.

    Returns:
        The ``(N, n_matches)`` group array in original row indices, plus one
        :class:`DedupeOutcome` per dataset (in input order) for provenance.
    """
    if len(coords) != len(names) or len(coords) != len(dedupe_radii_arcsec):
        raise ValueError("coords, names, and dedupe_radii_arcsec must align")
    outcomes: list[DedupeOutcome] = []
    active_indices: list[np.ndarray] = []
    filtered: list[tuple[np.ndarray, np.ndarray]] = []
    for (ra, dec), name, dedupe_radius in zip(coords, names, dedupe_radii_arcsec):
        ra, dec = float64_coordinates(ra, dec, name=name)
        outcome, active = dedupe_radec(ra, dec, dedupe_radius, name=name)
        outcomes.append(outcome)
        active_indices.append(np.flatnonzero(active))
        filtered.append((ra[active], dec[active]))
    groups = crossmatch_radec(
        filtered,
        radius_arcsec=radius_arcsec,
        match_policy=match_policy,
        matching_topology=matching_topology,
    )
    original = np.stack([index[rows] for index, rows in zip(active_indices, groups)])
    return original, outcomes


# --------------------------------------------------------------------------------------
# Collision detection (per-dataset exclusion masks)
# --------------------------------------------------------------------------------------
#
# Distinct from matching: these build no match groups. They answer "which rows of
# each dataset fall within ``radius_arcsec`` of a row in another dataset" and return
# a per-dataset boolean mask (``True`` = collides). Used to dedup overlapping rows
# when concatenating several datasets into one mixture. For "each physical object
# appears at most once" semantics prefer the entity-level index
# (``build_entity_index``), which resolves cross-dataset components explicitly
# instead of dropping both sides of every collision.


def detect_collisions(
    coords: Sequence[tuple[np.ndarray, np.ndarray]], radius_arcsec: float
) -> list[np.ndarray]:
    """Per-dataset boolean collision masks across *all* dataset pairs.

    Args:
        coords: One ``(ra, dec)`` array pair (degrees) per dataset.
        radius_arcsec: Collision radius in arcseconds.

    Returns:
        One boolean array per dataset (length = that dataset's row count);
        ``True`` marks a row within ``radius_arcsec`` of a row in *any* other
        dataset. Both rows of a colliding pair are flagged.
    """
    kernels = [
        CatalogKernel.from_xyz(radec_to_cartesian(ra, dec, degrees=True))
        for ra, dec in coords
    ]
    masks = [np.zeros(len(k), dtype=bool) for k in kernels]
    for i in range(len(kernels)):
        for j in range(i + 1, len(kernels)):
            row_i, row_j, _ = kernels[i].all_pairs(kernels[j], radius_arcsec)
            masks[i][row_i] = True
            masks[j][row_j] = True
    return masks


def detect_primary_collisions(
    coords: Sequence[tuple[np.ndarray, np.ndarray]], radius_arcsec: float
) -> list[np.ndarray]:
    """Per-dataset boolean collision masks against the *primary* (first) dataset.

    Like :func:`detect_collisions`, but only collisions with the first dataset
    are flagged: the primary's mask marks its rows hit by any other dataset, and
    each other dataset's mask marks its rows that hit the primary.
    """
    kernels = [
        CatalogKernel.from_xyz(radec_to_cartesian(ra, dec, degrees=True))
        for ra, dec in coords
    ]
    masks = [np.zeros(len(kernels[0]), dtype=bool)]
    for i in range(1, len(kernels)):
        row_0, row_i, _ = kernels[0].all_pairs(kernels[i], radius_arcsec)
        mask_i = np.zeros(len(kernels[i]), dtype=bool)
        mask_i[row_i] = True
        masks.append(mask_i)
        masks[0][row_0] = True
    return masks
