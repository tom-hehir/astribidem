"""Storage-independent coordinate, deduplication, and complete-edge primitives.

Adapted from Astral Projections. Everything here works in row positions: row
``i`` of a survey is the ``i``-th coordinate passed in.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from astro_crossmatch.graph import (
    clique_mask,
    component_sizes_and_edge_counts,
    connected_components,
)
from astro_crossmatch.inputs import PairEdges
from astro_crossmatch.kernel import CatalogKernel, radec_to_xyz


@dataclass(frozen=True)
class SurveyCoords:
    """One survey's unit sky vectors; row ``i`` is ``xyz[i]``."""

    name: str
    xyz: np.ndarray

    def __len__(self) -> int:
        return len(self.xyz)


def survey_coords_from_arrays(
    name: str,
    ra: np.ndarray,
    dec: np.ndarray,
) -> SurveyCoords:
    """Build coordinates from RA/Dec degree arrays."""
    ra = np.asarray(ra, dtype=np.float64)
    dec = np.asarray(dec, dtype=np.float64)
    if ra.ndim != 1 or dec.ndim != 1:
        raise ValueError(f"survey {name!r}: ra/dec must be 1-D")
    if len(ra) != len(dec):
        raise ValueError(
            f"survey {name!r}: ra/dec shapes differ: {ra.shape} vs {dec.shape}"
        )
    return SurveyCoords(name=name, xyz=radec_to_xyz(ra, dec))


@dataclass(frozen=True)
class DedupeOutcome:
    """One survey's dedupe verdicts, in internal positions."""

    survey: str
    radius_arcsec: float
    dropped_index: np.ndarray
    keeper_index: np.ndarray
    disputed_index: np.ndarray

    @property
    def n_dropped(self) -> int:
        return len(self.dropped_index)

    @property
    def n_disputed(self) -> int:
        return len(self.disputed_index)


def dedupe_survey(
    coords: SurveyCoords,
    radius_arcsec: float,
    kernel: CatalogKernel | None = None,
) -> tuple[DedupeOutcome, np.ndarray]:
    """De-duplicate one survey; return the outcome and the active-row mask.

    Within-radius components that are cliques are duplicates by the physical
    argument (below the survey's resolution floor, two rows cannot be two
    resolved objects): keep the **lowest row**, drop the rest. The kept row
    therefore depends on input order; sort the inputs first when that order
    is not meaningful. Non-clique components (chains from shredding, flanking
    geometries) are disputed: every member is flagged and excluded from
    cross-survey matching.

    ``kernel`` is the survey's already-built ``CatalogKernel`` (``crossmatch``
    shares one per survey with pair matching); omitted, one is built here.
    """
    n = len(coords)
    active = np.ones(n, dtype=bool)
    empty = np.empty(0, dtype=np.int64)
    if radius_arcsec <= 0 or n == 0:
        outcome = DedupeOutcome(
            survey=coords.name,
            radius_arcsec=radius_arcsec,
            dropped_index=empty,
            keeper_index=empty,
            disputed_index=empty,
        )
        return outcome, active

    if kernel is None:
        kernel = CatalogKernel.from_xyz(coords.xyz)
    # The complete within-radius pair set: nearest-only truncation would hide
    # exactly the extra candidates that make components disputed.
    first, second, _ = kernel.self_pairs(radius_arcsec)

    labels = connected_components(n, first, second)
    sizes, edge_counts = component_sizes_and_edge_counts(labels, first)
    is_clique = clique_mask(sizes, edge_counts)

    multi = sizes[labels] > 1
    clique_member = is_clique[labels] & multi
    disputed = ~is_clique[labels] & multi

    # Keeper per component = its lowest row.
    keeper = np.full(len(sizes), n, dtype=np.int64)
    np.minimum.at(keeper, labels, np.arange(n, dtype=np.int64))
    dropped = clique_member & (np.arange(n) != keeper[labels])

    active[dropped] = False
    active[disputed] = False

    outcome = DedupeOutcome(
        survey=coords.name,
        radius_arcsec=radius_arcsec,
        dropped_index=np.flatnonzero(dropped),
        keeper_index=keeper[labels[dropped]],
        disputed_index=np.flatnonzero(disputed),
    )
    return outcome, active


def build_pair_edges(
    a: SurveyCoords,
    b: SurveyCoords,
    active_a: np.ndarray,
    active_b: np.ndarray,
    radius_arcsec: float,
    kernel_a: CatalogKernel | None = None,
    kernel_b: CatalogKernel | None = None,
    *,
    workers: int = 1,
) -> PairEdges:
    """All candidate pairs between two surveys' active rows.

    The kernels always emit the *complete* in-radius pair set: the edge
    substrate's consumers assume completeness (the entitywise mode needs all
    edges to detect disputes, ``mutual_unique`` needs candidate counts), so
    it must not be truncated — the kernel's k-truncated fast path is only
    sound when a build is aligned to a single known degenerate policy, never
    for a substrate shared across modes. Kernels cover each survey's full
    row set (shared with dedupe); edges touching dedupe-inactive rows are
    dropped here.
    """
    if kernel_a is None:
        kernel_a = CatalogKernel.from_xyz(a.xyz)
    if kernel_b is None:
        kernel_b = CatalogKernel.from_xyz(b.xyz)
    i, j, sep_arcsec = kernel_a.all_pairs(kernel_b, radius_arcsec, workers=workers)
    keep = active_a[i] & active_b[j]
    return PairEdges(
        survey_a=a.name,
        survey_b=b.name,
        radius_arcsec=radius_arcsec,
        row_a=i[keep],
        row_b=j[keep],
        sep_arcsec=sep_arcsec[keep],
    )
