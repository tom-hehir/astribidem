"""Coordinates, deduplication and the candidate-edge build.

Adapted from Astral Projections. Everything here works in row positions: row
``i`` of a survey is the ``i``-th coordinate passed in.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from math import isfinite

import numpy as np

from astro_crossmatch.candidate_edges import CandidateEdges, DedupeOutcome, PairEdges
from astro_crossmatch.graph import (
    clique_mask,
    component_sizes_and_edge_counts,
    connected_components,
)
from astro_crossmatch.kernel import CatalogKernel, radec_to_xyz, resolve_workers


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
    """Build coordinates from RA/Dec degree arrays.

    Coordinates are converted to float64 before any calculation. Floating
    point inputs less precise than float64 trigger a ``UserWarning``, because
    their rounding already limits positional accuracy.
    """
    ra = np.asarray(ra)
    dec = np.asarray(dec)
    if ra.ndim != 1 or dec.ndim != 1:
        raise ValueError(f"survey {name!r}: ra/dec must be 1-D")
    if len(ra) != len(dec):
        raise ValueError(
            f"survey {name!r}: ra/dec shapes differ: {ra.shape} vs {dec.shape}"
        )
    return SurveyCoords(name=name, xyz=radec_to_xyz(ra, dec, name=name))


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
            name=coords.name,
            n_rows=n,
            dedupe_radius_arcsec=radius_arcsec,
            dropped_rows=empty,
            kept_rows=empty,
            disputed_rows=empty,
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
        name=coords.name,
        n_rows=n,
        dedupe_radius_arcsec=radius_arcsec,
        dropped_rows=np.flatnonzero(dropped),
        kept_rows=keeper[labels[dropped]],
        disputed_rows=np.flatnonzero(disputed),
    )
    return outcome, active


def pair_edge_chunks(
    kernel_a: CatalogKernel,
    kernel_b: CatalogKernel,
    active_a: np.ndarray,
    active_b: np.ndarray,
    radius_arcsec: float,
    *,
    workers: int = 1,
    chunk_rows: int | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Every in-radius pair of active rows, as ordered ``(row_a, row_b, sep)`` chunks.

    The kernels always emit the *complete* in-radius pair set: resolution
    assumes completeness (the entitywise mode needs all edges to detect
    disputes, ``mutual_unique`` needs candidate counts), so it must not be
    truncated. Kernels cover each survey's full row set (shared with dedupe);
    pairs touching dedupe-inactive rows are dropped here. ``chunk_rows`` sets
    how many of ``a``'s rows each chunk covers.
    """
    for i, j, separation in kernel_a.all_pairs_chunks(
        kernel_b, radius_arcsec, workers=workers, chunk_rows=chunk_rows
    ):
        keep = active_a[i] & active_b[j]
        yield i[keep], j[keep], separation[keep]


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
    """All candidate pairs between two surveys' active rows, held in memory."""
    if kernel_a is None:
        kernel_a = CatalogKernel.from_xyz(a.xyz)
    if kernel_b is None:
        kernel_b = CatalogKernel.from_xyz(b.xyz)
    chunks = list(
        pair_edge_chunks(
            kernel_a, kernel_b, active_a, active_b, radius_arcsec, workers=workers
        )
    )
    row_a, row_b, separation = (
        chunks[0]
        if len(chunks) == 1
        else tuple(np.concatenate([chunk[k] for chunk in chunks]) for k in range(3))
    )
    return PairEdges(
        survey_a=a.name,
        survey_b=b.name,
        radius_arcsec=radius_arcsec,
        row_a=row_a,
        row_b=row_b,
        separation_arcsec=separation,
    )


# --- the edge build ------------------------------------------------------------


@dataclass(frozen=True)
class _EdgeBuildPlan:
    """Validated edge-build settings, shared by in-memory and streamed builds."""

    surveys: tuple[SurveyCoords, ...]
    dedupe_radius_arcsec: dict[str, float]
    pair_radius_arcsec: dict[tuple[str, str], float]  # built pairs, survey order


def _validate_survey(survey: SurveyCoords) -> None:
    if not isinstance(survey.name, str) or not survey.name or "/" in survey.name:
        raise ValueError("survey names must be non-empty and must not contain '/'")
    xyz = np.asarray(survey.xyz)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"survey {survey.name!r}: xyz must have shape (n, 3)")
    if not np.isfinite(xyz).all():
        raise ValueError(f"survey {survey.name!r}: coordinates must be finite")


def _plan_edge_build(
    surveys: Sequence[SurveyCoords],
    radius_arcsec: float,
    dedupe_radius_arcsec: Mapping[str, float],
    pair_radius_overrides: Mapping[tuple[str, str] | frozenset[str], float] | None,
    pairs: Iterable[tuple[str, str]] | None,
) -> _EdgeBuildPlan:
    surveys = tuple(surveys)
    if not surveys:
        raise ValueError("at least one survey is required")
    for survey in surveys:
        _validate_survey(survey)
    names = tuple(survey.name for survey in surveys)
    known = set(names)
    if len(known) != len(names):
        raise ValueError("survey names must be unique")
    if set(dedupe_radius_arcsec) != known:
        raise ValueError("dedupe_radius_arcsec must name every survey exactly once")
    dedupe_radii = {name: float(dedupe_radius_arcsec[name]) for name in names}
    if any(not isfinite(value) or value < 0 for value in dedupe_radii.values()):
        raise ValueError("dedupe radii must be finite and >= 0")
    radius_arcsec = float(radius_arcsec)
    if not isfinite(radius_arcsec) or radius_arcsec <= 0:
        raise ValueError("radius_arcsec must be finite and > 0")

    radii = {frozenset(pair): radius_arcsec for pair in combinations(names, 2)}
    overridden: set[frozenset[str]] = set()
    for pair, radius in (pair_radius_overrides or {}).items():
        if isinstance(pair, str) or len(pair) != 2:
            raise ValueError("each radius override must name exactly two surveys")
        key = frozenset(pair)
        if len(key) != 2 or not key <= known:
            raise ValueError(f"radius override names invalid survey pair: {pair!r}")
        if key in overridden:
            raise ValueError(f"duplicate radius override for pair: {sorted(key)}")
        radius = float(radius)
        if not isfinite(radius) or radius <= 0:
            raise ValueError("override radii must be finite and > 0")
        overridden.add(key)
        radii[key] = radius

    selected = None
    if pairs is not None:
        selected = set()
        for pair in pairs:
            key = frozenset(pair)
            if (
                isinstance(pair, str)
                or len(pair) != 2
                or len(key) != 2
                or not key <= known
            ):
                raise ValueError(f"pairs names an invalid survey pair: {pair!r}")
            selected.add(key)
    return _EdgeBuildPlan(
        surveys=surveys,
        dedupe_radius_arcsec=dedupe_radii,
        pair_radius_arcsec={
            (a, b): radii[frozenset((a, b))]
            for a, b in combinations(names, 2)
            if selected is None or frozenset((a, b)) in selected
        },
    )


def _kernels_and_dedupe(
    plan: _EdgeBuildPlan,
) -> tuple[dict[str, CatalogKernel], tuple[DedupeOutcome, ...], dict[str, np.ndarray]]:
    """One kernel per survey, shared by its dedupe and every pair it joins."""
    kernels = {
        survey.name: CatalogKernel.from_xyz(survey.xyz) for survey in plan.surveys
    }
    outcomes = []
    active = {}
    for survey in plan.surveys:
        outcome, active[survey.name] = dedupe_survey(
            survey, plan.dedupe_radius_arcsec[survey.name], kernel=kernels[survey.name]
        )
        outcomes.append(outcome)
    return kernels, tuple(outcomes), active


def build_edges(
    surveys: Sequence[SurveyCoords],
    *,
    radius_arcsec: float,
    dedupe_radius_arcsec: Mapping[str, float],
    pair_radius_overrides: Mapping[tuple[str, str] | frozenset[str], float]
    | None = None,
    pairs: Iterable[tuple[str, str]] | None = None,
    workers: int = 1,
) -> CandidateEdges:
    """Dedupe every survey and find every in-radius pair between surveys.

    Every survey requires an explicit dedupe radius; ``0.0`` opts out.
    ``radius_arcsec`` is the default cross-survey radius. Optional pair
    overrides name exactly two surveys; the same unordered pair cannot be
    specified twice, even with equal radii. ``pairs`` restricts which survey
    pairs are joined (default: all); the result then serves only degenerate
    modes over those pairs. Coordinates must already be cleaned.
    """
    plan = _plan_edge_build(
        surveys, radius_arcsec, dedupe_radius_arcsec, pair_radius_overrides, pairs
    )
    workers = resolve_workers(workers)
    kernels, outcomes, active = _kernels_and_dedupe(plan)
    coords = {survey.name: survey for survey in plan.surveys}
    edges = {
        frozenset((a, b)): build_pair_edges(
            coords[a],
            coords[b],
            active[a],
            active[b],
            radius,
            kernel_a=kernels[a],
            kernel_b=kernels[b],
            workers=workers,
        )
        for (a, b), radius in plan.pair_radius_arcsec.items()
    }
    return CandidateEdges(surveys=outcomes, pairs=edges)
