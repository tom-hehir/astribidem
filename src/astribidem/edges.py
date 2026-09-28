"""Coordinates, deduplication and the candidate-edge build.

Adapted from Astral Projections. Everything here works in row positions: row
``i`` of a survey is the ``i``-th coordinate passed in.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from numpy.typing import ArrayLike

from astribidem._radii import dedupe_radii
from astribidem._radii import radius as validate_radius
from astribidem.candidate_edges import CandidateEdges, DedupeOutcome, PairEdges
from astribidem.graph import (
    clique_mask,
    component_sizes_and_edge_counts,
    connected_components,
)
from astribidem.kernel import (
    CatalogKernel,
    resolve_workers,
)


def dedupe_survey(
    name: str,
    radius_arcsec: float,
    kernel: CatalogKernel,
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
    shares one per survey with pair matching).
    """
    n = len(kernel)
    active = np.ones(n, dtype=bool)
    empty = np.empty(0, dtype=np.int64)
    if radius_arcsec <= 0 or n == 0:
        outcome = DedupeOutcome(
            name=name,
            n_rows=n,
            dedupe_radius_arcsec=radius_arcsec,
            dropped_rows=empty,
            kept_rows=empty,
            disputed_rows=empty,
        )
        return outcome, active

    # The complete within-radius pair set: nearest-only truncation would hide
    # exactly the extra candidates that make components disputed.
    first, second, _ = kernel.self_pairs(radius_arcsec)
    dropped, kept, disputed = dedupe_from_pairs(
        np.arange(n, dtype=np.int64), first, second
    )
    active[dropped] = False
    active[disputed] = False
    outcome = DedupeOutcome(
        name=name,
        n_rows=n,
        dedupe_radius_arcsec=radius_arcsec,
        dropped_rows=dropped,
        kept_rows=kept,
        disputed_rows=disputed,
    )
    return outcome, active


def dedupe_from_pairs(
    rows: np.ndarray, first: np.ndarray, second: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dedupe verdicts from one survey's within-radius pairs, without coordinates.

    ``rows`` holds the global row number of each of ``n`` nodes, in any order;
    ``first`` / ``second`` index those nodes, one entry per pair. Groups of
    rows joined by pairs that form cliques keep their lowest row and drop the
    rest; other groups of two or more rows are disputed. Returns global rows:
    ``dropped`` (ascending), the ``kept`` row of each dropped row, and
    ``disputed`` (ascending).
    """
    rows = np.asarray(rows, dtype=np.int64)
    n = len(rows)
    if n == 0 or len(first) == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, empty
    labels = connected_components(n, first, second)
    sizes, edge_counts = component_sizes_and_edge_counts(labels, first)
    is_clique = clique_mask(sizes, edge_counts)
    multi = sizes[labels] > 1
    disputed = ~is_clique[labels] & multi
    keeper = np.full(len(sizes), np.iinfo(np.int64).max, dtype=np.int64)
    np.minimum.at(keeper, labels, rows)
    dropped = is_clique[labels] & multi & (rows != keeper[labels])
    order = np.argsort(rows[dropped], kind="stable")
    return (
        rows[dropped][order],
        keeper[labels[dropped]][order],
        np.sort(rows[disputed]),
    )


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
    a: str,
    b: str,
    active_a: np.ndarray,
    active_b: np.ndarray,
    radius_arcsec: float,
    *,
    kernel_a: CatalogKernel,
    kernel_b: CatalogKernel,
    workers: int = 1,
) -> PairEdges:
    """All candidate pairs between two surveys' active rows, held in memory."""
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
        survey_a=a,
        survey_b=b,
        radius_arcsec=radius_arcsec,
        row_a=row_a,
        row_b=row_b,
        separation_arcsec=separation,
    )


# --- the edge build ------------------------------------------------------------


@dataclass(frozen=True)
class EdgeSettings:
    """Validated matching settings for an edge build, independent of coordinates.

    ``pair_radius_arcsec`` maps each built pair, in survey order, to its radius.
    """

    survey_names: tuple[str, ...]
    n_rows: dict[str, int]
    dedupe_radius_arcsec: dict[str, float]
    pair_radius_arcsec: dict[tuple[str, str], float]

    @property
    def max_radius_arcsec(self) -> float:
        """The largest pair or dedupe radius: the margin a region needs."""
        radii = [*self.pair_radius_arcsec.values(), *self.dedupe_radius_arcsec.values()]
        return max(radii, default=0.0)


def edge_settings(
    n_rows: Mapping[str, int],
    *,
    radius_arcsec: float,
    dedupe_radius_arcsec: float,
    dedupe_radius_arcsec_overrides: Mapping[str, float] | None = None,
    radius_arcsec_overrides: Mapping[tuple[str, str] | frozenset[str], float]
    | None = None,
    pairs: Iterable[tuple[str, str]] | None = None,
) -> EdgeSettings:
    """Validate matching settings for surveys with the given row counts.

    ``n_rows`` maps each survey name, in survey order, to its row count. The
    remaining arguments are those of ``build_edges``.
    """
    names = tuple(n_rows)
    if not names:
        raise ValueError("at least one survey is required")
    for name in names:
        if not isinstance(name, str) or not name or "/" in name:
            raise ValueError("survey names must be non-empty and must not contain '/'")
    known = set(names)
    expanded_dedupe = dedupe_radii(
        names, dedupe_radius_arcsec, dedupe_radius_arcsec_overrides
    )
    radius_arcsec = validate_radius(radius_arcsec, name="radius_arcsec")
    if radius_arcsec_overrides is not None and not isinstance(
        radius_arcsec_overrides, Mapping
    ):
        raise TypeError("radius_arcsec_overrides must be a mapping")

    radii = {frozenset(pair): radius_arcsec for pair in combinations(names, 2)}
    overridden: set[frozenset[str]] = set()
    for pair, radius in (radius_arcsec_overrides or {}).items():
        if isinstance(pair, str) or len(pair) != 2:
            raise ValueError("each radius override must name exactly two surveys")
        key = frozenset(pair)
        if len(key) != 2 or not key <= known:
            raise ValueError(f"radius override names invalid survey pair: {pair!r}")
        if key in overridden:
            raise ValueError(f"duplicate radius override for pair: {sorted(key)}")
        radius = validate_radius(radius, name="radius override")
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
    return EdgeSettings(
        survey_names=names,
        n_rows={name: int(n_rows[name]) for name in names},
        dedupe_radius_arcsec=expanded_dedupe,
        pair_radius_arcsec={
            (a, b): radii[frozenset((a, b))]
            for a, b in combinations(names, 2)
            if selected is None or frozenset((a, b)) in selected
        },
    )


def _prepare_edge_build(
    surveys: Mapping[str, tuple[ArrayLike, ArrayLike]],
    *,
    radius_arcsec: float,
    dedupe_radius_arcsec: float,
    dedupe_radius_arcsec_overrides: Mapping[str, float] | None = None,
    radius_arcsec_overrides: Mapping[tuple[str, str] | frozenset[str], float]
    | None = None,
    pairs: Iterable[tuple[str, str]] | None = None,
) -> tuple[EdgeSettings, dict[str, CatalogKernel]]:
    """Prepare kernels once; retain no RA/Dec references in the returned state."""
    if not isinstance(surveys, Mapping) or not surveys:
        raise ValueError("surveys must map at least one name to an (ra, dec) tuple")
    n_rows = {}
    for name, coordinates in surveys.items():
        if not isinstance(coordinates, tuple) or len(coordinates) != 2:
            raise ValueError(f"survey {name!r}: coordinates must be an (ra, dec) tuple")
        ra_shape, dec_shape = (np.shape(values) for values in coordinates)
        if len(ra_shape) != 1 or len(dec_shape) != 1:
            raise ValueError(f"survey {name!r}: ra/dec must be 1-D")
        if ra_shape != dec_shape:
            raise ValueError(f"survey {name!r}: ra/dec shapes differ")
        n_rows[name] = ra_shape[0]
    settings = edge_settings(
        n_rows,
        radius_arcsec=radius_arcsec,
        dedupe_radius_arcsec=dedupe_radius_arcsec,
        dedupe_radius_arcsec_overrides=dedupe_radius_arcsec_overrides,
        radius_arcsec_overrides=radius_arcsec_overrides,
        pairs=pairs,
    )
    kernels = {
        name: CatalogKernel(ra, dec, name=name) for name, (ra, dec) in surveys.items()
    }
    return settings, kernels


def _dedupe_kernels(
    settings: EdgeSettings, kernels: Mapping[str, CatalogKernel]
) -> tuple[tuple[DedupeOutcome, ...], dict[str, np.ndarray]]:
    """Deduplicate using the same kernels that serve cross-survey matching."""
    outcomes = []
    active = {}
    for name in settings.survey_names:
        outcome, active[name] = dedupe_survey(
            name, settings.dedupe_radius_arcsec[name], kernels[name]
        )
        outcomes.append(outcome)
    return tuple(outcomes), active


def _build_prepared_edges(
    settings: EdgeSettings, kernels: Mapping[str, CatalogKernel], *, workers: int
) -> CandidateEdges:
    outcomes, active = _dedupe_kernels(settings, kernels)
    edges = {
        frozenset((a, b)): build_pair_edges(
            a,
            b,
            active[a],
            active[b],
            radius,
            kernel_a=kernels[a],
            kernel_b=kernels[b],
            workers=workers,
        )
        for (a, b), radius in settings.pair_radius_arcsec.items()
    }
    return CandidateEdges(surveys=outcomes, pairs=edges)


def build_edges(
    surveys: Mapping[str, tuple[ArrayLike, ArrayLike]],
    *,
    radius_arcsec: float,
    dedupe_radius_arcsec: float,
    dedupe_radius_arcsec_overrides: Mapping[str, float] | None = None,
    radius_arcsec_overrides: Mapping[tuple[str, str] | frozenset[str], float]
    | None = None,
    pairs: Iterable[tuple[str, str]] | None = None,
    workers: int = 1,
) -> CandidateEdges:
    """Dedupe every survey and find every in-radius pair between surveys.

    ``surveys`` maps names to aligned one-dimensional ``(ra, dec)`` arrays in
    degrees. Mapping order is survey order; array order defines row positions.
    Each survey is converted to XYZ once when its matching kernel is built.

    Both radius arguments are required scalars. ``dedupe_radius_arcsec=0.0``
    disables deduplication. ``dedupe_radius_arcsec_overrides`` supplies named
    survey exceptions; ``radius_arcsec_overrides`` supplies survey-pair
    exceptions. Unknown names and duplicate unordered pair overrides fail.
    ``pairs`` restricts the pairs built (default: all), for degenerate modes
    over those pairs. Coordinates must already be cleaned.
    """
    workers = resolve_workers(workers)
    settings, kernels = _prepare_edge_build(
        surveys,
        radius_arcsec=radius_arcsec,
        dedupe_radius_arcsec=dedupe_radius_arcsec,
        dedupe_radius_arcsec_overrides=dedupe_radius_arcsec_overrides,
        radius_arcsec_overrides=radius_arcsec_overrides,
        pairs=pairs,
    )
    del surveys
    return _build_prepared_edges(settings, kernels, workers=workers)
