"""Coordinates, deduplication and the candidate-edge build.

Adapted from astral-projections. Everything here works in row positions: row
``i`` of a survey is the ``i``-th coordinate passed in.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from numpy.typing import ArrayLike

from astribidem.candidate_edges import CandidateEdges, DedupeOutcome, PairEdges
from astribidem.graph import (
    clique_mask,
    component_sizes_and_edge_counts,
    connected_components,
)
from astribidem.index_table import check_survey_names
from astribidem.sky_position_tree import (
    SkyPositionTree,
    check_match_radius,
    resolve_workers,
)


def dedupe_survey(
    name: str,
    radius_arcsec: float,
    tree: SkyPositionTree,
) -> tuple[DedupeOutcome, np.ndarray]:
    """De-duplicate one survey; return the outcome and the active-row mask.

    Within-radius components that are cliques are duplicates by the physical
    argument (below the survey's resolution floor, two rows cannot be two
    resolved objects): keep the lowest row, drop the rest. The kept row
    therefore depends on input order; sort the inputs first when that order
    is not meaningful. Non-clique components (chains from shredding, flanking
    geometries) are disputed: every member is flagged and excluded from
    cross-survey matching.

    ``tree`` is the survey's already-built ``SkyPositionTree`` (``crossmatch``
    shares one per survey with pair matching).
    """
    n = len(tree)
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
    first, second, _ = tree.self_pairs(radius_arcsec)
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
    tree_a: SkyPositionTree,
    tree_b: SkyPositionTree,
    active_a: np.ndarray,
    active_b: np.ndarray,
    radius_arcsec: float,
    *,
    workers: int = 1,
    chunk_rows: int | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Every in-radius pair of active rows, as ordered ``(row_a, row_b, sep)`` chunks.

    The trees always emit the *complete* in-radius pair set: resolution
    assumes completeness (the entitywise mode needs all edges to detect
    disputes, ``mutual_unique`` needs candidate counts), so it must not be
    truncated. Trees cover each survey's full row set (shared with dedupe);
    pairs touching dedupe-inactive rows are dropped here. ``chunk_rows`` sets
    how many of ``a``'s rows each chunk covers.
    """
    for i, j, separation in tree_a.all_pairs_chunks(
        tree_b, radius_arcsec, workers=workers, chunk_rows=chunk_rows
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
    tree_a: SkyPositionTree,
    tree_b: SkyPositionTree,
    workers: int = 1,
) -> PairEdges:
    """All candidate pairs between two surveys' active rows, held in memory."""
    chunks = list(
        pair_edge_chunks(
            tree_a, tree_b, active_a, active_b, radius_arcsec, workers=workers
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


def matching_radii(
    names: Iterable[str],
    *,
    radius_arcsec: float,
    dedupe_radius_arcsec: float,
    dedupe_radius_arcsec_overrides: Mapping[str, float] | None = None,
    radius_arcsec_overrides: Mapping[tuple[str, str] | frozenset[str], float]
    | None = None,
    pairs: Iterable[tuple[str, str]] | None = None,
) -> tuple[dict[str, float], dict[tuple[str, str], float]]:
    """Validate matching settings: each survey's dedupe radius and each pair's radius.

    ``names`` lists the surveys in survey order. The pair radii hold only the
    built pairs, in survey order, so pairs sharing a first survey are adjacent.
    The remaining arguments are those of ``build_edges``.
    """
    names = tuple(names)
    if not names:
        raise ValueError("at least one survey is required")
    check_survey_names(names)
    radii = radius_by_pair(names, radius_arcsec, radius_arcsec_overrides)
    selected = None
    if pairs is not None:
        selected = set()
        for pair in pairs:
            key = frozenset(pair)
            # A restriction naming an unknown pair would silently build nothing.
            if isinstance(pair, str) or key not in radii:
                raise ValueError(f"pairs names an invalid survey pair: {pair!r}")
            selected.add(key)
    dedupe_radii = dedupe_radius_by_survey(
        names, dedupe_radius_arcsec, dedupe_radius_arcsec_overrides
    )
    pair_radii = {
        (a, b): radii[frozenset((a, b))]
        for a, b in combinations(names, 2)
        if selected is None or frozenset((a, b)) in selected
    }
    return dedupe_radii, pair_radii


def edge_settings(n_rows: Mapping[str, int], **matching_settings) -> EdgeSettings:
    """Validate matching settings for surveys with the given row counts.

    ``n_rows`` maps each survey name, in survey order, to its row count;
    ``matching_settings`` are the keyword arguments of ``matching_radii``.
    """
    dedupe_radii, pair_radii = matching_radii(n_rows, **matching_settings)
    return EdgeSettings(
        survey_names=tuple(n_rows),
        n_rows={name: int(n) for name, n in n_rows.items()},
        dedupe_radius_arcsec=dedupe_radii,
        pair_radius_arcsec=pair_radii,
    )


def dedupe_radius_by_survey(
    names: Iterable[str],
    default: float,
    overrides: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Each survey's dedupe radius: ``overrides[name]`` if given, else ``default``.

    Raises:
        ValueError: ``overrides`` names a survey not in ``names``; a typo would
            otherwise leave that survey at the default without notice.
    """
    names = tuple(names)
    overrides = overrides or {}
    unknown = set(overrides) - set(names)
    if unknown:
        raise ValueError(
            f"dedupe_radius_arcsec_overrides names unknown surveys: {sorted(unknown)}"
        )
    return {name: float(overrides.get(name, default)) for name in names}


def radius_by_pair(
    names: Iterable[str],
    default: float,
    overrides: Mapping[tuple[str, str] | frozenset[str], float] | None = None,
) -> dict[frozenset[str], float]:
    """Each survey pair's matching radius: its entry in ``overrides``, else ``default``.

    Raises:
        ValueError: a radius is not a positive finite number (a non-positive
            one would silently match nothing, or everything as if positive), an
            override names something other than two of ``names``, or names one
            pair twice (for example both ``("a", "b")`` and ``("b", "a")``),
            which would make its radius ambiguous.
    """
    radii = {frozenset(pair): float(default) for pair in combinations(names, 2)}
    overridden: set[frozenset[str]] = set()
    for pair, radius in (overrides or {}).items():
        key = frozenset(pair)
        if isinstance(pair, str) or key not in radii:
            raise ValueError(
                f"radius_arcsec_overrides names an invalid survey pair: {pair!r}"
            )
        if key in overridden:
            raise ValueError(
                f"radius_arcsec_overrides names the pair {sorted(key)} twice"
            )
        overridden.add(key)
        radii[key] = float(radius)
    for key, radius in radii.items():
        check_match_radius(radius, f"matching radius for {sorted(key)}")
    return radii


def dedupe_trees(
    dedupe_radius_arcsec: Mapping[str, float], trees: Mapping[str, SkyPositionTree]
) -> tuple[tuple[DedupeOutcome, ...], dict[str, np.ndarray]]:
    """Dedupe each survey, in order, reading each tree once.

    Returns the outcomes and each survey's active-row mask. The trees are the
    same ones that serve cross-survey matching.
    """
    outcomes = []
    active = {}
    for name, radius in dedupe_radius_arcsec.items():
        outcome, active[name] = dedupe_survey(name, radius, trees[name])
        outcomes.append(outcome)
    return tuple(outcomes), active


def pairs_with_trees(
    pair_radius_arcsec: Mapping[tuple[str, str], float],
    trees: Mapping[str, SkyPositionTree],
) -> Iterator[tuple[str, str, float, SkyPositionTree, SkyPositionTree]]:
    """Each pair with its radius and two trees, reading trees as few times as possible.

    Pairs come in the given order, which groups them by first survey. Each
    first survey's tree is read once per group and each second survey's tree
    once per pair, and a tree is released before the next one is read. A
    caller that drops its own references before taking the next pair
    therefore holds at most two trees at a time.
    """
    group = tree_a = None
    for (a, b), radius in pair_radius_arcsec.items():
        if a != group:
            tree_a = None
            group, tree_a = a, trees[a]
        tree_b = trees[b]
        yield a, b, radius, tree_a, tree_b
        del tree_b


def build_edges_from_trees(
    trees: Mapping[str, SkyPositionTree],
    *,
    radius_arcsec: float,
    dedupe_radius_arcsec: float,
    dedupe_radius_arcsec_overrides: Mapping[str, float] | None = None,
    radius_arcsec_overrides: Mapping[tuple[str, str] | frozenset[str], float]
    | None = None,
    pairs: Iterable[tuple[str, str]] | None = None,
    workers: int = 1,
) -> CandidateEdges:
    """``build_edges`` from trees already built, or loaded with ``SkyPositionTree.load``.

    ``trees`` maps survey names, in survey order, to their trees. The other
    arguments are those of ``build_edges``, and the result is the same as
    ``build_edges`` over the coordinates the trees were built from. The trees
    are read in a fixed order, so a mapping that loads each tree when it is
    read holds few at a time: each tree once for dedupe, then the pairs grouped
    by first survey, with at most two trees held (see ``pairs_with_trees``).
    """
    workers = resolve_workers(workers)
    dedupe_radii, pair_radii = matching_radii(
        trees,
        radius_arcsec=radius_arcsec,
        dedupe_radius_arcsec=dedupe_radius_arcsec,
        dedupe_radius_arcsec_overrides=dedupe_radius_arcsec_overrides,
        radius_arcsec_overrides=radius_arcsec_overrides,
        pairs=pairs,
    )
    outcomes, active = dedupe_trees(dedupe_radii, trees)
    edges = {}
    for a, b, radius, tree_a, tree_b in pairs_with_trees(pair_radii, trees):
        edges[frozenset((a, b))] = build_pair_edges(
            a,
            b,
            active[a],
            active[b],
            radius,
            tree_a=tree_a,
            tree_b=tree_b,
            workers=workers,
        )
        del tree_a, tree_b
    return CandidateEdges(surveys=outcomes, pairs=edges)


def build_edges(
    coordinates: Mapping[str, tuple[ArrayLike, ArrayLike]],
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

    ``coordinates`` maps survey names to aligned one-dimensional ``(ra, dec)`` arrays in
    degrees. Mapping order is survey order; array order defines row positions.
    Each survey is converted to XYZ once when its tree is built; to build or
    load the trees yourself, use ``build_edges_from_trees``.

    Both radius arguments are required scalars. ``dedupe_radius_arcsec=0.0``
    disables deduplication. ``dedupe_radius_arcsec_overrides`` supplies named
    survey exceptions; ``radius_arcsec_overrides`` supplies survey-pair
    exceptions. Unknown names and duplicate unordered pair overrides fail.
    ``pairs`` restricts the pairs built (default: all), for modes over those
    pairs. Coordinates must already be cleaned.
    """
    trees = sky_position_trees(coordinates)
    del coordinates
    return build_edges_from_trees(
        trees,
        radius_arcsec=radius_arcsec,
        dedupe_radius_arcsec=dedupe_radius_arcsec,
        dedupe_radius_arcsec_overrides=dedupe_radius_arcsec_overrides,
        radius_arcsec_overrides=radius_arcsec_overrides,
        pairs=pairs,
        workers=workers,
    )


def sky_position_trees(
    coordinates: Mapping[str, tuple[ArrayLike, ArrayLike]],
) -> dict[str, SkyPositionTree]:
    """Each survey's tree, in survey order; the trees keep no RA/Dec references."""
    return {
        name: SkyPositionTree(ra, dec, name=name)
        for name, (ra, dec) in coordinates.items()
    }
