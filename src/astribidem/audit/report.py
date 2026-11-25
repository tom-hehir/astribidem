"""Diagnostics for candidate edges, computed from a ``CandidateEdges`` object.

The audit works the same on edges held in memory, reloaded with
``read_edges``, or streamed to disk and reloaded. It reports per-survey dedupe
outcomes, per-pair edge counts and candidate multiplicity, and a census of the
connected components over every survey: sizes, the largest (a percolation
guardrail), and clean-component counts by survey combination. A clean
component spanning surveys {A, B} is exactly one row of a clean A x B
product, so the census sizes subset joins before any are built.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from astribidem.candidate_edges import CandidateEdges, DedupeOutcome, PairEdges
from astribidem.graph import build_global_graph

SIZE_HISTOGRAM_CAP = 6


@dataclass(frozen=True)
class SurveyAudit:
    """One survey's rows and the outcome of its dedupe.

    A duplicate group is a kept row plus the rows dropped in its favour.
    """

    name: str
    n_rows: int
    dedupe_radius_arcsec: float
    n_active: int
    n_dropped: int
    n_disputed: int
    n_duplicate_groups: int
    largest_duplicate_group: int


@dataclass(frozen=True)
class SeparationPercentiles:
    """Percentiles of a pair's candidate separations, in arcseconds."""

    p5: float
    p25: float
    p50: float
    p75: float
    p95: float


@dataclass(frozen=True)
class PairAudit:
    """One survey pair's candidate edges.

    ``matched_fraction_*`` is the fraction of a survey's active rows with at
    least one candidate; ``multi_candidate_rows_*`` counts its rows with more
    than one. ``separation_arcsec`` is ``None`` when the pair has no edges.
    """

    survey_a: str
    survey_b: str
    radius_arcsec: float
    n_edges: int
    separation_arcsec: SeparationPercentiles | None
    matched_fraction_a: float
    matched_fraction_b: float
    multi_candidate_rows_a: int
    multi_candidate_rows_b: int


@dataclass(frozen=True)
class CombinationCount:
    """Components of two or more rows whose rows come from exactly ``surveys``."""

    surveys: tuple[str, ...]
    n_components: int
    n_clean: int


@dataclass(frozen=True)
class ComponentCensus:
    """Connected components over the active rows of every survey.

    ``size_counts[k]`` counts components of ``k + 1`` rows; the last entry
    counts every component of at least ``SIZE_HISTOGRAM_CAP`` rows. A clean
    component is a clique with one row per survey; an ambiguous one has two or
    more rows and is not clean.
    """

    n_rows: int
    n_components: int
    size_counts: tuple[int, ...]
    max_component_size: int
    n_clean_multi_survey: int
    n_ambiguous: int
    rows_in_ambiguous_components: int
    by_survey_combination: tuple[CombinationCount, ...]


@dataclass(frozen=True)
class EdgeAudit:
    """The audit of one set of candidate edges.

    ``all_pairs_built`` is false when the build was restricted to some pairs;
    the component census then omits the missing pairs' edges.
    """

    all_pairs_built: bool
    surveys: tuple[SurveyAudit, ...]
    pairs: tuple[PairAudit, ...]
    components: ComponentCensus


def _percentiles(values: np.ndarray) -> SeparationPercentiles | None:
    if len(values) == 0:
        return None
    points = np.percentile(values, (5, 25, 50, 75, 95))
    return SeparationPercentiles(*(float(point) for point in points))


def _size_counts(sizes: np.ndarray) -> tuple[int, ...]:
    capped = np.minimum(sizes, SIZE_HISTOGRAM_CAP)
    return tuple(
        int(count)
        for count in np.bincount(capped, minlength=SIZE_HISTOGRAM_CAP + 1)[1:]
    )


def _survey_audit(survey: DedupeOutcome) -> SurveyAudit:
    # Each distinct kept row stands for one duplicate group; the group also
    # contains every dropped row that points at it.
    _, dropped_per_group = np.unique(survey.kept_rows, return_counts=True)
    return SurveyAudit(
        name=survey.name,
        n_rows=survey.n_rows,
        dedupe_radius_arcsec=survey.dedupe_radius_arcsec,
        n_active=survey.n_active,
        n_dropped=survey.n_dropped,
        n_disputed=survey.n_disputed,
        n_duplicate_groups=len(dropped_per_group),
        largest_duplicate_group=int(dropped_per_group.max()) + 1
        if len(dropped_per_group)
        else 0,
    )


def _pair_audit(edges: CandidateEdges, pair: PairEdges) -> PairAudit:
    rows_a, counts_a = np.unique(pair.row_a, return_counts=True)
    rows_b, counts_b = np.unique(pair.row_b, return_counts=True)

    def fraction(n_matched: int, name: str) -> float:
        n_active = edges.survey(name).n_active
        return n_matched / n_active if n_active else 0.0

    return PairAudit(
        survey_a=pair.survey_a,
        survey_b=pair.survey_b,
        radius_arcsec=pair.radius_arcsec,
        n_edges=len(pair.row_a),
        separation_arcsec=_percentiles(pair.separation_arcsec),
        matched_fraction_a=fraction(len(rows_a), pair.survey_a),
        matched_fraction_b=fraction(len(rows_b), pair.survey_b),
        multi_candidate_rows_a=int((counts_a > 1).sum()),
        multi_candidate_rows_b=int((counts_b > 1).sum()),
    )


def _component_census(edges: CandidateEdges) -> ComponentCensus:
    names = edges.survey_names
    graph = build_global_graph(
        names,
        {survey.name: survey.active_rows() for survey in edges.surveys},
        [
            (p.survey_a, p.survey_b, p.row_a, p.row_b, p.separation_arcsec)
            for p in edges.pairs.values()
        ],
    )
    if graph.n_nodes == 0:
        return ComponentCensus(
            n_rows=0,
            n_components=0,
            size_counts=(0,) * SIZE_HISTOGRAM_CAP,
            max_component_size=0,
            n_clean_multi_survey=0,
            n_ambiguous=0,
            rows_in_ambiguous_components=0,
            by_survey_combination=(),
        )
    sizes = graph.sizes
    clean = graph.clean()
    multi = sizes > 1
    ambiguous = ~clean & multi
    combinations_found = []
    for bits in np.unique(graph.survey_bits[multi]):
        mask = (graph.survey_bits == bits) & multi
        combinations_found.append(
            CombinationCount(
                surveys=tuple(
                    name for i, name in enumerate(names) if (int(bits) >> i) & 1
                ),
                n_components=int(mask.sum()),
                n_clean=int((mask & clean).sum()),
            )
        )
    return ComponentCensus(
        n_rows=graph.n_nodes,
        n_components=len(sizes),
        size_counts=_size_counts(sizes),
        max_component_size=int(sizes.max()),
        n_clean_multi_survey=int((clean & multi).sum()),
        n_ambiguous=int(ambiguous.sum()),
        rows_in_ambiguous_components=int(sizes[ambiguous].sum()),
        by_survey_combination=tuple(combinations_found),
    )


def audit_edges(edges: CandidateEdges) -> EdgeAudit:
    """Summarise one set of candidate edges.

    Separations are in arcseconds. The component census covers the active
    rows of every survey and the edges of every built pair.
    ``dataclasses.asdict`` turns the result into plain JSON types.
    """
    expected = {frozenset(pair) for pair in combinations(edges.survey_names, 2)}
    return EdgeAudit(
        all_pairs_built=set(edges.pairs) == expected,
        surveys=tuple(_survey_audit(survey) for survey in edges.surveys),
        pairs=tuple(_pair_audit(edges, pair) for pair in edges.pairs.values()),
        components=_component_census(edges),
    )
