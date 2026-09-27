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

from itertools import combinations
from typing import Any

import numpy as np

from astro_crossmatch.candidate_edges import CandidateEdges, DedupeOutcome
from astro_crossmatch.graph import build_global_graph

_SEPARATION_PERCENTILES = (5, 25, 50, 75, 95)
_SIZE_HISTOGRAM_CAP = 6


def _percentiles(values: np.ndarray) -> dict[str, float] | None:
    if len(values) == 0:
        return None
    points = np.percentile(values, _SEPARATION_PERCENTILES)
    return {
        f"p{p}": round(float(v), 4) for p, v in zip(_SEPARATION_PERCENTILES, points)
    }


def _size_histogram(sizes: np.ndarray) -> dict[str, int]:
    histogram = {
        str(size): int((sizes == size).sum()) for size in range(1, _SIZE_HISTOGRAM_CAP)
    }
    histogram[f">={_SIZE_HISTOGRAM_CAP}"] = int((sizes >= _SIZE_HISTOGRAM_CAP).sum())
    return histogram


def _survey_audit(survey: DedupeOutcome) -> dict[str, Any]:
    # Each distinct kept row stands for one duplicate group; the group also
    # contains every dropped row that points at it.
    _, dropped_per_group = np.unique(survey.kept_rows, return_counts=True)
    return {
        "name": survey.name,
        "n_rows": survey.n_rows,
        "dedupe_radius_arcsec": survey.dedupe_radius_arcsec,
        "n_active": survey.n_rows - survey.n_dropped - survey.n_disputed,
        "n_dropped": survey.n_dropped,
        "n_disputed": survey.n_disputed,
        "n_duplicate_groups": len(dropped_per_group),
        "largest_duplicate_group": int(dropped_per_group.max()) + 1
        if len(dropped_per_group)
        else 0,
    }


def _pair_audit(edges: CandidateEdges, pair) -> dict[str, Any]:
    active = {
        name: edges.survey(name).n_rows
        - edges.survey(name).n_dropped
        - edges.survey(name).n_disputed
        for name in (pair.survey_a, pair.survey_b)
    }
    rows_a, counts_a = np.unique(pair.row_a, return_counts=True)
    rows_b, counts_b = np.unique(pair.row_b, return_counts=True)

    def fraction(n_matched: int, name: str) -> float:
        return round(n_matched / active[name], 6) if active[name] else 0.0

    return {
        "surveys": [pair.survey_a, pair.survey_b],
        "radius_arcsec": pair.radius_arcsec,
        "n_edges": len(pair.row_a),
        "separation_arcsec": _percentiles(pair.separation_arcsec),
        "matched_row_fraction": {
            pair.survey_a: fraction(len(rows_a), pair.survey_a),
            pair.survey_b: fraction(len(rows_b), pair.survey_b),
        },
        "multi_candidate_rows": {
            pair.survey_a: int((counts_a > 1).sum()),
            pair.survey_b: int((counts_b > 1).sum()),
        },
    }


def _component_census(edges: CandidateEdges) -> dict[str, Any]:
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
        return {"n_rows": 0, "n_components": 0}
    sizes = graph.sizes
    clean = graph.clean()
    multi = sizes > 1
    ambiguous = ~clean & multi
    by_combination = {}
    for bits in np.unique(graph.survey_bits[multi]):
        members = [name for i, name in enumerate(names) if (int(bits) >> i) & 1]
        mask = (graph.survey_bits == bits) & multi
        by_combination["+".join(members)] = {
            "n_components": int(mask.sum()),
            "n_clean": int((mask & clean).sum()),
        }
    return {
        "n_rows": graph.n_nodes,
        "n_components": len(sizes),
        "size_histogram": _size_histogram(sizes),
        "max_component_size": int(sizes.max()),
        "n_clean_multi_survey": int((clean & multi).sum()),
        "n_ambiguous": int(ambiguous.sum()),
        "rows_in_ambiguous_components": int(sizes[ambiguous].sum()),
        "by_survey_combination": by_combination,
    }


def audit_edges(edges: CandidateEdges) -> dict[str, Any]:
    """Summarise candidate edges as a JSON-serialisable dictionary.

    Separations are in arcseconds. The component census covers the active
    rows of every survey and the edges of every built pair; ``all_pairs_built``
    is false when the build was restricted to some pairs, in which case the
    census omits the missing pairs' edges.
    """
    expected = {frozenset(pair) for pair in combinations(edges.survey_names, 2)}
    return {
        "all_pairs_built": set(edges.pairs) == expected,
        "surveys": [_survey_audit(survey) for survey in edges.surveys],
        "pairs": [_pair_audit(edges, pair) for pair in edges.pairs.values()],
        "components": _component_census(edges),
    }
