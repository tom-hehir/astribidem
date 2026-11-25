"""Derive the candidate edges of stricter settings from one looser build.

A build finds every pair within its radius, so the edges for a smaller match
radius are those of a larger one with separation within the smaller radius:
``restrict_radius`` filters them. Dedupe needs one more ingredient, because a
deduped build keeps no within-survey pairs and no edges for inactive rows.
``build_edges_with_duplicates`` therefore builds with dedupe off and also
returns each survey's within-survey pairs; ``deduplicate`` then applies any
dedupe radius up to the one those pairs were found at. Both return ordinary
``CandidateEdges``, equal to a fresh ``build_edges`` at the stricter settings,
so a sweep resolves or audits each setting with no further searches::

    edges, duplicates = build_edges_with_duplicates(
        coordinates, radius_arcsec=2.0, dedupe_radius_arcsec=1.0
    )
    for dedupe_radius in (0.0, 0.5, 1.0):
        deduped = deduplicate(edges, duplicates, dedupe_radius)
        for radius in (0.5, 1.0, 2.0):
            audit = audit_edges(restrict_radius(deduped, radius))
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Mapping

import numpy as np
from numpy.typing import ArrayLike

from astribidem.candidate_edges import CandidateEdges, DedupeOutcome, PairEdges
from astribidem.edges import (
    build_prepared_edges,
    dedupe_from_pairs,
    dedupe_radius_by_survey,
    prepare_edge_build,
    radius_by_pair,
)
from astribidem.kernel import resolve_workers


def restrict_radius(
    edges: CandidateEdges,
    radius_arcsec: float,
    radius_arcsec_overrides: Mapping[tuple[str, str] | frozenset[str], float]
    | None = None,
) -> CandidateEdges:
    """The edges a build at a smaller match radius would have found.

    ``radius_arcsec`` applies to every built pair, and ``radius_arcsec_overrides``
    names exceptions, as in ``build_edges``. Dedupe outcomes are unchanged.
    Works on a single segment as well as on combined edges.

    Raises:
        ValueError: a radius exceeds the pair's built radius, or an override
            names a pair that was not built.
    """
    radii = radius_by_pair(edges.survey_names, radius_arcsec, radius_arcsec_overrides)
    unbuilt = {frozenset(pair) for pair in radius_arcsec_overrides or {}} - set(
        edges.pairs
    )
    if unbuilt:
        raise ValueError(
            "radius_arcsec_overrides names pairs that were not built: "
            f"{sorted(sorted(pair) for pair in unbuilt)}"
        )
    pairs = {}
    for key, pair in edges.pairs.items():
        radius = radii[key]
        if radius > pair.radius_arcsec:
            raise ValueError(
                f"radius {radius} arcsec for {sorted(key)} exceeds the built "
                f"radius {pair.radius_arcsec} arcsec"
            )
        keep = pair.separation_arcsec <= radius
        pairs[key] = PairEdges(
            survey_a=pair.survey_a,
            survey_b=pair.survey_b,
            radius_arcsec=radius,
            row_a=pair.row_a[keep],
            row_b=pair.row_b[keep],
            separation_arcsec=pair.separation_arcsec[keep],
        )
    return CandidateEdges(surveys=edges.surveys, pairs=pairs)


def build_edges_with_duplicates(
    coordinates: Mapping[str, tuple[ArrayLike, ArrayLike]],
    *,
    radius_arcsec: float,
    dedupe_radius_arcsec: float,
    dedupe_radius_arcsec_overrides: Mapping[str, float] | None = None,
    radius_arcsec_overrides: Mapping[tuple[str, str] | frozenset[str], float]
    | None = None,
    pairs: Iterable[tuple[str, str]] | None = None,
    workers: int = 1,
) -> tuple[CandidateEdges, dict[str, PairEdges]]:
    """Build edges with dedupe off, plus each survey's within-survey pairs.

    Takes the arguments of ``build_edges``. The edges join every row, as if
    every dedupe radius were zero. The second result maps each survey name to
    its pairs within its dedupe radius, as ``PairEdges`` whose two surveys are
    that survey. Pass both to ``deduplicate`` to apply a dedupe radius up to
    the one given here. Each survey's KD-tree serves both searches.
    """
    workers = resolve_workers(workers)
    settings, kernels = prepare_edge_build(
        coordinates,
        radius_arcsec=radius_arcsec,
        dedupe_radius_arcsec=dedupe_radius_arcsec,
        dedupe_radius_arcsec_overrides=dedupe_radius_arcsec_overrides,
        radius_arcsec_overrides=radius_arcsec_overrides,
        pairs=pairs,
    )
    del coordinates
    duplicates = {}
    for name, radius in settings.dedupe_radius_arcsec.items():
        if radius > 0 and len(kernels[name]):
            first, second, separation = kernels[name].self_pairs(radius)
        else:
            first = second = np.empty(0, dtype=np.int64)
            separation = np.empty(0, dtype=np.float64)
        duplicates[name] = PairEdges(
            survey_a=name,
            survey_b=name,
            radius_arcsec=radius,
            row_a=first,
            row_b=second,
            separation_arcsec=separation,
        )
    undeduped = dataclasses.replace(
        settings,
        dedupe_radius_arcsec=dict.fromkeys(settings.survey_names, 0.0),
    )
    return build_prepared_edges(undeduped, kernels, workers=workers), duplicates


def deduplicate(
    edges: CandidateEdges,
    duplicates: Mapping[str, PairEdges],
    dedupe_radius_arcsec: float,
    dedupe_radius_arcsec_overrides: Mapping[str, float] | None = None,
) -> CandidateEdges:
    """Apply dedupe to edges from ``build_edges_with_duplicates``.

    ``dedupe_radius_arcsec`` applies to every survey, and
    ``dedupe_radius_arcsec_overrides`` names exceptions, as in ``build_edges``.
    Each survey is deduped from its within-survey pairs inside its radius, and
    edges touching a dropped or disputed row are removed. Match radii are
    unchanged.

    Raises:
        ValueError: the edges were deduped already or cover only some rows, or
            a dedupe radius exceeds the one its survey's pairs were found at.
    """
    radii = dedupe_radius_by_survey(
        edges.survey_names, dedupe_radius_arcsec, dedupe_radius_arcsec_overrides
    )
    surveys = []
    active = {}
    for survey in edges.surveys:
        if survey.dedupe_radius_arcsec != 0 or survey.rows is not None:
            raise ValueError(
                "deduplicate needs edges from build_edges_with_duplicates; "
                f"survey {survey.name!r} was deduped already or is a segment"
            )
        found = duplicates[survey.name]
        radius = radii[survey.name]
        if radius > found.radius_arcsec:
            raise ValueError(
                f"dedupe radius {radius} arcsec for {survey.name!r} exceeds the "
                f"{found.radius_arcsec} arcsec its duplicate pairs were found at"
            )
        mask = np.ones(survey.n_rows, dtype=bool)
        empty = np.empty(0, dtype=np.int64)
        dropped = kept = disputed = empty
        if radius > 0:
            keep = found.separation_arcsec <= radius
            dropped, kept, disputed = dedupe_from_pairs(
                np.arange(survey.n_rows, dtype=np.int64),
                found.row_a[keep],
                found.row_b[keep],
            )
            mask[dropped] = False
            mask[disputed] = False
        active[survey.name] = mask
        surveys.append(
            DedupeOutcome(
                name=survey.name,
                n_rows=survey.n_rows,
                dedupe_radius_arcsec=radius,
                dropped_rows=dropped,
                kept_rows=kept,
                disputed_rows=disputed,
            )
        )
    pairs = {}
    for key, pair in edges.pairs.items():
        keep = active[pair.survey_a][pair.row_a] & active[pair.survey_b][pair.row_b]
        pairs[key] = PairEdges(
            survey_a=pair.survey_a,
            survey_b=pair.survey_b,
            radius_arcsec=pair.radius_arcsec,
            row_a=pair.row_a[keep],
            row_b=pair.row_b[keep],
            separation_arcsec=pair.separation_arcsec[keep],
        )
    return CandidateEdges(surveys=tuple(surveys), pairs=pairs)
