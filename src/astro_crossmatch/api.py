"""Resolve candidate edges into indexes, and map index rows to caller IDs."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from itertools import combinations

import pyarrow as pa
import pyarrow.compute as pc

from astro_crossmatch.candidate_edges import CandidateEdges
from astro_crossmatch.edges import SurveyCoords, build_edges
from astro_crossmatch.modes import CrossmatchModeConfig, ModeResult, mode_to_mapping
from astro_crossmatch.uids import _as_arrow


def _required_pairs(
    mode: CrossmatchModeConfig, survey_names: Iterable[str]
) -> set[frozenset[str]]:
    surveys = survey_names if mode.requires_all_surveys() else mode.required_surveys()
    return {frozenset(pair) for pair in combinations(surveys, 2)}


def _check_mode_fits_edges(mode: CrossmatchModeConfig, edges: CandidateEdges) -> None:
    """Fail loudly when the edges cannot answer the mode's question."""
    if not isinstance(mode, CrossmatchModeConfig):
        raise TypeError("mode must be a CrossmatchModeConfig")
    unknown = set(mode.required_surveys()) - set(edges.survey_names)
    if unknown:
        raise ValueError(f"mode names unknown surveys: {sorted(unknown)}")
    missing = _required_pairs(mode, edges.survey_names) - set(edges.pairs)
    if missing:
        raise ValueError(
            "the edges were built without survey pairs this mode needs: "
            f"{sorted(sorted(pair) for pair in missing)}"
        )


def resolve(edges: CandidateEdges, mode: CrossmatchModeConfig) -> ModeResult:
    """Resolve candidate edges into one row-position index.

    ``mode`` selects an Astral-derived subset join or exactly-once entity
    resolution. A degenerate mode reads only its own surveys' pairs, so it
    can use edges built for more surveys; an entitywise mode uses every
    survey in the edges. Entity selection is applied only after all surveys
    are resolved.

    The result contains int64 ``<survey>/row_index`` columns (row ``i`` is the
    survey's ``i``-th coordinate; null when absent), optional separations or
    dispute reasons, and policy/provenance metadata. Use ``rows_to_ids`` to
    replace rows with caller IDs.
    """
    _check_mode_fits_edges(mode, edges)
    result = mode.build(edges)
    used_pairs = _required_pairs(mode, edges.survey_names)
    provenance = {
        "surveys": [
            {"name": survey.name, "dedupe_radius_arcsec": survey.dedupe_radius_arcsec}
            for survey in edges.surveys
        ],
        "pair_radius_arcsec": [
            {
                "surveys": [pair.survey_a, pair.survey_b],
                "radius_arcsec": pair.radius_arcsec,
            }
            for key, pair in edges.pairs.items()
            if key in used_pairs
        ],
        "mode": mode_to_mapping(mode),
    }
    outcomes = {
        survey.name: {
            "radius_arcsec": survey.dedupe_radius_arcsec,
            "n_rows": survey.n_rows,
            "n_dropped": survey.n_dropped,
            "n_disputed": survey.n_disputed,
        }
        for survey in edges.surveys
    }
    metadata = dict(result.table.schema.metadata or {})
    metadata[b"astro_crossmatch.resolved_config"] = json.dumps(
        provenance, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    metadata[b"astro_crossmatch.dedupe"] = json.dumps(
        outcomes, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return ModeResult(result.table.replace_schema_metadata(metadata), result.summary)


def crossmatch(
    surveys: Sequence[SurveyCoords],
    *,
    radius_arcsec: float,
    dedupe_radius_arcsec: Mapping[str, float],
    mode: CrossmatchModeConfig,
    workers: int = 1,
    pair_radius_overrides: Mapping[tuple[str, str] | frozenset[str], float]
    | None = None,
) -> ModeResult:
    """Build candidate edges and resolve them in memory: one row-position index.

    Equivalent to ``resolve(build_edges(...), mode)``, except that a
    degenerate mode joins only the survey pairs it reads. See ``build_edges``
    for the matching settings and ``resolve`` for the result. No payloads are
    read and nothing is saved.
    """
    if not isinstance(mode, CrossmatchModeConfig):
        raise TypeError("mode must be a CrossmatchModeConfig")
    names = [survey.name for survey in surveys]
    unknown = set(mode.required_surveys()) - set(names)
    if unknown:
        raise ValueError(f"mode names unknown surveys: {sorted(unknown)}")
    edges = build_edges(
        surveys,
        radius_arcsec=radius_arcsec,
        dedupe_radius_arcsec=dedupe_radius_arcsec,
        pair_radius_overrides=pair_radius_overrides,
        pairs=None if mode.requires_all_surveys() else _required_pairs(mode, names),
        workers=workers,
    )
    return resolve(edges, mode)


def rows_to_ids(
    table: pa.Table,
    ids: Mapping[str, object],
    *,
    id_columns: Mapping[str, str] | None = None,
) -> pa.Table:
    """Replace ``<survey>/row_index`` columns of a match index by IDs.

    Each named survey's row column is used to index ``ids[survey]``, which
    holds one ID per input row in input order (the coordinates passed to the
    edge build, or the UIDs passed to ``match_uids``). The result replaces the
    row column in place as ``<survey>/<id column>`` (default ``id``); null
    rows stay null. Surveys not named in ``ids`` keep their row column. The
    IDs are not checked.
    """
    id_columns = id_columns or {}
    for survey, values in ids.items():
        row_column = f"{survey}/row_index"
        table = table.set_column(
            table.column_names.index(row_column),
            f"{survey}/{id_columns.get(survey, 'id')}",
            pc.take(_as_arrow(values, name=survey, kind="IDs"), table[row_column]),
        )
    return table
