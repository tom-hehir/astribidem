"""Resolve candidate edges into indexes, and map index rows to caller IDs."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from itertools import combinations

import pyarrow as pa
import pyarrow.compute as pc
from numpy.typing import ArrayLike

from astribidem.candidate_edges import CandidateEdges
from astribidem.edges import build_prepared_edges, prepare_edge_build
from astribidem.index_table import (
    DEDUPE_KEY,
    RESOLVED_CONFIG_KEY,
    encode_metadata,
    row_column,
)
from astribidem.kernel import resolve_workers
from astribidem.modes import CrossmatchModeConfig, mode_to_mapping
from astribidem.uids import as_arrow


def _required_pairs(
    mode: CrossmatchModeConfig, survey_names: Iterable[str]
) -> set[frozenset[str]]:
    surveys = survey_names if mode.requires_all_surveys() else mode.required_surveys()
    return {frozenset(pair) for pair in combinations(surveys, 2)}


def _check_mode_fits_edges(mode: CrossmatchModeConfig, edges: CandidateEdges) -> None:
    """Fail loudly when the edges cannot answer the mode's question."""
    unknown = set(mode.required_surveys()) - set(edges.survey_names)
    if unknown:
        raise ValueError(f"mode names unknown surveys: {sorted(unknown)}")
    missing = _required_pairs(mode, edges.survey_names) - set(edges.pairs)
    if missing:
        raise ValueError(
            "the edges were built without survey pairs this mode needs: "
            f"{sorted(sorted(pair) for pair in missing)}"
        )


def resolve(edges: CandidateEdges, mode: CrossmatchModeConfig) -> pa.Table:
    """Resolve candidate edges into one row-position index.

    ``mode`` selects a subset join or exactly-once entity
    resolution. A degenerate mode reads only its own surveys' pairs, so it
    can use edges built for more surveys; an entitywise mode uses every
    survey in the edges. Entity selection is applied only after all surveys
    are resolved.

    The result contains int64 ``<survey>/row_index`` columns (row ``i`` is the
    survey's ``i``-th coordinate; null when absent), optional separations or
    dispute reasons, and metadata recording the configuration, dedupe
    outcomes and the result summary (read with ``index_summary``), so a table
    saved with ``pyarrow.parquet.write_table`` is complete. Use ``rows_to_ids`` to
    replace rows with caller IDs.
    """
    _check_mode_fits_edges(mode, edges)
    table = mode.build(edges)
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
    metadata = dict(table.schema.metadata or {})
    metadata[RESOLVED_CONFIG_KEY] = encode_metadata(provenance)
    metadata[DEDUPE_KEY] = encode_metadata(outcomes)
    return table.replace_schema_metadata(metadata)


def crossmatch(
    coordinates: Mapping[str, tuple[ArrayLike, ArrayLike]],
    *,
    radius_arcsec: float,
    dedupe_radius_arcsec: float,
    dedupe_radius_arcsec_overrides: Mapping[str, float] | None = None,
    mode: CrossmatchModeConfig,
    workers: int = 1,
    radius_arcsec_overrides: Mapping[tuple[str, str] | frozenset[str], float]
    | None = None,
) -> pa.Table:
    """Build candidate edges and resolve them in memory: one row-position index.

    Equivalent to ``resolve(build_edges(...), mode)``, except that a
    degenerate mode joins only the survey pairs it reads. See ``build_edges``
    for the matching settings and ``resolve`` for the result. No payloads are
    read and nothing is saved.
    """
    names = tuple(coordinates)
    workers = resolve_workers(workers)
    settings, kernels = prepare_edge_build(
        coordinates,
        radius_arcsec=radius_arcsec,
        dedupe_radius_arcsec=dedupe_radius_arcsec,
        dedupe_radius_arcsec_overrides=dedupe_radius_arcsec_overrides,
        radius_arcsec_overrides=radius_arcsec_overrides,
        pairs=None if mode.requires_all_surveys() else _required_pairs(mode, names),
    )
    del coordinates
    edges = build_prepared_edges(settings, kernels, workers=workers)
    del kernels
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
    row column in place, named ``<survey>/id`` unless ``id_columns`` gives
    another column name; null rows stay null. Surveys not named in ``ids``
    keep their row column. The IDs are not checked.
    """
    id_columns = id_columns or {}
    unknown = set(id_columns) - set(ids)
    if unknown:
        raise ValueError(f"id_columns names surveys not in ids: {sorted(unknown)}")
    for survey, values in ids.items():
        rows = row_column(survey)
        table = table.set_column(
            table.column_names.index(rows),
            id_columns.get(survey, f"{survey}/id"),
            pc.take(as_arrow(values, name=survey, kind="IDs"), table[rows]),
        )
    return table


def ids_to_rows(
    matches: pa.Table,
    ids: Mapping[str, object],
    *,
    id_columns: Mapping[str, str] | None = None,
) -> pa.Table:
    """Replace each survey's ID column of a match table by its row positions.

    The inverse of ``rows_to_ids``. Each survey's ID column, ``<survey>/id``
    unless ``id_columns`` gives another column name, is looked up in
    ``ids[survey]``, which holds one ID per input row in input order, and
    becomes ``<survey>/row_index``. Null IDs stay null. Surveys not named in
    ``ids``, and surveys whose default ID column is absent, are left as they
    are; a survey named in ``id_columns`` must have that column. Integer IDs
    of different widths, and string IDs of either string type, are compared
    after casting the table's IDs to the type of ``ids[survey]``.

    Raises:
        ValueError: ``ids[survey]`` holds a duplicate or null ID, which would
            make the lookup ambiguous; or the table holds an ID missing from
            ``ids[survey]``, which would otherwise become a null row.
    """
    id_columns = id_columns or {}
    unknown = set(id_columns) - set(ids)
    if unknown:
        raise ValueError(f"id_columns names surveys not in ids: {sorted(unknown)}")
    for survey, values in ids.items():
        column = id_columns.get(survey, f"{survey}/id")
        if column not in matches.column_names:
            if survey in id_columns:
                raise ValueError(f"match table has no ID column {column!r}")
            continue
        lookup = as_arrow(values, name=survey, kind="IDs")
        # count_distinct skips nulls, so a null lookup ID also fails here.
        if pc.count_distinct(lookup).as_py() != len(lookup):
            raise ValueError(f"ids[{survey!r}] must hold unique, non-null IDs")
        table_ids = matches[column]
        if table_ids.type != lookup.type:
            table_ids = pc.cast(table_ids, lookup.type)
        positions = pc.index_in(table_ids, value_set=lookup)
        missing = pc.and_(pc.is_valid(table_ids), pc.is_null(positions))
        if pc.any(missing).as_py():
            examples = pc.filter(table_ids, missing).slice(0, 5).to_pylist()
            raise ValueError(
                f"match table holds IDs missing from ids[{survey!r}]: {examples}"
            )
        matches = matches.set_column(
            matches.column_names.index(column),
            row_column(survey),
            pc.cast(positions, pa.int64()),
        )
    return matches
