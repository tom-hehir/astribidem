"""Save and reload candidate edges, or stream them to disk while building.

A saved edge directory holds::

    metadata.json                   surveys (name, n_rows, dedupe radius), pair radii
    deduplication/<survey>.parquet  row, status ("dropped" | "disputed"), kept_row
    edges/<a>__<b>.parquet          row_a, row_b, separation_arcsec

``kept_row`` is null for disputed rows. Rows not listed in a deduplication
file are active. Every value is a row position, as in ``CandidateEdges``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from astro_crossmatch.candidate_edges import CandidateEdges, DedupeOutcome, PairEdges
from astro_crossmatch.edges import (
    SurveyCoords,
    _kernels_and_dedupe,
    _plan_edge_build,
    pair_edge_chunks,
)
from astro_crossmatch.kernel import resolve_workers

_DROPPED = "dropped"
_DISPUTED = "disputed"
_EDGE_SCHEMA = pa.schema(
    [
        ("row_a", pa.int64()),
        ("row_b", pa.int64()),
        ("separation_arcsec", pa.float64()),
    ]
)


def _edge_file(directory: Path, survey_a: str, survey_b: str) -> Path:
    return directory / "edges" / f"{survey_a}__{survey_b}.parquet"


def _dedupe_file(directory: Path, survey: str) -> Path:
    return directory / "deduplication" / f"{survey}.parquet"


def _prepare_directory(directory: Path, pairs: Iterable[tuple[str, str]]) -> None:
    """Create the layout, refusing survey names whose pair files would collide."""
    files = [_edge_file(directory, a, b) for a, b in pairs]
    if len(set(files)) != len(files):
        raise ValueError(
            "two survey pairs share an edge file name; rename surveys so that "
            "no name contains '__'"
        )
    (directory / "edges").mkdir(parents=True, exist_ok=True)
    (directory / "deduplication").mkdir(exist_ok=True)


def _write_metadata(
    directory: Path,
    surveys: Sequence[DedupeOutcome],
    pair_radii: Iterable[tuple[str, str, float]],
) -> None:
    metadata = {
        "surveys": [
            {
                "name": survey.name,
                "n_rows": survey.n_rows,
                "dedupe_radius_arcsec": survey.dedupe_radius_arcsec,
            }
            for survey in surveys
        ],
        "pairs": [
            {"surveys": [a, b], "radius_arcsec": radius} for a, b, radius in pair_radii
        ],
    }
    text = json.dumps(metadata, allow_nan=False, indent=2)
    (directory / "metadata.json").write_text(text + "\n")


def _write_dedupe(directory: Path, survey: DedupeOutcome) -> None:
    n_dropped, n_disputed = survey.n_dropped, survey.n_disputed
    kept = np.concatenate([survey.kept_rows, np.zeros(n_disputed, dtype=np.int64)])
    no_kept_row = np.r_[
        np.zeros(n_dropped, dtype=bool), np.ones(n_disputed, dtype=bool)
    ]
    table = pa.table(
        {
            "row": pa.array(
                np.concatenate([survey.dropped_rows, survey.disputed_rows]), pa.int64()
            ),
            "status": pa.array(
                [_DROPPED] * n_dropped + [_DISPUTED] * n_disputed, pa.string()
            ),
            "kept_row": pa.array(kept, pa.int64(), mask=no_kept_row),
        }
    )
    pq.write_table(table, _dedupe_file(directory, survey.name))


def _edge_table(row_a, row_b, separation) -> pa.Table:
    return pa.table([row_a, row_b, separation], schema=_EDGE_SCHEMA)


def write_edges(edges: CandidateEdges, directory: str | Path) -> None:
    """Save candidate edges held in memory; ``read_edges`` reloads them."""
    directory = Path(directory)
    _prepare_directory(
        directory, [(p.survey_a, p.survey_b) for p in edges.pairs.values()]
    )
    for survey in edges.surveys:
        _write_dedupe(directory, survey)
    for pair in edges.pairs.values():
        pq.write_table(
            _edge_table(pair.row_a, pair.row_b, pair.separation_arcsec),
            _edge_file(directory, pair.survey_a, pair.survey_b),
        )
    _write_metadata(
        directory,
        edges.surveys,
        [(p.survey_a, p.survey_b, p.radius_arcsec) for p in edges.pairs.values()],
    )


def read_edges(directory: str | Path) -> CandidateEdges:
    """Reload candidate edges saved by ``write_edges`` or ``build_edges_to_directory``."""
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text())
    surveys = []
    for entry in metadata["surveys"]:
        table = pq.read_table(
            _dedupe_file(directory, entry["name"]),
            columns=["row", "status", "kept_row"],
        )
        rows = table["row"].to_numpy()
        dropped = table["status"].to_numpy(zero_copy_only=False) == _DROPPED
        surveys.append(
            DedupeOutcome(
                name=entry["name"],
                n_rows=entry["n_rows"],
                dedupe_radius_arcsec=entry["dedupe_radius_arcsec"],
                dropped_rows=rows[dropped],
                kept_rows=table["kept_row"].filter(pa.array(dropped)).to_numpy(),
                disputed_rows=rows[~dropped],
            )
        )
    pairs = {}
    for entry in metadata["pairs"]:
        a, b = entry["surveys"]
        table = pq.read_table(
            _edge_file(directory, a, b), columns=["row_a", "row_b", "separation_arcsec"]
        )
        pairs[frozenset((a, b))] = PairEdges(
            survey_a=a,
            survey_b=b,
            radius_arcsec=entry["radius_arcsec"],
            row_a=table["row_a"].to_numpy(),
            row_b=table["row_b"].to_numpy(),
            separation_arcsec=table["separation_arcsec"].to_numpy(),
        )
    return CandidateEdges(surveys=tuple(surveys), pairs=pairs)


def build_edges_to_directory(
    surveys: Sequence[SurveyCoords],
    directory: str | Path,
    *,
    radius_arcsec: float,
    dedupe_radius_arcsec: Mapping[str, float],
    pair_radius_overrides: Mapping[tuple[str, str] | frozenset[str], float]
    | None = None,
    pairs: Iterable[tuple[str, str]] | None = None,
    workers: int = 1,
    chunk_rows: int | None = None,
) -> None:
    """Build candidate edges, writing each chunk of pairs to disk as it is found.

    Produces the same files as ``write_edges(build_edges(...), directory)``
    without holding every edge in memory. At most ``workers`` chunks of
    ``chunk_rows`` query rows are in memory at once; the default chunk size is
    one chunk per worker, which bounds nothing, so set ``chunk_rows`` when
    edge memory is the limit. Coordinates, KD-trees and dedupe results stay in
    memory throughout.
    """
    plan = _plan_edge_build(
        surveys, radius_arcsec, dedupe_radius_arcsec, pair_radius_overrides, pairs
    )
    workers = resolve_workers(workers)
    directory = Path(directory)
    _prepare_directory(directory, plan.pair_radius_arcsec)
    kernels, outcomes, active = _kernels_and_dedupe(plan)
    for outcome in outcomes:
        _write_dedupe(directory, outcome)
    for (a, b), radius in plan.pair_radius_arcsec.items():
        with pq.ParquetWriter(_edge_file(directory, a, b), _EDGE_SCHEMA) as writer:
            for row_a, row_b, separation in pair_edge_chunks(
                kernels[a],
                kernels[b],
                active[a],
                active[b],
                radius,
                workers=workers,
                chunk_rows=chunk_rows,
            ):
                writer.write_table(_edge_table(row_a, row_b, separation))
    _write_metadata(
        directory,
        outcomes,
        [(a, b, radius) for (a, b), radius in plan.pair_radius_arcsec.items()],
    )
