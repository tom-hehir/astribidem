"""Save and reload candidate edges, or stream them to disk while building.

A saved edge directory holds ``metadata.json`` and one directory per segment::

    metadata.json                           surveys (name, n_rows, dedupe radius),
                                            pair radii, and the segment names
    <segment>/rows/<survey>.parquet         row: the rows the segment covers
    <segment>/deduplication/<survey>.parquet  row, status, kept_row
    <segment>/edges/<a>__<b>.parquet        row_a, row_b, separation_arcsec

A segment that covers every row, such as ``segment-0`` of an in-memory build,
has no ``rows`` directory. ``status`` is ``"dropped"`` or ``"disputed"``;
``kept_row`` is null for disputed rows; covered rows not listed in a
deduplication file are active. Every value is a global row position.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from astro_crossmatch.candidate_edges import (
    CandidateEdges,
    DedupeOutcome,
    PairEdges,
    combine_segments,
)
from astro_crossmatch.edges import (
    SurveyCoords,
    _kernels_and_dedupe,
    _plan_edge_build,
    pair_edge_chunks,
)
from astro_crossmatch.kernel import resolve_workers

SINGLE_SEGMENT = "segment-0"
_DROPPED = "dropped"
_DISPUTED = "disputed"
_EDGE_SCHEMA = pa.schema(
    [
        ("row_a", pa.int64()),
        ("row_b", pa.int64()),
        ("separation_arcsec", pa.float64()),
    ]
)


def _edge_file(segment: Path, survey_a: str, survey_b: str) -> Path:
    return segment / "edges" / f"{survey_a}__{survey_b}.parquet"


def _dedupe_file(segment: Path, survey: str) -> Path:
    return segment / "deduplication" / f"{survey}.parquet"


def _rows_file(segment: Path, survey: str) -> Path:
    return segment / "rows" / f"{survey}.parquet"


def _check_pair_file_names(pairs: Iterable[tuple[str, str]]) -> None:
    names = [f"{a}__{b}" for a, b in pairs]
    if len(set(names)) != len(names):
        raise ValueError(
            "two survey pairs share an edge file name; rename surveys so that "
            "no name contains '__'"
        )


def _check_segment_name(name: str) -> None:
    if not isinstance(name, str) or not name or "/" in name or name.startswith("."):
        raise ValueError(f"invalid segment name: {name!r}")
    if name == "metadata.json":
        raise ValueError("a segment cannot be named metadata.json")


def write_metadata(
    directory: str | Path,
    surveys: Sequence[DedupeOutcome],
    pair_radii: Iterable[tuple[str, str, float]],
    segments: Sequence[str],
    extra: Mapping[str, object] | None = None,
) -> None:
    """Write ``metadata.json``: the build settings and the segment names."""
    for name in segments:
        _check_segment_name(name)
    if len(set(segments)) != len(segments):
        raise ValueError("segment names must be unique")
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
        "segments": list(segments),
        **(extra or {}),
    }
    text = json.dumps(metadata, allow_nan=False, indent=2)
    (Path(directory) / "metadata.json").write_text(text + "\n")


def read_metadata(directory: str | Path) -> dict:
    """The contents of a saved edge directory's ``metadata.json``."""
    return json.loads((Path(directory) / "metadata.json").read_text())


def _write_dedupe(segment: Path, survey: DedupeOutcome) -> None:
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
    pq.write_table(table, _dedupe_file(segment, survey.name))


def _edge_table(row_a, row_b, separation) -> pa.Table:
    return pa.table([row_a, row_b, separation], schema=_EDGE_SCHEMA)


def _make_segment_directories(segment: Path, *, with_rows: bool) -> None:
    (segment / "edges").mkdir(parents=True, exist_ok=True)
    (segment / "deduplication").mkdir(exist_ok=True)
    if with_rows:
        (segment / "rows").mkdir(exist_ok=True)


def write_segment(directory: str | Path, name: str, edges: CandidateEdges) -> None:
    """Write one segment's files; ``write_metadata`` lists the segments."""
    _check_segment_name(name)
    _check_pair_file_names((p.survey_a, p.survey_b) for p in edges.pairs.values())
    segment = Path(directory) / name
    with_rows = any(survey.rows is not None for survey in edges.surveys)
    _make_segment_directories(segment, with_rows=with_rows)
    for survey in edges.surveys:
        _write_dedupe(segment, survey)
        if with_rows:
            rows = (
                np.arange(survey.n_rows, dtype=np.int64)
                if survey.rows is None
                else survey.rows
            )
            pq.write_table(
                pa.table({"row": pa.array(rows, pa.int64())}),
                _rows_file(segment, survey.name),
            )
    for pair in edges.pairs.values():
        pq.write_table(
            _edge_table(pair.row_a, pair.row_b, pair.separation_arcsec),
            _edge_file(segment, pair.survey_a, pair.survey_b),
        )


def write_edges(edges: CandidateEdges, directory: str | Path) -> None:
    """Save candidate edges held in memory as the single segment ``segment-0``."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    write_segment(directory, SINGLE_SEGMENT, edges)
    write_metadata(
        directory,
        edges.surveys,
        [(p.survey_a, p.survey_b, p.radius_arcsec) for p in edges.pairs.values()],
        [SINGLE_SEGMENT],
    )


def read_segment(directory: str | Path, name: str) -> CandidateEdges:
    """Load one saved segment."""
    directory = Path(directory)
    metadata = read_metadata(directory)
    if name not in metadata["segments"]:
        raise ValueError(f"no segment {name!r}; segments: {metadata['segments']}")
    segment = directory / name
    has_rows = (segment / "rows").is_dir()
    surveys = []
    for entry in metadata["surveys"]:
        table = pq.read_table(
            _dedupe_file(segment, entry["name"]),
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
                disputed_rows=np.sort(rows[~dropped]),
                rows=pq.read_table(_rows_file(segment, entry["name"]))["row"].to_numpy()
                if has_rows
                else None,
            )
        )
    pairs = {}
    for entry in metadata["pairs"]:
        a, b = entry["surveys"]
        table = pq.read_table(
            _edge_file(segment, a, b), columns=["row_a", "row_b", "separation_arcsec"]
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


def segment_names(directory: str | Path) -> list[str]:
    """The segments of a saved edge directory, in their recorded order."""
    return list(read_metadata(directory)["segments"])


def read_edges(directory: str | Path) -> CandidateEdges:
    """Reload saved candidate edges as one segment covering all of their rows.

    Every segment is loaded and combined, so this suits edges that fit in
    memory; use ``read_segment`` to process large builds one segment at a time.
    """
    return combine_segments(
        [read_segment(directory, name) for name in segment_names(directory)]
    )


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

    Produces the same single segment as ``write_edges(build_edges(...),
    directory)`` without holding every edge in memory. At most ``workers`` chunks of
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
    _check_pair_file_names(plan.pair_radius_arcsec)
    segment = directory / SINGLE_SEGMENT
    _make_segment_directories(segment, with_rows=False)
    kernels, outcomes, active = _kernels_and_dedupe(plan)
    for outcome in outcomes:
        _write_dedupe(segment, outcome)
    for (a, b), radius in plan.pair_radius_arcsec.items():
        with pq.ParquetWriter(_edge_file(segment, a, b), _EDGE_SCHEMA) as writer:
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
    write_metadata(
        directory,
        outcomes,
        [(a, b, radius) for (a, b), radius in plan.pair_radius_arcsec.items()],
        [SINGLE_SEGMENT],
    )
