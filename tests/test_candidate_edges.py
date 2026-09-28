"""Candidate edges: building, resolving, saving, reloading and streaming."""

import json

import numpy as np
import pyarrow.parquet as pq
import pytest

from astribidem import (
    CatalogKernel,
    DegenerateCrossmatchConfig,
    EntitywiseCrossmatchConfig,
    build_edges,
    build_edges_to_directory,
    crossmatch,
    read_edges,
    resolve,
    write_edges,
)
from astribidem import kernel as kernel_module


def survey(offsets_arcsec):
    offsets = np.asarray(offsets_arcsec, dtype=float)
    return 180 + offsets / 3600, np.zeros(len(offsets))


# Survey a: rows 0-2 are three copies, rows 3-5 a dedupe chain, row 6 alone.
SURVEYS = {
    "a": survey([0.0, 0.1, 0.2, 10.0, 10.4, 10.8, 20.0]),
    "b": survey([0.3, 10.6, 20.2, 30.0]),
    "c": survey([0.25, 20.1]),
}
SETTINGS = {
    "radius_arcsec": 1.0,
    "dedupe_radius_arcsec": 0.0,
    "dedupe_radius_arcsec_overrides": {"a": 0.5, "b": 0.0, "c": 0.0},
    "radius_arcsec_overrides": {("a", "c"): 0.8},
}


def assert_same_edges(left, right):
    assert left.survey_names == right.survey_names
    for x, y in zip(left.surveys, right.surveys):
        assert (x.name, x.n_rows, x.dedupe_radius_arcsec) == (
            y.name,
            y.n_rows,
            y.dedupe_radius_arcsec,
        )
        for field in ("dropped_rows", "kept_rows", "disputed_rows"):
            np.testing.assert_array_equal(getattr(x, field), getattr(y, field))
    assert list(left.pairs) == list(right.pairs)
    for key, x in left.pairs.items():
        y = right.pairs[key]
        assert (x.survey_a, x.survey_b, x.radius_arcsec) == (
            y.survey_a,
            y.survey_b,
            y.radius_arcsec,
        )
        for field in ("row_a", "row_b", "separation_arcsec"):
            np.testing.assert_array_equal(getattr(x, field), getattr(y, field))


def test_resolving_built_edges_equals_crossmatch():
    edges = build_edges(SURVEYS, **SETTINGS)
    for mode in (
        EntitywiseCrossmatchConfig(),
        EntitywiseCrossmatchConfig(resolver="split"),
        DegenerateCrossmatchConfig(surveys=["a", "b"]),
    ):
        expected = crossmatch(SURVEYS, mode=mode, **SETTINGS)
        assert resolve(edges, mode).equals(expected, check_metadata=True)


def test_saved_files_hold_rows_dedupe_outcomes_and_settings(tmp_path):
    write_edges(build_edges(SURVEYS, **SETTINGS), tmp_path)
    from astribidem import __version__

    assert json.loads((tmp_path / "metadata.json").read_text()) == {
        "format": "astribidem.edges",
        "format_version": 1,
        "astribidem_version": __version__,
        "surveys": [
            {"name": "a", "n_rows": 7, "dedupe_radius_arcsec": 0.5},
            {"name": "b", "n_rows": 4, "dedupe_radius_arcsec": 0.0},
            {"name": "c", "n_rows": 2, "dedupe_radius_arcsec": 0.0},
        ],
        "pairs": [
            {"surveys": ["a", "b"], "radius_arcsec": 1.0},
            {"surveys": ["a", "c"], "radius_arcsec": 0.8},
            {"surveys": ["b", "c"], "radius_arcsec": 1.0},
        ],
        "segments": ["segment-0"],
    }
    segment = tmp_path / "segment-0"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["metadata.json", "segment-0"]
    assert sorted(p.name for p in segment.iterdir()) == ["deduplication", "edges"]
    assert pq.read_table(segment / "deduplication" / "a.parquet").to_pydict() == {
        "row": [1, 2, 3, 4, 5],
        "status": ["dropped", "dropped", "disputed", "disputed", "disputed"],
        "kept_row": [0, 0, None, None, None],
    }
    assert pq.read_table(segment / "deduplication" / "b.parquet").num_rows == 0
    edges = pq.read_table(segment / "edges" / "a__b.parquet")
    assert edges.column_names == ["row_a", "row_b", "separation_arcsec"]
    assert edges["row_a"].to_pylist() == [0, 6]
    assert edges["row_b"].to_pylist() == [0, 2]


def test_read_edges_restores_the_written_object(tmp_path):
    edges = build_edges(SURVEYS, **SETTINGS)
    write_edges(edges, tmp_path)
    assert_same_edges(read_edges(tmp_path), edges)


@pytest.mark.parametrize("workers,chunk_rows", [(1, None), (1, 2), (3, 1)])
def test_streamed_build_writes_the_same_edges(tmp_path, workers, chunk_rows):
    build_edges_to_directory(
        SURVEYS, tmp_path, workers=workers, chunk_rows=chunk_rows, **SETTINGS
    )
    assert_same_edges(read_edges(tmp_path), build_edges(SURVEYS, **SETTINGS))


def test_restricted_pairs_serve_only_degenerate_modes_over_them():
    edges = build_edges(SURVEYS, pairs=[("b", "a")], **SETTINGS)
    assert list(edges.pairs) == [frozenset({"a", "b"})]
    degenerate = DegenerateCrossmatchConfig(surveys=["a", "b"])
    assert resolve(edges, degenerate)["a/row_index"].to_pylist() == [0, 6]
    with pytest.raises(ValueError, match="without survey pairs"):
        resolve(edges, EntitywiseCrossmatchConfig())
    with pytest.raises(ValueError, match="without survey pairs"):
        resolve(edges, DegenerateCrossmatchConfig(surveys=["a", "c"]))
    with pytest.raises(ValueError, match="unknown surveys"):
        resolve(edges, DegenerateCrossmatchConfig(surveys=["a", "z"]))


@pytest.mark.parametrize("pairs", [[("a", "z")], [("a", "a")], ["ab"]])
def test_invalid_pair_selection_is_rejected(pairs):
    with pytest.raises(ValueError, match="invalid survey pair"):
        build_edges(SURVEYS, pairs=pairs, **SETTINGS)


def test_chunk_stream_concatenates_to_the_complete_join():
    rng = np.random.default_rng(0)
    a = CatalogKernel(rng.uniform(0, 0.01, 400), rng.uniform(0, 0.01, 400))
    b = CatalogKernel(rng.uniform(0, 0.01, 300), rng.uniform(0, 0.01, 300))
    chunks = list(a.all_pairs_chunks(b, 5.0, workers=3, chunk_rows=37))
    assert len(chunks) == 11
    joined = [np.concatenate([chunk[k] for chunk in chunks]) for k in range(3)]
    for part, whole in zip(joined, a.all_pairs(b, 5.0)):
        np.testing.assert_array_equal(part, whole)


def test_chunk_stream_keeps_at_most_workers_chunks_in_flight(monkeypatch):
    rng = np.random.default_rng(1)
    a = CatalogKernel(rng.uniform(0, 0.01, 400), rng.uniform(0, 0.01, 400))
    b = CatalogKernel(rng.uniform(0, 0.01, 300), rng.uniform(0, 0.01, 300))
    started = []
    tree = kernel_module.cKDTree

    def counting_tree(*args, **kwargs):
        started.append(1)
        return tree(*args, **kwargs)

    monkeypatch.setattr(kernel_module, "cKDTree", counting_tree)
    stream = a.all_pairs_chunks(b, 5.0, workers=2, chunk_rows=10)
    next(stream)
    # Two chunks submitted up front, one more when the first is handed over.
    assert len(started) <= 3
    rest = list(stream)
    assert len(rest) == 39 and len(started) == 40


def test_writers_refuse_colliding_pair_file_names(tmp_path):
    surveys = {
        "a__b": survey([0]),
        "c": survey([0]),
        "a": survey([0]),
        "b__c": survey([0]),
    }
    settings = {
        "radius_arcsec": 1.0,
        "dedupe_radius_arcsec": 0.0,
        "dedupe_radius_arcsec_overrides": {name: 0.0 for name in surveys},
    }
    with pytest.raises(ValueError, match="share an edge file name"):
        write_edges(build_edges(surveys, **settings), tmp_path)
    with pytest.raises(ValueError, match="share an edge file name"):
        build_edges_to_directory(surveys, tmp_path, **settings)


def restrict(edges, rows_by_survey):
    """The segment of ``edges`` covering the given rows of each survey."""
    from astribidem import CandidateEdges, DedupeOutcome, PairEdges

    surveys = []
    for survey in edges.surveys:
        rows = np.asarray(rows_by_survey[survey.name], dtype=np.int64)
        keep = np.isin(survey.dropped_rows, rows)
        surveys.append(
            DedupeOutcome(
                name=survey.name,
                n_rows=survey.n_rows,
                dedupe_radius_arcsec=survey.dedupe_radius_arcsec,
                dropped_rows=survey.dropped_rows[keep],
                kept_rows=survey.kept_rows[keep],
                disputed_rows=survey.disputed_rows[np.isin(survey.disputed_rows, rows)],
                rows=rows,
            )
        )
    pairs = {}
    for key, pair in edges.pairs.items():
        keep = np.isin(pair.row_a, rows_by_survey[pair.survey_a])
        pairs[key] = PairEdges(
            pair.survey_a,
            pair.survey_b,
            pair.radius_arcsec,
            pair.row_a[keep],
            pair.row_b[keep],
            pair.separation_arcsec[keep],
        )
    return CandidateEdges(surveys=tuple(surveys), pairs=pairs)


# Two sky patches 1 degree apart: rows of each patch form separate segments.
PATCHES = {
    "a": survey([0.0, 0.1, 0.2, 10.0, 10.4, 10.8, 3600.0, 3600.1, 3610.0]),
    "b": survey([0.3, 10.6, 3600.2, 3610.6, 3630.0]),
    "c": survey([0.25, 3600.15]),
}
PATCH_ROWS = [
    {"a": [0, 1, 2, 3, 4, 5], "b": [0, 1], "c": [0]},
    {"a": [6, 7, 8], "b": [2, 3, 4], "c": [1]},
]


def test_combined_segments_equal_the_whole_build():
    edges = build_edges(PATCHES, **SETTINGS)
    segments = [restrict(edges, rows) for rows in PATCH_ROWS]
    from astribidem import combine_segments

    assert_same_edges(combine_segments(segments), edges)


def test_segments_save_their_rows_and_reload(tmp_path):
    from astribidem import read_segment, segment_names, write_segment
    from astribidem.edge_files import write_metadata

    edges = build_edges(PATCHES, **SETTINGS)
    segments = [restrict(edges, rows) for rows in PATCH_ROWS]
    for name, segment in zip(["band-0", "band-1"], segments):
        write_segment(tmp_path, name, segment)
    write_metadata(
        tmp_path,
        edges.surveys,
        [(p.survey_a, p.survey_b, p.radius_arcsec) for p in edges.pairs.values()],
        ["band-0", "band-1"],
    )
    assert segment_names(tmp_path) == ["band-0", "band-1"]
    assert sorted(p.name for p in (tmp_path / "band-1").iterdir()) == [
        "deduplication",
        "edges",
        "rows",
    ]
    assert pq.read_table(tmp_path / "band-1" / "rows" / "a.parquet")[
        "row"
    ].to_pylist() == [6, 7, 8]
    assert_same_edges(read_segment(tmp_path, "band-1"), segments[1])
    assert_same_edges(read_edges(tmp_path), edges)


def test_resolving_segments_separately_gives_the_same_entities():
    edges = build_edges(PATCHES, **SETTINGS)
    segments = [restrict(edges, rows) for rows in PATCH_ROWS]
    for mode in (
        EntitywiseCrossmatchConfig(),
        EntitywiseCrossmatchConfig(resolver="split"),
        DegenerateCrossmatchConfig(surveys=["a", "b"]),
    ):
        whole = resolve(edges, mode)
        parts = [resolve(segment, mode) for segment in segments]
        split = sorted(json.dumps(r) for part in parts for r in part.to_pylist())
        assert split == sorted(json.dumps(r) for r in whole.to_pylist())
