"""Candidate edges: building, resolving, saving, reloading and streaming."""

import json

import numpy as np
import pyarrow.parquet as pq
import pytest

from astro_crossmatch import (
    CatalogKernel,
    DegenerateCrossmatchConfig,
    EntitywiseCrossmatchConfig,
    build_edges,
    build_edges_to_directory,
    crossmatch,
    read_edges,
    resolve,
    survey_coords_from_arrays,
    write_edges,
)
from astro_crossmatch import kernel as kernel_module


def survey(name, offsets_arcsec):
    offsets = np.asarray(offsets_arcsec, dtype=float)
    return survey_coords_from_arrays(name, 180 + offsets / 3600, np.zeros(len(offsets)))


# Survey a: rows 0-2 are three copies, rows 3-5 a dedupe chain, row 6 alone.
SURVEYS = [
    survey("a", [0.0, 0.1, 0.2, 10.0, 10.4, 10.8, 20.0]),
    survey("b", [0.3, 10.6, 20.2, 30.0]),
    survey("c", [0.25, 20.1]),
]
SETTINGS = {
    "radius_arcsec": 1.0,
    "dedupe_radius_arcsec": {"a": 0.5, "b": 0.0, "c": 0.0},
    "pair_radius_overrides": {("a", "c"): 0.8},
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
        expected = crossmatch(SURVEYS, mode=mode, **SETTINGS).table
        assert resolve(edges, mode).table.equals(expected, check_metadata=True)


def test_saved_files_hold_rows_dedupe_outcomes_and_settings(tmp_path):
    write_edges(build_edges(SURVEYS, **SETTINGS), tmp_path)
    assert json.loads((tmp_path / "metadata.json").read_text()) == {
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
    }
    assert pq.read_table(tmp_path / "deduplication" / "a.parquet").to_pydict() == {
        "row": [1, 2, 3, 4, 5],
        "status": ["dropped", "dropped", "disputed", "disputed", "disputed"],
        "kept_row": [0, 0, None, None, None],
    }
    assert pq.read_table(tmp_path / "deduplication" / "b.parquet").num_rows == 0
    edges = pq.read_table(tmp_path / "edges" / "a__b.parquet")
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
    assert resolve(edges, degenerate).table["a/row_index"].to_pylist() == [0, 6]
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
    surveys = [
        survey("a__b", [0]),
        survey("c", [0]),
        survey("a", [0]),
        survey("b__c", [0]),
    ]
    settings = {
        "radius_arcsec": 1.0,
        "dedupe_radius_arcsec": {item.name: 0.0 for item in surveys},
    }
    with pytest.raises(ValueError, match="share an edge file name"):
        write_edges(build_edges(surveys, **settings), tmp_path)
    with pytest.raises(ValueError, match="share an edge file name"):
        build_edges_to_directory(surveys, tmp_path, **settings)
