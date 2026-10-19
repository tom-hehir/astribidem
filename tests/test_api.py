"""Public API validation and end-to-end scientific boundary checks."""

import json
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from astribidem import (
    DegenerateCrossmatchConfig,
    EntitySelectionConfig,
    EntitywiseCrossmatchConfig,
    crossmatch,
    ids_to_rows,
    index_summary,
    rows_to_ids,
)


def survey(offsets):
    offsets = np.asarray(offsets, dtype=float)
    return 180 + offsets / 3600, np.zeros(len(offsets))


def resolve(surveys, *, mode=None, **kwargs):
    return crossmatch(
        surveys,
        radius_arcsec=kwargs.pop("radius_arcsec", 1.0),
        dedupe_radius_arcsec=kwargs.pop("dedupe_radius_arcsec", 0.0),
        mode=mode or EntitywiseCrossmatchConfig(),
        **kwargs,
    )


def test_public_entity_index_emits_rows_nulls_and_round_trips_parquet(tmp_path):
    sources = {"a": survey([0, 20]), "b": survey([0.2])}
    result = resolve(sources)
    assert result.schema.field("a/row_index").type == pa.int64()
    assert result.to_pylist() == [
        {"a/row_index": 0, "b/row_index": 0, "disputed_reason": None},
        {"a/row_index": 1, "b/row_index": None, "disputed_reason": None},
    ]
    path = tmp_path / "index.parquet"
    pq.write_table(result, path)
    assert pq.read_table(path).equals(result, check_metadata=True)


def test_public_dedupe_keeps_lowest_row_and_reports_outcome():
    sources = {"a": survey([0.1, 0]), "b": survey([0.05])}
    result = resolve(
        sources,
        dedupe_radius_arcsec=0.0,
        dedupe_radius_arcsec_overrides={"a": 0.2, "b": 0},
    )
    assert result["a/row_index"].to_pylist() == [0]
    assert result["b/row_index"].to_pylist() == [0]
    outcomes = json.loads(result.schema.metadata[b"astribidem.dedupe"])
    assert outcomes["a"]["n_dropped"] == 1


def test_rows_to_ids_keeps_types_nulls_column_order_and_metadata():
    big = 2**60 + 7
    result = resolve({"a": survey([0, 20]), "b": survey([0.2])})
    table = rows_to_ids(
        result,
        {"a": pa.array([big, big + 1], pa.uint64()), "b": ["b-0"]},
        id_columns={"a": "object_id"},
    )
    assert table.column_names == ["a/object_id", "b/id", "disputed_reason"]
    assert table.schema.field("a/object_id").type == pa.uint64()
    assert table.to_pylist() == [
        {"a/object_id": big, "b/id": "b-0", "disputed_reason": None},
        {"a/object_id": big + 1, "b/id": None, "disputed_reason": None},
    ]
    assert table.schema.metadata == result.schema.metadata


def test_rows_to_ids_leaves_unnamed_surveys_as_rows():
    result = resolve({"a": survey([0]), "b": survey([0.2])})
    table = rows_to_ids(result, {"b": [42]})
    assert table.column_names == ["a/row_index", "b/id", "disputed_reason"]


def test_pair_override_changes_dispute_classification():
    sources = {"a": survey([0]), "b": survey([0.3]), "c": survey([0.6])}
    clean = resolve(sources)
    assert len(clean) == 1
    disputed = resolve(sources, radius_arcsec_overrides={("a", "c"): 0.1})
    assert len(disputed) == 3
    assert disputed["disputed_reason"].to_pylist() == ["ambiguous_component"] * 3


def test_selection_does_not_hide_optional_survey_ambiguity():
    sources = {"a": survey([0]), "b": survey([0.2]), "c": survey([0.3, 0.4])}
    mode = EntitywiseCrossmatchConfig(
        selection=EntitySelectionConfig(min_surveys=2, must_include_surveys=["a", "b"])
    )
    assert len(resolve(sources, mode=mode)) == 0


def test_subset_join_builds_only_selected_relations():
    sources = {"a": survey([0]), "b": survey([0.2]), "c": survey([20])}
    result = resolve(sources, mode=DegenerateCrossmatchConfig(surveys=["a", "b"]))
    assert result.column_names == [
        "a/row_index",
        "b/row_index",
        "a__b/separation_arcsec",
    ]
    assert result["a/row_index"].to_pylist() == [0]


def test_empty_survey_is_valid_and_missing_members_stay_null():
    sources = {"a": survey([]), "b": survey([0])}
    result = resolve(sources)
    assert result.to_pylist() == [
        {"a/row_index": None, "b/row_index": 0, "disputed_reason": None}
    ]


def test_dedupe_override_for_an_unknown_survey_is_rejected():
    with pytest.raises(ValueError, match="unknown surveys"):
        resolve({"a": survey([0])}, dedupe_radius_arcsec_overrides={"x": 0.5})


@pytest.mark.parametrize(
    "override",
    [
        {("a", "b"): 1, ("b", "a"): 1},
        {("a", "z"): 1},
        {("a", "a"): 1},
    ],
)
def test_invalid_pair_override_is_rejected(override):
    sources = {"a": survey([0]), "b": survey([0])}
    with pytest.raises(ValueError, match="override"):
        resolve(sources, radius_arcsec_overrides=override)


def test_unknown_mode_survey_is_rejected_before_matching():
    with pytest.raises(ValueError, match="invalid survey pair"):
        resolve({"a": survey([0])}, mode=DegenerateCrossmatchConfig(surveys=["a", "b"]))


def test_no_upstream_or_storage_dependencies_are_imported():
    assert not any(
        name.startswith(("astral_projections", "aion2", "lsdb", "hats"))
        for name in sys.modules
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_low_precision_coordinates_warn_but_compute_in_float64(dtype):
    ra = np.array([180.0, 180.0001], dtype=dtype)
    with pytest.warns(UserWarning, match="16-bit|32-bit"):
        result = resolve({"a": (ra, np.zeros(2, dtype=np.float64))})
    assert result.num_rows == 2


@pytest.mark.parametrize(
    "ra", [[180.0, 181.0], np.array([180, 181]), np.array([180.0, 181.0])]
)
def test_float64_integer_and_list_coordinates_do_not_warn(ra, recwarn):
    resolve({"a": (ra, [0.0, 0.0])})
    assert not recwarn.list


def test_saved_index_keeps_its_summary(tmp_path):
    result = resolve({"a": survey([0, 20]), "b": survey([0.2])})
    path = tmp_path / "index.parquet"
    pq.write_table(result, path)
    assert index_summary(pq.read_table(path)) == index_summary(result)


@pytest.mark.parametrize(
    "ra,dec,message",
    [
        ([0.0], [100.0], "declination"),
        ([0.0], [-90.00001], "declination"),
        ([float("nan")], [0.0], "finite"),
        ([0.0], [float("inf")], "finite"),
    ],
)
@pytest.mark.parametrize("entry", ["arrays", "banded"])
def test_invalid_coordinates_fail_at_every_array_entry(
    tmp_path, ra, dec, message, entry
):
    from astribidem import write_band_layout

    with pytest.raises(ValueError, match=message):
        if entry == "arrays":
            resolve({"a": (ra, dec)})
        else:
            write_band_layout(tmp_path, {"a": [(ra, dec)]}, band_height_deg=1.0)


def test_poles_and_wrapped_ra_remain_valid():
    result = resolve(
        {
            "a": ([-1.0, 361.0], [-90.0, 90.0]),
            "b": ([123.0, 456.0], [-90.0, 90.0]),
        }
    )
    assert result["a/row_index"].to_pylist() == [0, 1]
    assert result["b/row_index"].to_pylist() == [0, 1]


def test_ids_to_rows_inverts_rows_to_ids_with_typed_and_null_ids():
    big = 2**63 + 5
    ids = {"s": pa.array([big, big + 1], pa.uint64())}
    matches = pa.table({"s/row_index": pa.array([1, None, 0], pa.int64())})
    assert ids_to_rows(rows_to_ids(matches, ids), ids).equals(matches)


def test_ids_to_rows_reads_a_named_id_column_and_casts_integer_widths():
    matches = pa.table({"key": pa.array([20, 10], pa.int32()), "other": [1, 2]})
    result = ids_to_rows(matches, {"s": [10, 20]}, id_columns={"s": "key"})
    assert result.column_names == ["s/row_index", "other"]
    assert result["s/row_index"].to_pylist() == [1, 0]


def test_ids_to_rows_leaves_surveys_without_their_default_column():
    matches = pa.table({"a/id": ["x"]})
    assert ids_to_rows(matches, {"b": ["y"]}).equals(matches)


@pytest.mark.parametrize(
    "matches,ids,kwargs,message",
    [
        (pa.table({"s/id": [1]}), {"s": [1, 1]}, {}, "unique, non-null"),
        (pa.table({"s/id": [1]}), {"s": [1, None]}, {}, "unique, non-null"),
        (
            pa.table({"key": [3]}),
            {"s": [1, 2]},
            {"id_columns": {"s": "key"}},
            "missing",
        ),
        (
            pa.table({"s/id": [1]}),
            {"s": [1]},
            {"id_columns": {"s": "key"}},
            "no ID column",
        ),
        (
            pa.table({"s/id": [1]}),
            {"s": [1]},
            {"id_columns": {"t": "key"}},
            "not in ids",
        ),
    ],
)
def test_ids_to_rows_rejects_ambiguous_or_missing_ids(matches, ids, kwargs, message):
    with pytest.raises(ValueError, match=message):
        ids_to_rows(matches, ids, **kwargs)
