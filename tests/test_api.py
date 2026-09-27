"""Public API validation and end-to-end scientific boundary checks."""

import json
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from astro_crossmatch import (
    DegenerateCrossmatchConfig,
    EntitySelectionConfig,
    EntitywiseCrossmatchConfig,
    crossmatch,
    rows_to_ids,
    survey_coords_from_arrays,
)


def survey(name, offsets):
    offsets = np.asarray(offsets, dtype=float)
    return survey_coords_from_arrays(name, 180 + offsets / 3600, np.zeros(len(offsets)))


def resolve(surveys, *, mode=None, **kwargs):
    return crossmatch(
        surveys,
        radius_arcsec=kwargs.pop("radius_arcsec", 1.0),
        dedupe_radius_arcsec=kwargs.pop(
            "dedupe_radius_arcsec", {item.name: 0.0 for item in surveys}
        ),
        mode=mode or EntitywiseCrossmatchConfig(),
        **kwargs,
    )


def test_public_entity_index_emits_rows_nulls_and_round_trips_parquet(tmp_path):
    sources = [survey("a", [0, 20]), survey("b", [0.2])]
    result = resolve(sources)
    assert result.table.schema.field("a/row_index").type == pa.int64()
    assert result.table.to_pylist() == [
        {"a/row_index": 0, "b/row_index": 0, "disputed_reason": None},
        {"a/row_index": 1, "b/row_index": None, "disputed_reason": None},
    ]
    path = tmp_path / "index.parquet"
    pq.write_table(result.table, path)
    assert pq.read_table(path).equals(result.table, check_metadata=True)


def test_public_dedupe_keeps_lowest_row_and_reports_outcome():
    sources = [survey("a", [0.1, 0]), survey("b", [0.05])]
    result = resolve(sources, dedupe_radius_arcsec={"a": 0.2, "b": 0})
    assert result.table["a/row_index"].to_pylist() == [0]
    assert result.table["b/row_index"].to_pylist() == [0]
    outcomes = json.loads(result.table.schema.metadata[b"astro_crossmatch.dedupe"])
    assert outcomes["a"]["n_dropped"] == 1


def test_rows_to_ids_keeps_types_nulls_column_order_and_metadata():
    big = 2**60 + 7
    result = resolve([survey("a", [0, 20]), survey("b", [0.2])])
    table = rows_to_ids(
        result.table,
        {"a": pa.array([big, big + 1], pa.uint64()), "b": ["b-0"]},
        id_columns={"a": "object_id"},
    )
    assert table.column_names == ["a/object_id", "b/id", "disputed_reason"]
    assert table.schema.field("a/object_id").type == pa.uint64()
    assert table.to_pylist() == [
        {"a/object_id": big, "b/id": "b-0", "disputed_reason": None},
        {"a/object_id": big + 1, "b/id": None, "disputed_reason": None},
    ]
    assert table.schema.metadata == result.table.schema.metadata


def test_rows_to_ids_leaves_unnamed_surveys_as_rows():
    result = resolve([survey("a", [0]), survey("b", [0.2])])
    table = rows_to_ids(result.table, {"b": [42]})
    assert table.column_names == ["a/row_index", "b/id", "disputed_reason"]


def test_pair_override_changes_dispute_classification():
    sources = [survey("a", [0]), survey("b", [0.3]), survey("c", [0.6])]
    clean = resolve(sources)
    assert len(clean.table) == 1
    disputed = resolve(sources, pair_radius_overrides={("a", "c"): 0.1})
    assert len(disputed.table) == 3
    assert disputed.table["disputed_reason"].to_pylist() == ["ambiguous_component"] * 3


def test_selection_does_not_hide_optional_survey_ambiguity():
    sources = [
        survey("a", [0]),
        survey("b", [0.2]),
        survey("c", [0.3, 0.4]),
    ]
    mode = EntitywiseCrossmatchConfig(
        selection=EntitySelectionConfig(min_surveys=2, must_include_surveys=["a", "b"])
    )
    assert len(resolve(sources, mode=mode).table) == 0


def test_subset_join_builds_only_selected_relations():
    sources = [survey("a", [0]), survey("b", [0.2]), survey("c", [20])]
    result = resolve(sources, mode=DegenerateCrossmatchConfig(surveys=["a", "b"]))
    assert result.table.column_names == [
        "a/row_index",
        "b/row_index",
        "a__b/separation_arcsec",
    ]
    assert result.table["a/row_index"].to_pylist() == [0]


def test_empty_survey_is_valid_and_missing_members_stay_null():
    sources = [survey("a", []), survey("b", [0])]
    result = resolve(sources)
    assert result.table.to_pylist() == [
        {"a/row_index": None, "b/row_index": 0, "disputed_reason": None}
    ]


@pytest.mark.parametrize(
    "radii", [{}, {"a": -1}, {"a": float("nan")}, {"a": 0, "x": 0}]
)
def test_dedupe_choice_is_explicit_and_valid(radii):
    with pytest.raises(ValueError, match="dedupe"):
        resolve([survey("a", [0])], dedupe_radius_arcsec=radii)


@pytest.mark.parametrize(
    "override",
    [
        {("a", "b"): 1, ("b", "a"): 1},
        {("a", "z"): 1},
        {("a", "a"): 1},
        {("a", "b"): float("nan")},
    ],
)
def test_invalid_pair_override_is_rejected(override):
    sources = [survey("a", [0]), survey("b", [0])]
    with pytest.raises(ValueError, match="override"):
        resolve(sources, pair_radius_overrides=override)


def test_unknown_mode_survey_is_rejected_before_matching():
    with pytest.raises(ValueError, match="unknown surveys"):
        resolve([survey("a", [0])], mode=DegenerateCrossmatchConfig(surveys=["a", "b"]))


def test_no_upstream_or_storage_dependencies_are_imported():
    assert not any(
        name.startswith(("astral_projections", "aion2", "lsdb", "hats"))
        for name in sys.modules
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_low_precision_coordinates_warn_but_compute_in_float64(dtype):
    ra = np.array([180.0, 180.0001], dtype=dtype)
    with pytest.warns(UserWarning, match="16-bit|32-bit"):
        coords = survey_coords_from_arrays("a", ra, np.zeros(2, dtype=np.float64))
    assert coords.xyz.dtype == np.float64


@pytest.mark.parametrize(
    "ra", [[180.0, 181.0], np.array([180, 181]), np.array([180.0, 181.0])]
)
def test_float64_integer_and_list_coordinates_do_not_warn(ra, recwarn):
    survey_coords_from_arrays("a", ra, [0.0, 0.0])
    assert not recwarn.list


def test_geometry_adapter_warns_on_low_precision_coordinates():
    from astro_crossmatch.geometry import crossmatch_radec, dedupe_and_crossmatch_radec

    low = (np.array([180.0], np.float32), np.array([0.0], np.float32))
    high = (np.array([180.0]), np.array([0.0]))
    with pytest.warns(UserWarning, match="32-bit"):
        crossmatch_radec([low, high], radius_arcsec=1.0)
    with pytest.warns(UserWarning, match="32-bit"):
        dedupe_and_crossmatch_radec(
            [low, high], 1.0, names=["a", "b"], dedupe_radii_arcsec=[0.0, 0.0]
        )


def test_saved_index_keeps_its_summary(tmp_path):
    result = resolve([survey("a", [0, 20]), survey("b", [0.2])])
    path = tmp_path / "index.parquet"
    pq.write_table(result.table, path)
    metadata = pq.read_table(path).schema.metadata
    assert json.loads(metadata[b"astro_crossmatch.summary"]) == result.summary
