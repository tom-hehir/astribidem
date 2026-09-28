"""Hub-and-spoke matching composes independent full-input links through one anchor."""

import json
from dataclasses import FrozenInstanceError

import numpy as np
import pyarrow as pa
import pytest

from astro_crossmatch import SpatialLink, UIDLink, match_hub_and_spoke


def catalog(ids, *, uid=None, offsets=None, id_type=None):
    columns = {"id": pa.array(ids, type=id_type)}
    if uid is not None:
        columns["uid"] = pa.array(uid)
    if offsets is not None:
        columns["ra"] = pa.array(180 + np.asarray(offsets, dtype=float) / 3600)
        columns["dec"] = pa.array(np.zeros(len(offsets)))
    return pa.table(columns)


def test_uid_link_ignores_closer_spatial_neighbor_and_uses_distinct_ids():
    result = match_hub_and_spoke(
        {
            "a": catalog([90], uid=["correct"], offsets=[0]),
            "uid": catalog([7, 8], uid=["wrong", "correct"], offsets=[0, 20]),
            "sky": catalog([3], offsets=[0.1]),
        },
        anchor="a",
        links={"uid": UIDLink(), "sky": SpatialLink(1)},
        dedupe_radius_arcsec={"a": 0, "sky": 0},
    )
    assert result.to_pydict() == {
        "entity_id": [0],
        "a/id": [90],
        "uid/id": [8],
        "sky/id": [3],
    }


@pytest.mark.parametrize("anchor_first", [True, False])
def test_four_sources_intersect_links_in_anchor_order_and_keep_column_order(
    anchor_first,
):
    tables = {
        "anchor": catalog([40, 10, 30, 20], uid=[4, 1, 3, 2], offsets=[40, 10, 30, 20]),
        "uid": catalog([104, 103, 101], uid=[4, 3, 1]),
        "sky": catalog([230, 240, 220], offsets=[30.1, 40.1, 20.1]),
        "last": catalog([303, 304, 302], uid=[3, 4, 2]),
    }
    if not anchor_first:
        tables = {name: tables[name] for name in ("uid", "sky", "anchor", "last")}
    result = match_hub_and_spoke(
        tables,
        anchor="anchor",
        links={"last": UIDLink(), "sky": SpatialLink(1), "uid": UIDLink()},
        dedupe_radius_arcsec={"anchor": 0, "sky": 0},
    )
    assert result.column_names == ["entity_id", *(f"{name}/id" for name in tables)]
    assert result.to_pydict() == {
        "entity_id": [0, 1],
        "anchor/id": [40, 30],
        "uid/id": [104, 103],
        "sky/id": [240, 230],
        "last/id": [304, 303],
    }
    metadata = json.loads(result.schema.metadata[b"astro_crossmatch.resolved_config"])
    assert metadata["method"] == "hub_and_spoke"
    assert metadata["anchor"] == "anchor"
    assert metadata["join"] == "inner"
    assert metadata["topology"] == "anchor-pairs"
    assert metadata["links"]["uid"]["n_matches"] == 3
    assert metadata["links"]["sky"]["policy"] == "mutual_nearest"
    assert metadata["links"]["sky"]["dedupe"]["anchor"]["n_rows"] == 4


@pytest.mark.parametrize("uid_first", [True, False])
def test_earlier_uid_filter_does_not_remove_full_input_spatial_competitor(uid_first):
    tables = {
        "a": catalog([10, 20], uid=[1, 2], offsets=[0, 0.2]),
        "uid": catalog([100], uid=[1]),
        "sky": catalog([200], offsets=[0.19]),
    }
    if not uid_first:
        tables = {name: tables[name] for name in ("a", "sky", "uid")}
    result = match_hub_and_spoke(
        tables,
        anchor="a",
        links={"uid": UIDLink(), "sky": SpatialLink(1)},
        dedupe_radius_arcsec={"a": 0, "sky": 0},
    )
    assert result.num_rows == 0
    metadata = json.loads(result.schema.metadata[b"astro_crossmatch.resolved_config"])
    assert metadata["links"]["uid"]["n_matches"] == 1
    assert metadata["links"]["sky"]["n_matches"] == 1


def test_uid_eligibility_does_not_change_full_anchor_dedupe_keeper():
    result = match_hub_and_spoke(
        {
            # Dedupe keeps a's first row; the UID link targets the dropped one.
            "a": catalog([20, 10], uid=[2, 1], offsets=[0, 0.1]),
            "uid": catalog([100], uid=[1]),
            "sky": catalog([200], offsets=[0.05]),
        },
        anchor="a",
        links={"uid": UIDLink(), "sky": SpatialLink(1)},
        dedupe_radius_arcsec={"a": 0.2, "sky": 0},
    )
    assert result.num_rows == 0
    metadata = json.loads(result.schema.metadata[b"astro_crossmatch.resolved_config"])
    assert metadata["links"]["sky"]["dedupe"]["a"]["n_dropped"] == 1


@pytest.mark.parametrize(
    "invalid", ["duplicate_uid", "null_uid", "duplicate_id", "nan"]
)
def test_empty_uid_link_does_not_skip_validation_of_other_full_inputs(invalid):
    third = catalog([200], uid=[1], offsets=[0.1])
    links = {"empty": UIDLink(), "third": UIDLink()}
    radii = {}
    if invalid == "duplicate_uid":
        third = catalog([200, 201], uid=[1, 1])
    elif invalid == "null_uid":
        third = catalog([200], uid=[None])
    elif invalid == "duplicate_id":
        third = catalog([200, 200], uid=[1, 2])
    else:
        third = catalog([200], offsets=[np.nan])
        links["third"] = SpatialLink(1)
        radii = {"a": 0, "third": 0}
    with pytest.raises(ValueError, match="duplicate|non-null|finite"):
        match_hub_and_spoke(
            {
                "a": catalog([10], uid=[1], offsets=[0]),
                "empty": catalog([], uid=pa.array([], pa.int64()), id_type=pa.int64()),
                "third": third,
            },
            anchor="a",
            links=links,
            dedupe_radius_arcsec=radii,
        )


@pytest.mark.parametrize(
    "id_type,values",
    [
        (pa.large_string(), ["z", "a\0", "a"]),
        (pa.uint64(), [2**64 - 1, 2**63 + 1, 2**63]),
    ],
)
def test_spatial_dedupe_keeps_first_row_and_original_arrow_type(id_type, values):
    table = catalog(values, uid=[1, 2, 3], offsets=[0, 0.05, 0.1], id_type=id_type)
    # Chunked IDs must remain native through matching.
    table = table.set_column(
        0, "id", pa.chunked_array([[], values[:1], [], values[1:]], type=id_type)
    )
    result = match_hub_and_spoke(
        {
            "a": table,
            "uid": catalog([1], uid=[1]),
            "sky": catalog([2**64 - 1], offsets=[0.15], id_type=pa.uint64()),
        },
        anchor="a",
        links={"uid": UIDLink(), "sky": SpatialLink(1)},
        dedupe_radius_arcsec={"a": 0.2, "sky": 0},
    )
    assert result["a/id"].type == id_type
    assert result["a/id"].to_pylist() == [values[0]]
    assert result["sky/id"].type == pa.uint64()
    assert result["sky/id"].to_pylist() == [2**64 - 1]
    metadata = json.loads(result.schema.metadata[b"astro_crossmatch.resolved_config"])
    assert metadata["links"]["sky"]["dedupe"]["a"]["n_dropped"] == 2


def test_uid_links_compare_full_unsigned_range_exactly_and_default_to_id():
    result = match_hub_and_spoke(
        {
            "a": catalog([2**63 - 1, 0], id_type=pa.int64()),
            "b": catalog([2**63, 0], id_type=pa.uint64()),
            "c": catalog([0], id_type=pa.uint8()),
        },
        anchor="a",
        links={"b": UIDLink(), "c": UIDLink()},
        dedupe_radius_arcsec={},
    )
    assert result.to_pydict() == {
        "entity_id": [0],
        "a/id": [0],
        "b/id": [0],
        "c/id": [0],
    }
    assert result["a/id"].type == pa.int64()
    assert result["b/id"].type == pa.uint64()
    assert result["c/id"].type == pa.uint8()


def test_anchored_spatial_policy_can_repeat_counterpart_observation():
    result = match_hub_and_spoke(
        {"a": catalog([2, 1], offsets=[0, 0.2]), "b": catalog([30], offsets=[0.1])},
        anchor="a",
        links={"b": SpatialLink(1, "anchored_nearest")},
        dedupe_radius_arcsec={"a": 0, "b": 0},
    )
    assert result.to_pydict() == {"entity_id": [0, 1], "a/id": [2, 1], "b/id": [30, 30]}


def test_counterparts_do_not_need_to_spatially_match_each_other():
    result = match_hub_and_spoke(
        {
            "a": catalog([1], offsets=[0]),
            "b": catalog([2], offsets=[-0.8]),
            "c": catalog([3], offsets=[0.8]),
        },
        anchor="a",
        links={"b": SpatialLink(1), "c": SpatialLink(1)},
        dedupe_radius_arcsec={"a": 0, "b": 0, "c": 0},
    )
    assert result.num_rows == 1


@pytest.mark.parametrize("radius", [True, "1", 0, -1, np.nan, np.inf])
def test_spatial_link_rejects_invalid_radius(radius):
    with pytest.raises(ValueError, match="radius"):
        SpatialLink(radius)


@pytest.mark.parametrize("policy", [None, 1, "unknown"])
def test_spatial_link_rejects_unknown_policy(policy):
    with pytest.raises(ValueError, match="policy"):
        SpatialLink(1, policy)


def test_link_configurations_are_frozen():
    with pytest.raises(FrozenInstanceError):
        SpatialLink(1).radius_arcsec = 2


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"anchor": "missing"}, "anchor"),
        ({"links": {}}, "links"),
        ({"links": {"a": UIDLink(), "b": UIDLink()}}, "links"),
        ({"links": {"b": object()}}, "configurations"),
        ({"dedupe_radius_arcsec": {"a": 0}}, "spatial participants"),
        ({"workers": True}, "workers"),
        ({"workers": 0}, "workers"),
    ],
)
def test_invalid_composition_configuration(kwargs, message):
    settings = {"anchor": "a", "links": {"b": UIDLink()}, "dedupe_radius_arcsec": {}}
    settings.update(kwargs)
    with pytest.raises((ValueError, TypeError), match=message):
        match_hub_and_spoke({"a": catalog([1]), "b": catalog([1])}, **settings)


@pytest.mark.parametrize(
    "radii",
    [
        {},
        {"a": 0},
        {"a": 0, "b": 0, "extra": 0},
        {"a": True, "b": 0},
        {"a": -1, "b": 0},
    ],
)
def test_spatial_dedupe_configuration_is_explicit_and_exact(radii):
    with pytest.raises(ValueError, match="radius|spatial participants"):
        match_hub_and_spoke(
            {"a": catalog([1], offsets=[0]), "b": catalog([2], offsets=[0])},
            anchor="a",
            links={"b": SpatialLink(1)},
            dedupe_radius_arcsec=radii,
        )


@pytest.mark.parametrize("empty_name", ["a", "b"])
def test_empty_spatial_inputs_keep_original_id_types(empty_name):
    tables = {
        "a": catalog(["a"], offsets=[0], id_type=pa.large_string()),
        "b": catalog([2**64 - 1], offsets=[0], id_type=pa.uint64()),
    }
    tables[empty_name] = tables[empty_name].slice(0, 0)
    result = match_hub_and_spoke(
        tables,
        anchor="a",
        links={"b": SpatialLink(1)},
        dedupe_radius_arcsec={"a": 0, "b": 0},
    )
    assert result.num_rows == 0
    assert result["a/id"].type == pa.large_string()
    assert result["b/id"].type == pa.uint64()


def test_low_precision_catalog_coordinates_warn():
    table = pa.table(
        {
            "id": [1],
            "ra": pa.array([180.0], pa.float32()),
            "dec": pa.array([0.0], pa.float64()),
        }
    )
    with pytest.warns(UserWarning, match="ra is 32-bit"):
        match_hub_and_spoke(
            {"a": table, "b": catalog([2], offsets=[0.1])},
            anchor="a",
            links={"b": SpatialLink(1)},
            dedupe_radius_arcsec={"a": 0, "b": 0},
        )


def test_non_numeric_catalog_coordinates_are_rejected():
    table = pa.table({"id": [1], "ra": ["180.0"], "dec": [0.0]})
    with pytest.raises(ValueError, match="ra must be numeric"):
        match_hub_and_spoke(
            {"a": table, "b": catalog([2], offsets=[0.1])},
            anchor="a",
            links={"b": SpatialLink(1)},
            dedupe_radius_arcsec={"a": 0, "b": 0},
        )


def test_low_precision_column_warns_once_through_every_conversion(recwarn):
    table = pa.table(
        {
            "id": [1],
            "ra": pa.array([180.0], pa.float32()),
            "dec": pa.array([0.0], pa.float64()),
        }
    )
    match_hub_and_spoke(
        {"a": table, "b": catalog([2], offsets=[0.1])},
        anchor="a",
        links={"b": SpatialLink(1)},
        dedupe_radius_arcsec={"a": 0, "b": 0},
    )
    assert [str(w.message)[:28] for w in recwarn.list] == [
        "survey 'a': ra is 32-bit flo"
    ]
