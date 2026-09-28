"""Saved products carry a version, and unsupported readers fail explicitly."""

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from astribidem import (
    EntitywiseCrossmatchConfig,
    UIDLink,
    __version__,
    build_band,
    build_edges,
    crossmatch,
    index_summary,
    match_hub_and_spoke,
    match_uids,
    prepare_band_build,
    read_edges,
    rows_to_ids,
    survey_coords_from_arrays,
    write_band_layout,
    write_edges,
)


def spatial_index():
    return crossmatch(
        [survey_coords_from_arrays("a", [10.0], [0.0])],
        radius_arcsec=1.0,
        dedupe_radius_arcsec={"a": 0.0},
        mode=EntitywiseCrossmatchConfig(),
    )


@pytest.mark.parametrize("kind", ["spatial", "uid", "hub_and_spoke"])
def test_index_version_survives_id_mapping_and_parquet(kind, tmp_path):
    if kind == "spatial":
        index = spatial_index()
    elif kind == "uid":
        index = match_uids({"a": [1], "b": [1]})
    else:
        index = match_hub_and_spoke(
            {"a": pa.table({"uid": [1]}), "b": pa.table({"uid": [1]})},
            anchor="a",
            links={"b": UIDLink()},
            dedupe_radius_arcsec={},
        )
    index = rows_to_ids(index, {"a": ["object-a"]})
    pq.write_table(index, tmp_path / "index.parquet")
    saved = pq.read_table(tmp_path / "index.parquet")
    assert saved.schema.metadata[b"astribidem.index_format_version"] == b"1"
    assert saved.schema.metadata[b"astribidem.version"] == __version__.encode()
    assert saved.equals(index, check_metadata=True)


@pytest.mark.parametrize("version", [None, b"0", b"2"])
def test_summary_rejects_unversioned_or_unsupported_indexes(version):
    index = spatial_index()
    metadata = dict(index.schema.metadata)
    if version is None:
        metadata.pop(b"astribidem.index_format_version")
    else:
        metadata[b"astribidem.index_format_version"] = version
    with pytest.raises(ValueError, match="unsupported astribidem index format"):
        index_summary(index.replace_schema_metadata(metadata))


@pytest.mark.parametrize("version", [None, 0, 2, "1", True])
def test_saved_edges_reject_unversioned_or_unsupported_files(tmp_path, version):
    edges = build_edges(
        [survey_coords_from_arrays("a", [10.0], [0.0])],
        radius_arcsec=1.0,
        dedupe_radius_arcsec={"a": 0.0},
    )
    write_edges(edges, tmp_path)
    path = tmp_path / "metadata.json"
    metadata = json.loads(path.read_text())
    if version is None:
        metadata.pop("format_version")
    else:
        metadata["format_version"] = version
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="unsupported astribidem.edges"):
        read_edges(tmp_path)


def test_layout_and_worker_plan_reject_unsupported_versions(tmp_path):
    layout, edges = tmp_path / "layout", tmp_path / "edges"
    write_band_layout(layout, {"a": [([10.0], [0.0])]}, band_height_deg=1.0)
    settings = {"radius_arcsec": 1.0, "dedupe_radius_arcsec": {"a": 0.0}}
    bands = prepare_band_build(layout, edges, **settings)
    plan_path = edges / "band-build.json"
    plan = json.loads(plan_path.read_text())
    plan["format_version"] = 2
    plan_path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="unsupported astribidem.band_build"):
        build_band(edges, bands[0])
    layout_path = layout / "layout.json"
    metadata = json.loads(layout_path.read_text())
    metadata["format_version"] = 2
    layout_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="unsupported astribidem.band_layout"):
        prepare_band_build(layout, edges, **settings)
