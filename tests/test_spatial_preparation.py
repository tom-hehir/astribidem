"""Public coordinate/radius contracts and per-call matching resource lifetimes."""

import json
import weakref
from typing import NamedTuple

import numpy as np
import pytest

from astribidem import (
    CatalogKernel,
    EntitywiseCrossmatchConfig,
    build_edges,
    build_edges_by_band,
    build_edges_to_directory,
    crossmatch,
    read_edges,
    resolve,
    write_band_layout,
)
from astribidem import kernel as kernel_module


def coordinates(offsets):
    return 10.0 + np.asarray(offsets) / 3600, np.zeros(len(offsets))


@pytest.mark.parametrize("route", ["memory", "streamed", "banded", "crossmatch"])
def test_scalar_defaults_and_overrides_have_identical_results(tmp_path, route):
    surveys = {
        "a": coordinates([0, 0.1]),
        "b": coordinates([0.05, 0.15]),
        "c": coordinates([3]),
    }
    settings = {
        "radius_arcsec": 0.3,
        "radius_arcsec_overrides": {("a", "c"): 4.0},
        "dedupe_radius_arcsec": 0.2,
        "dedupe_radius_arcsec_overrides": {"b": 0.0},
    }
    mode = EntitywiseCrossmatchConfig()
    expected_edges = build_edges(surveys, **settings)
    assert [s.n_dropped for s in expected_edges.surveys] == [1, 0, 0]
    assert expected_edges.pairs[frozenset(("a", "b"))].row_b.tolist() == [0, 1]
    assert expected_edges.pairs[frozenset(("a", "c"))].row_a.tolist() == [0]
    expected = resolve(expected_edges, mode)
    if route == "crossmatch":
        actual = crossmatch(surveys, mode=mode, **settings)
    elif route == "memory":
        actual = resolve(build_edges(surveys, **settings), mode)
    else:
        directory = tmp_path / "edges"
        if route == "streamed":
            build_edges_to_directory(surveys, directory, chunk_rows=1, **settings)
        else:
            layout = tmp_path / "layout"
            write_band_layout(
                layout, {n: [c] for n, c in surveys.items()}, band_height_deg=0.01
            )
            build_edges_by_band(layout, directory, **settings)
        actual = resolve(read_edges(directory), mode)
    assert actual.equals(expected, check_metadata=True)
    dedupe = json.loads(actual.schema.metadata[b"astribidem.dedupe"])
    assert {n: d["radius_arcsec"] for n, d in dedupe.items()} == {
        "a": 0.2,
        "b": 0.0,
        "c": 0.2,
    }


@pytest.mark.parametrize("argument", ["radius_arcsec", "dedupe_radius_arcsec"])
@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), True, "1", {}, None])
def test_radius_defaults_require_finite_scalars(argument, value):
    settings = {"radius_arcsec": 1.0, "dedupe_radius_arcsec": 0.0, argument: value}
    with pytest.raises(ValueError, match="scalar radius"):
        build_edges({"a": coordinates([0])}, **settings)


@pytest.mark.parametrize(
    "overrides",
    [{"missing": 0}, {"a": -1}, {"a": float("inf")}, {"a": True}, {"a": "1"}, []],
)
def test_invalid_dedupe_overrides_fail(overrides):
    with pytest.raises((ValueError, TypeError), match="dedupe"):
        build_edges(
            {"a": coordinates([0])},
            radius_arcsec=1,
            dedupe_radius_arcsec=0,
            dedupe_radius_arcsec_overrides=overrides,
        )


@pytest.mark.parametrize(
    "surveys",
    [
        [],
        {},
        {"a": ([0],)},
        {"a": {"ra": [0], "dec": [0]}},
        {"a": ([0], [0, 1])},
        {"a": ([[0]], [[0]])},
        {"a/b": ([0], [0])},
    ],
)
def test_invalid_coordinate_mappings_fail(surveys):
    with pytest.raises(ValueError):
        build_edges(surveys, radius_arcsec=1, dedupe_radius_arcsec=0)


def test_named_tuple_coordinates_preserve_input_arrays_and_survey_order():
    class Coordinates(NamedTuple):
        ra: np.ndarray
        dec: np.ndarray

    ra, dec = coordinates([0, 2])
    original_ra, original_dec = ra.copy(), dec.copy()
    result = build_edges(
        {"b": Coordinates(ra, dec), "a": (ra, dec)},
        radius_arcsec=1,
        dedupe_radius_arcsec=0,
    )
    assert result.survey_names == ("b", "a")
    np.testing.assert_array_equal(ra, original_ra)
    np.testing.assert_array_equal(dec, original_dec)


@pytest.mark.parametrize("route", ["crossmatch", "edges", "streamed"])
def test_temporary_inputs_are_released_before_spatial_queries(
    monkeypatch, tmp_path, route
):
    refs = []

    def inputs():
        surveys = {n: coordinates([0, 0.1]) for n in ("a", "b")}
        refs.extend(weakref.ref(array) for pair in surveys.values() for array in pair)
        return surveys

    queried = []
    original = CatalogKernel.self_pairs

    def self_pairs(kernel, radius):
        assert all(ref() is None for ref in refs)
        queried.append(1)
        return original(kernel, radius)

    monkeypatch.setattr(CatalogKernel, "self_pairs", self_pairs)
    # Explicit keywords leave ownership with the callee. Python 3.11 keeps
    # an extra caller-side argument tuple alive when forwarding **kwargs.
    if route == "crossmatch":
        crossmatch(
            inputs(),
            radius_arcsec=1.0,
            dedupe_radius_arcsec=0.2,
            mode=EntitywiseCrossmatchConfig(),
        )
    elif route == "edges":
        build_edges(inputs(), radius_arcsec=1.0, dedupe_radius_arcsec=0.2)
    else:
        build_edges_to_directory(
            inputs(),
            tmp_path,
            radius_arcsec=1.0,
            dedupe_radius_arcsec=0.2,
        )
    assert len(queried) == 2


def test_each_call_prepares_each_survey_once_and_retains_no_kernels(monkeypatch):
    original_init, original_conversion = (
        CatalogKernel.__init__,
        kernel_module.radec_to_xyz,
    )
    kernels, conversions = [], []

    def initialize(kernel, *args, **kwargs):
        original_init(kernel, *args, **kwargs)
        assert kernel.xyz.dtype == np.float64
        assert np.shares_memory(kernel.xyz, kernel.tree.data)
        kernels.append(weakref.ref(kernel))

    def convert(*args, **kwargs):
        conversions.append(kwargs.get("name"))
        return original_conversion(*args, **kwargs)

    monkeypatch.setattr(CatalogKernel, "__init__", initialize)
    monkeypatch.setattr(kernel_module, "radec_to_xyz", convert)
    surveys = {n: coordinates([0, 0.1]) for n in ("a", "b", "c")}
    for _ in range(2):
        crossmatch(
            surveys,
            radius_arcsec=1,
            dedupe_radius_arcsec=0.2,
            mode=EntitywiseCrossmatchConfig(),
        )
        assert all(ref() is None for ref in kernels)
    assert conversions == ["a", "b", "c", "a", "b", "c"]
    assert len(kernels) == 6


@pytest.mark.parametrize("offsets,radius", [([0, 0], 0), ([], 1)])
def test_geometry_dedupe_fast_path_needs_no_tree(monkeypatch, offsets, radius):
    from astribidem.geometry import dedupe_radec

    def unexpected_tree(*args, **kwargs):
        pytest.fail("a no-op dedupe must not build a tree")

    monkeypatch.setattr(kernel_module, "cKDTree", unexpected_tree)
    outcome, active = dedupe_radec(*coordinates(offsets), radius, name="a")
    assert outcome.n_rows == len(offsets)
    assert outcome.n_dropped == outcome.n_disputed == 0
    assert active.tolist() == [True] * len(offsets)
