"""Candidate-edge audits: dedupe, pair and component diagnostics."""

import json

import numpy as np

from astro_crossmatch import (
    audit_edges,
    build_edges,
    read_edges,
    survey_coords_from_arrays,
    write_edges,
)


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


def test_audit_reports_dedupe_groups_pairs_and_clean_components():
    audit = audit_edges(build_edges(SURVEYS, **SETTINGS))
    assert audit["all_pairs_built"]
    assert audit["surveys"][0] == {
        "name": "a",
        "n_rows": 7,
        "dedupe_radius_arcsec": 0.5,
        "n_active": 2,
        "n_dropped": 2,
        "n_disputed": 3,
        "n_duplicate_groups": 1,
        "largest_duplicate_group": 3,
    }
    a_b = audit["pairs"][0]
    assert a_b["surveys"] == ["a", "b"] and a_b["n_edges"] == 2
    assert a_b["matched_row_fraction"] == {"a": 1.0, "b": 0.5}
    assert audit["pairs"][1]["radius_arcsec"] == 0.8
    components = audit["components"]
    assert components["n_rows"] == 8
    assert components["size_histogram"]["1"] == 2
    assert components["size_histogram"]["3"] == 2
    assert components["n_ambiguous"] == 0
    assert components["by_survey_combination"] == {
        "a+b+c": {"n_components": 2, "n_clean": 2}
    }
    json.dumps(audit)  # plain JSON types throughout


def test_audit_counts_contested_rows_and_ambiguous_components():
    edges = build_edges(
        [survey("x", [0.0]), survey("y", [0.2, 0.4])],
        radius_arcsec=1.0,
        dedupe_radius_arcsec={"x": 0.0, "y": 0.0},
    )
    audit = audit_edges(edges)
    assert audit["pairs"][0]["multi_candidate_rows"] == {"x": 1, "y": 0}
    components = audit["components"]
    assert components["n_ambiguous"] == 1
    assert components["rows_in_ambiguous_components"] == 3
    assert components["max_component_size"] == 3
    assert components["by_survey_combination"] == {
        "x+y": {"n_components": 1, "n_clean": 0}
    }


def test_audit_of_reloaded_edges_matches_in_memory_edges(tmp_path):
    edges = build_edges(SURVEYS, **SETTINGS)
    write_edges(edges, tmp_path)
    assert audit_edges(read_edges(tmp_path)) == audit_edges(edges)


def test_audit_flags_restricted_pair_builds():
    edges = build_edges(SURVEYS, pairs=[("a", "b")], **SETTINGS)
    audit = audit_edges(edges)
    assert not audit["all_pairs_built"]
    assert [pair["surveys"] for pair in audit["pairs"]] == [["a", "b"]]


def test_audit_of_empty_surveys():
    edges = build_edges(
        [survey("x", []), survey("y", [])],
        radius_arcsec=1.0,
        dedupe_radius_arcsec={"x": 0.0, "y": 0.0},
    )
    audit = audit_edges(edges)
    assert audit["components"] == {"n_rows": 0, "n_components": 0}
    assert audit["pairs"][0]["separation_arcsec"] is None
    assert audit["pairs"][0]["matched_row_fraction"] == {"x": 0.0, "y": 0.0}
