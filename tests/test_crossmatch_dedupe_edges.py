"""Dedupe and edge-substrate tests over hand-built geometries."""

from __future__ import annotations

import numpy as np
from astro_crossmatch.edges import (
    build_pair_edges,
    dedupe_survey,
    survey_coords_from_arrays,
)

ARCSEC = 1.0 / 3600.0


def _coords(name, ids, ra_arcsec, dec_arcsec=None):
    ra_arcsec = np.asarray(ra_arcsec, dtype=np.float64)
    dec = (
        np.zeros_like(ra_arcsec)
        if dec_arcsec is None
        else np.asarray(dec_arcsec, dtype=np.float64) * ARCSEC
    )
    return survey_coords_from_arrays(
        name, np.asarray(ids), 180.0 + ra_arcsec * ARCSEC, dec
    )


def test_dedupe_keeps_lowest_id_in_clean_pair():
    # Two coincident rows; the lower ID wins regardless of load order.
    coords = _coords("s", ids=[42, 7], ra_arcsec=[0.0, 0.1])
    outcome, active = dedupe_survey(coords, 0.5)
    assert active.tolist() == [False, True]  # position 1 holds id 7
    assert coords.ids[outcome.dropped_index].tolist() == [42]
    assert coords.ids[outcome.keeper_index].tolist() == [7]
    assert outcome.n_disputed == 0


def test_dedupe_keeper_rule_works_for_string_ids():
    coords = _coords("s", ids=np.array(["b", "a", "c"]), ra_arcsec=[0.0, 0.1, 30.0])
    outcome, active = dedupe_survey(coords, 0.5)
    assert coords.ids[outcome.dropped_index].tolist() == ["b"]
    assert coords.ids[outcome.keeper_index].tolist() == ["a"]
    assert active.tolist() == [False, True, True]


def test_dedupe_flags_non_clique_chain_as_disputed():
    # a--b within radius, b--c within radius, a--c outside: a chain, not a
    # clique. All three are disputed and barred from matching.
    coords = _coords("s", ids=[1, 2, 3], ra_arcsec=[0.0, 0.45, 0.9])
    outcome, active = dedupe_survey(coords, 0.5)
    assert outcome.n_dropped == 0
    assert sorted(coords.ids[outcome.disputed_index].tolist()) == [1, 2, 3]
    assert not active.any()


def test_dedupe_zero_radius_is_explicit_opt_out():
    coords = _coords("s", ids=[1, 2], ra_arcsec=[0.0, 0.0])
    outcome, active = dedupe_survey(coords, 0.0)
    assert active.all()
    assert outcome.n_dropped == 0 and outcome.n_disputed == 0


def test_pair_edges_drop_dedupe_inactive_rows():
    a = _coords("a", ids=[1, 2], ra_arcsec=[0.0, 0.1])
    b = _coords("b", ids=[10], ra_arcsec=[0.05])
    active_a = np.array([True, False])
    active_b = np.array([True])
    edges = build_pair_edges(a, b, active_a, active_b, 1.0)
    assert edges.row_a.tolist() == [0]
    assert edges.row_b.tolist() == [0]


def test_dedupe_converts_multiplicity_into_clean_match():
    # Survey a has a duplicate pair straddling one b row: without dedupe the
    # component is ambiguous (two a rows), with dedupe it is a clean 2-clique.
    a = _coords("a", ids=[5, 4], ra_arcsec=[0.0, 0.05])
    b = _coords("b", ids=[9], ra_arcsec=[0.02])
    outcome, active_a = dedupe_survey(a, 0.5)
    assert coords_dropped(a, outcome) == [5]
    edges = build_pair_edges(a, b, active_a, np.ones(1, dtype=bool), 1.0)
    assert len(edges.row_a) == 1
    assert a.ids[edges.row_a[0]] == 4


def coords_dropped(coords, outcome):
    return coords.ids[outcome.dropped_index].tolist()
