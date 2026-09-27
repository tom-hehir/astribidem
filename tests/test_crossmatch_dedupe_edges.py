"""Dedupe and edge-substrate tests over hand-built geometries."""

from __future__ import annotations

import numpy as np

from astro_crossmatch.edges import (
    build_pair_edges,
    dedupe_survey,
    survey_coords_from_arrays,
)

ARCSEC = 1.0 / 3600.0


def _coords(name, ra_arcsec, dec_arcsec=None):
    ra_arcsec = np.asarray(ra_arcsec, dtype=np.float64)
    dec = (
        np.zeros_like(ra_arcsec)
        if dec_arcsec is None
        else np.asarray(dec_arcsec, dtype=np.float64) * ARCSEC
    )
    return survey_coords_from_arrays(name, 180.0 + ra_arcsec * ARCSEC, dec)


def test_dedupe_keeps_lowest_row_in_clean_pair():
    coords = _coords("s", ra_arcsec=[0.0, 0.1])
    outcome, active = dedupe_survey(coords, 0.5)
    assert active.tolist() == [True, False]
    assert outcome.dropped_rows.tolist() == [1]
    assert outcome.kept_rows.tolist() == [0]
    assert outcome.n_disputed == 0


def test_dedupe_keeper_follows_row_order_not_position_on_sky():
    coords = _coords("s", ra_arcsec=[30.0, 0.1, 0.0])
    outcome, active = dedupe_survey(coords, 0.5)
    assert outcome.dropped_rows.tolist() == [2]
    assert outcome.kept_rows.tolist() == [1]
    assert active.tolist() == [True, True, False]


def test_dedupe_drops_every_extra_copy_onto_one_kept_row():
    coords = _coords("s", ra_arcsec=[0.2, 0.0, 30.0, 0.1])
    outcome, active = dedupe_survey(coords, 0.5)
    assert outcome.dropped_rows.tolist() == [1, 3]
    assert outcome.kept_rows.tolist() == [0, 0]
    assert active.tolist() == [True, False, True, False]


def test_dedupe_flags_non_clique_chain_as_disputed():
    # a--b within radius, b--c within radius, a--c outside: a chain, not a
    # clique. All three are disputed and barred from matching.
    coords = _coords("s", ra_arcsec=[0.0, 0.45, 0.9])
    outcome, active = dedupe_survey(coords, 0.5)
    assert outcome.n_dropped == 0
    assert outcome.disputed_rows.tolist() == [0, 1, 2]
    assert not active.any()


def test_dedupe_zero_radius_is_explicit_opt_out():
    coords = _coords("s", ra_arcsec=[0.0, 0.0])
    outcome, active = dedupe_survey(coords, 0.0)
    assert active.all()
    assert outcome.n_dropped == 0 and outcome.n_disputed == 0


def test_pair_edges_drop_dedupe_inactive_rows():
    a = _coords("a", ra_arcsec=[0.0, 0.1])
    b = _coords("b", ra_arcsec=[0.05])
    active_a = np.array([True, False])
    active_b = np.array([True])
    edges = build_pair_edges(a, b, active_a, active_b, 1.0)
    assert edges.row_a.tolist() == [0]
    assert edges.row_b.tolist() == [0]


def test_dedupe_converts_multiplicity_into_clean_match():
    # Survey a has a duplicate pair straddling one b row: without dedupe the
    # component is ambiguous (two a rows), with dedupe it is a clean 2-clique.
    a = _coords("a", ra_arcsec=[0.0, 0.05])
    b = _coords("b", ra_arcsec=[0.02])
    outcome, active_a = dedupe_survey(a, 0.5)
    assert outcome.dropped_rows.tolist() == [1]
    edges = build_pair_edges(a, b, active_a, np.ones(1, dtype=bool), 1.0)
    assert edges.row_a.tolist() == [0]
