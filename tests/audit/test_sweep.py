"""Edges derived for stricter settings equal a fresh build at those settings."""

import numpy as np
import pytest

from astribidem import build_edges
from astribidem.audit import (
    audit_edges,
    build_edges_with_duplicates,
    deduplicate,
    restrict_radius,
)

LOOSE = {
    "radius_arcsec": 2.0,
    "dedupe_radius_arcsec": 1.0,
    "radius_arcsec_overrides": {("a", "c"): 3.0},
}


def dense_surveys(seed=0):
    """Crowded surveys with close duplicates, so dedupe drops and disputes rows."""
    rng = np.random.default_rng(seed)
    side = 40 / 3600  # 40 arcsec square

    def survey(n):
        ra = 150 + rng.uniform(0, side, n)
        dec = rng.uniform(0, side, n)
        copies = rng.choice(n, n // 4, replace=False)
        jitter = rng.normal(0, 0.3 / 3600, (2, len(copies)))
        return np.concatenate([ra, ra[copies] + jitter[0]]), np.concatenate(
            [dec, dec[copies] + jitter[1]]
        )

    return {"a": survey(120), "b": survey(90), "c": survey(60)}


def assert_same_edges(left, right):
    assert left.survey_names == right.survey_names
    for x, y in zip(left.surveys, right.surveys, strict=True):
        assert (x.name, x.n_rows, x.dedupe_radius_arcsec, x.rows) == (
            y.name,
            y.n_rows,
            y.dedupe_radius_arcsec,
            y.rows,
        )
        for field in ("dropped_rows", "kept_rows", "disputed_rows"):
            np.testing.assert_array_equal(getattr(x, field), getattr(y, field))
    assert set(left.pairs) == set(right.pairs)
    for key, x in left.pairs.items():
        y = right.pairs[key]
        assert (x.survey_a, x.survey_b, x.radius_arcsec) == (
            y.survey_a,
            y.survey_b,
            y.radius_arcsec,
        )
        for field in ("row_a", "row_b", "separation_arcsec"):
            np.testing.assert_array_equal(getattr(x, field), getattr(y, field))


def test_restrict_radius_equals_a_build_at_the_smaller_radius():
    surveys = dense_surveys()
    loose = build_edges(surveys, **LOOSE)
    for radius, overrides in [(0.5, None), (1.2, {("c", "a"): 2.5}), (2.0, None)]:
        assert_same_edges(
            restrict_radius(loose, radius, overrides),
            build_edges(
                surveys,
                radius_arcsec=radius,
                dedupe_radius_arcsec=1.0,
                radius_arcsec_overrides=overrides,
            ),
        )


def test_deduplicate_then_restrict_equals_a_fresh_build():
    surveys = dense_surveys(seed=1)
    edges, duplicates = build_edges_with_duplicates(surveys, **LOOSE)
    assert all(survey.n_dropped == 0 for survey in edges.surveys)
    for dedupe_radius, overrides in [
        (0.0, None),
        (0.4, None),
        (1.0, None),
        (0.6, {"b": 0.0, "c": 1.0}),
    ]:
        deduped = deduplicate(edges, duplicates, dedupe_radius, overrides)
        fresh = build_edges(
            surveys,
            radius_arcsec=1.5,
            dedupe_radius_arcsec=dedupe_radius,
            dedupe_radius_arcsec_overrides=overrides,
        )
        assert_same_edges(restrict_radius(deduped, 1.5), fresh)
        assert audit_edges(restrict_radius(deduped, 1.5)) == audit_edges(fresh)
    # The sweep must exercise both dedupe verdicts.
    deduped = deduplicate(edges, duplicates, 1.0)
    assert sum(s.n_dropped for s in deduped.surveys) > 0
    assert sum(s.n_disputed for s in deduped.surveys) > 0


def test_restricted_pair_builds_keep_their_pairs():
    surveys = dense_surveys()
    edges, duplicates = build_edges_with_duplicates(
        surveys, radius_arcsec=2.0, dedupe_radius_arcsec=1.0, pairs=[("a", "b")]
    )
    narrowed = restrict_radius(deduplicate(edges, duplicates, 0.5), 1.0)
    assert set(narrowed.pairs) == {frozenset("ab")}
    assert_same_edges(
        narrowed,
        build_edges(
            surveys, radius_arcsec=1.0, dedupe_radius_arcsec=0.5, pairs=[("a", "b")]
        ),
    )


def test_settings_looser_than_the_build_are_refused():
    surveys = dense_surveys()
    edges, duplicates = build_edges_with_duplicates(surveys, **LOOSE)
    with pytest.raises(ValueError, match="exceeds the built radius"):
        restrict_radius(edges, 2.5)
    with pytest.raises(ValueError, match="exceeds the 1.0 arcsec"):
        deduplicate(edges, duplicates, 1.5)
    with pytest.raises(ValueError, match="invalid survey pair"):
        restrict_radius(edges, 1.0, {("a", "z"): 1.0})


def test_overrides_must_name_built_pairs():
    edges = build_edges(
        dense_surveys(), radius_arcsec=2.0, dedupe_radius_arcsec=0.0, pairs=[("a", "b")]
    )
    with pytest.raises(ValueError, match="not built"):
        restrict_radius(edges, 1.0, {("a", "c"): 1.0})


def test_deduplicate_refuses_edges_that_were_deduped_already():
    surveys = dense_surveys()
    _, duplicates = build_edges_with_duplicates(surveys, **LOOSE)
    with pytest.raises(ValueError, match="deduped already"):
        deduplicate(build_edges(surveys, **LOOSE), duplicates, 0.5)
