"""Mode-builder tests: policy reductions, degenerate joins, entity index."""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest
from astro_crossmatch.inputs import (
    PairEdges,
    ResolveInputs,
)
from astro_crossmatch.modes import (
    DegenerateCrossmatchConfig,
    EntitySelectionConfig,
    EntitywiseCrossmatchConfig,
    one_to_one_pairs,
)


def _inputs(surveys, edges, *, disputed=None):
    """Assemble ResolveInputs from {name: n_rows} and per-pair edge triples.

    ``edges`` maps (a, b) -> (row_a, row_b, sep) in positions. All rows are
    active unless ``disputed`` lists positions for a survey (disputed rows
    are also removed from active, as dedupe would).
    """
    disputed = disputed or {}
    active = {}
    for name, n in surveys.items():
        mask = np.ones(n, dtype=bool)
        mask[disputed.get(name, [])] = False
        active[name] = np.flatnonzero(mask)
    pairs = {}
    radii = {}
    for (a, b), (row_a, row_b, sep) in edges.items():
        pairs[frozenset((a, b))] = PairEdges(
            survey_a=a,
            survey_b=b,
            radius_arcsec=1.0,
            row_a=np.asarray(row_a, dtype=np.int64),
            row_b=np.asarray(row_b, dtype=np.int64),
            sep_arcsec=np.asarray(sep, dtype=np.float64),
        )
    names = list(surveys)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            radii[frozenset((a, b))] = 1.0
    return ResolveInputs(
        survey_names=tuple(names),
        active_row_index=active,
        dedupe_disputed={
            name: np.asarray(rows, dtype=np.int64) for name, rows in disputed.items()
        },
        pairs=pairs,
        pair_radius_arcsec=radii,
    )


# --- policy reductions -------------------------------------------------------


def test_one_to_one_policies_on_flanked_geometry():
    # a0 -- b0 at 0.2; a1 -- b0 at 0.4: b0 is contested.
    row_a = np.array([0, 1])
    row_b = np.array([0, 0])
    sep = np.array([0.2, 0.4])

    ka, kb, ks = one_to_one_pairs(row_a, row_b, sep, "anchored_nearest")
    assert ka.tolist() == [0, 1] and kb.tolist() == [0, 0]

    ka, kb, ks = one_to_one_pairs(row_a, row_b, sep, "mutual_nearest")
    assert ka.tolist() == [0] and kb.tolist() == [0] and ks[0] == 0.2

    ka, kb, _ = one_to_one_pairs(row_a, row_b, sep, "mutual_unique")
    assert len(ka) == 0  # b0 has two candidates

    ka, kb, _ = one_to_one_pairs(row_a, row_b, sep, "anchored_unique")
    assert ka.tolist() == [0, 1]  # anchors unique; no reverse check

    with pytest.raises(ValueError, match="unknown subset-join policy"):
        one_to_one_pairs(row_a, row_b, sep, "nope")


# --- degenerate joins --------------------------------------------------------


def test_degenerate_two_survey_join_emits_rows_and_separations():
    inputs = _inputs(
        {"a": 2, "b": 2},
        {("a", "b"): ([0, 1], [0, 1], [0.1, 0.2])},
    )
    result = DegenerateCrossmatchConfig(surveys=["a", "b"]).build(inputs)
    table = result.table
    assert table.column("a/row_index").to_pylist() == [0, 1]
    assert table.column("b/row_index").to_pylist() == [0, 1]
    seps = table.column("a__b/separation_arcsec").to_pylist()
    assert seps == pytest.approx([0.1, 0.2], abs=1e-6)
    metadata = table.schema.metadata
    assert metadata[b"astro_crossmatch.match_index.mode"] == b"degenerate"
    assert result.summary["n_groups"] == 2


def test_degenerate_three_way_requires_all_pairwise_agreement():
    # a0-b0-c0 fully consistent; a1-b1 matches but c disagrees.
    edges = {
        ("a", "b"): ([0, 1], [0, 1], [0.1, 0.1]),
        ("a", "c"): ([0, 1], [0, 1], [0.1, 0.1]),
        ("b", "c"): ([0], [0], [0.1]),  # b1-c1 relation missing
    }
    inputs = _inputs({"a": 2, "b": 2, "c": 2}, edges)
    result = DegenerateCrossmatchConfig(surveys=["a", "b", "c"]).build(inputs)
    assert result.summary["n_groups"] == 1
    assert result.table.column("a/row_index").to_pylist() == [0]


def test_degenerate_anchored_policy_restricted_to_two_surveys():
    with pytest.raises(ValueError, match="two-survey legacy policy"):
        DegenerateCrossmatchConfig(surveys=["a", "b", "c"], policy="anchored_nearest")
    with pytest.raises(ValueError, match="at least two surveys"):
        DegenerateCrossmatchConfig(surveys=["a"])
    with pytest.raises(ValueError, match="twice"):
        DegenerateCrossmatchConfig(surveys=["a", "a"])


# --- entitywise --------------------------------------------------------------


def test_entitywise_clean_pair_and_singletons():
    # a0--b0 clean pair; a1 and b1 isolated singletons.
    inputs = _inputs(
        {"a": 2, "b": 2},
        {("a", "b"): ([0], [0], [0.2])},
    )
    result = EntitywiseCrossmatchConfig().build(inputs)
    table = result.table
    assert table.num_rows == 3
    rows = set(zip(table.column("a/row_index").to_pylist(), table.column("b/row_index").to_pylist()))
    assert rows == {(0, 0), (1, None), (None, 1)}
    assert table.column("disputed_reason").to_pylist() == [None, None, None]
    assert result.summary["n_clean_multi_survey"] == 1


def test_entitywise_refuse_dissolves_flanked_component():
    # Two a-rows both within radius of one b-row: multiplicity, ambiguous.
    inputs = _inputs(
        {"a": 2, "b": 1},
        {("a", "b"): ([0, 1], [0, 0], [0.2, 0.4])},
    )
    result = EntitywiseCrossmatchConfig().build(inputs)
    table = result.table
    assert table.num_rows == 3  # three flagged singletons
    assert set(table.column("disputed_reason").to_pylist()) == {"ambiguous_component"}
    assert result.summary["n_ambiguous_components"] == 1
    assert result.summary["rows_in_ambiguous_components"] == 3


def test_entitywise_dedupe_disputed_rows_surface_as_flagged_singletons():
    inputs = _inputs(
        {"a": 3, "b": 1},
        {("a", "b"): ([0], [0], [0.1])},
        disputed={"a": [1, 2]},
    )
    result = EntitywiseCrossmatchConfig().build(inputs)
    table = result.table
    reasons = table.column("disputed_reason").to_pylist()
    assert reasons.count("dedupe_disputed") == 2
    dedupe_rows = [
        table.column("a/row_index").to_pylist()[k]
        for k, reason in enumerate(reasons)
        if reason == "dedupe_disputed"
    ]
    assert sorted(dedupe_rows) == [1, 2]

    dropped = EntitywiseCrossmatchConfig(disputed="drop").build(inputs)
    assert dropped.table.num_rows == 1  # only the clean pair survives
    assert dropped.table.column("disputed_reason").to_pylist() == [None]


def test_entitywise_split_resolver_partitions_chain():
    # Chain a0--b0 (0.2), b0--a1 (0.9): split keeps the tight pair, a1 singleton.
    inputs = _inputs(
        {"a": 2, "b": 1},
        {("a", "b"): ([0, 1], [0, 0], [0.2, 0.9])},
    )
    result = EntitywiseCrossmatchConfig(resolver="split").build(inputs)
    table = result.table
    assert result.summary["n_resolved_groups"] == 2
    rows = set(zip(table.column("a/row_index").to_pylist(), table.column("b/row_index").to_pylist()))
    assert rows == {(0, 0), (1, None)}
    assert table.column("disputed_reason").to_pylist() == [None, None]


def test_entitywise_sequential_resolver_uses_priority():
    # Same flanked geometry; priority decides that b attaches its row to the
    # nearest a-entity, leaving the other a-row a singleton entity.
    inputs = _inputs(
        {"a": 2, "b": 1},
        {("a", "b"): ([0, 1], [0, 0], [0.4, 0.2])},
    )
    result = EntitywiseCrossmatchConfig(
        resolver="sequential", priority=["a", "b"]
    ).build(inputs)
    rows = set(
        zip(
            result.table.column("a/row_index").to_pylist(),
            result.table.column("b/row_index").to_pylist(),
        )
    )
    assert rows == {(1, 0), (0, None)}

    with pytest.raises(ValueError, match="permutation"):
        EntitywiseCrossmatchConfig(resolver="sequential", priority=["a"]).build(inputs)


def test_entitywise_size_cap_dissolves_oversized_components():
    inputs = _inputs(
        {"a": 2, "b": 1},
        {("a", "b"): ([0, 1], [0, 0], [0.2, 0.4])},
    )
    result = EntitywiseCrossmatchConfig(resolver="split", size_cap=2).build(inputs)
    assert result.summary["n_resolved_groups"] == 0
    assert set(result.table.column("disputed_reason").to_pylist()) == {
        "ambiguous_component"
    }


def test_entitywise_selection_filters_after_resolution():
    inputs = _inputs(
        {"a": 2, "b": 2},
        {("a", "b"): ([0], [0], [0.2])},
    )
    result = EntitywiseCrossmatchConfig(
        selection=EntitySelectionConfig(min_surveys=2)
    ).build(inputs)
    assert result.table.num_rows == 1
    assert result.table.column("a/row_index").to_pylist() == [0]
    assert result.summary["n_entities_before_selection"] == 3

    must = EntitywiseCrossmatchConfig(
        selection=EntitySelectionConfig(must_include_surveys=["b"])
    ).build(inputs)
    rows = set(
        zip(
            must.table.column("a/row_index").to_pylist(),
            must.table.column("b/row_index").to_pylist(),
        )
    )
    assert rows == {(0, 0), (None, 1)}

    with pytest.raises(ValueError, match="unknown surveys"):
        EntitywiseCrossmatchConfig(
            selection=EntitySelectionConfig(must_include_surveys=["nope"])
        ).build(inputs)


def test_entitywise_absent_surveys_are_null_rows():
    inputs = _inputs({"a": 1, "b": 1}, {})
    table = EntitywiseCrossmatchConfig().build(inputs).table
    assert table.schema.field("a/row_index").type == pa.int64()
    rows = set(
        zip(
            table.column("a/row_index").to_pylist(),
            table.column("b/row_index").to_pylist(),
        )
    )
    assert rows == {(0, None), (None, 0)}


def test_entitywise_validation():
    with pytest.raises(ValueError, match="unknown resolver"):
        EntitywiseCrossmatchConfig(resolver="nope")
    with pytest.raises(ValueError, match="requires 'priority'"):
        EntitywiseCrossmatchConfig(resolver="sequential")
    with pytest.raises(ValueError, match="applies only to"):
        EntitywiseCrossmatchConfig(resolver="refuse", priority=["a"])
    with pytest.raises(ValueError, match="size_cap"):
        EntitywiseCrossmatchConfig(size_cap=0)
    with pytest.raises(ValueError, match="min_surveys"):
        EntitySelectionConfig(min_surveys=0)
    with pytest.raises(ValueError, match="duplicate"):
        EntitySelectionConfig(must_include_surveys=["a", "a"])
