"""Exact UID joins must never round integer keys or invent sky coordinates."""

import json

import numpy as np
import pyarrow as pa
import pytest

from astribidem import match_uids as match_uid_rows
from astribidem import rows_to_ids


def match_uids(uids, *, ids=None, **kwargs):
    """Rows mapped to observation IDs, or back to the UIDs when none are given.

    Most tests below assert typed-ID results; this composes them from the row
    output exactly as a caller would.
    """
    return rows_to_ids(match_uid_rows(uids, **kwargs), ids if ids is not None else uids)


@pytest.mark.parametrize(
    "join,expected",
    [
        ("inner", {"entity_id": [0], "a/id": ["a-one"], "b/id": [21], "c/id": [31]}),
        (
            "left",
            {
                "entity_id": [0, 1, 2],
                "a/id": ["a-two", "a-one", "a-four"],
                "b/id": [None, 21, None],
                "c/id": [32, 31, None],
            },
        ),
        (
            "outer",
            {
                "entity_id": [0, 1, 2, 3, 4],
                "a/id": ["a-two", "a-one", "a-four", None, None],
                "b/id": [None, 21, None, 23, None],
                "c/id": [32, 31, None, None, 35],
            },
        ),
    ],
)
def test_three_source_joins_preserve_order_and_distinct_observation_ids(join, expected):
    result = match_uids(
        {"a": [2, 1, 4], "b": [3, 1], "c": [1, 2, 5]},
        ids={
            "a": ["a-two", "a-one", "a-four"],
            "b": np.array([23, 21], dtype="int16"),
            "c": [31, 32, 35],
        },
        join=join,
    )
    assert result.to_pydict() == expected
    assert result["b/id"].type == pa.int16()
    metadata = json.loads(result.schema.metadata[b"astribidem.resolved_config"])
    assert metadata["method"] == "exact_uid"
    assert metadata["join"] == join
    assert metadata["anchor"] == "a"


def test_explicit_anchor_changes_rows_not_column_order():
    result = match_uids({"a": [1, 2], "b": [3, 2, 1]}, join="outer", anchor="b")
    assert result.to_pydict() == {
        "entity_id": [0, 1, 2],
        "a/id": [None, 2, 1],
        "b/id": [3, 2, 1],
    }


def test_signed_unsigned_uids_and_nullable_uint64_ids_are_exact():
    big = 2**63
    result = match_uids(
        {
            "a": pa.array([big + 1, big, 1], pa.uint64()),
            "b": pa.array([-1, 1], pa.int64()),
        },
        join="outer",
    )
    assert result.to_pydict() == {
        "entity_id": [0, 1, 2, 3],
        "a/id": [big + 1, big, 1, None],
        "b/id": [None, None, 1, -1],
    }
    assert result["a/id"].type == pa.uint64()
    assert result["b/id"].type == pa.int64()
    assert match_uids(
        {"a": [2**64 - 1, 2**64 - 2], "b": np.array([2**64 - 2], dtype="uint64")}
    )["a/id"].to_pylist() == [2**64 - 2]


def test_string_matching_is_exact_and_accepts_chunked_large_strings():
    result = match_uids(
        {
            "a": pa.chunked_array([["A", "001"], ["é"]], type=pa.large_string()),
            "b": ["a", "1", "é"],
        },
    )
    assert result["a/id"].to_pylist() == ["é"]
    assert result["a/id"].type == pa.large_string()


@pytest.mark.parametrize("join", ["inner", "left", "outer"])
def test_empty_sources_retain_output_types(join):
    result = match_uids({"a": pa.array([], pa.string()), "b": ["x"]}, join=join)
    assert result["a/id"].type == pa.string()
    assert result.to_pydict() == (
        {"entity_id": [0], "a/id": [None], "b/id": ["x"]}
        if join == "outer"
        else {"entity_id": [], "a/id": [], "b/id": []}
    )


@pytest.mark.parametrize(
    "uids,kwargs,message",
    [
        ({}, {}, "at least one"),
        ({"a/b": [1]}, {}, "source names"),
        ({"a": [1, 1]}, {}, "duplicate"),
        ({"a": [1, None]}, {}, "non-null"),
        ({"a": [1.0]}, {}, "integer or string"),
        ({"a": [True]}, {}, "integer or string"),
        ({"a": [1], "b": ["1"]}, {}, "no coercion"),
        ({"a": [1]}, {"join": "right"}, "join must"),
        ({"a": [1]}, {"anchor": "b"}, "anchor must"),
        ({"a": [-1, 2**63]}, {}, "fit one Arrow integer type"),
    ],
)
def test_invalid_or_ambiguous_input_is_rejected(uids, kwargs, message):
    with pytest.raises(ValueError, match=message):
        match_uids(uids, **kwargs)


def test_empty_python_sequence_infers_other_source_key_type():
    # An untyped empty list must not be taken as integers and clash with strings.
    result = match_uid_rows({"a": [], "b": ["x"]}, join="outer")
    assert result.to_pydict() == {
        "entity_id": [0],
        "a/row_index": [None],
        "b/row_index": [0],
    }
    assert match_uid_rows({"a": [], "b": []}).num_rows == 0


def test_rows_are_int64_positions():
    result = match_uid_rows(
        {"a": [2, 1, 4], "b": np.array([3, 1], dtype="int8")}, join="outer"
    )
    assert result.to_pydict() == {
        "entity_id": [0, 1, 2, 3],
        "a/row_index": [0, 1, 2, None],
        "b/row_index": [None, 1, None, 0],
    }
    assert result.schema.field("b/row_index").type == pa.int64()


def test_python_integer_below_int64_is_rejected_clearly():
    with pytest.raises(ValueError, match="fit one Arrow integer type"):
        match_uids({"a": [-(2**63) - 1]})


def _reference_join(keys, ids, *, join, anchor):
    """Small, deliberately brute-force oracle using Python's exact equality."""
    ordered = list(keys[anchor])
    if join == "inner":
        ordered = [
            key for key in ordered if all(key in values for values in keys.values())
        ]
    elif join == "outer":
        for values in keys.values():
            for key in values:
                if key not in ordered:
                    ordered.append(key)
    return {
        "entity_id": list(range(len(ordered))),
        **{
            f"{name}/id": [
                ids[name][values.index(key)] if key in values else None
                for key in ordered
            ]
            for name, values in keys.items()
        },
    }


@pytest.mark.parametrize("join", ["inner", "left", "outer"])
@pytest.mark.parametrize("anchor", ["signed", "unsigned", "small"])
def test_three_way_integer_extremes_do_not_round_or_wrap(join, anchor):
    keys = {
        "signed": [-(2**63), -1, 2**53, 2**53 + 1, 2**63 - 1, 0, 42],
        "unsigned": [2**64 - 1, 2**63, 2**63 - 1, 2**53 + 1, 2**53 + 2, 0, 42],
        "small": [42, 0, -1, 32767],
    }
    ids = {
        "signed": pa.array(range(7), pa.int8()),
        "unsigned": pa.array([2**64 - 1 - index for index in range(7)], pa.uint64()),
        "small": pa.array(["forty-two", "zero", "minus-one", "max-small"]),
    }
    result = match_uids(
        {
            "signed": pa.array(keys["signed"], pa.int64()),
            "unsigned": pa.array(keys["unsigned"], pa.uint64()),
            "small": pa.array(keys["small"], pa.int16()),
        },
        ids=ids,
        join=join,
        anchor=anchor,
    )
    assert result.to_pydict() == _reference_join(
        keys,
        {name: values.to_pylist() for name, values in ids.items()},
        join=join,
        anchor=anchor,
    )
    assert [result[f"{name}/id"].type for name in keys] == [
        pa.int8(),
        pa.uint64(),
        pa.string(),
    ]


@pytest.mark.parametrize("join", ["inner", "left", "outer"])
@pytest.mark.parametrize("bits", [8, 16, 32, 64])
def test_integer_width_boundaries_remain_distinct(bits, join):
    maximum = 2 ** (bits - 1) - 1
    keys = {
        "signed": [-maximum - 1, -1, 0, maximum],
        "unsigned": [0, maximum, maximum + 1, 2 * maximum + 1],
    }
    result = match_uids(
        {
            "signed": pa.array(keys["signed"], pa.type_for_alias(f"int{bits}")),
            "unsigned": pa.array(keys["unsigned"], pa.type_for_alias(f"uint{bits}")),
        },
        join=join,
    )
    assert result.to_pydict() == _reference_join(keys, keys, join=join, anchor="signed")
    assert result["signed/id"].type == pa.type_for_alias(f"int{bits}")
    assert result["unsigned/id"].type == pa.type_for_alias(f"uint{bits}")


@pytest.mark.parametrize("join", ["inner", "left", "outer"])
@pytest.mark.parametrize("anchor", ["empty", "unsigned"])
def test_typed_empty_integer_input_does_not_force_unsigned_overflow(join, anchor):
    keys = {"empty": [], "unsigned": [2**64 - 1, 2**63, 0]}
    result = match_uids(
        {
            "empty": pa.chunked_array([[], []], type=pa.int8()),
            "unsigned": pa.chunked_array(
                [[2**64 - 1], [], [2**63, 0]], type=pa.uint64()
            ),
        },
        join=join,
        anchor=anchor,
    )
    assert result.to_pydict() == _reference_join(keys, keys, join=join, anchor=anchor)
    assert result["empty/id"].type == pa.int8()
    assert result["unsigned/id"].type == pa.uint64()


@pytest.mark.parametrize("join", ["inner", "left", "outer"])
@pytest.mark.parametrize("anchor", ["a", "b", "c"])
@pytest.mark.parametrize("family", ["integer", "string"])
def test_randomized_three_source_joins_match_exact_reference(join, anchor, family):
    rng = np.random.default_rng(1024)
    for _ in range(12):
        keys = {}
        arrays = {}
        ids = {}
        for index, name in enumerate(("a", "b", "c")):
            selected = rng.choice(20, size=int(rng.integers(0, 15)), replace=False)
            values = [int(value) for value in selected]
            if family == "string":
                values = [f"é-{value:03d}" for value in values]
                dtype = pa.large_string() if index == 1 else pa.string()
            else:
                dtype = (pa.int8(), pa.uint16(), pa.int64())[index]
            keys[name] = values
            midpoint = len(values) // 2
            arrays[name] = pa.chunked_array(
                [[], values[:midpoint], [], values[midpoint:]], type=dtype
            )
            ids[name] = [f"{name}-{row}" for row in range(len(values))]
        result = match_uids(
            arrays,
            ids={name: pa.array(values, pa.string()) for name, values in ids.items()},
            join=join,
            anchor=anchor,
        )
        assert result.to_pydict() == _reference_join(
            keys, ids, join=join, anchor=anchor
        )


@pytest.mark.parametrize("invalid", ["duplicates", "nulls"])
def test_chunked_invalid_uids_are_checked_across_chunk_boundaries(invalid):
    values = pa.chunked_array(
        [["a", "b"], [], ["a" if invalid == "duplicates" else None]],
        type=pa.large_string(),
    )
    message = "duplicate" if invalid == "duplicates" else "non-null"
    with pytest.raises(ValueError, match=message):
        match_uids({"source": values})


def test_python_integer_above_uint64_is_rejected_clearly():
    with pytest.raises(ValueError, match="fit one Arrow integer type"):
        match_uids({"a": [2**64]})


def test_outer_late_source_matches_prior_non_anchor_keys_across_empty_source():
    keys = {
        "first": ["70", "10"],
        "anchor": ["20", "10"],
        "empty": [],
        "last": ["70", "30", "20"],
    }
    ids = {
        name: [f"{name}-{row}" for row in range(len(values))]
        for name, values in keys.items()
    }
    result = match_uids(
        {
            name: pa.chunked_array([[], values[:1], [], values[1:]], pa.string())
            for name, values in keys.items()
        },
        ids={name: pa.array(values, pa.string()) for name, values in ids.items()},
        join="outer",
        anchor="anchor",
    )
    assert result.to_pydict() == _reference_join(
        keys, ids, join="outer", anchor="anchor"
    )


@pytest.mark.parametrize("last", [[7], []])
def test_four_source_inner_successive_reductions_keep_original_observations(last):
    keys = {"a": [9, 1, 7, 3], "b": [7, 1, 8], "c": [1, 7], "d": last}
    ids = {
        "a": [40, 10, 30, 20],
        "b": [103, 101, 102],
        "c": [202, 201],
        "d": [301] if last else [],
    }
    result = match_uids(keys, ids=ids)
    assert result.to_pydict() == _reference_join(keys, ids, join="inner", anchor="a")


@pytest.mark.parametrize("join", ["inner", "left", "outer"])
@pytest.mark.parametrize("values", [["z", "é", "a\0"], []])
def test_single_source_chunked_keys_keep_order_and_separate_typed_ids(join, values):
    identifiers = pa.array([2**64 - 1 - row for row in range(len(values))], pa.uint64())
    result = match_uids(
        {"only": pa.chunked_array([[], values[:1], [], values[1:]], pa.large_string())},
        ids={"only": identifiers},
        join=join,
    )
    expected = pa.table(
        {
            "entity_id": pa.array(range(len(values)), pa.int64()),
            "only/id": identifiers,
        }
    )
    assert result.equals(expected)
