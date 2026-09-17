"""Exact UID joins must never round integer keys or invent sky coordinates."""

import json

import numpy as np
import pyarrow as pa
import pytest

from astro_crossmatch import match_uids


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
    metadata = json.loads(result.schema.metadata[b"astro_crossmatch.resolved_config"])
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
        ({"a": [1]}, {"ids": {"b": [1]}}, "every input source"),
        ({"a": [1, 2]}, {"ids": {"a": [10, 10]}}, "duplicate"),
        ({"a": [1]}, {"ids": {"a": [None]}}, "non-null"),
        ({"a": [1, 2]}, {"ids": {"a": [10]}}, "same length"),
        ({"a": [-1, 2**63]}, {}, "fit one Arrow integer type"),
    ],
)
def test_invalid_or_ambiguous_input_is_rejected(uids, kwargs, message):
    with pytest.raises(ValueError, match=message):
        match_uids(uids, **kwargs)


def test_empty_python_sequence_infers_other_source_key_type():
    result = match_uids({"a": [], "b": ["x"]}, join="outer")
    assert result.to_pydict() == {"entity_id": [0], "a/id": [None], "b/id": ["x"]}
    assert result["a/id"].type == pa.string()
    assert match_uids({"a": [], "b": []}).num_rows == 0


def test_python_integer_below_int64_is_rejected_clearly():
    with pytest.raises(ValueError, match="fit one Arrow integer type"):
        match_uids({"a": [-(2**63) - 1]})
