"""Exact UID joins with typed, stable source-observation identifiers."""

from __future__ import annotations

from collections.abc import Mapping
import json
from numbers import Integral
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc


def _identifiers(values: Any, *, name: str, kind: str) -> pa.Array | pa.ChunkedArray:
    """Keep integer identity exact, including unsigned values above int64."""
    if isinstance(values, (pa.Array, pa.ChunkedArray)):
        array = values
    else:
        # Arrow infers Python integers as int64. Explicitly select uint64 for
        # ordinary integer sequences that exceed int64, without a NumPy/float
        # round trip. Typed NumPy and pandas inputs retain their own dtype.
        if isinstance(values, (list, tuple)) and all(
            isinstance(value, Integral) and not isinstance(value, bool)
            for value in values
        ):
            dtype = pa.int64()
            if values and min(values) < -(2**63):
                raise ValueError(
                    f"source {name!r}: {kind} must fit one Arrow integer type"
                )
            if values and max(values) > 2**63 - 1:
                if min(values) < 0 or max(values) > 2**64 - 1:
                    raise ValueError(
                        f"source {name!r}: {kind} must fit one Arrow integer type"
                    )
                dtype = pa.uint64()
            array = pa.array(values, type=dtype)
        else:
            array = pa.array(values, from_pandas=True)
    if array.null_count:
        raise ValueError(f"source {name!r}: {kind} must be non-null")
    if not (
        pa.types.is_integer(array.type)
        or pa.types.is_string(array.type)
        or pa.types.is_large_string(array.type)
    ):
        raise ValueError(f"source {name!r}: {kind} must be integer or string values")
    if pc.count_distinct(array).as_py() != len(array):
        raise ValueError(f"source {name!r}: {kind} must be unique; duplicate values")
    return array


def match_uids(
    uids: Mapping[str, Any],
    *,
    ids: Mapping[str, Any] | None = None,
    join: str = "inner",
    anchor: str | None = None,
) -> pa.Table:
    """Join equal UIDs and return ``entity_id`` and ``<source>/id`` columns.

    Each source must have unique, non-null integer or string UIDs. All sources
    must use the same UID family, but integer widths/signedness and string
    widths may differ. Integer comparison is exact, including uint64. Strings
    are case-sensitive and are not normalized or coerced to integers.

    ``ids`` optionally supplies distinct unique source-observation identifiers;
    otherwise each source's UIDs are its identifiers. Output identifiers keep
    their Arrow types, including nullable outer/left memberships.

    The anchor defaults to the first mapping entry. Inner and left joins follow
    its input order. Outer joins start with the anchor and append unseen UIDs
    in the other sources' mapping/input order. Column order follows ``uids``.
    Entity IDs are consecutive int64 values in that deterministic result order.
    This in-memory join intentionally rejects duplicate keys rather than
    silently choosing an observation or producing a Cartesian product.
    """
    if not isinstance(uids, Mapping) or not uids:
        raise ValueError("uids must map at least one source name to UID values")
    names = tuple(uids)
    if any(not isinstance(name, str) or not name or "/" in name for name in names):
        raise ValueError("source names must be non-empty and must not contain '/'")
    if join not in ("inner", "left", "outer"):
        raise ValueError("join must be 'inner', 'left', or 'outer'")
    if anchor is None:
        anchor = names[0]
    if anchor not in uids:
        raise ValueError("anchor must name an input source")
    if ids is not None and (not isinstance(ids, Mapping) or set(ids) != set(names)):
        raise ValueError("ids must name every input source exactly once")

    keys = {name: _identifiers(uids[name], name=name, kind="UIDs") for name in names}
    # An untyped empty Python sequence has no key family. Infer it from another
    # source so {"empty": [], "strings": ["x"]} remains a valid empty/outer join.
    untyped_empty = {
        name
        for name in names
        if isinstance(uids[name], (list, tuple)) and not uids[name]
    }
    typed_keys = [array for name, array in keys.items() if name not in untyped_empty]
    inferred_type = typed_keys[0].type if typed_keys else pa.int64()
    for name in untyped_empty:
        keys[name] = pa.array([], type=inferred_type)
    families = {
        "integer" if pa.types.is_integer(array.type) else "string"
        for array in keys.values()
    }
    if len(families) != 1:
        raise ValueError("UID types must all be integer or all be string; no coercion")
    identifiers = (
        keys
        if ids is None
        else {name: _identifiers(ids[name], name=name, kind="IDs") for name in names}
    )
    for name in names:
        if len(identifiers[name]) != len(keys[name]):
            raise ValueError(f"source {name!r}: IDs and UIDs must have the same length")

    # Python ints preserve all Arrow signed/unsigned integer values. Arrow hash
    # joins can coerce mixed signedness to a lossy common representation.
    positions = {
        name: {value: row for row, value in enumerate(keys[name].to_pylist())}
        for name in names
    }
    ordered = list(positions[anchor])
    if join == "inner":
        ordered = [
            value
            for value in ordered
            if all(value in positions[name] for name in names)
        ]
    elif join == "outer":
        seen = set(ordered)
        for name in names:
            if name == anchor:
                continue
            for value in positions[name]:
                if value not in seen:
                    ordered.append(value)
                    seen.add(value)
    columns = {"entity_id": pa.array(range(len(ordered)), type=pa.int64())}
    for name in names:
        lookup = positions[name]
        rows = pa.array([lookup.get(value) for value in ordered], type=pa.int64())
        columns[f"{name}/id"] = pc.take(identifiers[name], rows)
    provenance = {
        "method": "exact_uid",
        "join": join,
        "anchor": anchor,
        "surveys": [
            {
                "name": name,
                "uid_type": str(keys[name].type),
                "id_type": str(identifiers[name].type),
            }
            for name in names
        ],
        "duplicate_keys": "reject",
        "null_keys": "reject",
        "ordering": "anchor_then_source_input_order",
    }
    metadata = {
        b"astro_crossmatch.resolved_config": json.dumps(
            provenance, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    }
    return pa.table(columns).replace_schema_metadata(metadata)
