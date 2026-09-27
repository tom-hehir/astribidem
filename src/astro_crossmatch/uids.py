"""Exact UID joins with typed, stable source-observation identifiers."""

from __future__ import annotations

import json
from collections.abc import Mapping
from numbers import Integral
from typing import Any

import numpy as np
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


def _common_key_type(keys: Mapping[str, pa.Array | pa.ChunkedArray]) -> pa.DataType:
    """Choose a lossless native representation for Arrow equality kernels."""
    types = {array.type for array in keys.values()}
    if all(pa.types.is_integer(dtype) for dtype in types):
        signed = [
            dtype.bit_width for dtype in types if pa.types.is_signed_integer(dtype)
        ]
        unsigned = [
            dtype.bit_width for dtype in types if pa.types.is_unsigned_integer(dtype)
        ]
        if not signed or not unsigned:
            return max(types, key=lambda dtype: dtype.bit_width)
        if max(unsigned) < 64:
            # The next signed width holds every value of a smaller unsigned
            # type. Never let a signed/unsigned comparison fall back to float.
            return getattr(pa, f"int{max(max(signed), 2 * max(unsigned))}")()
        # No native 64-bit integer holds both negative int64 and full uint64.
        # Decimal128 with scale zero is a fixed-width, exact integer encoding;
        # casting and hashing stay in Arrow, including for chunked input.
        return pa.decimal128(20, 0)
    return pa.large_string() if pa.large_string() in types else pa.string()


def _append_arrays(
    existing: pa.Array | pa.ChunkedArray, extra: pa.Array | pa.ChunkedArray
) -> pa.ChunkedArray:
    """Append native buffers without copying accumulated keys or row positions."""
    existing_chunks = (
        existing.chunks if isinstance(existing, pa.ChunkedArray) else [existing]
    )
    extra_chunks = extra.chunks if isinstance(extra, pa.ChunkedArray) else [extra]
    return pa.chunked_array([*existing_chunks, *extra_chunks], type=existing.type)


def _unmatched_rows(
    matched: pa.Array | pa.ChunkedArray, source_length: int
) -> pa.Array:
    """Find unseen source rows using the existing lookup, without hashing again."""
    unseen = np.ones(source_length, dtype=np.bool_)
    # Only non-null integer row positions enter NumPy, never UID values.
    unseen[pc.drop_null(matched).to_numpy(zero_copy_only=False)] = False
    return pa.array(np.flatnonzero(unseen), type=matched.type)


def match_uids(
    uids: Mapping[str, Any],
    *,
    join: str = "inner",
    anchor: str | None = None,
) -> pa.Table:
    """Join equal UIDs; return ``entity_id`` and ``<source>/row_index`` columns.

    Each source must have unique, non-null integer or string UIDs. All sources
    must use the same UID family, but integer widths/signedness and string
    widths may differ. Integer comparison is exact, including uint64. Strings
    are case-sensitive and are not normalized or coerced to integers.

    Row ``i`` of a source is its ``i``-th UID; a null row means the source is
    absent from that entity (outer and left joins). Map rows to observation
    IDs, or back to the UIDs, with ``rows_to_ids``.

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

    # Normalize explicitly before calling equality kernels: their implicit
    # signed/unsigned coercion need not preserve full-range integer identity.
    # Only per-source metadata is held in Python; keys and lookups stay native.
    key_type = _common_key_type(keys)
    comparable = {name: pc.cast(array, key_type) for name, array in keys.items()}
    ordered = comparable[anchor]
    row_maps = {anchor: pa.array(np.arange(len(ordered), dtype=np.int64))}
    others = [name for name in names if name != anchor]
    for index, name in enumerate(others):
        # One lookup per non-anchor source supplies both membership and rows.
        matched = pc.index_in(ordered, value_set=comparable[name])
        more_sources = index + 1 < len(others)
        if join == "inner":
            keep = pc.is_valid(matched)
            for source, rows in row_maps.items():
                row_maps[source] = pc.filter(rows, keep)
            del rows
            matched = pc.filter(matched, keep)
            if more_sources:
                ordered = pc.filter(ordered, keep)
            del keep
        elif join == "outer":
            extra_rows = _unmatched_rows(matched, len(comparable[name]))
            if len(extra_rows):
                if more_sources:
                    ordered = _append_arrays(
                        ordered, pc.take(comparable[name], extra_rows)
                    )
                # Share null padding across maps of the same integer width.
                padding = {
                    dtype: pa.nulls(len(extra_rows), type=dtype)
                    for dtype in {rows.type for rows in row_maps.values()}
                }
                row_maps = {
                    source: _append_arrays(rows, padding[rows.type])
                    for source, rows in row_maps.items()
                }
                matched = _append_arrays(matched, extra_rows)
                del padding
            del extra_rows
        row_maps[name] = matched
        del matched

    del comparable, ordered
    columns = {"entity_id": pa.array(np.arange(len(row_maps[anchor]), dtype=np.int64))}
    for name in names:
        columns[f"{name}/row_index"] = pc.cast(row_maps.pop(name), pa.int64())
    provenance = {
        "method": "exact_uid",
        "join": join,
        "anchor": anchor,
        "surveys": [{"name": name, "uid_type": str(keys[name].type)} for name in names],
        "duplicate_keys": "reject",
        "null_keys": "reject",
        "ordering": "anchor_then_source_input_order",
    }
    metadata = {
        b"astro_crossmatch.resolved_config": json.dumps(
            provenance, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode(),
        b"astro_crossmatch.n_rows": json.dumps(
            {name: len(keys[name]) for name in names}, sort_keys=True
        ).encode(),
    }
    return pa.table(columns).replace_schema_metadata(metadata)
