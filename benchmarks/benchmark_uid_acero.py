"""Two-source Arrow UID prototypes; no production matcher is replaced.

Run with --module pointing to astribidem/uids.py. Each measurement runs
in a fresh, sequential process; all strategies use the same native validation
and final Arrow output. No pandas or Polars imports are required.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import platform
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.acero as acero
import pyarrow.compute as pc


def load_baseline(path):
    spec = importlib.util.spec_from_file_location("baseline_uids", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def array_view(array):
    if isinstance(array, pa.ChunkedArray):
        array = array.combine_chunks()
    return array.to_numpy(zero_copy_only=True)


def metadata(keys, ids, join, anchor):
    config = {
        "method": "exact_uid",
        "join": join,
        "anchor": anchor,
        "surveys": [
            {"name": name, "uid_type": str(value.type), "id_type": str(ids[name].type)}
            for name, value in keys.items()
        ],
        "duplicate_keys": "reject",
        "null_keys": "reject",
        "ordering": "anchor_then_source_input_order",
    }
    return {
        b"astribidem.resolved_config": json.dumps(
            config, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    }


def finish(keys, ids, rows, join, anchor):
    columns = {"entity_id": pa.array(np.arange(len(rows[anchor]), dtype=np.int64))}
    for name in keys:
        columns[f"{name}/id"] = pc.take(ids[name], rows[name])
    return pa.table(columns).replace_schema_metadata(metadata(keys, ids, join, anchor))


def match_pair(
    baseline, keys, *, strategy, threads=1, join="inner", ids=None, anchor=None
):
    """Small two-source prototype of the typed-ID output contract.

    ``current`` gathers IDs from the committed matcher's row output, so every
    strategy is compared on the same typed-ID table.
    """
    if len(keys) != 2 or join not in ("inner", "left", "outer"):
        raise ValueError("prototype requires two sources and a supported join")
    names = list(keys)
    anchor = names[0] if anchor is None else anchor
    if anchor not in keys:
        raise ValueError("invalid anchor")
    other = next(name for name in names if name != anchor)
    keys = {
        name: baseline._identifiers(value, name=name, kind="UIDs")
        for name, value in keys.items()
    }
    if (
        len({"int" if pa.types.is_integer(v.type) else "str" for v in keys.values()})
        != 1
    ):
        raise ValueError("incompatible key families")
    if ids is None:
        ids = keys
    else:
        if set(ids) != set(keys):
            raise ValueError("IDs must name every source")
        ids = {
            name: baseline._identifiers(value, name=name, kind="IDs")
            for name, value in ids.items()
        }
        if any(len(ids[name]) != len(keys[name]) for name in keys):
            raise ValueError("IDs and keys must have equal lengths")
    dtype = baseline._common_key_type(keys)
    values = {name: pc.cast(value, dtype) for name, value in keys.items()}
    a, b = values[anchor], values[other]
    a_rows = pa.array(np.arange(len(a), dtype=np.int64))

    if strategy == "current":
        result = baseline.match_uids(keys, anchor=anchor, join=join)
        rows = {name: result[f"{name}/row_index"] for name in keys}
    elif strategy == "lookup":
        b_rows = pc.index_in(a, value_set=b)
        valid = pc.is_valid(b_rows)
        if join == "inner":
            a_rows, b_rows = pc.filter(a_rows, valid), pc.filter(b_rows, valid)
        elif join == "outer":
            unseen = np.ones(len(b), dtype=np.bool_)
            unseen[array_view(pc.filter(b_rows, valid))] = False
            extras = pa.array(np.flatnonzero(unseen))
            a_rows = pa.chunked_array([a_rows, pa.nulls(len(extras), pa.int64())])
            b_rows = pc.cast(b_rows, pa.int64())
            chunks = (
                list(b_rows.chunks) if isinstance(b_rows, pa.ChunkedArray) else [b_rows]
            )
            b_rows = pa.chunked_array([*chunks, extras])
        rows = {anchor: a_rows, other: b_rows}
    else:
        left = pa.table({"key": a, "a_row": a_rows})
        right = pa.table(
            {"key": b, "b_row": pa.array(np.arange(len(b), dtype=np.int64))}
        )
        join_type = {"inner": "inner", "left": "left outer", "outer": "full outer"}[
            join
        ]
        if strategy == "table":
            joined = left.join(
                right, keys="key", join_type=join_type, use_threads=threads > 1
            ).select(["a_row", "b_row"])
        elif strategy == "acero":
            sources = [
                acero.Declaration("table_source", acero.TableSourceNodeOptions(t))
                for t in (left, right)
            ]
            options = acero.HashJoinNodeOptions(
                join_type,
                ["key"],
                ["key"],
                left_output=["a_row"],
                right_output=["b_row"],
            )
            joined = acero.Declaration("hashjoin", options, inputs=sources).to_table(
                use_threads=threads > 1
            )
            assert joined.column_names == ["a_row", "b_row"]
        else:
            raise ValueError("unknown strategy")
        # Nulls default to the end in both supported Arrow versions.
        order = pc.sort_indices(
            joined, sort_keys=[("a_row", "ascending"), ("b_row", "ascending")]
        )
        joined = joined.take(order)
        rows = {anchor: joined["a_row"], other: joined["b_row"]}
    return finish(keys, ids, rows, join, anchor)


def check(baseline, threads):
    cases = [
        {"a": pa.array([9, 1, 7]), "b": pa.array([7, 9, 2])},
        {"a": pa.array(["α", "a\0", "a", "é"]), "b": pa.array(["a", "α", "b", "a\0"])},
        {"a": pa.array(["b", "a"], pa.large_string()), "b": pa.array(["b", "c"])},
        {"a": pa.array([], pa.string()), "b": pa.array([], pa.string())},
        {"a": pa.array([], pa.string()), "b": pa.array(["b", "a"])},
        {"a": pa.array(["b", "a"]), "b": pa.array([], pa.string())},
        {"a": pa.array(["b", "a"]), "b": pa.array(["c", "d"])},
        {
            "a": pa.array([-(2**63), 2**63 - 1, 0], pa.int64()),
            "b": pa.array([2**63, 0, 2**64 - 1, 2**63 - 1], pa.uint64()),
        },
        {"a": pa.array([2**63], pa.uint64()), "b": pa.array([2**63 - 1], pa.int64())},
        {
            "a": pa.chunked_array([[], ["α"], [], ["b", "a"]], pa.string()),
            "b": pa.chunked_array([["a", "c"], [], ["α"]], pa.string()),
        },
        {
            "a": pa.chunked_array([], pa.int64()),
            "b": pa.chunked_array([[], [2, 1], []], pa.int64()),
        },
    ]
    count = 0
    for strategy in ("lookup", "table", "acero"):
        for keys in cases:
            for join in ("inner", "left", "outer"):
                for anchor in keys:
                    for separate_ids in (False, True):
                        ids = (
                            None
                            if not separate_ids
                            else {
                                name: (
                                    pc.cast(
                                        pa.array(np.arange(len(value))), pa.string()
                                    )
                                    if name == "a"
                                    else pa.array(np.arange(len(value)), pa.uint64())
                                )
                                for name, value in keys.items()
                            }
                        )
                        got = match_pair(
                            baseline,
                            keys,
                            strategy=strategy,
                            threads=threads,
                            join=join,
                            anchor=anchor,
                            ids=ids,
                        )
                        expected = match_pair(
                            baseline,
                            keys,
                            strategy="current",
                            join=join,
                            anchor=anchor,
                            ids=ids,
                        )
                        assert got.equals(expected, check_metadata=True), (
                            strategy,
                            join,
                            anchor,
                        )
                        count += 1
        for a, b in [
            (["x", "x"], ["y"]),
            (["x", "x"], []),
            ([], ["x", "x"]),
            ([None], ["x"]),
        ]:
            for join in ("inner", "left", "outer"):
                keys = {"a": pa.array(a, pa.string()), "b": pa.array(b, pa.string())}
                try:
                    match_pair(
                        baseline, keys, strategy=strategy, threads=threads, join=join
                    )
                except ValueError:
                    pass
                else:
                    raise AssertionError((strategy, join, "accepted invalid keys"))
                count += 1
    return count


def fixture(kind, n):
    rng = np.random.default_rng(20260917)
    ranks = {
        "a": rng.permutation(np.arange(n, dtype=np.int64)),
        "b": rng.permutation(np.arange(n // 2, n + n // 2, dtype=np.int64)),
    }
    keys = {
        name: pa.array(np.int64(2**60) + values * np.int64(104729))
        for name, values in ranks.items()
    }
    if kind == "string":
        keys = {name: pc.cast(value, pa.string()) for name, value in keys.items()}
    return keys, ranks


def validate(result, keys, ranks, n, join):
    if join == "inner":
        order = ranks["a"][ranks["a"] >= n // 2]
    else:
        order = np.concatenate([ranks["a"], ranks["b"][ranks["b"] >= n]])
    rows = {}
    for name, values in ranks.items():
        inverse = np.full(n + n // 2, -1, dtype=np.int64)
        inverse[values] = np.arange(n, dtype=np.int64)
        selected = inverse[order]
        rows[name] = pa.array(selected, mask=selected < 0)
    expected = finish(keys, keys, rows, join, "a")
    assert result.equals(expected, check_metadata=True), (
        "result differs from rank oracle"
    )


def peak_bytes():
    n = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return n if sys.platform == "darwin" else n * 1024


def worker(args):
    pa.set_cpu_count(args.threads)
    baseline = load_baseline(args.module)
    if args.check_only:
        return {
            "checks_passed": check(baseline, args.threads),
            "threads": args.threads,
            "pyarrow": pa.__version__,
        }
    warm, _ = fixture(args.kind, 128)
    match_pair(
        baseline, warm, strategy=args.strategy, threads=args.threads, join=args.join
    )
    keys, ranks = fixture(args.kind, args.rows)
    del warm
    gc.collect()
    before = peak_bytes()
    start = time.perf_counter()
    result = match_pair(
        baseline, keys, strategy=args.strategy, threads=args.threads, join=args.join
    )
    elapsed = time.perf_counter() - start
    after = peak_bytes()
    validate(result, keys, ranks, args.rows, args.join)
    return {
        "kind": args.kind,
        "join": args.join,
        "strategy": args.strategy,
        "threads": args.threads,
        "rows_per_source": args.rows,
        "seconds": elapsed,
        "baseline_rss_mb": before / 1e6,
        "peak_rss_mb": after / 1e6,
        "extra_peak_rss_mb": (after - before) / 1e6,
        "output_rows": result.num_rows,
        "validated": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--rows", type=int, default=500_000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--kind", choices=["int64", "string"])
    parser.add_argument("--join", choices=["inner", "outer"])
    parser.add_argument("--strategy", choices=["current", "lookup", "table", "acero"])
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--compact", action="store_true", help="Only lookup and 4-thread joins"
    )
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(args)))
        return

    def invoke(extra):
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--module",
            str(args.module.resolve()),
            *extra,
        ]
        proc = subprocess.run(command, text=True, capture_output=True, check=False)
        if proc.returncode:
            raise RuntimeError(proc.stderr + proc.stdout)
        return json.loads(proc.stdout)

    checks = [invoke(["--check-only", "--threads", str(t)]) for t in (1, 4)]
    print(json.dumps({"semantic_checks": checks}), flush=True)
    if args.check_only:
        return
    strategies = (
        [("lookup", 1), ("table", 4), ("acero", 4)]
        if args.compact
        else [
            ("current", 1),
            ("lookup", 1),
            ("table", 1),
            ("table", 4),
            ("acero", 1),
            ("acero", 4),
        ]
    )
    configurations = [
        (kind, join, strategy, threads)
        for kind in ("int64", "string")
        for join in ("inner", "outer")
        for strategy, threads in strategies
    ]
    samples = []
    # Rotate strategies between repeats to reduce fixed-order scheduling bias.
    for repeat in range(args.repeats):
        for kind, join, strategy, threads in (
            configurations[repeat:] + configurations[:repeat]
        ):
            sample = invoke(
                [
                    "--kind",
                    kind,
                    "--join",
                    join,
                    "--strategy",
                    strategy,
                    "--threads",
                    str(threads),
                    "--rows",
                    str(args.rows),
                ]
            )
            sample["repeat"] = repeat + 1
            samples.append(sample)
        print(
            json.dumps({"repeat_completed": repeat + 1, "measurements": len(samples)}),
            flush=True,
        )
    summaries = []
    for kind, join, strategy, threads in configurations:
        selected = [
            s
            for s in samples
            if (s["kind"], s["join"], s["strategy"], s["threads"])
            == (kind, join, strategy, threads)
        ]
        row = {
            "kind": kind,
            "join": join,
            "strategy": strategy,
            "threads": threads,
            "rows_per_source": args.rows,
            "repeats": args.repeats,
        }
        for field in ("seconds", "baseline_rss_mb", "peak_rss_mb", "extra_peak_rss_mb"):
            row[f"median_{field}"] = statistics.median(s[field] for s in selected)
        summaries.append(row)
        print(json.dumps(row), flush=True)
    report = {
        "summary": summaries,
        "measurements": samples,
        "semantic_checks": checks,
        "python": sys.version,
        "numpy": np.__version__,
        "pyarrow": pa.__version__,
        "platform": platform.platform(),
        "module": str(args.module.resolve()),
        "notes": "Two sources, shuffled 50% overlap, sparse int64 >2**60 or 19-digit UTF8 strings. Full validation, normalization, row matching, ordering and Arrow output timed. Sequential isolated warmed processes, all with same Arrow/NumPy imports; no pandas or Polars. Extra RSS is difference between high-water marks after input creation and matching, not exact allocations. Output checked against rank oracle outside measured region. Prototypes are two-source only; current strategy is committed production code.",
    }
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
