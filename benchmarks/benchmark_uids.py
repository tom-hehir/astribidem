"""Compare UID implementations in sequential, isolated worker processes.

Example:
    python benchmark_uids.py --module src/astribidem/uids.py --label arrays

Inputs and correctness checks are outside the timed region. Every worker warms
up its implementation first, builds typed arrays, then measures one match call.
Peak RSS increment is measured from the post-input process high-water mark:
it is not a precise allocation count and excludes input storage. Compare like
environments and repeat counts; Arrow allocation and OS accounting add noise.
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
import pyarrow.compute as pc


def load_matcher(path: Path):
    spec = importlib.util.spec_from_file_location("benchmark_uid_impl", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.match_uids


def fixture(case: str, n: int):
    """Return typed keys and logical ranks with exactly n/2 shared values."""
    if n % 2:
        raise ValueError("rows must be even")
    half = n // 2
    rng = np.random.default_rng(20260917)
    ranks = {
        "a": rng.permutation(np.arange(n, dtype=np.int64)),
        "b": rng.permutation(np.arange(half, n + half, dtype=np.int64)),
    }
    if case == "int64":
        keys = {name: pa.array(values) for name, values in ranks.items()}
    elif case == "string":
        keys = {
            name: pc.cast(pa.array(values), pa.string())
            for name, values in ranks.items()
        }
    elif case == "mixed_integer":
        # Negative signed keys, shared keys ending at INT64_MAX, and unsigned
        # keys starting at INT64_MAX + 1. The adjacent boundary values must not
        # match each other, even though float64 cannot distinguish them.
        a = ranks["a"]
        a_values = np.empty(n, dtype=np.int64)
        negative = a < half
        a_values[negative] = -(a[negative] + 1)
        a_values[~negative] = np.iinfo(np.int64).max - (a[~negative] - half)
        b = ranks["b"]
        b_values = np.empty(n, dtype=np.uint64)
        shared = b < n
        b_values[shared] = np.uint64(np.iinfo(np.int64).max) - (
            b[shared] - half
        ).astype(np.uint64)
        b_values[~shared] = np.uint64(2**63) + (b[~shared] - n).astype(np.uint64)
        keys = {"a": pa.array(a_values), "b": pa.array(b_values)}
    else:
        raise ValueError(case)
    return keys, ranks


def validate(result, keys, ranks, n: int, join: str):
    half = n // 2
    if join == "inner":
        order = ranks["a"][ranks["a"] >= half]
    else:
        order = np.concatenate([ranks["a"], ranks["b"][ranks["b"] >= n]])
    assert result.num_rows == len(order)
    assert result.column_names == ["entity_id", "a/row_index", "b/row_index"]
    assert (
        result["entity_id"]
        .combine_chunks()
        .equals(pa.array(np.arange(len(order), dtype=np.int64)))
    )
    for name, key_array in keys.items():
        inverse = np.full(n + half, -1, dtype=np.int64)
        inverse[ranks[name]] = np.arange(n, dtype=np.int64)
        rows = inverse[order]
        expected = pa.array(rows, pa.int64(), mask=rows < 0)
        actual = result[f"{name}/row_index"].combine_chunks()
        assert actual.equals(expected), f"incorrect {name} rows or ordering"


def peak_rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if sys.platform == "darwin" else value * 1024


def worker(args):
    match_uids = load_matcher(args.module)
    warm_keys, _ = fixture(args.case, 128)
    match_uids(warm_keys, join=args.join)
    keys, ranks = fixture(args.case, args.rows)
    del warm_keys
    gc.collect()
    rss_before = peak_rss_bytes()
    start = time.perf_counter()
    result = match_uids(keys, join=args.join)
    elapsed = time.perf_counter() - start
    rss_after = peak_rss_bytes()
    # Check the complete ordered results using source row positions, never a
    # potentially lossy comparison between mixed integer families.
    validate(result, keys, ranks, args.rows, args.join)
    return {
        "case": args.case,
        "join": args.join,
        "input_rows_per_source": args.rows,
        "output_rows": result.num_rows,
        "seconds": elapsed,
        "rss_baseline_mb": rss_before / 1e6,
        "peak_rss_mb": rss_after / 1e6,
        "peak_rss_increment_mb": (rss_after - rss_before) / 1e6,
        "validated": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", type=Path, required=True)
    parser.add_argument("--label", default="candidate")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--case", choices=["int64", "string", "mixed_integer"])
    parser.add_argument("--join", choices=["inner", "outer"])
    parser.add_argument("--rows", type=int)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(args)))
        return
    if args.repeats < 1:
        parser.error("repeats must be positive")
    cases = [("int64", 500_000), ("string", 100_000), ("mixed_integer", 100_000)]
    measurements = []
    summary = []
    for case, rows in cases:
        for join in ["inner", "outer"]:
            samples = []
            for repeat in range(args.repeats):
                cmd = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--module",
                    str(args.module.resolve()),
                    "--worker",
                    "--case",
                    case,
                    "--join",
                    join,
                    "--rows",
                    str(rows),
                ]
                # Deliberately serial: concurrent workers distort RSS/time.
                raw = subprocess.check_output(cmd, text=True)
                measurement = json.loads(raw)
                measurement["repeat"] = repeat + 1
                measurements.append(measurement)
                samples.append(measurement)
            row = {
                "case": case,
                "join": join,
                "input_rows_per_source": rows,
                "output_rows": samples[0]["output_rows"],
                "median_seconds": statistics.median(s["seconds"] for s in samples),
                "median_peak_rss_increment_mb": statistics.median(
                    s["peak_rss_increment_mb"] for s in samples
                ),
            }
            summary.append(row)
            print(json.dumps(row), flush=True)
    report = {
        "label": args.label,
        "module": str(args.module.resolve()),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pyarrow": pa.__version__,
        "repeats": args.repeats,
        "seed": 20260917,
        "notes": "Isolated, sequential, warmed workers; timed matching only; full ordered output checked after measurement. RSS increment is relative to the process high-water mark after creating inputs, not a precise allocation count. Decimal MB.",
        "summary": summary,
        "measurements": measurements,
    }
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
