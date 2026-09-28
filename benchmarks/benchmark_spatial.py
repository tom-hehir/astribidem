"""Compare spatial routes in fresh processes, checking complete output equality.

Peak RSS uses resource.getrusage, so this benchmark runs on Linux/macOS.
Each timed route writes a Parquet index. Validation runs after measurement.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from astribidem import (
    EntitywiseCrossmatchConfig,
    build_edges_by_band,
    crossmatch,
    resolve_to_file,
    survey_coords_from_arrays,
    write_band_layout,
)
from astribidem.banded import as_chunks


def peak_mb():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / (1e6 if sys.platform == "darwin" else 1e3)


def worker(args):
    rng = np.random.default_rng(args.seed)
    width = 10.0 if args.field == "sparse" else 0.2
    ra = rng.uniform(10.0, 10.0 + width, args.rows)
    dec = rng.uniform(-width / 2, width / 2, args.rows)
    coordinates = {
        "a": (ra, dec),
        "b": (
            ra + rng.normal(0.0, 0.15 / 3600, args.rows),
            dec + rng.normal(0.0, 0.15 / 3600, args.rows),
        ),
    }
    settings = {"radius_arcsec": 1.0, "dedupe_radius_arcsec": {"a": 0.0, "b": 0.0}}
    mode = EntitywiseCrossmatchConfig()
    baseline = peak_mb()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        start = time.perf_counter()
        if args.route == "in_memory":
            surveys = [survey_coords_from_arrays(n, *c) for n, c in coordinates.items()]
            table = crossmatch(surveys, mode=mode, **settings)
            pq.write_table(table, root / "index.parquet")
            del table, surveys
        else:
            write_band_layout(
                root / "layout",
                {n: as_chunks(*c, 10_000) for n, c in coordinates.items()},
                band_height_deg=width / 8,
            )
            build_edges_by_band(root / "layout", root / "edges", **settings)
            resolve_to_file(
                root / "edges",
                mode,
                root / "index.parquet",
                memory_limit="128MB",
                threads=1,
            )
        seconds = time.perf_counter() - start
        peak = peak_mb()
        disk_bytes = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
        # Canonical Arrow serialization checks all rows and columns; chunk
        # boundaries and metadata insertion order are not semantic differences.
        table = pq.read_table(root / "index.parquet").combine_chunks()
        metadata = sorted((table.schema.metadata or {}).items())
        table = table.replace_schema_metadata(dict(metadata))
        buffer = pa.BufferOutputStream()
        with pa.ipc.new_stream(buffer, table.schema) as writer:
            writer.write_table(table)
        digest = hashlib.sha256(buffer.getvalue()).hexdigest()
        return {
            "field": args.field,
            "route": args.route,
            "rows_per_source": args.rows,
            "seconds": seconds,
            "peak_rss_mb": peak,
            "additional_peak_rss_mb": max(0, peak - baseline),
            "retained_disk_bytes": disk_bytes,
            "output_rows": table.num_rows,
            "output_sha256": digest,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=100_000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--output", type=Path, default=Path("spatial-results.json"))
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--field", choices=["sparse", "crowded"], default="sparse")
    parser.add_argument("--route", choices=["in_memory", "banded"], default="in_memory")
    args = parser.parse_args()
    if args.rows < 1 or args.repeats < 1:
        parser.error("rows and repeats must be positive")
    if args.worker:
        print(json.dumps(worker(args)))
        return
    results = []
    for field in ("sparse", "crowded"):
        for repeat in range(args.repeats):
            # Alternate route order while keeping independent processes.
            for route in (
                ("in_memory", "banded") if repeat % 2 == 0 else ("banded", "in_memory")
            ):
                output = subprocess.check_output(
                    [
                        sys.executable,
                        __file__,
                        "--worker",
                        "--field",
                        field,
                        "--route",
                        route,
                        "--rows",
                        str(args.rows),
                        "--seed",
                        str(args.seed),
                    ],
                    text=True,
                )
                result = json.loads(output)
                result["repeat"] = repeat
                results.append(result)
                print(
                    f"{field} / {route}: {result['seconds']:.3f}s, {result['peak_rss_mb']:.1f} MB"
                )
        hashes = {r["output_sha256"] for r in results if r["field"] == field}
        if len(hashes) != 1:
            raise RuntimeError(f"{field}: spatial routes produced different indexes")
    report = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "dependencies": {
            p: importlib.metadata.version(p)
            for p in (
                "astribidem",
                "numpy",
                "scipy",
                "pyarrow",
                "duckdb",
            )
        },
        "seed": args.seed,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Validated equal indexes; wrote {args.output}")


if __name__ == "__main__":
    main()
