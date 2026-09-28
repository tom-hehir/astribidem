# Spatial matching benchmark

`benchmark_spatial.py` compares in-memory spatial matching and the banded
layout/build/resolve route using the same two generated catalogues. Both routes
write a Parquet index with `EntitywiseCrossmatchConfig()` and disabled dedupe.
It checks complete ordered table and metadata equality using canonical Arrow
hashes after measurement. A difference aborts the run.

```sh
uv sync --locked --extra dev
uv run --no-sync python benchmarks/benchmark_spatial.py \
  --rows 100000 --repeats 3 --output benchmarks/results/spatial-100k.json
```

The sparse field covers a 10-degree square and the crowded field a 0.2-degree
square, both near the equator. The counterpart positions have independent
0.15-arcsec Gaussian perturbations; the matching radius is 1 arcsec. Increasing
the row count also increases density in these fixed footprints. Banded builds
use a band height one eighth of the field width, one worker, a 128 MB DuckDB
sort limit and one sort thread.

Each repetition runs in a fresh process. Route order alternates between
repetitions. Timing includes coordinate conversion, matching/resolution and
writing the index; the banded route also includes layout and intermediate edge
I/O. Input generation, validation and process startup are outside the timer.
No warm-up call is made. Use a quiet machine for measurements.

The JSON records elapsed seconds, absolute peak process RSS, the increase over
the post-input high-water mark, retained disk bytes, output row count, output
hash and software versions. RSS is measured before loading the final index for
validation. It includes the generated input arrays and is not a memory budget
or allocation count. Disk figures exclude deleted temporary sort files.

This is a reproducible synthetic comparison of two equivalent workflows, not
a throughput guarantee or a comparison against tools with different matching
policies. In-memory input generation also means it does not demonstrate a
catalogue larger than physical RAM. Real surveys, extra participants, dedupe,
larger radii and boundary-spanning groups need their own measurements. The
benchmark uses `resource.getrusage` and supports Linux and macOS.

## Recorded small-catalogue baseline

Recorded on 2026-09-28 with 20,000 rows per source, three fresh processes per
case, Python 3.13.5 on macOS arm64. This small run validates the
benchmark and quantifies overhead; the banded route is intended for inputs
that exceed the in-memory route's capacity. It need not be faster or use less
memory on small inputs.

| Field | Route | Median seconds | Median peak RSS (MB) |
| --- | --- | ---: | ---: |
| sparse | in_memory | 0.100 | 112.1 |
| sparse | banded | 0.397 | 146.6 |
| crowded | in_memory | 0.116 | 113.5 |
| crowded | banded | 0.643 | 145.2 |

Both routes produced identical complete indexes for each field. See
[raw repetitions and environment](results/spatial-20k.json).
