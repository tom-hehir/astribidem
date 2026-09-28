# Direct Acero UID join experiment

**Follow-up:** production now uses the optimized `index_in` approach generalized
to N sources; see the [implementation decision](uid-matching-implementation.md) and
[production measurements](../benchmarks/README.md).
The experiment and its "Current" measurements below describe the earlier
implementation at `80aedb8`, before that change.

Evaluated 2026-09-17. Direct Acero works with our native integer/string and
two-source matching semantics, but these measurements do not justify replacing
the production matcher. Threaded joins win some cases; direct `index_in` wins
others and generally uses less memory. Keep the production implementation
unchanged while deciding the next optimization.

## What was compared

All candidates use the production validation and lossless type normalization,
restore anchor/source input ordering, gather original observation IDs, and
produce the same typed Arrow table and metadata:

- **Current:** committed `match_uids`, including its repeated membership and
  row-index lookups.
- **Lookup prototype:** one `pc.index_in` probe, reusing its row map for outer
  membership instead of another key lookup. Keys remain Arrow arrays; NumPy
  handles only integer row positions and a boolean membership mask.
- **Table join:** join key and row-position columns with `Table.join`, discard
  the resulting key column, sort row positions, then gather IDs.
- **Direct Acero:** the same hash-join engine with
  [`HashJoinNodeOptions`](https://arrow.apache.org/docs/python/generated/pyarrow.acero.HashJoinNodeOptions.html)
  configured with `left_output=["a_row"]`, `right_output=["b_row"]`.
  This avoids emitting the key column from the join itself. Sorting and final
  ID gathers remain necessary, as does internal state for matching keys.

The prototype uses public APIs available on the minimum supported PyArrow 15.
It requires no additional dependency. The benchmark is deliberately two-source;
it is not a replacement for the N-source public matcher or its complete input
validation. Intermediate N-way joins would generally need to retain or
reconstruct a representative key, especially for full outer joins, reducing
the scope for dropping keys early.

## Results

Two shuffled sources with 50% overlap; sparse int64 values above 2**60 or their
19-digit UTF-8 representations. Inner results contain half as many rows as one
source; outer results contain 1.5 times as many. Tables show medians of three
repetitions per case.

At **2,000,000 rows per source**, cells show **seconds / total peak process RSS
in decimal MB**, rounded:

| Keys / join | Lookup, 1 thread | Table join, 4 threads | Acero, 4 threads |
| --- | ---: | ---: | ---: |
| int64 / inner | 0.463 / 655 | 0.418 / 704 | 0.418 / 705 |
| int64 / outer | 0.488 / 686 | 0.603 / 729 | 0.598 / 721 |
| string / inner | 1.495 / 771 | 1.232 / 894 | 1.213 / 889 |
| string / outer | 1.548 / 798 | 1.462 / 921 | 1.436 / 933 |

Direct Acero is approximately 10% faster than lookup for integer inner joins,
19% faster for string inner joins, and 7% faster for string outer joins. It is
23% slower for integer outer joins. Its peak RSS is higher in all four cases.
In the 2m sweep, differences between Table join and direct Acero timings are
only about 0–2%;
with three runs, do not treat those small differences as established wins.

At **500,000 rows per source**, cells show **milliseconds / additional peak
process RSS in decimal MB**:

| Keys / join | Current | Lookup | Table join, 4 threads | Acero, 4 threads |
| --- | ---: | ---: | ---: | ---: |
| int64 / inner | 171.8 / 130.0 | 117.5 / 119.6 | 93.3 / 138.1 | 93.0 / 142.8 |
| int64 / outer | 209.2 / 134.0 | 123.6 / 125.6 | 142.0 / 148.3 | 139.9 / 169.0 |
| string / inner | 350.2 / 130.1 | 261.4 / 127.2 | 217.1 / 140.3 | 221.3 / 135.2 |
| string / outer | 415.3 / 140.6 | 274.3 / 135.2 | 279.8 / 168.9 | 268.0 / 140.2 |

The lookup optimization is 25–41% faster than current production in these
500k two-source cases. Direct Acero reduces memory relative to Table join in
some 500k string cases, but that advantage is not consistent at two million
rows. Omitting keys from the join output does not establish a lower overall
peak: validation, hash state, sorting and final ID output all contribute.

The recommendation is to prioritize eliminating redundant Arrow hash passes
while preserving the full public contract. Keep threaded Acero as a candidate
if representative builds later justify trading memory and implementation
complexity for speed. This is not a universal engine ranking.

## Validation and measurement limits

Each strategy ran in a fresh, sequential subprocess with the same Arrow/NumPy
imports and a small warm-up. Timing includes validation, normalization, matching,
ordering and Arrow output. Fixture construction and complete result validation
are outside timing. All 108 measured outputs matched an independent rank-based
oracle. The 500k sweep also includes one-thread Table/Acero measurements.

Small semantic checks cover inner/left/outer joins, either anchor, separate IDs,
chunked and empty inputs, Unicode and embedded NUL, `string`/`large_string`,
signed/unsigned integer extrema including full uint64, and rejection of
duplicate/null keys. All 432 checks passed at each of one and four threads on
both PyArrow 15.0.0 and 25.0.1. These compare prototypes against the production
matcher; they are not an independent proof of every public API behavior.

Performance environment: Python 3.13.5, NumPy 2.5.3, PyArrow 25.0.1, macOS arm64.
The production baseline is `94184f515a8b8e8006f31147a2814ad2ea3833fc`
(also unchanged at documentation commit `a5fc170`). No pandas or Polars is
imported. Four threads means an Arrow CPU pool limited to four; lookup has no
explicit multithreaded execution option here.

RSS is the process high-water mark. Additional RSS is its increase after
fixture creation, not a precise allocation count. Total RSS includes inputs,
benchmark rank arrays, interpreter and allocator retention. Fixture preparation
and allocator reuse can affect both measures; raw files retain baseline,
absolute and incremental RSS for every repetition. Do not extrapolate these
figures directly into a catalog memory budget. N-way performance, unequal source
sizes, other overlaps and key lengths remain unmeasured.

## Reproduce

From the repository root with its NumPy/PyArrow dependencies installed:

```sh
python benchmarks/benchmark_uid_acero.py \
  --module src/astribidem/uids.py --check-only
python benchmarks/benchmark_uid_acero.py \
  --module src/astribidem/uids.py --rows 500000 \
  --output /tmp/uid-acero-500k.json
python benchmarks/benchmark_uid_acero.py \
  --module src/astribidem/uids.py --rows 2000000 --compact \
  --output /tmp/uid-acero-2m.json
```

Keep the baseline module revision fixed when comparing future changes: the
script imports its private validation/normalization helpers. The prototype
supports typed two-source inputs, not every public input form such as untyped
empty Python lists. RSS collection uses Unix `resource` (macOS/Linux units).

Raw results: [500k/source](../benchmarks/results/uid-acero-500k.json),
[2m/source](../benchmarks/results/uid-acero-2m.json), and
[minimum-version semantic checks](../benchmarks/results/uid-acero-arrow15-checks.json).
