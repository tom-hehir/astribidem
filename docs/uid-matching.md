# UID matching: Arrow-native implementation

Decision recorded 2026-09-17: retain canonical PyArrow operations for exact UID
matching. Do not add Polars or pandas as dependencies, or implement custom
shared-dictionary matching. Arrow already supports native integer and string
keys. Dependency simplicity is the reason for this choice; the alternatives
were not uniformly slower or more memory hungry.

## Required behavior

- Support signed and unsigned integers of every Arrow integer width, plus
  `string` and `large_string`. Native key arrays must never be converted into
  per-key Python objects, including transiently during validation or joining.
- Compare integer values exactly across widths and signedness, including full
  uint64. Retain the existing lossless normalization, which uses decimal128 at
  scale zero when no integer type can represent both domains. Original output
  observation-ID types remain unchanged.
- Match strings exactly and case-sensitively. Do not implicitly convert between
  numeric strings and integers, strip leading zeros, or normalize identifiers.
  Callers establish the shared identifier namespace and explicitly convert
  representations when needed.
- Preserve the existing null/duplicate rejection, N-source inner/left/outer
  joins, deterministic anchor/source ordering, absent memberships, and
  provenance. Like coordinate matching, the matcher returns row positions;
  `rows_to_ids` maps them to observation IDs or back to the UIDs afterwards.
- Keep indexes in memory for now. Capacity includes inputs, temporary matching
  state and output; a raw key column fitting in RAM is insufficient. Payload
  cache limits do not bound matching memory.
- Prefer canonical public library APIs. Zero-copy conversion is desirable,
  but native representation and total peak memory matter more than avoiding
  every buffer allocation. Matching and output construction allocate too.

Floats, binary/UUIDs, temporal types, and composite keys remain outside the
current public UID contract. Add types for demonstrated catalog requirements.
Dictionary-encoded integers/strings are possible future input representations,
but are not accepted directly by the current validator. Public decimal UID
support is distinct from the internal exact integer normalization above.

## Current implementation and threading

`match_uids` validates with Arrow `count_distinct` and performs one `index_in`
lookup per non-anchor source. `index_in` provides row positions in probe order,
with null for a miss. Inner joins filter the accumulated row maps; outer joins
reuse matched positions to mark unseen source rows in a NumPy boolean bitmap.
Only native integer row positions enter NumPy; integer/string keys remain in
Arrow. The wrapper supplies validation, exact type normalization, deterministic
ordering and N-source semantics. IDs are gathered once with `take`, releasing
each row map as its output column is built.
[`Table.join`](https://arrow.apache.org/docs/python/generated/pyarrow.Table.html#pyarrow.Table.join)
is another canonical option, but would still need explicit duplicate rejection,
lossless common key types and restoration of our output order.

The normal HF workflow performs UID matching once on complete in-memory indexes
in one process before materialization; resumes reuse saved match tables.
Internal threading is welcome where beneficial. This does not require our own
process pool or a consumer training/distributed-loading layer. The current UID
API has no worker/thread setting; the HF CLI's `--workers` is spatial-only.
Arrow table joins expose `use_threads`, with their pool governed by
[`pa.set_cpu_count`](https://arrow.apache.org/docs/python/generated/pyarrow.set_cpu_count.html).
A future threaded path must respect the host application's CPU budget, including
concurrent independent builds. Threading can increase peak memory.

## Options considered

| Approach | Assessment |
| --- | --- |
| Arrow `index_in` | Selected primitive. Native integer/string support with existing dependencies and direct row-index results. One lookup per non-anchor source; reuse row maps rather than hashing again for membership and final output. Uniqueness validation remains a separate pass. |
| Arrow `Table.join` | Canonical internally threaded alternative; modest gains in some tested cases, with ordering/validation costs. Consider if representative measurements justify a change. |
| Direct Arrow Acero hash join | Same engine as `Table.join`, with explicit row-position-only output. Evaluated at 500k and 2m rows/source; no consistent overall improvement over lookup. See the [Acero evaluation](uid-acero-evaluation.md). |
| NumPy sorting + `searchsorted` | Low-memory integer option using native numeric views. A shuffled 500k/source experiment took about 141 ms and 30 MB additional peak RSS. Ordinary Arrow strings have no equivalent simple zero-copy NumPy representation; this would require a separate string path. |
| pandas integer `Index.get_indexer` | Native integer path was competitive: about 38 ms / 42 MB in the same integer experiment. Adds a dependency and does not provide the required string path. |
| pandas Arrow-backed string indexing/merge | Tested pandas 3.0.5 paths still construct Python string objects internally. Native input/output dtypes do not guarantee native intermediates. Index ordering/uniqueness checks cause object conversion even though part of merge factorization uses Arrow. |
| Polars `DataFrame.join` | Canonical native integer/string join with built-in uniqueness validation, ordering and threading. Strong measured performance; deferred to avoid adding a dependency. Arrow import is mostly zero-copy, not universally so. |
| Custom shared Arrow dictionary encoding + NumPy code lookup | A native prototype improved some string measurements, but adds bespoke matching machinery despite available canonical APIs. Not selected. |

The NumPy/pandas timings above are a separate single-threaded, two-source inner
join experiment with the same validation/output requirements, not a comparison
against the threaded results below. These alternatives remain research notes,
not supported selectable backends.

The pandas finding is specific to the inspected public paths/version, not every
pandas operation. Source references:
[`get_join_indexers` and `_factorize_keys`](https://github.com/pandas-dev/pandas/blob/v3.0.5/pandas/core/reshape/merge.py),
[`Index._get_engine_target`](https://github.com/pandas-dev/pandas/blob/v3.0.5/pandas/core/indexes/base.py).

## Canonical join measurements

Exploratory prototypes used two shuffled sources of 500,000 unique IDs, with
50% overlap. Integers were sparse int64 values above 2**60; strings were their
19-digit Arrow UTF-8 representations. Inner outputs contained 250,000 rows;
outer outputs contained 750,000. Medians of three sequential isolated warmed
processes are shown as **milliseconds / additional peak RSS in decimal MB**.

| Keys / join | Arrow lookup | Arrow join, 4 threads | Polars join, 1 thread | Polars join, 4 threads |
| --- | ---: | ---: | ---: | ---: |
| int64 / inner | 121.3 / 134.8 | 95.9 / 138.6 | 49.7 / 40.9 | 27.9 / 51.8 |
| int64 / outer | 140.4 / 154.4 | 154.1 / 156.9 | 271.2 / 84.7 | 123.0 / 104.7 |
| string / inner | 262.9 / 117.7 | 226.2 / 141.2 | 91.6 / 102.1 | 54.1 / 102.3 |
| string / outer | 284.7 / 136.3 | 291.3 / 175.9 | 346.7 / 132.8 | 184.3 / 181.5 |

The direct Arrow reference validates keys, performs one anchor-to-target lookup,
and reuses its row map for outer membership. It is an optimization prototype,
not an exact benchmark of the committed N-source matcher at `94184f5`.
Table joins use only keys and source row positions, then gather original IDs
from Arrow. Arrow restores order explicitly; Polars uses `validate="1:1"` and
`maintain_order="left"`/`"left_right"`. Polars imports with `rechunk=False` and
exports only row positions, avoiding unnecessary string reserialization.

Timing includes validation, normalization, conversions, ordering and typed
Arrow output. Fixture preparation and independent result checks are excluded.
All timed outputs were checked; small checks also covered Unicode/NUL, mixed
integer extrema, string widths, empty inputs and rejected duplicate/null keys.

Python 3.13.5, NumPy 2.5.3, PyArrow 25.0.1 and Polars 1.44.2; macOS arm64.
[Raw repetitions and environment details](../benchmarks/results/uid-canonical-options.json)
include one-thread Arrow joins and absolute RSS. All workers imported the same
dependencies. RSS increments are changes in process high-water marks after
fixture creation, not exact allocation counts; allocator reuse and preparation
peaks affect them. These prototypes do not establish N-source, left-join,
chunked-input, minimum-version or survey-scale performance.

## Direct Acero follow-up

A subsequent [Acero experiment](uid-acero-evaluation.md) compares the committed
matcher, a lookup optimization, `Table.join`, and a public Acero plan that emits
only row positions. It includes 500k and 2m rows/source, a reproducible prototype,
and compatibility checks on PyArrow 15 and 25. Threaded Acero speeds up some
joins but costs more memory at 2m; omitting key output provides no consistent
overall advantage. The experiment retained production at its historical
baseline; the subsequent production update adopts the optimized lookup approach.

## When to revisit

Revisit if a representative build approaches RAM capacity or spends material
time matching. Measure the complete Arrow-input-to-Arrow-output operation,
including validation, casts, chunk handling, ordering and output. Vary source
count, imbalance, overlap, key length and CPU budget; use the full API tests to
protect exact identity and semantics. Preserve signed/unsigned boundary tests.

The current implementation already reuses lookup row maps. Consider a canonical
threaded Arrow join if further representative measurements justify it. Reconsider another dependency only for a demonstrated benefit worth
its maintenance cost. Keep scientific UID policy and Arrow-facing interfaces
independent of this implementation choice; no backend-selection framework is
needed now.
