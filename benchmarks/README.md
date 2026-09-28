# UID matching benchmark

The [UID implementation decision](../docs/uid-matching.md) records a separate
comparison of canonical Arrow, NumPy, pandas and Polars approaches, including
threading and native-string requirements. Arrow remains the chosen dependency;
the measurements below describe the earlier Python-object-to-Arrow update.

The [direct Acero evaluation](../docs/uid-acero-evaluation.md) preserves a
reproducible two-source comparison of the production matcher, optimized
`index_in`, `Table.join`, and row-position-only Acero joins at 500k and 2m
rows/source. Run `benchmark_uid_acero.py` using the commands in that report.
It is an experiment, not a selectable production backend.

`benchmark_uids.py` measures the UID matcher in fresh, sequential subprocesses.
Each process warms its implementation, creates native typed inputs, times one
match call and then validates the complete ordered row output and nullable
memberships. The matcher now returns rows, so mapping them to IDs is a separate
`rows_to_ids` step outside the timing; earlier measurements below included the
ID gather. Validating an older typed-ID implementation needs the script from
that commit. There are three repeats per case; the table reports medians.

Run from the repository root with the package's dependencies installed:

```sh
python benchmarks/benchmark_uids.py \
  --module src/astribidem/uids.py \
  --label native_arrays --output /tmp/uid-native.json
```

To reproduce the previous Python-dictionary implementation separately:

```sh
git show 9c075412265e941cd94ed49d8adf44eb64284a22:src/astro_crossmatch/uids.py > /tmp/uid-baseline.py
python benchmarks/benchmark_uids.py \
  --module /tmp/uid-baseline.py \
  --label python_baseline --output /tmp/uid-baseline.json
```

Recorded on 2026-09-17 with Python 3.13.5, NumPy 2.5.3 and PyArrow 25.0.1,
on macOS arm64. Each case has two shuffled sources with half their keys shared.
Mixed integers include adjacent signed/unsigned keys across `INT64_MAX`.

| Rows per source / key type | Join | Previous time | Native time | Previous additional peak RSS | Native additional peak RSS |
| --- | --- | ---: | ---: | ---: | ---: |
| 500,000 int64 | inner | 0.486 s | 0.162 s | 249.9 MB | 130.0 MB |
| 500,000 int64 | outer | 0.430 s | 0.200 s | 310.7 MB | 134.0 MB |
| 100,000 strings | inner | 0.104 s | 0.040 s | 50.2 MB | 24.1 MB |
| 100,000 strings | outer | 0.102 s | 0.053 s | 53.3 MB | 28.3 MB |
| 100,000 mixed integers | inner | 0.090 s | 0.027 s | 60.3 MB | 30.2 MB |
| 100,000 mixed integers | outer | 0.087 s | 0.033 s | 66.8 MB | 31.1 MB |

Raw repetitions and environment metadata are in
[the previous baseline](results/uid-python-baseline.json) and
[the native-array result](results/uid-native.json).

RSS is the increase in the process high-water mark over the value measured
after input creation; it excludes input storage and is not an exact allocation
count. Decimal MB are used. Allocator retention and OS accounting affect the
result. These local measurements are not a general throughput or memory bound:
native hash tables remain a material memory cost, indexes remain in RAM, and
different numbers of sources, overlaps and key lengths can change the balance.

Validation for the implementation change: 251 core tests and 186 downstream
HF tests passed on PyArrow 25.0.1. All 75 UID tests also passed with Python 3.11,
NumPy 1.26.4 and the minimum declared PyArrow version, 15.0.0.

## `index_in` production update

The production matcher now performs one `pc.index_in` lookup per non-anchor
source and reuses the resulting row positions for membership and final output.
It retains N-source inner/left/outer semantics, exact integer and string keys,
and all validation. NumPy handles only integer row positions and an outer-join
boolean bitmap. Normalized keys are released before final ID gathers; row maps
are released as each output column is built.

A before/after run on 2026-09-17 used the same fixtures and worker from
`benchmark_uid_acero.py`, calling the production `current` strategy against
`80aedb8` and the updated module. Each case used 500,000 rows/source, 50% overlap,
one Arrow thread, and three fresh sequential warmed workers per implementation;
implementation order alternated between repeats. Times include validation,
normalization, matching and ordered typed output; every output was checked
against the independent rank oracle outside timing.

| Keys / join | Before | Updated | Before additional peak RSS | Updated additional peak RSS |
| --- | ---: | ---: | ---: | ---: |
| int64 / inner | 171.8 ms | 119.5 ms | 130.0 MB | 119.5 MB |
| int64 / outer | 216.1 ms | 139.5 ms | 134.0 MB | 125.6 MB |
| string / inner | 316.0 ms | 274.6 ms | 130.2 MB | 127.2 MB |
| string / outer | 434.2 ms | 258.9 ms | 140.7 MB | 134.1 MB |

Medians are shown. The updated matcher took 13–40% less time on these two-source
workloads, with modest memory reductions. RSS has the high-water-mark limitations
described above; this does not establish N-source performance or survey-scale
capacity. [Raw results and module checksum](results/uid-index-in.json) preserve
all 24 measurements and the environment (Python 3.13.5, NumPy 2.5.3,
PyArrow 25.0.1, macOS arm64).

For a before/after comparison using the existing benchmark suite:

```sh
git show 80aedb8:src/astro_crossmatch/uids.py > /tmp/uids-before.py
python benchmarks/benchmark_uid_acero.py \
  --module /tmp/uids-before.py --output /tmp/uids-before.json
python benchmarks/benchmark_uid_acero.py \
  --module src/astribidem/uids.py --output /tmp/uids-after.json
```

Compare the `strategy="current"` measurements in those files. The suite also
runs the older two-source prototypes. To reproduce individual production workers
or alternate before/after modules, use `--worker --strategy current --threads 1
--rows 500000 --kind int64 --join inner`, varying key family and join as needed.

Validation: 260 core tests and 186 downstream HF tests passed against the updated
source. All 84 UID tests passed on minimum-supported PyArrow 15.0.0 as well.
New regressions cover late-source matches to previously introduced outer keys,
successive four-source inner reductions, and single-source chunked/empty inputs.
