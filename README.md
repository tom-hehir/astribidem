# astro-crossmatch

Standalone astronomical crossmatching, extracted from Tom Hehir’s existing
Astral and AstroBench implementations. Produces match indexes of row positions,
optionally mapped to caller IDs, rather than downloading or joining scientific
payloads. Private development repository.

## Install

With GitHub credentials configured for this private repository:

```bash
uv pip install "git+https://github.com/tom-hehir/astro-crossmatch.git"
```

Pin an immutable commit in applications. Runtime dependencies are NumPy, SciPy
and PyArrow. There is no Torch, Lightning, AION, Astral, HATS or LSDB dependency.

## Match coordinate arrays

```python
from astro_crossmatch import (
    crossmatch, rows_to_ids, survey_coords_from_arrays, DegenerateCrossmatchConfig,
)

sources = [
    survey_coords_from_arrays("a", ra=[10.0, 20.0], dec=[0.0, 0.0]),
    survey_coords_from_arrays("b", ra=[10.0001, 30.0], dec=[0.0, 0.0]),
]
result = crossmatch(
    sources,
    radius_arcsec=1.0,
    dedupe_radius_arcsec={"a": 0.0, "b": 0.0},
    mode=DegenerateCrossmatchConfig(surveys=["a", "b"], policy="mutual_nearest"),
)
# a/row_index | b/row_index | a__b/separation_arcsec
#           0 |           0 |                   0.36

index = rows_to_ids(result.table, {"a": [101, 102], "b": ["b-201", "b-202"]})
# a/id | b/id  | a__b/separation_arcsec
#  101 | b-201 |                   0.36
```

Every index column refers to rows: row `i` of a survey is the `i`-th coordinate
passed in, and a null row means the survey is absent. Use the rows directly for
positional access, such as `table.take(index["a/row_index"])`, or map them to
IDs with `rows_to_ids`. The IDs must be unique, non-null and in the same order
as the coordinates; their Arrow types are preserved.

Dedupe radii are explicit per survey; zero opts out. Input coordinates must
already have catalog-specific cleaning applied.

## Match exact UIDs

```python
from astro_crossmatch import match_uids

matches = match_uids(
    {"images": ["object-B", "object-A"], "spectra": ["object-A", "object-C"]},
    ids={"images": [101, 102], "spectra": [201, 202]},
    join="outer",
)
# entity_id | images/id | spectra/id
#         0 |       101 |       null
#         1 |       102 |        201
#         2 |      null |        202
```

UIDs are exact equality keys, without coordinates or a radius. Omit `ids` when
UIDs themselves identify source observations. Keys and IDs must each be unique
and non-null within each source; duplicates are rejected rather than silently
expanded into a Cartesian join. Integer keys compare exactly across integer
widths/signedness, including uint64; string keys are case-sensitive. Integer and
string keys cannot be mixed. Output IDs retain their source Arrow types.

`join="inner"` (default) retains keys present in every source; `"left"` retains
all anchor keys; `"outer"` retains all keys. The anchor defaults to the first
mapping entry and can be named with `anchor="spectra"`. Rows follow anchor input
order, with outer joins appending unseen keys in the remaining sources' input
order. Column order follows the mapping. Matching policy and types are recorded
in Arrow schema metadata. This join, like coordinate matching, runs in memory.

Keys and row lookups stay in native Arrow arrays and hash kernels; the matcher
does not create a Python object per key. Integer types are normalized losslessly
before comparison. Mixed signed/uint64 keys use fixed-width decimal128 with
scale zero when no standard integer type can represent both domains. Original
output ID types are preserved. The [UID implementation decision](docs/uid-matching.md)
records the Arrow-native choice, requirements, alternatives and threading
tradeoffs. See the [UID benchmark](benchmarks/README.md) for measured time and
memory use; native hash tables still consume memory.

## Scientific contracts

For independent links to one anchor, use the separate
[`match_hub_and_spoke` API](docs/hub-and-spoke-matching.md). Each link can use UID
or spatial matching; all-UID, all-spatial and mixed-link configurations are
supported. It intersects full-input link results, requiring every spoke to
match the anchor without checking spoke-to-spoke relationships, and returns
typed ID columns. The richer spatial `crossmatch` modes and N-source
`match_uids` joins remain separate APIs.

- The kernel emits the complete inclusive-radius candidate set.
- Pair policies: anchored_nearest, anchored_unique, mutual_nearest, mutual_unique.
- Degenerate subset joins require mutual pairwise agreement for N > 2; anchored
  policies in this interface accept two surveys.
- EntitywiseCrossmatchConfig supports refuse, sequential and split resolvers,
  nullable absent memberships, disputed singleton/drop handling and selection.
- The separate `astro_crossmatch.geometry` array adapter preserves AstroBench’s
  N-way anchor-pairs and all-pairs topology API, returning positional arrays.
- Per-survey dedupe keeps the lowest row of each duplicate group, so the kept
  row follows input order. Sort the inputs first when their order is arbitrary.
- The spatial `crossmatch` core reproduces AION-2's crossmatch output exactly;
  `tests/test_aion_parity.py` checks this against recorded AION-2 indexes.

Coordinate arrays and candidate relations are currently held in memory. This
package is not yet a distributed or out-of-core survey processing engine.

## Design proposals

The [grouped UID/spatial design](docs/design/grouped-uid-spatial-matching.md)
records a deferred possible extension: select one representative position per
UID-defined object, then apply richer spatial resolution and partial-membership
policies without requiring a universal anchor. Current UID/spatial composition
uses hub-and-spoke matching. The grouped design will only be implemented when a
concrete use case needs it; no implementation or AION-2 integration is planned.

## Development

```bash
uv venv
uv pip install -e '.[dev]'
uv run --no-project pytest
uv run --no-project ruff check src tests
```

The inherited tests include independent brute-force and Astropy comparisons,
threshold boundaries, dedupe, disputes, and N-way matching. The AION-2 parity
test uses indexes recorded by `tests/fixtures/generate_aion_parity.py`, which
runs in an AION-2 environment.
See [provenance](docs/provenance.md).
