# Independent UID/spatial anchor links

`match_mixed` composes existing two-source matchers through one explicit anchor.
Each other source has one `UIDLink` or `SpatialLink`. The output is an inner
intersection: an anchor observation is retained only when every link succeeds.
Rows follow the anchor's input order; ID columns follow the input mapping order.

```python
import pyarrow as pa
from astro_crossmatch import UIDLink, SpatialLink, match_mixed

matches = match_mixed(
    {
        "labels": pa.table({
            "id": [10, 20], "uid": [101, 102],
            "ra": [10.0, 20.0], "dec": [0.0, 0.0],
        }),
        "spectra": pa.table({"id": [200, 100], "uid": [102, 101]}),
        "images": pa.table({
            "id": [300], "ra": [10.0001], "dec": [0.0],
        }),
    },
    anchor="labels",
    links={"spectra": UIDLink(), "images": SpatialLink(1.0)},
    dedupe_radius_arcsec={"labels": 0.0, "images": 0.0},
)
# entity_id | labels/id | spectra/id | images/id
#         0 |        10 |        100 |       300
```

Tables require unique non-null integer or string `id` columns. UID links use
`uid`, falling back to `id` when absent. Keys keep the existing exact integer/
string semantics and duplicate/null rejection. This first interface provides
one shared UID field per source, including the anchor; it does not support
different anchor key columns for different UID links or composite keys.

Spatial participants additionally need finite `ra`/`dec` in degrees in a common
celestial frame. Cleaning and coordinate-frame conversion remain the caller's
responsibility. Every spatial participant, including the anchor, requires an
explicit dedupe radius; `0.0` opts out. UID-only sources need no coordinates or
dedupe radius. `SpatialLink` supports the existing `mutual_nearest` (default),
`mutual_unique`, `anchored_nearest` and `anchored_unique` policies. Anchored
policies may select the same counterpart for multiple anchors.

## Scientific contract

Every link sees its full configured input catalogs. Only completed link results
are intersected. Earlier UID matches therefore cannot remove competitors from
a subsequent spatial match or alter its dedupe decisions. An empty link does
not skip validation or matching of the remaining inputs.

The anchor defines which coordinates drive each spatial relationship. The
result guarantees each declared anchor relationship; counterparts do not need
to match one another. There is no transitive identity inference, many-to-many
expansion, UID-then-spatial fallback or outer join. Existing `crossmatch` and
`match_uids` behavior is unchanged, including the existing N-way spatial
all-pairs contract. A configuration containing only one link type still follows
this explicitly anchor-based composition contract.

The result contains consecutive int64 `entity_id` and original typed source
IDs. Link policies, UID types, spatial dedupe outcomes and pre-intersection match
counts are recorded in `astro_crossmatch.resolved_config` schema metadata.
This initial mixed API emits no separation columns.

## Implementation boundary

The implementation lives in `mixed.py` and calls the existing matchers. Native
Arrow IDs/UIDs stay native. For the spatial API, Arrow-sorted ID ranks serve as
integer surrogates, preserving its lowest-stable-ID dedupe rule without creating
Python string objects. Original IDs are gathered into the final table.

All matching remains in memory. Spatial links currently execute sequentially;
`workers` controls the existing spatial matcher within each link. Repeated
spatial links can repeat anchor kernel/dedupe work; no shared-kernel framework
or per-link checkpoint system is introduced. Core tests exercise ambiguity,
exact identity, dedupe and order; realistic mixed-build performance is not yet
benchmarked.

The independent-link composition follows AstroBench's mixed MMU workflow.
Unlike its positional result/first-link ordering, this API exposes typed stable
IDs in anchor input order. HF scanning, persistence and payload materialization
remain responsibilities of `hf-crossmatch`.

## Validation

The complete core suite passes (299 tests), including 39 new mixed-matching
regressions. The 39 mixed and 84 UID tests also pass on minimum-supported
PyArrow 15.0.0. The downstream HF suite passes against the new core (228 tests),
and all 42 new HF mixed/index tests pass on PyArrow 15.0.0. Tests use local
fixtures; survey-scale performance has not been measured for this extension.
