# Hub-and-spoke matching

`match_hub_and_spoke` matches each counterpart (spoke) independently to one
explicit anchor (hub). Each link uses `UIDLink` or `SpatialLink`; configurations
may be entirely UID-based, entirely spatial, or combine both methods. The
topology is the distinguishing feature: there are no spoke-to-spoke checks.

The output is an inner intersection: an anchor observation is retained only when
every link succeeds. Partial groups and groups without the anchor are not
emitted. Rows follow the anchor's input order; ID columns follow the input
mapping order.

For a spatial example, put A at 0 arcsec, B at -0.8 arcsec and C at +0.8 arcsec
along a small angular axis. With A as anchor and a 1 arcsec radius, both A-B and
A-C pass. The group can contain A, B and C even though B-C is 1.6 arcsec apart.
An all-pairs radius requirement would reject that triple. UID equality is
transitive; angular proximity is not.

```python
import pyarrow as pa
from astribidem import UIDLink, SpatialLink, match_hub_and_spoke, rows_to_ids

catalogs = {
    "labels": pa.table(
        {
            "id": [10, 20],
            "uid": [101, 102],
            "ra": [10.0, 20.0],
            "dec": [0.0, 0.0],
        }
    ),
    "spectra": pa.table({"id": [200, 100], "uid": [102, 101]}),
    "images": pa.table(
        {
            "id": [300],
            "ra": [10.0001],
            "dec": [0.0],
        }
    ),
}
matches = match_hub_and_spoke(
    catalogs,
    anchor="labels",
    links={"spectra": UIDLink(), "images": SpatialLink(1.0)},
    dedupe_radius_arcsec=0.0,
)
# entity_id | labels/row_index | spectra/row_index | images/row_index
#         0 |                0 |                 1 |                0

matches = rows_to_ids(matches, {name: table["id"] for name, table in catalogs.items()})
# entity_id | labels/id | spectra/id | images/id
#         0 |        10 |        100 |       300
```

The result refers to rows: row `i` of a source is its table's `i`-th row.
`rows_to_ids` maps rows to any ID column. UID links use a table's `uid` column,
falling back to `id` when absent. Keys keep the existing exact integer/
string semantics and duplicate/null rejection. This first interface provides one
shared UID field per source, including the anchor; it does not support different
anchor key columns for different UID links or composite keys.

Spatial participants additionally need finite `ra`/`dec` in degrees in a common
celestial frame. Cleaning and coordinate-frame conversion remain the caller's
responsibility. The scalar `dedupe_radius_arcsec` applies to every spatial
participant, including the anchor; `0.0` opts out. Use
`dedupe_radius_arcsec_overrides` for exceptions among those participants.
UID-only sources need no coordinates and cannot have a spatial dedupe override. `SpatialLink` supports the existing `mutual_nearest` (default),
`mutual_unique`, `anchored_nearest` and `anchored_unique` policies. Anchored
policies may select the same counterpart for multiple anchors.

## Scientific contract

Every link sees its full configured input catalogs. Only completed link results
are intersected. Earlier UID matches therefore cannot remove competitors from a
subsequent spatial match or alter its dedupe decisions. An empty link does not
skip validation or matching of the remaining inputs.

The anchor defines which coordinates drive each spatial relationship. The result
guarantees each declared anchor relationship; counterparts do not need to match
one another. There is no transitive identity inference, many-to-many expansion,
UID-then-spatial fallback or outer join. Existing `crossmatch` and `match_uids`
behavior is unchanged, including the existing N-way spatial all-pairs contract.
A configuration containing only one link type still follows this explicitly
anchor-based composition contract.

The result contains consecutive int64 `entity_id` and one int64
`<source>/row_index` column per table.
Link policies, UID types, spatial dedupe outcomes and pre-intersection match
counts are recorded in `astribidem.resolved_config` schema metadata. This
API emits no separation columns.

The main `crossmatch` API retains its richer spatial strategies, including
N-source all-pairs subset joins and entitywise `refuse`, `sequential` and
`split` resolution. These are not applied across the independent hub-and-spoke
links. The lower-level `geometry.crossmatch_radec` adapter also supports
`matching_topology="anchor-pairs"` with complete matches, returning positional
arrays rather than this typed-ID product.

## Implementation boundary

The implementation lives in `hub_and_spoke.py` and uses the shared UID,
spatial-kernel, deduplication and resolution primitives.
Arrow IDs/UIDs stay native. Spatial links match row positions, so spatial
matching never sees the IDs. Spatial dedupe keeps the lowest row of each
duplicate group and therefore follows the catalog's row order. The final table
holds the matched rows, like every other matcher.

All matching remains in memory. Spatial links currently execute sequentially;
`workers` controls the spatial queries within each link. Each call prepares
the full anchor kernel and dedupe result once, then reuses them for every
spatial spoke. Each spoke kernel is released after its candidate edges are
built, before the next spoke is prepared. No kernels are cached across calls.
Core tests check these lifetimes alongside ambiguity,
exact identity, dedupe and order; realistic hub-and-spoke build performance is
not yet benchmarked.

The independent-link composition follows AstroBench's mixed MMU workflow. Unlike
its first-link ordering, this API returns rows in anchor input order. HF scanning, persistence and payload materialization remain
responsibilities of `astribidem-hf`.

## Renaming from the original interface

`match_hub_and_spoke` replaces `match_mixed`, and `hub_and_spoke.py` replaces
`mixed.py`; there are no compatibility aliases. Update imports and calls.
Matching behavior is unchanged. The resolved-config provenance method is now
`hub_and_spoke`, so saved workflow identities using the old method are
different.

`astribidem-hf` now exposes `match_catalog_hub_and_spoke` and the
`hub_and_spoke` CLI method, with its dependency pinned to a core commit
providing the renamed API. Its
[migration guide](https://github.com/tom-hehir/astribidem-hf/blob/5de543ac020e8a89854c2b97a1e0911772b362d4/docs/cli.md#migrating-the-original-anchor-link-interface)
covers the import and configuration changes and the new work directory needed
for workflows saved under `mixed`.

## Initial implementation validation

Before this rename, the original anchor-link implementation passed the complete
core suite (299 tests), including 39 new composition regressions. Those 39 tests
and 84 UID tests also passed on minimum-supported PyArrow 15.0.0. The downstream
HF suite passed against that core (228 tests), and all 42 new HF
composition/index tests passed on PyArrow 15.0.0. These historical checks used
local fixtures; they do not establish survey-scale performance or validation of
subsequent changes.

## Related design notes

The [grouped UID/spatial proposal](design/grouped-uid-spatial-matching.md)
describes a deferred possible extension: UID grouping, coordinate-source
priority, and richer spatial resolution between representative objects without a
universal anchor. Its stricter coordinate requirement and partial membership
handling do not apply to this API. It will only be implemented when a concrete
use case needs it. Broader hub-and-spoke retention modes likewise remain future
work; this API currently supports inner results only.
