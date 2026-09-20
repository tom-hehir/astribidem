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
from astro_crossmatch import UIDLink, SpatialLink, match_hub_and_spoke

matches = match_hub_and_spoke(
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
string semantics and duplicate/null rejection. This first interface provides one
shared UID field per source, including the anchor; it does not support different
anchor key columns for different UID links or composite keys.

Spatial participants additionally need finite `ra`/`dec` in degrees in a common
celestial frame. Cleaning and coordinate-frame conversion remain the caller's
responsibility. Every spatial participant, including the anchor, requires an
explicit dedupe radius; `0.0` opts out. UID-only sources need no coordinates or
dedupe radius. `SpatialLink` supports the existing `mutual_nearest` (default),
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

The result contains consecutive int64 `entity_id` and original typed source IDs.
Link policies, UID types, spatial dedupe outcomes and pre-intersection match
counts are recorded in `astro_crossmatch.resolved_config` schema metadata. This
API emits no separation columns.

The main `crossmatch` API retains its richer spatial strategies, including
N-source all-pairs subset joins and entitywise `refuse`, `sequential` and
`split` resolution. These are not applied across the independent hub-and-spoke
links. The lower-level `geometry.crossmatch_radec` adapter also supports
`matching_topology="anchor-pairs"` with complete matches, returning positional
arrays rather than this stable-ID product.

## Implementation boundary

The implementation lives in `hub_and_spoke.py` and calls the existing matchers.
Arrow IDs/UIDs stay native. For the spatial API, Arrow-sorted ID ranks serve as
integer surrogates, preserving its lowest-stable-ID dedupe rule without creating
Python string objects. Original IDs are gathered into the final table.

All matching remains in memory. Spatial links currently execute sequentially;
`workers` controls the existing spatial matcher within each link. Repeated
spatial links can repeat anchor kernel/dedupe work; no shared-kernel framework
or per-link checkpoint system is introduced. Core tests exercise ambiguity,
exact identity, dedupe and order; realistic hub-and-spoke build performance is
not yet benchmarked.

The independent-link composition follows AstroBench's mixed MMU workflow. Unlike
its positional result/first-link ordering, this API exposes typed stable IDs in
anchor input order. HF scanning, persistence and payload materialization remain
responsibilities of `hf-crossmatch`.

## Renaming from the original interface

`match_hub_and_spoke` replaces `match_mixed`, and `hub_and_spoke.py` replaces
`mixed.py`; there are no compatibility aliases. Update imports and calls.
Matching behavior is unchanged. The resolved-config provenance method is now
`hub_and_spoke`, so saved workflow identities using the old method are
different.

`hf-crossmatch` remains pinned to the original core commit and continues to use
its existing `match_catalog_mixed` API and `mixed` CLI method. Its caller
migration and dependency-pin update are deferred until the higher-level API
decision and any resulting core changes are complete. Upgrade that consumer's
imports and core pin together; the historical pin does not expose the renamed
API.

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
