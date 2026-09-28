# Hub-and-spoke matching

`match_hub_and_spoke` matches each counterpart (spoke) independently to one
explicit anchor (hub). Each link uses `UIDLink` or `SpatialLink`; configurations
may be entirely UID-based, entirely spatial, or combine both methods. The
topology is the distinguishing feature: there are no spoke-to-spoke checks.

The output is an inner intersection: an anchor observation is retained only when
every link succeeds. Partial groups and groups without the anchor are not
emitted. Rows follow the anchor's input order; row columns follow the input
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
# Contents of `matches`: the input row positions that matched every link.
# entity_id | labels/row_index | spectra/row_index | images/row_index
#         0 |                0 |                 1 |                0

matches = rows_to_ids(matches, {name: table["id"] for name, table in catalogs.items()})
# Contents of `matches` after mapping the row positions to catalogue IDs.
# entity_id | labels/id | spectra/id | images/id
#         0 |        10 |        100 |       300
```

The result refers to rows: row `i` of a source is its table's `i`-th row.
`rows_to_ids` maps rows to any ID column. UID links use a table's `uid` column,
falling back to `id` when absent. Keys keep the existing exact integer/
string semantics and duplicate/null rejection. Each table supplies one
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
provide separate N-way joins and spatial association policies.
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
arrays instead of an Arrow table.

## Memory and execution

All matching runs in memory. Spatial links execute sequentially; `workers`
controls queries within each link. The full anchor's spatial kernel and dedupe
result are prepared once per call and reused across spatial links. Each spoke's
kernel is released after its candidate edges are built, before preparing the
next spoke. Calls do not share cached kernels.
