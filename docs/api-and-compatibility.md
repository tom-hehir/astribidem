# API and compatibility

The first public version is `0.0.0a0`. During alpha development, signatures,
result schemas and matching behavior may change between releases. Pin an exact
version for a reproducible application. Changes will be recorded in
[the changelog](../CHANGELOG.md); there is no promise of deprecation shims
between alpha releases.

## Public entry points

Names explicitly exported from `astribidem` are the supported alpha interface.

| Task | Entry points |
| --- | --- |
| Match positions | `crossmatch` with a mapping of names to `(ra, dec)` tuples |
| Choose association policy | `DegenerateCrossmatchConfig`, `EntitywiseCrossmatchConfig`, `EntitySelectionConfig`, `CrossmatchModeConfig` |
| Apply a pair policy to candidate pairs you found yourself | `one_to_one_pairs` (with `CatalogKernel.all_pairs`) |
| Match shared identifiers | `match_uids`, `match_hub_and_spoke`, `UIDLink`, `SpatialLink` |
| Use results | `rows_to_ids`, `ids_to_rows`, `index_summary` |
| Reuse candidate searches | `build_edges`, `resolve`, `audit_edges`, `CandidateEdges`, `PairEdges`, `DedupeOutcome` |
| Save candidate searches | `write_edges`, `read_edges`, `build_edges_to_directory` |
| Build and resolve large catalogues | `write_band_layout`, `build_edges_by_band`, `resolve_to_file` |
| Coordinate separate workers | `prepare_band_build`, `build_band`, `sweep_boundaries`, `read_segment`, `write_segment`, `segment_names`, `combine_segments` |
| Query the geometric kernel | `CatalogKernel` |
| Inspect the installed release | `__version__` |

The worker/segment and kernel interfaces are advanced and may evolve during
alpha. Use the high-level functions unless you need their explicit controls.
The `astribidem.banded.as_chunks` helper is also available. Other module helpers,
underscore-prefixed names and intermediate files are implementation details.

## Coordinate and radius inputs

`crossmatch`, `build_edges` and `build_edges_to_directory` accept a mapping
of survey names to `(ra, dec)` tuples, always in degrees. Each pair contains
aligned one-dimensional array-like values; an ordinary tuple or compatible
`NamedTuple` works. Mapping order defines survey order, and array order defines
row positions.

Both radius arguments are required finite scalars: `radius_arcsec > 0` and
`dedupe_radius_arcsec >= 0`. Zero disables spatial deduplication. Optional
`radius_arcsec_overrides` maps survey pairs to positive radii;
`dedupe_radius_arcsec_overrides` maps survey names to nonnegative radii.
Defaults never accept dictionaries. Unknown names and duplicate unordered pair
overrides fail. Saved provenance records the expanded radii, so equivalent
settings have the same scientific metadata.

Banded builds use the same radius arguments. `match_hub_and_spoke` uses the
same dedupe default/overrides for spatial participants, while each `SpatialLink`
sets its matching radius. It takes the same `(ra, dec)` mapping as `crossmatch`
for spatial participants, and the same mapping of names to identifier arrays as
`match_uids` for UID participants.

Every matching call is self-contained. RA/Dec becomes XYZ at kernel
construction, once per survey in an in-memory build or once per loaded region
in a banded build. Ordinary builds share each kernel across deduplication and
all its pairs. Mixed builds reuse the hub and prepare one spatial spoke at a
time. Input arrays are not mutated, and no kernels are retained across calls.
The in-memory entry points drop their input references after preparation;
arrays still referenced by the caller remain allocated. Kernels retain XYZ
and the tree, whose coordinate buffer shares the XYZ allocation. Resolution
uses row indices and separations and retains no coordinates or trees.

## Rows and ordering

All matchers return ordinary `pyarrow.Table` objects. `<survey>/row_index`
columns contain int64 input positions; null means absent membership. They are
meaningful only with the exact corresponding input order. Dedupe keeps the
lowest input row, not the lowest caller ID.

Spatial entities are ordered by their first present survey in input survey
order, then that survey's row. UID joins follow anchor order and append unseen
outer-join keys in remaining source/input order. Hub-and-spoke follows anchor
order. UID and hub-and-spoke results also contain a consecutive `entity_id`;
it is not a persistent identifier across independently rebuilt inputs.

`rows_to_ids` preserves Arrow types, null membership and schema metadata. It
does not validate caller IDs for uniqueness or scientific meaning. Supply one
ID per original input row in the same order. A saved row index alone cannot
identify a reordered catalogue: retain the source revision and row-order
provenance alongside it, or map to your own persistent IDs.

`ids_to_rows` is the inverse: it turns a match table's ID columns back into
row positions, given the same per-row IDs, which must then be unique and
non-null. Its `id_columns` names whole columns (`{"s": "key"}` reads column
`key`), so it can read match tables built elsewhere; an ID missing from the
lookup raises instead of becoming null.

## Saved files

Index tables carry their matching settings, dedupe outcomes and summary in
Arrow schema metadata, which Parquet preserves. Saved edge `metadata.json`,
band `layout.json` and `band-build.json` hold the settings their readers need.

Saved files carry no format or package version, and readers do not check one.
A release that changes a saved layout says so in the changelog; rebuild the
files with that release rather than reading older ones.
