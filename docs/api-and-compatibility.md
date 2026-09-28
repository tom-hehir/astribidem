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
| Match positions | `survey_coords_from_arrays`, `SurveyCoords`, `crossmatch` |
| Choose association policy | `DegenerateCrossmatchConfig`, `EntitywiseCrossmatchConfig`, `EntitySelectionConfig`, `CrossmatchModeConfig` |
| Match shared identifiers | `match_uids`, `match_hub_and_spoke`, `UIDLink`, `SpatialLink` |
| Use results | `rows_to_ids`, `index_summary` |
| Reuse candidate searches | `build_edges`, `resolve`, `audit_edges`, `CandidateEdges`, `PairEdges`, `DedupeOutcome` |
| Save candidate searches | `write_edges`, `read_edges`, `build_edges_to_directory` |
| Build and resolve large catalogues | `write_band_layout`, `build_edges_by_band`, `resolve_to_file` |
| Coordinate separate workers | `prepare_band_build`, `build_band`, `sweep_boundaries`, `read_segment`, `write_segment`, `segment_names`, `combine_segments` |
| Query the geometric kernel | `CatalogKernel` |
| Inspect the installed release | `__version__` |

The worker/segment and kernel interfaces are advanced and may evolve during
alpha. Use the high-level functions unless you need their explicit controls.
The documented `astribidem.geometry` array adapter and
`astribidem.banded.as_chunks` helper are also available. Other module helpers,
underscore-prefixed names and intermediate files are implementation details.

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

## Saved formats

Package versions and saved-format versions are independent. Format version 1
is introduced with `0.0.0a0`:

- Index tables carry `astribidem.index_format_version` and `astribidem.version`
  in Arrow schema metadata. Parquet preserves them, along with matching policy,
  dedupe outcomes and summaries where those apply.
- Saved edge `metadata.json`, band `layout.json` and `band-build.json` carry
  `format`, integer `format_version`, and `astribidem_version` fields.
- Edge/layout/worker readers reject missing or unsupported format versions
  before interpreting their files. `index_summary` checks the index format
  version before reading a spatial summary.

Plain `pyarrow.parquet.read_table` remains a general-purpose loader and does
not enforce astribidem compatibility. `rows_to_ids` is also usable with caller
constructed row tables. Consumers interpreting saved index metadata should
check the format version explicitly. Reading data with Arrow does not imply
that its matching semantics are compatible with a newer package.

Unversioned files from private development must be regenerated. Later breaking
storage changes will increment the format version and be documented; alpha
releases need not provide migration readers. A producer package version is
provenance, not by itself a compatibility check.
