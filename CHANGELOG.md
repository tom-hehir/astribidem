# Changelog

## 0.0.0a1 — 2026-10-19

Breaking changes, with no compatibility shims; rebuild saved edges and
indexes with this release.

- Saved files and index tables no longer carry format or package versions,
  and readers no longer check them. A `0.0.0a0` reader rejects files written
  by this release.
- A pair of surveys is written `a+b`: separation columns are
  `<a>+<b>/separation_arcsec` and saved edge files `edges/<a>+<b>.parquet`.
  Survey names may not contain `+` (nor `/`).
- `match_hub_and_spoke(anchor=, links=, coordinates=, uids=, ...)` takes the
  `(ra, dec)` mapping of `crossmatch` and the identifier mapping of
  `match_uids` instead of Arrow tables with `ra`/`dec`/`uid`/`id` columns.
  Output columns are the anchor's, then each catalogue's in `links` order.
- `crossmatch`, `build_edges` and `build_edges_to_directory` call their first
  parameter `coordinates` (was `surveys`).
- `astribidem.geometry` is removed: use `crossmatch` (all-pairs subset joins),
  `match_hub_and_spoke` (anchor-pairs), `CatalogKernel.all_pairs` and
  `one_to_one_pairs`. `CatalogKernel.from_xyz` is removed, and
  `CatalogKernel.all_pairs` no longer takes `chunk_rows` (`all_pairs_chunks`
  does).
- The band build and segment-wise resolution move to
  `astribidem.out_of_core` (`banded`, `regions`, `index_files`); root exports
  are unchanged. The `large` extra is renamed `out-of-core`.
- `combine_segments` and `write_segment` are no longer exported from the root.
- `rows_to_ids(id_columns=)` gives whole column names, as `ids_to_rows` does:
  `{"s": "s/objid"}` writes `s/objid` (was `{"s": "objid"}`). Survey names in
  `id_columns` that are not in `ids` raise.
- `DegenerateCrossmatchConfig.surveys` is required. Plain-value checks on
  mode settings are gone (for example a `size_cap` below one); unknown names,
  ambiguous overrides and unknown policy values still raise. A matching
  radius that is not positive and finite still raises, at every entry point
  (`SpatialLink`, `CatalogKernel.all_pairs`, `crossmatch`): without the check,
  zero matched nothing and a negative radius matched as if it were positive.
  Zero still disables dedupe.

Added

- `ids_to_rows`, the inverse of `rows_to_ids`, moved from astribidem-hf.
- Root exports `one_to_one_pairs`, `read_edges_metadata` and `as_chunks`.

Fixed

- A banded build of catalogues with no rows no longer crashes, and
  `read_edges` and `resolve_to_file` handle saved edges with no segments.

Development

- Test and development tools are dependency groups (`tests`, `dev`) rather
  than published extras. Ruff checks `E, F, I, B, UP`. Process-starting
  tests are marked `slow`, and warnings from astribidem fail tests.
- Benchmarks and the Acero experiment were removed from `main`; they remain
  at commit `47a1a14`.

## 0.0.0a0 — 2026-09-28

First public alpha. The earlier `0.1.0a1` metadata was
used only during private development and was never published.

- Spatial, exact UID and hub-and-spoke matching returning Arrow row indexes.
- Spatial inputs are name-to-`(ra, dec)` mappings, converted at kernel creation.
- Scalar radius defaults with `radius_arcsec_overrides` for pairs and
  `dedupe_radius_arcsec_overrides` for surveys.
- Each mixed match prepares the hub kernel and dedupe result once and releases
  spoke kernels between links; calls remain independent.
- Explicit deduplication, pair and entity-resolution policies, with optional
  mapping from input rows to caller IDs.
- Reusable candidate edges, saved segments and banded spatial processing.
- Versioned saved formats and producer version metadata. Unversioned private
  development files must be regenerated.
- Reject nonfinite coordinates and declinations outside [-90, 90].
- Require DuckDB 1.5 for sorted segment resolution; SciPy starts at 1.11.1.
- CI checks formatting, built distributions and optional installation boundaries
  on Ubuntu with Python 3.14.
- Publish tested distributions to PyPI through Trusted Publishing on GitHub release.

This is an alpha interface. Pin the version and consult
[API and compatibility](docs/api-and-compatibility.md) before upgrading.
