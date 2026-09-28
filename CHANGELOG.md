# Changelog

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
