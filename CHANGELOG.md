# Changelog

## 0.0.0a0 — unreleased

First public alpha, prepared for release. The earlier `0.1.0a1` metadata was
used only during private development and was never published.

- Spatial, exact UID and hub-and-spoke matching returning Arrow row indexes.
- Explicit deduplication, pair and entity-resolution policies, with optional
  mapping from input rows to caller IDs.
- Reusable candidate edges, saved segments and banded spatial processing.
- Versioned saved formats and producer version metadata. Unversioned private
  development files must be regenerated.
- Reject nonfinite coordinates and declinations outside [-90, 90].
- Require DuckDB 1.5 for sorted segment resolution; SciPy starts at 1.11.1.
- CI checks built distributions, supported Python versions, minimum
  dependencies, operating systems and optional installation boundaries.
- Publish tested distributions through Trusted Publishing on GitHub release;
  manual release runs from version tags target TestPyPI.

This is an alpha interface. Pin the version and consult
[API and compatibility](docs/api-and-compatibility.md) before upgrading.
