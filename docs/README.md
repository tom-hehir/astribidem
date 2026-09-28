# Documentation

astribidem builds catalogue associations as Arrow tables of row positions.
Start with the [README examples](../README.md), then choose a matching policy
using the [matching guide](matching-guide.md).

- [Matching guide](matching-guide.md): policies, scientific assumptions and an
  ambiguous two-catalogue example.
- [API and compatibility](api-and-compatibility.md): supported entry points,
  ordering, saved formats and the alpha compatibility policy.
- [Candidate edges](candidate-edges.md): reuse candidate searches, save and
  reload edges, and process catalogues in declination bands.
- [Hub-and-spoke matching](hub-and-spoke-matching.md): independent UID/spatial
  associations through a shared anchor.
- [Benchmarks](../benchmarks/README.md): reproducible UID and spatial timings.
- [Development and releases](releasing.md): local checks, CI, TestPyPI and PyPI.
- [Provenance](provenance.md): the extraction from Astral and AstroBench.

The design documents record implementation decisions and deferred proposals;
the guides above describe the supported package.
