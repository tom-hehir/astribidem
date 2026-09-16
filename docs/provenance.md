# Provenance

The initial extraction preserves the MIT notice, Copyright (c) 2026 Tom Hehir.

- Astral: https://github.com/PolymathicAI/astral-projections at
  `9b61fd3cedc8a61b26bdb2d65df2b4cab14d9ada`.
  Source: `packages/toolkit/src/astral_projections/toolkit/data/crossmatch/`.
  Kernel, graph, inputs and mode builders are retained with import/metadata
  namespace changes. The storage-independent portions of edges.py are extracted.
  Existing kernel, mode and dedupe tests retain their scientific assertions.
- AstroBench: https://github.com/PolymathicAI/astrobench at
  `c3e3b8aeaef26e1ad11d4dbabd6030cf74332100`.
  Source: `src/astrobench/datasets/crossmatch/geometry.py` and
  `tests/crossmatch_geometry/`. Imports point at the standalone primitives.

The new array-facing API owns input validation and in-memory orchestration.
No AION-2, Astral catalog loaders, HATS materializers or training components are
runtime dependencies. Upstream repositories were not changed by this extraction.

One inherited empty-survey entity-resolution defect was corrected: an entirely
absent survey now produces typed null IDs rather than indexing its empty ID array.
