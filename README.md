# astro-crossmatch

Standalone astronomical crossmatching, extracted from Tom Hehir’s existing
Astral and AstroBench implementations. Produces typed stable-ID indexes rather
than downloading or joining scientific payloads. Private development repository.

## Install

With GitHub credentials configured for this private repository:

```bash
uv pip install "git+https://github.com/tom-hehir/astro-crossmatch.git"
```

Pin an immutable commit in applications. Runtime dependencies are NumPy, SciPy
and PyArrow. There is no Torch, Lightning, AION, Astral, HATS or LSDB dependency.

## Match coordinate arrays

```python
from astro_crossmatch import (
    crossmatch, survey_coords_from_arrays, DegenerateCrossmatchConfig,
)

sources = [
    survey_coords_from_arrays("a", [101, 102], [10.0, 20.0], [0.0, 0.0]),
    survey_coords_from_arrays("b", [201, 202], [10.0001, 30.0], [0.0, 0.0]),
]
result = crossmatch(
    sources,
    radius_arcsec=1.0,
    dedupe_radius_arcsec={"a": 0.0, "b": 0.0},
    mode=DegenerateCrossmatchConfig(surveys=["a", "b"], policy="mutual_nearest"),
)
print(result.table)  # Stable-ID columns a/id and b/id, plus diagnostics.
```

Dedupe radii are explicit per survey; zero opts out. IDs must be unique and
non-null. Input coordinates must already have catalog-specific cleaning applied.

## Scientific contracts

- The kernel emits the complete inclusive-radius candidate set.
- Pair policies: anchored_nearest, anchored_unique, mutual_nearest, mutual_unique.
- Degenerate subset joins require mutual pairwise agreement for N > 2; anchored
  policies in this interface accept two surveys.
- EntitywiseCrossmatchConfig supports refuse, sequential and split resolvers,
  nullable absent memberships, disputed singleton/drop handling and selection.
- The separate `astro_crossmatch.geometry` array adapter preserves AstroBench’s
  N-way anchor-pairs and all-pairs topology API. Its transient positions are not
  the persisted stable-ID product.
- Per-survey dedupe keeps the lowest stable ID, independent of input row order.

Coordinate arrays and candidate relations are currently held in memory. This
package is not yet a distributed or out-of-core survey processing engine.

## Development

```bash
uv venv
uv pip install -e '.[dev]'
uv run --no-project pytest
uv run --no-project ruff check src tests
```

The inherited tests include independent brute-force and Astropy comparisons,
threshold boundaries, dedupe, disputes, and N-way matching.
See [provenance](docs/provenance.md).
