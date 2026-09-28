# Choosing a matching policy

Use astribidem when you need explicit choices about duplicate observations,
ambiguous associations or membership across several catalogues. Results contain
row positions and metadata; retrieving payloads remains the caller's job.

## Scientific assumptions

Pass aligned, finite RA/Dec arrays in degrees, preferably float64. Declination
must lie in [-90, 90]; both poles are allowed. RA may wrap outside [0, 360).
Every spatial participant must already share the appropriate celestial frame
and epoch. The package does not transform frames, propagate proper motions,
interpret astrometric uncertainties, or estimate association probabilities.
Apply catalogue-specific quality cuts and remove sentinel values upstream:
(0, 0) is a valid sky position and is not automatically discarded.

Matching computes in float64. Float32 or float16 inputs trigger a warning
because their rounding already limits positional accuracy: float32 RA is
spaced up to 0.11 arcsec apart near 360 degrees. Converting a rounded input to
float64 cannot recover that precision.

An exact candidate search emits the complete inclusive-radius set under the
floating-point spherical geometry. It does not establish that nearby rows are
the same physical source. Radius and resolution policy are scientific choices.
The candidate search is not truncated to the nearest neighbour before policies
that need every candidate are applied.

Set a scalar `dedupe_radius_arcsec` for all surveys; zero disables dedupe.
Supply exceptions with `dedupe_radius_arcsec_overrides={"a": 0.5}`. The
matching radius follows the same pattern: `radius_arcsec` and
`radius_arcsec_overrides={("a", "b"): 2.0}`. Both defaults are required;
overrides are optional dictionaries and unknown survey names fail. A
duplicate group whose members all lie within the dedupe radius keeps its lowest
input row. A connected chain that is not such a group is disputed and excluded
from pair matching. Sort inputs first if you want a preferred observation to
survive; preserve that order when mapping rows to IDs.

## Pair policies

For two surveys, pass `DegenerateCrossmatchConfig(surveys=["a", "b"],
policy=...)` to `crossmatch`. Survey `a` is the anchor.

| Policy | Accepted association |
| --- | --- |
| `anchored_nearest` | Each anchor's nearest in-radius candidate; a counterpart can be reused. |
| `anchored_unique` | An anchor with exactly one in-radius candidate; a counterpart can be reused. |
| `mutual_nearest` | The two rows select each other as nearest candidates. |
| `mutual_unique` | Both rows have exactly one in-radius candidate. |

For example, A has rows at offsets 0 and 0.8 arcsec, and B has one row at 0.2
arcsec. Dedupe is disabled. Both anchored policies produce two associations;
mutual-nearest keeps only A's first row with B; mutual-unique keeps neither
because B has two candidates.

```python
from astribidem import crossmatch, DegenerateCrossmatchConfig

surveys = {
    "a": ([10.0, 10.0 + 0.8 / 3600], [0.0, 0.0]),
    "b": ([10.0 + 0.2 / 3600], [0.0]),
}
for policy in (
    "anchored_nearest",
    "anchored_unique",
    "mutual_nearest",
    "mutual_unique",
):
    index = crossmatch(
        surveys,
        radius_arcsec=1.0,
        dedupe_radius_arcsec=0.0,
        mode=DegenerateCrossmatchConfig(surveys=["a", "b"], policy=policy),
    )
    print(policy, index.num_rows)  # 2, 2, 1, 0
```

For more than two surveys, degenerate joins require mutual agreement between
every pair. Anchored policies in this API accept only two surveys. Use
`match_hub_and_spoke` when each counterpart should instead match one common
anchor without requiring counterparts to match each other.

## Entities and partial memberships

`EntitywiseCrossmatchConfig()` preserves clean groups and can retain unmatched
or disputed rows as singleton entities. Missing survey membership is null.
Its default `resolver="refuse"` declines ambiguous groups; `"sequential"`
resolves using survey priority and `"split"` uses complete-linkage grouping.
Those alternatives encode additional assumptions and can change membership.
`disputed="drop"` discards disputed rows instead of retaining them as flagged
singletons. Apply `EntitySelectionConfig` after resolution to select required
survey combinations. Selection does not hide ambiguity from other surveys.

## Shared identifiers

Use [`match_uids`](uid-matching.md) only when keys refer to the same namespace
across sources. It performs exact equality with inner, left or outer membership,
without any position test. Keys must be unique and non-null within each source.
Equal numbers from unrelated survey-specific ID systems are not shared identifiers.
For mixed spatial/UID links, see [hub-and-spoke matching](hub-and-spoke-matching.md).

## Memory

`crossmatch` and `match_uids` run in memory. Streaming candidate edges to disk
still holds coordinates, trees and dedupe results in memory. Banded spatial
matching reduces this to individual bands and boundary groups; memory depends
on density, candidate count and concurrency, not just catalogue row count.
Choose a band height that fits the densest band and a boundary carry limit.
`resolve_to_file` sorts with DuckDB from `astribidem[large]`; its sort memory
limit does not bound the memory of resolving one segment. See
[candidate edges](candidate-edges.md) for the full workflow.
