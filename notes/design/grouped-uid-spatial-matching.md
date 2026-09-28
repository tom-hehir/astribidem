# Grouped UID and spatial matching

Status: deferred possible future extension, recorded 2026-09-20 and clarified
2026-09-21; not implemented or scheduled. This preserves the behavior discussed
for a more general UID/spatial composition if a concrete use case needs it. The
detailed design is provisional, not the contract of the existing
`match_hub_and_spoke` API. No AION-2 integration is planned.

## Motivation and scope

Current UID/spatial composition uses hub-and-spoke matching: every configured
counterpart must match one anchor through a UID or spatial link. Pure UID
matching is already more flexible through `match_uids`, which supports N-source
inner, left and outer joins without positions. The deferred extension here is
specifically about combining UID identity with richer spatial resolution.

The proposed structure more closely follows the existing all-pairs and
entitywise spatial schemes. It first groups rows sharing a trusted UID, selects
one representative position per object, and then applies a chosen spatial policy
between those objects. It needs no universal anchor and can preserve partial
original-dataset memberships. Other positions within a UID object do not impose
additional spatial consistency tests.

This broader structure may be useful when independent anchor links are too
restrictive, but no current consumer has established that need. Keep the
existing hub-and-spoke composition as the supported approach and implement this
extension only when a concrete workflow requires it.

This belongs in the general catalog-matching package. AION-2 already
materializes multiple modalities from the same ordered source rows into one wide
catalog table and spatially matches that catalog once. No current AION workflow
was identified that requires this extension. Preserve the design here until a
concrete consumer justifies implementation. General UID and mixed matching are
intentionally not being added to AION-2.

## Current APIs are different

- [`match_uids`](../../docs/uid-matching.md) already supports N-source inner, left and
  outer joins of exact integer or string keys. It requires unique, non-null keys
  and preserves separate typed source-observation IDs.
- [`match_hub_and_spoke`](../../docs/hub-and-spoke-matching.md) composes independent
  links through one explicit anchor. Each link can be UID or spatial, including
  all-UID and all-spatial configurations. Every link sees its full input
  catalogs; results are combined by inner intersection. It does not construct
  the grouped representative objects described here, support coordinate-source
  priority, or emit partial matches.
- The main spatial `crossmatch` API supports all-pairs subset joins and
  entitywise `refuse`, `sequential` and `split` resolution. This proposal would
  let UID-defined objects participate in those spatial strategies through their
  representative positions; that integration is not currently implemented.

Existing public behavior remains unchanged by this proposal. In particular,
sources used only in UID links in `match_hub_and_spoke` may omit coordinates,
whereas the grouped mixed design below deliberately requires them for every
dataset.

## Dataset grouping

Declare an ordered list of datasets and explicitly declare only the sets that
share a UID system. Each dataset belongs to at most one such set and supplies
one UID column for that set. Datasets not in a declared set automatically
participate as separate spatial inputs; no singleton-set declarations are
required.

For example, with datasets `[A, B, C, D, E]` and declared UID sets `{A, B}` and
`{D, E}`, the spatial inputs are `{A, B}`, `C`, and `{D, E}`. Within `{A, B}`,
each distinct UID identifies one object containing at most one retained row from
A and one from B. Use the union of member UIDs when constructing these objects
so partial objects remain available for spatial decisions.

UID values are comparable only within their declared shared identifier system.
Two independently numbered catalogs having the same numeric value does not
establish identity. Preserve the current exact integer/string rules and null
rejection. Composite keys, multiple UID systems within one dataset, arbitrary
pair-specific key graphs, and automatic UID-to-spatial fallback are outside this
initial design.

## Duplicate UIDs

Resolve duplicate keys within each input dataset before UID grouping:

| Policy      | Behavior                                                      |
| ----------- | ------------------------------------------------------------- |
| `raise`     | Default: reject duplicate UID values.                         |
| `warn_keep` | Warn and retain the first occurrence in original input order. |
| `warn_drop` | Warn and remove all occurrences of each duplicated UID.       |

Dropping copies in one dataset does not remove the same UID from other datasets.
Preserve the original observation IDs of retained rows; do not renumber
observations after filtering. These policies avoid one-to-many and many-to-many
expansion. They concern shared UID keys and do not relax the requirement that
source-observation IDs uniquely identify input rows.

Only rejection exists in the current package. The two warning policies are part
of this proposed extension; they were inspired by AstroBench's existing UID
duplicate handling.

## Representative coordinates

Each UID-defined object supplies exactly one position for spatial matching. Take
both coordinate components from the first dataset present for that object in the
effective coordinate-priority order. Other members' positions do not add
distance tests or veto the UID association. Do not average positions.

Coordinate priority is independently configurable. When omitted, it defaults to
the user-declared dataset order. Resolve that default before execution; an
optimized internal lookup order must not change representative positions. For
priority `[B, A, C]`, an object present in A and B uses B, while an object
present only in A uses A. The priority applies within each declared UID set.

Every input dataset must provide valid coordinates when the run includes spatial
matching. Do not silently skip a coordinate-free member or fall back past an
invalid position. Coordinates remain optional for a pure UID run. Catalog
cleaning and conversion into a common coordinate frame remain caller
responsibilities.

Dataset processing order, coordinate priority, and a spatial resolver's
scientifically meaningful priority are distinct concerns. Execution may be
optimized only when associations and representative choices remain unchanged.

## Spatial association and retention

Treat each UID-defined object as an indivisible spatial unit. Apply the chosen
existing spatial policy between the grouped inputs and ungrouped datasets.
Grouping does not select hub-and-spoke topology or weaken an all-pairs policy.
If a spatial policy permits reusing an observation, the associated UID object is
reused with it; grouping alone does not promise exactly-once output.

Configure radii between spatial inputs, such as `{A, B}` and C. The radius is
the same whether a particular `{A, B}` object uses A's position or B's. There
are no per-member radius overrides within that spatial relationship.

All UID objects participate before final completeness filtering. A row lacking
an optional UID companion must remain a spatial competitor. Removing it early
could turn an ambiguous association into an apparently unique one. Existing
spatial deduplication and ambiguity policies act on whole UID objects and must
not split an established UID association. A configured drop disposition can
still discard an entire object.

After spatial resolution, expand membership to the original datasets and apply
the requested coverage requirements. For datasets A, B and C:

| Available rows | Require A, B and C | Require A and C | Require at least two datasets |
| -------------- | ------------------ | --------------- | ----------------------------- |
| A, B, C        | keep               | keep            | keep                          |
| A, C           | drop               | keep            | keep                          |
| A, B           | drop               | drop            | keep                          |

The temporary presence of spatial input `{A, B}` does not imply that both
original datasets contributed data. Selection counts original datasets, not
collapsed inputs. Retention does not resurrect observations discarded by the
configured duplicate or spatial policies.

## Data and performance boundary

Carry source-observation references through grouping and expansion; do not copy
payload tables. Results should make the effective coordinate priority, chosen
coordinate source, original memberships, duplicate disposition, radii and
spatial policy traceable. Exact public configuration and provenance field names
remain implementation details; the names in this note are conceptual.

UID equality is transitive, so N-source UID grouping does not require checking
every source pair or creating a clique of UID edges. Group by keys or use native
lookup maps, then build output memberships. Angular proximity is not transitive
and cannot use that shortcut. Preserve the package's native Arrow key handling
and exact integer identity; benchmark alternatives before adding new matching
machinery. Existing in-memory scope remains the starting point; out-of-core
processing is separate work.

## Validation required before implementation can be accepted

- Coordinate priority and its declared-order default select the expected member,
  including objects absent from higher-priority datasets.
- Execution-order optimizations leave coordinates and memberships unchanged.
- Optional UID companions do not remove spatial competitors.
- Spatial uncertainty preserves UID associations as complete units.
- Radii depend on grouped inputs, not the selected coordinate-source member.
- Coverage counts original datasets after expansion.
- Duplicate policies preserve first-occurrence ordering and original IDs;
  `warn_drop` removes every copy only within its source.
- Pure UID runs accept coordinate-free inputs; mixed runs validate coordinates
  for all datasets before matching.
- Exact integer boundaries, string identity, null handling, empty inputs and
  deterministic results retain the existing UID guarantees.

This note records behavior and ownership only. It does not add a public API,
change the existing hub-and-spoke matcher, or schedule an implementation.
