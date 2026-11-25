# Auditing a crossmatch

The audit summarises a set of [candidate edges](candidate-edges.md): what
dedupe removed, how many candidates each pair found, and how the rows group
into components. Because the edge build is the expensive stage, one build can
also be audited at stricter settings without searching again. Everything here
lives in `astribidem.audit`; the matching code does not depend on it.

## Audit one set of edges

```python
from astribidem.audit import audit_edges

audit = audit_edges(edges)  # an EdgeAudit
audit.components.max_component_size
```

`audit_edges` works on edges built in memory or reloaded with `read_edges`.
It returns frozen dataclasses:

- `audit.surveys`, one `SurveyAudit` per survey: its row count, active rows,
  dropped and disputed rows, and the number and largest size of its duplicate
  groups;
- `audit.pairs`, one `PairAudit` per built pair: its radius, edge count,
  separation percentiles, the fraction of each survey's active rows with at
  least one candidate, and the rows with several candidates;
- `audit.components`, a `ComponentCensus` of the connected components over
  every survey: counts by size, the largest component, clean and ambiguous
  counts, and counts by survey combination.

A growing largest component warns that the radius is close to percolation. A
clean component spanning surveys A and B is one row of a clean A x B product,
so the combination counts size subset joins before building them. The census
omits pairs that were not built; `audit.all_pairs_built` is false in that case.
`dataclasses.asdict(audit)` gives plain JSON types for saving.

## Audit stricter settings from one build

A build finds every pair within its radius, so the edges for a smaller radius
are those with a smaller separation. `restrict_radius` takes them, and gives
the same edges as a fresh build at that radius:

```python
from astribidem.audit import restrict_radius

edges = build_edges(coordinates, radius_arcsec=2.0, dedupe_radius_arcsec=0.5)
for radius in (0.5, 1.0, 1.5, 2.0):
    audit = audit_edges(restrict_radius(edges, radius))
```

`radius_arcsec_overrides` holds named pairs at their own radius, as in
`build_edges`. A radius larger than the built one is refused.

Dedupe needs more than the usual build, which keeps neither the pairs within
a survey nor the edges of the rows dedupe removes. `build_edges_with_duplicates`
takes the arguments of `build_edges` and builds with dedupe off, returning the
edges and each survey's pairs within its dedupe radius. `deduplicate` then
applies any dedupe radius up to that one:

```python
from astribidem.audit import build_edges_with_duplicates, deduplicate

edges, duplicates = build_edges_with_duplicates(
    coordinates, radius_arcsec=2.0, dedupe_radius_arcsec=1.0
)
for dedupe_radius in (0.0, 0.5, 1.0):
    deduped = deduplicate(edges, duplicates, dedupe_radius)
    for radius in (0.5, 1.0, 2.0):
        audit = audit_edges(restrict_radius(deduped, radius))
```

Each derived set of edges equals `build_edges` at its settings, so it can also
be passed to `resolve`. The edges without dedupe include the duplicates'
candidates, so they are somewhat larger than a deduped build's. This route
works in memory only.

## Command line

Installing astribidem adds an `astribidem` command. It prints formatted
tables; `--save PATH` also writes the result as JSON.

```bash
astribidem audit edges-directory
```

audits a saved edge directory. `--radius R` narrows every pair to `R`
arcseconds first, and `--radius-override a+b=R`, which can be repeated, sets
one pair's radius; pairs not named keep their built radius.

```bash
astribidem audit-sweep edges-directory --radius 0.5 1 1.5 2
```

reads the edges once and audits them at each radius, printing one table row
per radius. A pair given with `--radius-override` stays at that radius
throughout. The command line sweeps only the match radius: saved edges do not
keep the within-survey pairs that a dedupe sweep needs.

```bash
astribidem summary index.parquet
```

prints an index file's row count, columns, and the settings, dedupe outcomes
and summary recorded in its metadata. It reads only the Parquet footer, so it
is quick on any size of index.
