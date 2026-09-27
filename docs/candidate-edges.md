# Candidate edges

Crossmatching runs in two stages. The edge build dedupes each survey and finds
every in-radius pair between surveys. Resolution then turns those candidate
edges into one index under a chosen mode. The edge build is the expensive
stage; resolution is a comparatively cheap pass over the edges.

`crossmatch` runs both stages in memory and keeps nothing. Split the stages when
you want to resolve the same edges under several modes, or keep the edges for
later.

## Build and resolve in memory

```python
from astro_crossmatch import (
    DegenerateCrossmatchConfig, EntitywiseCrossmatchConfig, build_edges, resolve,
)

edges = build_edges(
    surveys,
    radius_arcsec=1.0,
    dedupe_radius_arcsec={"a": 0.5, "b": 0.0, "c": 0.0},
)
entities = resolve(edges, EntitywiseCrossmatchConfig())
pairs = resolve(edges, DegenerateCrossmatchConfig(surveys=["a", "b"]))
```

`build_edges` returns a `CandidateEdges` object. It holds one `DedupeOutcome`
per survey and one `PairEdges` per survey pair, all in row positions. `resolve`
takes every matching setting from the edges, so the settings are not repeated.

A degenerate mode reads only the pairs between its own surveys. It can
therefore use edges built for more surveys, and gives the same index as a build
over its surveys alone. An entitywise mode uses every survey in the edges,
because another survey's rows can make a component ambiguous.

`build_edges(..., pairs=[("a", "b")])` joins only the named pairs. Such edges
serve only degenerate modes over those pairs; `resolve` rejects any other mode.

## Save and reload

```python
from astro_crossmatch import read_edges, write_edges

write_edges(edges, "edges-directory")
edges = read_edges("edges-directory")
```

Saved edges are made of segments. A segment is a self-contained slice of the
answer: every edge in it joins two of its rows, and no connected group of rows
is split between segments. `write_edges` saves an in-memory build as a single
segment named `segment-0`:

```text
edges-directory/
├── metadata.json                surveys (name, n_rows, dedupe radius), pair radii, segments
└── segment-0/
    ├── deduplication/
    │   ├── a.parquet            row, status, kept_row
    │   └── b.parquet
    └── edges/
        └── a__b.parquet         row_a, row_b, separation_arcsec
```

Each deduplication file lists only the rows that dedupe removed from matching.
A `dropped` row is a duplicate, and `kept_row` is the lowest row of its
duplicate group, which remains active. A `disputed` row belongs to a group that
is not a clique; its `kept_row` is null. Rows that are not listed are active.
Edges are sorted by `(row_a, row_b)`, and row numbers are global.

A segment that covers only some rows also has a `rows/<survey>.parquet` file
listing them; a segment without a `rows` directory covers every row.
`read_segment(directory, name)` loads one segment, `segment_names(directory)`
lists them, and `read_edges` loads every segment and combines them with
`combine_segments`. Resolving segments separately gives the same entities as
resolving them combined.

## Stream edges to disk

When the edges are too large to hold in memory, write them while they are
found:

```python
from astro_crossmatch import build_edges_to_directory

build_edges_to_directory(
    surveys,
    "edges-directory",
    radius_arcsec=1.0,
    dedupe_radius_arcsec={"a": 0.5, "b": 0.0, "c": 0.0},
    workers=4,
    chunk_rows=1_000_000,
)
edges = read_edges("edges-directory")
```

This writes the same `segment-0` as `write_edges(build_edges(...), ...)`. Each
chunk covers `chunk_rows` rows of the first survey in a pair, and at most
`workers` chunks are in memory at once. By default there is one chunk per
worker, which holds a pair's complete edge set at once, so set `chunk_rows` when
edge memory is the limit. Very small chunks can slow the build.

Streaming bounds only the edges. Every survey's coordinates, KD-tree and dedupe
results stay in memory for the whole build, at roughly 50 bytes per row.
Resolution later loads all edges of the surveys it uses, but no coordinates.

## Audit the edges

```python
from astro_crossmatch import audit_edges

audit = audit_edges(edges)  # a dictionary of plain JSON types
```

The audit works on edges held in memory or reloaded with `read_edges`. It
reports:

- for each survey, its row count, active rows, dropped and disputed rows, and
  the number and largest size of its duplicate groups;
- for each pair, its radius, edge count, separation percentiles, the fraction of
  active rows with at least one candidate, and the rows with several
  candidates;
- a census of connected components over every survey: a size histogram, the
  largest component, clean and ambiguous counts, and counts by survey
  combination.

A growing largest component warns that the radius is close to percolation. A
clean component spanning surveys A and B is one row of a clean A x B product,
so the combination counts size subset joins before building them. The census
omits pairs that were not built; `all_pairs_built` is false in that case.

