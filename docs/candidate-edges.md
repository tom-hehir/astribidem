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
from astribidem import (
    DegenerateCrossmatchConfig,
    EntitywiseCrossmatchConfig,
    build_edges,
    resolve,
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
from astribidem import read_edges, write_edges

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
from astribidem import build_edges_to_directory

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

## Build edges band by band

When coordinates and KD-trees do not fit in memory, build the edges one
declination band at a time. The build first streams every survey's
coordinates into a layout grouped by band:

```python
from astribidem import build_edges_by_band, write_band_layout
from astribidem.banded import as_chunks

write_band_layout(
    "layout-directory",
    {"a": chunks_a, "b": chunks_b},  # iterables of (ra, dec) chunks, in row order
    band_height_deg=0.1,
)
build_edges_by_band(
    "layout-directory",
    "edges-directory",
    radius_arcsec=1.0,
    dedupe_radius_arcsec={"a": 0.5, "b": 0.0},
)
```

`as_chunks(ra, dec, chunk_rows)` splits in-memory arrays into chunks. The layout
costs 24 bytes per row per survey on disk and can be deleted after the build.

Each band task loads its band plus the rows within the largest radius of its
two boundaries. It saves the groups of rows that lie wholly inside the band as
segment `band-<k>` and hands over the groups that reach a boundary. A sweep
over the boundaries, from south to north, completes those groups and saves
them as segments `boundary-<k>`. The saved edges equal `build_edges` on the
same coordinates exactly, and each segment can be resolved on its own.

The band height must exceed the largest pair or dedupe radius. Memory scales
with the rows in the largest band; the share of rows handed to the sweep is
about twice the largest radius divided by the band height. `band_processes`
runs band tasks in parallel processes. `max_carried_rows` makes the sweep fail
loudly when groups span many bands, which happens only when the radius is too
large for the source density.

To resolve large saved edges without loading every segment at once, use
`resolve_to_file`:

```python
from astribidem import EntitywiseCrossmatchConfig, resolve_to_file

resolve_to_file(
    "edges-directory",
    EntitywiseCrossmatchConfig(),
    "index.parquet",
    processes=8,
)
```

It resolves each segment on its own, optionally in parallel processes, and
sorts the per-segment indexes into the usual entity order: first present
survey, then that survey's row. The file equals
`resolve(read_edges(...), mode)`, metadata included.

The sort runs in DuckDB, installed with the `large` extra
(`astribidem[large]`). DuckDB spills to disk beyond `memory_limit`, such
as `"4GB"`, using `threads` threads; when the limit is too small for the
threads, the sort stops with a `MemoryError`. DuckDB could write the sorted
index to Parquet itself, which is faster, but it would choose the column types
and drop the index metadata, so its rows are streamed back and written with
exactly the resolved schema instead. `sort=False` skips the sort, and needs no
DuckDB, listing each segment's entities in segment order.

For separate processes, such as a job array, `prepare_band_build` records the
settings and returns the bands, `build_band(edges_directory, band)` builds one
band, and `sweep_boundaries(edges_directory)` finishes the build once every
band has run.

## Audit the edges

```python
from astribidem import audit_edges

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
