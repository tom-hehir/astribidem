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
    dedupe_radius_arcsec=0.0,
    dedupe_radius_arcsec_overrides={"a": 0.5, "b": 0.0, "c": 0.0},
)
entities = resolve(edges, EntitywiseCrossmatchConfig())
pairs = resolve(edges, DegenerateCrossmatchConfig(surveys=["a", "b"]))
```

`build_edges` returns a `CandidateEdges` object. It holds one `DedupeOutcome`
per survey and one `PairEdges` per survey pair, all in row positions. `resolve`
takes every matching setting from the edges, so the settings are not repeated.

### Which surveys a mode covers

Both modes take an optional `surveys` list: the surveys the index covers, in
column order. `resolve` reads only those surveys' dedupe outcomes and the pairs
among them. Edges built for more surveys therefore give the same index as edges
built for those surveys alone, so one edge build can serve several indexes over
different surveys. Without `surveys`, a mode covers every survey in the edges,
in the order they were built; that is the usual case when the edges were built
for exactly the surveys you need.

The list matters most for an entitywise mode. Every survey it resolves takes
part in forming entities: another survey's rows can join a component or make
it ambiguous, so adding a survey can change which entities form. Listing the
surveys keeps the others out of resolution entirely. To resolve several surveys
but keep only some memberships, list them all in `surveys` and filter with
`EntitySelectionConfig`, which applies after resolution.

```python
edges = build_edges(surveys, radius_arcsec=1.0, dedupe_radius_arcsec=0.0)
ab_entities = resolve(edges, EntitywiseCrossmatchConfig(surveys=["a", "b"]))
ab_pairs = resolve(edges, DegenerateCrossmatchConfig(surveys=["a", "b"]))
```

`build_edges(..., pairs=[("a", "b")])` joins only the named pairs. Such edges
serve only modes whose surveys' pairs were all built; `resolve` rejects any
other mode.

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
        └── a+b.parquet          row_a, row_b, separation_arcsec
```

Each deduplication file lists only the rows that dedupe removed from matching.
A `dropped` row is a duplicate, and `kept_row` is the lowest row of its
duplicate group, which remains active. A `disputed` row belongs to a group that
is not a clique; its `kept_row` is null. Rows that are not listed are active.
Edges are sorted by `(row_a, row_b)`, and row numbers are global.

A segment that covers only some rows also has a `rows/<survey>.parquet` file
listing them; a segment without a `rows` directory covers every row.
`read_segment(directory, name)` loads one segment, `segment_names(directory)`
lists them, `read_edges_metadata(directory)` returns the saved settings, and
`read_edges` loads every segment and combines them. Resolving segments separately gives the same entities as
resolving them combined.

`read_edges` and `read_segment` take an optional `surveys` list and then read
only those surveys' files and the saved pairs among them, in that order. Pass a
mode's surveys to avoid loading edges saved for many surveys when the mode
covers a few: `resolve(read_edges(directory, mode.surveys), mode)` equals
`resolve(read_edges(directory), mode)`. `resolve_to_file` does this itself.

## Stream edges to disk

When the edges are too large to hold in memory, write them while they are
found:

```python
from astribidem import build_edges_to_directory

build_edges_to_directory(
    surveys,
    "edges-directory",
    radius_arcsec=1.0,
    dedupe_radius_arcsec=0.0,
    dedupe_radius_arcsec_overrides={"a": 0.5, "b": 0.0, "c": 0.0},
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

Streaming bounds only the edges. Every survey's XYZ, KD-tree and dedupe
results stay in memory for the build. Caller-retained RA/Dec arrays also
consume memory; the library drops its input references after preparing trees.
Float64 XYZ occupies 24 bytes per row, in addition to tree structures,
dedupe state, candidate buffers and conversion temporaries.
Resolution later loads all edges of the surveys it uses, but no coordinates.

## Build from trees

Each survey's positions are matched through a `SkyPositionTree`: its unit
vectors and a KD-tree over them, 40 to 50 bytes per row. The builders above
make one per survey from the coordinates. `build_edges_from_trees` and
`build_edges_from_trees_to_directory` take the trees instead, as a mapping of
survey names to trees, and otherwise match `build_edges` and
`build_edges_to_directory` exactly.

Building a tree costs much more than loading a saved one, so trees you will
use again can be saved once and loaded when needed:

```python
from astribidem import SkyPositionTree, build_edges_from_trees_to_directory

for name, (ra, dec) in surveys.items():
    SkyPositionTree(ra, dec, name=name).save(f"trees/{name}.pkl")
trees = {name: SkyPositionTree.load(f"trees/{name}.pkl") for name in surveys}
build_edges_from_trees_to_directory(
    trees, "edges-directory", radius_arcsec=1.0, dedupe_radius_arcsec=0.0
)
```

A saved tree is a pickle, so load only files you saved yourself.

The trees are read from the mapping in a fixed order: each tree once for
dedupe, then the pairs grouped by their first survey, with at most two trees
held at a time. A mapping that loads a tree each time it is read therefore
keeps only two trees in memory, when every survey's tree would not fit at
once; each first survey's tree is loaded once per group of pairs and each
second survey's once per pair. Only the dedupe results, about one byte per
row, stay in memory throughout.

## Build edges band by band

When coordinates and KD-trees do not fit in memory, build the edges one
declination band at a time. The result is the same as an in-memory build, but
the layout and intermediate files cost time: on 20,000 rows per catalogue the
banded route took four to six times as long. Use it only when the in-memory
route does not fit. The build first streams every survey's coordinates into a
layout grouped by band:

```python
from astribidem import as_chunks, build_edges_by_band, write_band_layout

write_band_layout(
    "layout-directory",
    {"a": chunks_a, "b": chunks_b},  # iterables of (ra, dec) chunks, in row order
    band_height_deg=0.1,
)
build_edges_by_band(
    "layout-directory",
    "edges-directory",
    radius_arcsec=1.0,
    dedupe_radius_arcsec=0.0,
    dedupe_radius_arcsec_overrides={"a": 0.5, "b": 0.0},
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

The sort runs in DuckDB, installed with the `out-of-core` extra
(`astribidem[out-of-core]`). DuckDB spills to disk beyond `memory_limit`, such
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

`astribidem.audit` summarises candidate edges, audits one build at smaller
radii or dedupe radii without searching again, and provides the `astribidem`
command line. See [auditing a crossmatch](auditing.md).
