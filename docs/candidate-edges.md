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

The directory contains:

```text
metadata.json                    surveys (name, n_rows, dedupe radius) and pair radii
deduplication/<survey>.parquet   row, status, kept_row
edges/<a>__<b>.parquet           row_a, row_b, separation_arcsec
```

Each deduplication file lists only the rows that dedupe removed from matching.
A `dropped` row is a duplicate, and `kept_row` is the lowest row of its
duplicate group, which remains active. A `disputed` row belongs to a group that
is not a clique; its `kept_row` is null. Rows that are not listed are active.
Edges are sorted by `(row_a, row_b)`.

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

This writes the same files as `write_edges(build_edges(...), ...)`. Each chunk
covers `chunk_rows` rows of the first survey in a pair, and at most `workers`
chunks are in memory at once. By default there is one chunk per worker, which
holds a pair's complete edge set at once, so set `chunk_rows` when edge memory is
the limit. Very small chunks can slow the build.

Streaming bounds only the edges. Every survey's coordinates, KD-tree and dedupe
results stay in memory for the whole build, at roughly 50 bytes per row.
Resolution later loads all edges of the surveys it uses, but no coordinates.
