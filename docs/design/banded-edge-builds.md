# Banded edge builds for catalogs larger than memory

Status: agreed design, recorded 2026-09-27; not yet implemented. This note
records the plan for building candidate edges and resolving them when survey
catalogs do not fit in memory, together with every option considered and the
reason each was accepted or rejected. The implementation stages at the end
describe the intended order of work.

## The problem

The edge build currently holds every survey's coordinates and KD-tree in memory
for the whole build, which costs about 50 bytes per row per survey. Streaming
edges to disk with `build_edges_to_directory` bounds only the memory used by
the edges. The memory floor therefore grows with the catalogs:

| Rows per survey | Coordinates and tree per survey | Three surveys |
| --------------- | ------------------------------- | ------------- |
| 10⁸             | 5 GB                            | 15 GB         |
| 10⁹             | 50 GB                           | 150 GB        |
| 10¹⁰            | 500 GB                          | 1.5 TB        |

Resolution adds its own cost: `resolve` loads every edge and builds a graph
over every active row. At 10⁹ rows and several surveys the build needs a very
large node, and at 10¹⁰ rows it does not fit on one node at all.

## What is local and what is global

Only one step needs coordinates: finding pairs of rows within a radius, both
between surveys and within a survey for dedupe. Every later step works on
pairs of row numbers.

| Step                                       | Needs coordinates | Depends on                              |
| ------------------------------------------ | ----------------- | --------------------------------------- |
| Pairs within a survey's dedupe radius      | yes               | rows within the dedupe radius           |
| Pairs within each survey pair's radius     | yes               | rows within the pair radius             |
| Dedupe groups and verdicts                 | no                | one dedupe group                        |
| Entity groups, clean or ambiguous          | no                | one connected component                 |
| Resolvers (`refuse`, `sequential`, `split`) | no               | one connected component                 |
| Degenerate policies                        | no                | each row's own candidate pairs          |
| Selection and summary counts               | no                | each entity                             |

Every step after pair finding therefore depends only on one connected group of
rows. Splitting the sky into pieces gives exactly the in-memory result
whenever every group is handled completely by exactly one worker. Groups can
be long chains, so they can cross any boundary, and handling them is the heart
of the design.

## The plan

### Overview

1. A **regrouping pass** writes every survey's row number, RA and Dec into a
   scratch file grouped by declination band.
2. A **band task** for each band loads that band's rows plus a margin of rows
   from the neighbouring bands, finds pairs, and separates groups that lie
   wholly inside the band from groups that touch the margin. The first kind is
   finished and saved; the second kind is handed over.
3. A **boundary sweep** processes the handed-over groups one band boundary at a
   time, from south to north, and saves them once they are complete.
4. `resolve` processes each saved segment independently, and an optional merge
   puts the index into a defined order.

### Declination bands

A band is the part of the sky between two declinations, covering every RA.
Bands have a fixed height set by the user. The key property is that two points
are never closer on the sky than their difference in declination. A row
within the largest radius `r_max` of another row therefore lies within `r_max`
in declination, so the margin a band needs from each neighbour is exactly the
rows within `r_max` of the shared boundary. No sphere geometry is needed, RA
wrap-around does not arise, and each band has only two neighbours.

The planner rejects a band height smaller than `r_max`, because a margin could
then reach past the neighbouring band. Heights in practice are around a tenth
of a degree or more, far larger than radii of 0.1–1 arcsec.

This partitioning follows the Zones algorithm, which splits the sky into
declination zones of fixed height and matches each zone against itself and its
neighbours ([Gray, Nieto-Santisteban and Szalay 2007](https://arxiv.org/pdf/cs/0701171)).
A parallel version of it crossmatched SDSS DR3 against 2MASS across several
database servers
([Nieto-Santisteban et al.](https://esto.nasa.gov/conferences/nstc2007/papers/Nieto-Santisteban_Maria_A10P2_NSTC-07-0074.pdf)).

### The regrouping pass

The regrouping pass reads each survey's coordinates once, in row order, and
writes `(row, ra, dec)` with the rows grouped by band. Each band is found as
`floor((dec + 90) / band_height)`. The pass exists so that a band task reads
only its own band and the edges of its two neighbours. Without it, every band
task would scan the whole catalog, and the total reading would grow with the
number of bands: at 10⁹ rows and 500 bands that is about 12 TB instead of
about 72 GB.

The scratch file costs 24 bytes per row per survey, which is 24 GB at 10⁹ rows
and 240 GB at 10¹⁰ rows. It sits on disk, is needed only during the build, and
can be deleted afterwards. Coordinates stay float64 throughout, following the
rule that received coordinates become float64 before any calculation.

### Band tasks

A band task loads, for every survey, the rows the band owns and the margin
rows within `r_max` of either boundary. It then runs the existing kernel and
graph code on those rows:

1. It finds the dedupe pairs within each survey and forms dedupe groups. A
   dedupe group whose members are all owned by the band is final, and its
   verdict is decided here. A dedupe group that includes a margin row is
   deferred, and its rows are marked as deferred.
2. It finds the pairs between surveys and forms entity groups. An edge joins
   the local graph when both of its rows are final and active, or when either
   row is deferred, because a deferred row's dedupe verdict is not yet known.
3. It classifies each entity group. A group whose members are all owned and
   none deferred is final. Every other group is deferred.
4. It saves the final groups as a segment named after the band, and hands over
   the deferred groups: their owned rows, their dedupe pairs whose lower row is
   owned, and their cross-survey pairs whose first row is owned.

The ownership rules mean every row and every pair is saved or handed over by
exactly one band.

A final group is complete and correct. Every true neighbour of one of its rows
lies within `r_max`, so it is either owned by the band or in the margin. A
margin row or deferred row in the group would have made the group deferred, so
every neighbour is owned and final, and every edge was judged with known
verdicts. A group that truly crosses a boundary has a row within `r_max` of
that boundary on each side, so every band that owns part of it sees a margin
row in it and defers it. Edges through rows later found inactive can only join
deferred groups together; they never make a deferred group look final, and the
sweep recomputes the groups after applying the verdicts.

### The boundary sweep

Each handed-over group records which of its band's boundaries it touches. The
sweep then visits the boundaries in order of declination:

```text
open = nothing
for each boundary k, between band k and band k+1, in order of declination:
    take the groups handed over at boundary k, and the groups still open
    join them into complete groups on row numbers alone
    save the groups that reach no later boundary as segment boundary-k
    keep the groups that reach boundary k+1 open
```

Memory is bounded by one boundary's handed-over groups plus the groups carried
forward. A group is carried only when it is longer than a band's height, which
is rare. The number of rows handed over is roughly `2 r_max / band_height`
times the average group size of 1–3, typically well under 1 %.

In crowded fields, groups grow and chains lengthen. Near percolation a single
group can cross every boundary, and neither this build nor the in-memory build
can bound its memory. The sweep therefore fails loudly when the rows it
carries forward exceed a memory budget, with an error explaining that the
radius is too large for the density. The audit's largest-component figure
warns about this before it happens.

### Saved edges: segments

Saved edges consist of segments. A segment is a self-contained slice of the
answer: every edge in a segment joins two rows listed in that segment, and no
group is split between segments. Each segment has its own directory beside
`metadata.json`, which lists the segments:

```text
edges-directory/
├── metadata.json                  surveys, row counts, radii; segments: ["band-0", "band-1"]
├── band-0/
│   ├── rows/
│   │   ├── a.parquet              the rows of a that this segment covers
│   │   └── b.parquet
│   ├── deduplication/
│   │   ├── a.parquet              row, status, kept_row
│   │   └── b.parquet
│   └── edges/
│       └── a__b.parquet           row_a, row_b, separation_arcsec
└── band-1/
    └── ...
```

A banded build writes `band-0`, `band-1`, … and `boundary-0`, `boundary-1`, ….
An in-memory build writes a single segment, `segment-0`, whose `rows/`
directory is left out because it covers every row. Row numbers are global in
every segment. There is one format for every kind of build, so every reader has
a single code path.

### Resolution and index order

`resolve` runs on each segment independently, and segments can be resolved in
parallel. Because every group lies within one segment, the combined result is
the same set of entities as resolving everything at once.

Entities are ordered by their first present survey, in the configured survey
order, and then by that survey's row. Degenerate mode already uses this order,
because every entity contains the first survey. Entitywise mode currently lists
clean groups, then ambiguous rows, then dedupe-disputed rows, and adopts the
same rule instead. Each segment's output is already in this order for its own
entities, so a banded build restores the global order by merging segments, not
by sorting the whole index. The merge is on by default and can be turned off.

### Inputs

The banded build reads only the regrouped coordinates, so any input that can
stream coordinates in row order can feed it:

- HATS catalogs on Hugging Face stream through hf-crossmatch's coordinate scan
  in shard order. Margin catalogs are never scanned, because they duplicate
  rows.
- AION-2 Lance token tables stream coordinates in fragment order, so row
  numbers equal the Lance row offsets that `<catalog>/row_index` already means.
- Local Parquet files stream row groups in file order.

hf-crossmatch currently builds an in-memory table of IDs, coordinates and file
locations during its scan. Using the banded build on catalogs of 10⁹ rows
needs that scan to write coordinates into the regrouped file and IDs and
locations into an on-disk table in row order, so that materialization can look
up matched rows by number. That change belongs in hf-crossmatch and follows the
core work.

### Memory and I/O

| Quantity                         | Estimate                                   |
| -------------------------------- | ------------------------------------------ |
| One band task                    | about 100 bytes per row across all surveys, about 8 GB for 2 × 10⁷ rows per survey and four surveys |
| Scratch file                     | 24 bytes per row per survey                |
| Reading during band tasks        | about three times the scratch file         |
| Boundary sweep                   | a few GB even at 10¹⁰ rows, at normal densities |
| Resolving one segment            | a few GB                                   |
| Saved edges                      | about 24 bytes per pair and 8 bytes per row |

### Exactness and testing

The banded build reproduces the in-memory build exactly: the same dedupe
verdicts, the same pairs with the same separations, and the same entities.
Separations come from the same float64 unit vectors and the same inclusive
boundary; kept rows are chosen on global row numbers; and the resolvers'
tie-breaks depend only on the relative order of rows within a group, which
splitting into bands does not change. Only the order of entities changes, and
the ordering rule makes it identical in both builds.

The tests extend `tests/test_aion_parity.py` with a banded source whose band
height is a few arcseconds, so that many groups cross one or more boundaries.
Its dedupe outcomes, pairs and resolved indexes must equal the in-memory build
and the recorded AION-2 indexes, compared after sorting where AION-2's order
differs. Further tests cover random catalogs at several band heights and
densities, and a chain that spans three bands.

## Implementation stages

1. The segment format for saved edges, and the new entity order for in-memory
   builds.
2. Dedupe from saved pairs, and the band-task core that classifies groups as
   final or deferred.
3. The regrouping pass and the boundary sweep.
4. Resolving segments in parallel, and merging their outputs in order.
5. The streamed scan in hf-crossmatch, as a separate pull request.

Lance input follows once the regrouping pass exists, since a Lance table
becomes one more source of streamed coordinates.

## Options considered

### How to split the work

| Option | Outcome | Reason |
| ------ | ------- | ------ |
| Process each region alone, ignoring boundaries | rejected | Pairs across a boundary are lost, which makes ambiguous groups look clean and leaves duplicates split across the boundary. This is a purity failure, not only lost recall. |
| Process each region alone, dropping rows near boundaries | rejected | Chains carry the problem inward, so false clean groups remain, and real objects are discarded. |
| Resolve each region locally using its margin, as LSDB does for pairwise crossmatches ([LSDB crossmatching](https://docs.lsdb.io/en/latest/tutorials/pre_executed/crossmatching.html)) | rejected | This is exact for pair joins, but a group crossing a boundary would be resolved twice, breaking the exactly-once entitywise output. |
| Find pairs per region with a margin, then resolve in one global pass | rejected | This is exact, but the global pass still needs about 1 TB at 10¹⁰ rows. |
| Keep all coordinates in memory, sorted by declination, and build trees, pairs and graphs per band | rejected | This is simpler and fits about 10⁹ rows per survey, but it does not scale to 10¹⁰ rows, which is the purpose of this work. |
| Bands with a margin, deferral of boundary groups, and a boundary sweep | accepted | This is exact and bounds memory everywhere. |

macauff also splits the sky into small regions that are matched separately
([EPCC](https://www.epcc.ed.ac.uk/whats-happening/articles/parallelising-macauff-photometric-catalogue-software-lsstuk-cross-match)).

### Region shape

| Option | Outcome | Reason |
| ------ | ------- | ------ |
| HEALPix pixels | rejected for now | Pixels would reuse the partitioning of HATS inputs and skip the regrouping pass, but they need a HEALPix dependency, margin geometry around irregular pixel boundaries, handling for the mixed pixel orders of HATS, and separate routes for Lance and local Parquet. The band-task core does not depend on the region shape, so HEALPix regions can be added later. |
| Tiles in RA and declination | rejected | RA margins scale with 1/cos(declination) and break down near the poles and at the RA wrap-around; HEALPix is the cleaner choice for two-axis regions. |
| Declination bands | accepted | The margin is one comparison, each band has two neighbours, and no dependency is needed. The cost is that a band draws rows from every RA, which the regrouping pass absorbs. |

### Band sizes

| Option | Outcome | Reason |
| ------ | ------- | ------ |
| Fixed height set by the user | accepted | This is the simplest option. Row counts vary with area, which is proportional to cos(declination), and with survey density, so the height must keep the largest band within memory. |
| Bands holding equal numbers of rows | recorded for later | A counting pass over declination builds a histogram, band edges are placed every `rows_per_task` rows, and rows find their band with `np.searchsorted`. Tasks are balanced, at the cost of an extra pass and a table of band edges. |
| Thin fixed bands grouped into tasks by row count | rejected | Tasks would be balanced without an extra pass, but each task spans several bands, which is more complex than a table of band edges, and thin bands must be at least `r_max` tall. |

### Boundary groups

| Option | Outcome | Reason |
| ------ | ------- | ------ |
| One boundary segment processed in a single pass | rejected | Its memory is unbounded in crowded fields. |
| A sweep over boundaries in order of declination | accepted | Memory is bounded by one boundary's groups plus the groups carried forward, and the result stays exact. The sweep runs in order, but its work is small next to the band tasks. |

### Crowded fields

| Option | Outcome | Reason |
| ------ | ------- | ------ |
| Rely on the audit's largest-component figure alone | rejected | A giant group could still exhaust memory silently. |
| Fail loudly when the sweep carries more rows than a memory budget | accepted | This is one check on a quantity the sweep already tracks. |
| Drop groups larger than a generous size limit during the build | recorded as reasonable | Such groups would never be accepted scientifically, and dropping them bounds memory. It breaks the principle that saved edges are complete and that resolution makes every decision, so it is not the default. |

### Scratch space

| Option | Outcome | Reason |
| ------ | ------- | ------ |
| 24 bytes per row per survey | accepted | This is small next to the inputs and suits HPC scratch filesystems. |
| float32 coordinates or implicit row numbers | rejected | float32 conflicts with the float64 rule, and implicit row numbers make the layout fragile. |

### Saved edge format

| Option | Outcome | Reason |
| ------ | ------- | ------ |
| One format for every build, with an in-memory build as a single segment | accepted | Every reader has one code path. |
| Separate flat and segmented formats | rejected | Writing both is easy, but every reader would need two code paths. |
| Segment directories inside a `parts/` directory | rejected | The extra level adds nothing, because `metadata.json` already lists the segments. |
| The terms part, chunk, shard, fragment or partition | rejected | "Part" names hf-crossmatch output files, "chunk" names `chunk_rows` streaming, "shard" names hf-crossmatch and HATS files, "fragment" names Lance fragments, and "partition" names HATS and Arrow partitions. "Segment" has no existing meaning in either repository. |
| Numbered segments with their kind recorded in metadata | rejected | Descriptive names (`band-k`, `boundary-k`, `segment-0`) are clearer. |

### Index order

| Option | Outcome | Reason |
| ------ | ------- | ------ |
| First present survey, then its row | accepted | One rule covers both modes and both builds, and each survey's entities then come in ascending row order. |
| By the combination of surveys present, then row | recorded | Each combination forms a contiguous block, but the combinations need their own order, and filtering on non-null row columns already selects a combination cheaply. |
| By entity kind (clean, ambiguous, disputed), then row | recorded | This resembles the current order; filtering on `disputed_reason` already separates the kinds cheaply. |
| By the number of surveys present, then row | recorded | This is a coarser form of ordering by combination. |
| By position on the sky | recorded | `resolve` has no coordinates; hf-crossmatch's `spatial_order` already orders materialized output for payload locality. |
| Restore the current in-memory order exactly | rejected | The current entitywise order has three blocks with different rules, which is complicated to reproduce and serves no consumer. |
| Unordered | available | The merge can be turned off, leaving segment order. |

### Sequencing with hf-crossmatch

| Option | Outcome | Reason |
| ------ | ------- | ------ |
| Build and test the banded core first, then change hf-crossmatch's scan | accepted | Each change stays focused, and the core is checked against the in-memory build before anything depends on it. |
| Change both together | rejected | It would be one much larger change across two repositories. |
