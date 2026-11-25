# Spatial join implementation decisions

Development record. For supported behavior and settings, see
[candidate edges](../docs/candidate-edges.md).

Decision recorded 2026-11-25: keep the tree-against-tree join in
`CatalogKernel`, parallelized by chunking the first survey's rows in row order.
We also looked at two ways of making the chunks spatially compact. Neither is
worth implementing, because catalogs stored in sky order already give compact
chunks.

## The join

Every cross-survey candidate search uses scipy's
[`cKDTree.sparse_distance_matrix`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.sparse_distance_matrix.html)
between two trees, with `output_type="ndarray"`. It walks both trees together
and skips every pair of nodes whose bounding boxes lie farther apart than the
radius, so a whole block of rows is ruled out in one comparison. It runs on one
thread and has no `workers` option. To use several cores, `CatalogKernel`
splits the first survey's rows into contiguous ranges, about one per worker and
at least `MIN_CHUNK_ROWS` each, builds a tree for each range, and joins those
trees against the second survey's full tree in a thread pool; the C join
releases the GIL. Each chunk is sorted by `(i, j)` and the chunks follow row
order, so the result equals a single call bit for bit, and the chunks can be
streamed to disk in order.

scipy offers two point-against-tree queries that run on several threads and need
no tree for the query side. Neither fits:

- [`query`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.query.html)
  with `distance_upper_bound` returns NumPy arrays, but at most `k` neighbours
  per row. Resolution needs the complete candidate set: the entitywise mode
  needs every edge to detect disputes, and `mutual_unique` counts candidates. A
  truncated search is used only on the k-limited path, never for saved edges.
  Re-querying the rows whose `k` slots all came back full would make it
  complete, at the cost of a loop of growing `k`.
- [`query_ball_point`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.query_ball_point.html)
  is complete, but returns one Python list per row (or, with `return_length`,
  only the counts). There is no flat NumPy form, which makes it slow and
  memory-hungry at 10^8 rows.

Tree building is single-threaded too, and in the experiment below it cost as
much as the joins or more. This is why one kernel per survey serves its dedupe
and every pair it takes part in.

## Spatially compact chunks

When rows are in random sky order, each chunk of contiguous rows is spread over
the whole footprint, so its tree's bounding boxes overlap most of the other
survey's tree and the walk prunes less. We considered two ways of making each
chunk cover a compact patch of sky:

- **Take chunks in tree order.** A built tree records its rows in spatial order
  as `cKDTree.indices`, so `xyz[tree.indices[lo:hi]]` is a compact chunk. Each
  chunk's rows must then be mapped back, and the whole result re-sorted by
  `(i, j)`, since the chunks no longer follow row order. That re-sort also
  prevents streaming the chunks to disk in order.
- **Sort the coordinates inside `CatalogKernel`,** for example by HEALPix index,
  and map rows back on output. Contiguous ranges would then be compact, but the
  output needs the same re-sort. We did not time this.

Neither is worth adding. Tree-order chunks were slower than row-order chunks in
the experiment below, because gathering scattered rows and re-sorting cost more
than the better pruning saved. Catalogs stored in sky order, such as HEALPix
order, already have compact row ranges and get the fast case with no change
here: tree builds about twice as fast and the chunked join about five times as
fast as with random order. Callers who control their catalogs' row order should
store them in sky order. The same applies to banded builds, whose regions call
the same chunked join.

## Experiment

Run 2026-11-25 on a laptop with 8 threads, with other work running, so only the
ratios are meaningful. Synthetic catalogs on a 200 deg² patch: 6M rows at a
Legacy Survey-like density, and 16M rows at an HSC-like density that contain
80% of the first catalog's objects jittered by 0.1″. The radius is 1″. Each time
is the best of three runs. Every method produced the same 4,916,546 pairs in
the same order.

| Row order | Tree build, 6M and 16M rows | One call, 1 thread | Row-order chunks | Tree-order chunks |
| --------- | --------------------------- | ------------------ | ---------------- | ----------------- |
| Random    | 4.1 s, 12.8 s               | 9.0 s              | 4.6 s            | 5.4 s             |
| Sky order | 2.4 s, 6.9 s                | 3.2 s              | 0.9 s            | 2.0 s             |

"Sky order" sorts rows into quarter-degree declination strips, then by right
ascension, standing in for HEALPix order.
