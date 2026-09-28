"""Resolve saved edges segment by segment into one index file.

``resolve_to_file`` resolves each segment of a saved edge directory on its own,
optionally in parallel processes, and writes the index as one Parquet file.
Segments never share a connected group of rows, so the entities are exactly
those of resolving every segment together. With ``sort=True`` (the default)
the per-segment indexes, each already in entity order, are merged into the
global entity order: first present survey, then that survey's row. The result
then equals ``resolve(read_edges(directory), mode).table``, metadata included,
while holding only one batch per segment in memory.
"""

from __future__ import annotations

import json
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from astro_crossmatch.api import resolve
from astro_crossmatch.edge_files import read_segment, segment_names
from astro_crossmatch.modes import CrossmatchModeConfig

_ROW_BITS = 40  # row numbers below 2**40; survey positions fill the bits above


def _resolve_segment(arguments) -> None:
    edges_directory, name, mode, path = arguments
    table = resolve(read_segment(edges_directory, name), mode).table
    pq.write_table(table, path)


def _is_count(key: str) -> bool:
    """Per-segment counts that add up; ``n_rows`` is each survey's total."""
    return key != "n_rows" and key.startswith(("n_", "rows_in_"))


def _merged(values: list, key: str | None = None):
    """Add counts across segments; every other value must agree."""
    first = values[0]
    if isinstance(first, dict):
        return {k: _merged([v[k] for v in values], k) for k in first}
    if key is not None and _is_count(key):
        return sum(values)
    if any(value != first for value in values):
        raise ValueError(f"segments disagree on {key!r}: {first!r}")
    return first


def _merged_metadata(metadatas: list[dict[bytes, bytes]]) -> dict[bytes, bytes]:
    merged = dict(metadatas[0])
    for key in (b"astro_crossmatch.dedupe", b"astro_crossmatch.summary"):
        combined = _merged([json.loads(metadata[key]) for metadata in metadatas])
        merged[key] = json.dumps(
            combined, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    return merged


def _entity_keys(batch: pa.RecordBatch | pa.Table, row_columns: list[str]):
    """One sortable integer per entity: first present survey, then its row."""
    keys = np.full(batch.num_rows, -1, dtype=np.int64)
    for position, column in enumerate(row_columns):
        rows = batch[column]
        present = ~np.asarray(rows.is_null())
        if hasattr(rows, "combine_chunks"):
            rows = rows.combine_chunks()
        values = rows.fill_null(0).to_numpy()
        take = present & (keys < 0)
        keys[take] = (position << _ROW_BITS) + values[take]
    return keys


def _merge_sorted(paths: list[Path], writer: pq.ParquetWriter, batch_rows: int):
    """K-way merge of per-segment indexes that are each sorted by entity key."""
    readers = [pq.ParquetFile(path) for path in paths]
    schema = readers[0].schema_arrow
    row_columns = [name for name in schema.names if name.endswith("/row_index")]
    streams = [reader.iter_batches(batch_size=batch_rows) for reader in readers]
    buffers: list[pa.Table | None] = [None] * len(streams)
    keys: list[np.ndarray | None] = [None] * len(streams)

    def refill(i):
        for batch in streams[i]:
            if batch.num_rows:
                buffers[i] = pa.Table.from_batches([batch])
                keys[i] = _entity_keys(buffers[i], row_columns)
                return
        buffers[i], keys[i] = None, None

    for i in range(len(streams)):
        refill(i)
    while any(buffer is not None for buffer in buffers):
        active = [i for i, buffer in enumerate(buffers) if buffer is not None]
        # Every buffered key up to the smallest buffer tail is final: no
        # segment can later produce a smaller key.
        threshold = min(keys[i][-1] for i in active)
        tables, table_keys = [], []
        for i in active:
            count = int(np.searchsorted(keys[i], threshold, side="right"))
            if count:
                tables.append(buffers[i].slice(0, count))
                table_keys.append(keys[i][:count])
                buffers[i] = buffers[i].slice(count)
                keys[i] = keys[i][count:]
            if buffers[i].num_rows == 0:
                refill(i)
        chunk = pa.concat_tables(tables)
        order = np.argsort(np.concatenate(table_keys), kind="stable")
        writer.write_table(chunk.take(order))
    for reader in readers:
        reader.close()


def resolve_to_file(
    edges_directory: str | Path,
    mode: CrossmatchModeConfig,
    index_path: str | Path,
    *,
    processes: int = 1,
    sort: bool = True,
    batch_rows: int = 65_536,
) -> None:
    """Resolve every segment of saved edges and write one index file.

    ``processes`` resolves segments in parallel. With ``sort=False`` the index
    lists each segment's entities in segment order instead of merging them.
    ``batch_rows`` sets how many rows of each segment the merge holds at once.
    """
    edges_directory = Path(edges_directory)
    index_path = Path(index_path)
    names = segment_names(edges_directory)
    work = index_path.with_name(f".{index_path.name}.segments")
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    try:
        paths = [work / f"{name}.parquet" for name in names]
        tasks = [
            (str(edges_directory), name, mode, str(path))
            for name, path in zip(names, paths)
        ]
        if processes > 1:
            with ProcessPoolExecutor(processes) as pool:
                list(pool.map(_resolve_segment, tasks))
        else:
            for task in tasks:
                _resolve_segment(task)
        metadatas = [pq.read_schema(path).metadata for path in paths]
        schema = pq.read_schema(paths[0]).with_metadata(_merged_metadata(metadatas))
        with pq.ParquetWriter(index_path, schema) as writer:
            if sort:
                _merge_sorted(paths, writer, batch_rows)
            else:
                for path in paths:
                    with pq.ParquetFile(path) as reader:
                        for batch in reader.iter_batches(batch_size=batch_rows):
                            writer.write_table(pa.Table.from_batches([batch]))
    finally:
        shutil.rmtree(work, ignore_errors=True)
